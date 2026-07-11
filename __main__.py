from __future__ import annotations

import argparse
import asyncio
import ctypes
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import mss
import numpy as np
from PIL import Image


LOGGER = logging.getLogger("ocr_game_tts")
_POCKET_TTS_LOCK = threading.RLock()


@dataclass(frozen=True)
class HsvRange:
    low: tuple[int, int, int]
    high: tuple[int, int, int]


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int
    area: float

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class DialogueRegion:
    kind: str
    box: Box
    text_crop: Image.Image
    name_crop: Image.Image | None = None


@dataclass(frozen=True)
class Args:
    voice: str
    monitor: int
    region: tuple[int, int, int, int] | None
    image: Path | None
    once: bool
    interval: float
    debug_dir: Path | None
    max_boxes: int
    box_style: str
    min_area: int
    brown_hsv: HsvRange
    padding: int
    confidence_repeat: int
    speak: bool
    local_ocr_min_confidence: float
    warm_up_ocr: bool
    ocr_backend: str
    repeat_cooldown: float
    require_fullscreen: bool
    tts_language: str
    tts_device: str
    stream_tts: bool
    tts_stream_max_tokens: int


def parse_hsv(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected H,S,V")
    try:
        parsed = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("HSV values must be integers") from exc
    if any(component < 0 or component > 255 for component in parsed):
        raise argparse.ArgumentTypeError("HSV values must be between 0 and 255")
    return parsed  # type: ignore[return-value]


def parse_region(value: str) -> tuple[int, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Expected left,top,width,height")
    try:
        left, top, width, height = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Region values must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Region width and height must be positive")
    return left, top, width, height


def parse_args(argv: Sequence[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(
        description="Read game dialogue boxes with local OCR and Pocket TTS."
    )
    parser.add_argument("--voice", default=os.getenv("POCKET_TTS_VOICE", "anna"))
    parser.add_argument("--tts-language", default=os.getenv("POCKET_TTS_LANGUAGE", "english"))
    parser.add_argument("--tts-device", default=os.getenv("POCKET_TTS_DEVICE", "auto"))
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--region", type=parse_region)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--max-boxes", type=int, default=2)
    parser.add_argument("--box-style", choices=("auto", "visual-novel", "brown"), default="auto")
    parser.add_argument("--min-area", type=int, default=8000)
    parser.add_argument("--brown-hsv-low", type=parse_hsv, default=(5, 40, 25))
    parser.add_argument("--brown-hsv-high", type=parse_hsv, default=(30, 255, 210))
    parser.add_argument("--padding", type=int, default=10)
    parser.add_argument("--confidence-repeat", type=int, default=1)
    parser.add_argument("--no-speak", dest="speak", action="store_false")
    parser.add_argument("--local-ocr-min-confidence", type=float, default=0.85)
    parser.add_argument("--no-warm-up-ocr", dest="warm_up_ocr", action="store_false")
    parser.add_argument("--ocr-backend", choices=("windows", "rapidocr"), default="windows")
    parser.add_argument("--repeat-cooldown", type=float, default=30.0)
    parser.add_argument("--require-fullscreen", action="store_true")
    parser.add_argument("--stream-tts", dest="stream_tts", action="store_true")
    parser.add_argument("--no-stream-tts", dest="stream_tts", action="store_false")
    parser.add_argument("--tts-stream-max-tokens", type=int, default=50)
    parser.set_defaults(stream_tts=False)
    parser.add_argument("--verbose", action="store_true")
    namespace = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if namespace.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if namespace.max_boxes < 1:
        parser.error("--max-boxes must be at least 1")
    if namespace.confidence_repeat < 1:
        parser.error("--confidence-repeat must be at least 1")
    if namespace.interval <= 0:
        parser.error("--interval must be positive")
    if namespace.local_ocr_min_confidence < 0 or namespace.local_ocr_min_confidence > 1:
        parser.error("--local-ocr-min-confidence must be between 0 and 1")
    if namespace.repeat_cooldown < 0:
        parser.error("--repeat-cooldown must be zero or greater")
    if namespace.tts_stream_max_tokens < 8:
        parser.error("--tts-stream-max-tokens must be at least 8")

    return Args(
        voice=namespace.voice,
        monitor=namespace.monitor,
        region=namespace.region,
        image=namespace.image,
        once=namespace.once,
        interval=namespace.interval,
        debug_dir=namespace.debug_dir,
        max_boxes=namespace.max_boxes,
        box_style=namespace.box_style,
        min_area=namespace.min_area,
        brown_hsv=HsvRange(namespace.brown_hsv_low, namespace.brown_hsv_high),
        padding=namespace.padding,
        confidence_repeat=namespace.confidence_repeat,
        speak=namespace.speak,
        local_ocr_min_confidence=namespace.local_ocr_min_confidence,
        warm_up_ocr=namespace.warm_up_ocr,
        ocr_backend=namespace.ocr_backend,
        repeat_cooldown=namespace.repeat_cooldown,
        require_fullscreen=namespace.require_fullscreen,
        tts_language=namespace.tts_language,
        tts_device=namespace.tts_device,
        stream_tts=namespace.stream_tts,
        tts_stream_max_tokens=namespace.tts_stream_max_tokens,
    )


def get_capture_monitor(args: Args, screen_capture: mss.mss) -> dict[str, int]:
    if args.region is None:
        monitors = screen_capture.monitors
        if args.monitor < 1 or args.monitor >= len(monitors):
            raise ValueError(f"Monitor {args.monitor} is unavailable. Found {len(monitors) - 1} monitors.")
        return monitors[args.monitor]

    left, top, width, height = args.region
    center_x = left + width // 2
    center_y = top + height // 2
    for monitor in screen_capture.monitors[1:]:
        if (
            monitor["left"] <= center_x < monitor["left"] + monitor["width"]
            and monitor["top"] <= center_y < monitor["top"] + monitor["height"]
        ):
            return monitor
    return screen_capture.monitors[1]


def capture_screen(args: Args) -> Image.Image:
    with mss.mss() as screen_capture:
        if args.region is not None:
            left, top, width, height = args.region
            monitor = {"left": left, "top": top, "width": width, "height": height}
        else:
            monitor = get_capture_monitor(args, screen_capture)
        shot = screen_capture.grab(monitor)
        return Image.frombytes("RGB", shot.size, shot.rgb)


def load_frame(args: Args) -> Image.Image:
    if args.image is None:
        return capture_screen(args)
    return Image.open(args.image).convert("RGB")


def is_foreground_window_fullscreen(args: Args, tolerance: int = 12) -> bool:
    if os.name != "nt":
        return True

    with mss.mss() as screen_capture:
        monitor = get_capture_monitor(args, screen_capture)

    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return False

    class Rect(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    rect = Rect()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return False

    monitor_left = monitor["left"]
    monitor_top = monitor["top"]
    monitor_right = monitor_left + monitor["width"]
    monitor_bottom = monitor_top + monitor["height"]

    return (
        rect.left <= monitor_left + tolerance
        and rect.top <= monitor_top + tolerance
        and rect.right >= monitor_right - tolerance
        and rect.bottom >= monitor_bottom - tolerance
    )



def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def find_brown_boxes(image: Image.Image, args: Args) -> tuple[list[Box], np.ndarray]:
    bgr = pil_to_bgr(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(args.brown_hsv.low), np.array(args.brown_hsv.high))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = image.width * image.height
    boxes: list[Box] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < args.min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue
        aspect_ratio = width / height
        fill_ratio = area / float(width * height)
        area_ratio = area / float(frame_area)
        if aspect_ratio < 1.8 or fill_ratio < 0.35 or area_ratio > 0.65:
            continue
        boxes.append(Box(x=x, y=y, width=width, height=height, area=area))

    boxes = merge_overlapping_boxes(boxes)
    boxes.sort(key=lambda box: (box.bottom, box.area), reverse=True)
    return boxes[: args.max_boxes], mask


def find_visual_novel_dialogues(image: Image.Image, args: Args) -> tuple[list[DialogueRegion], np.ndarray]:
    rgb = np.array(image)
    cream_mask = build_light_dialogue_mask(rgb)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9))
    cream_mask = cv2.morphologyEx(cream_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cream_mask = cv2.morphologyEx(cream_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cream_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = image.width * image.height
    panels: list[Box] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(args.min_area, frame_area * 0.035):
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue
        aspect_ratio = width / height
        fill_ratio = area / float(width * height)
        if aspect_ratio < 3.0 or width < image.width * 0.45 or fill_ratio < 0.35:
            continue
        candidate = Box(x=x, y=y, width=width, height=height, area=area)
        if not has_light_dialogue_fill(rgb, candidate, min_ratio=0.42):
            continue
        panels.append(candidate)

    if not panels:
        lower_region = find_lower_third_dialogue(image)
        return ([lower_region], cream_mask) if lower_region else ([], cream_mask)

    regions: list[DialogueRegion] = []
    for panel in sorted(panels, key=lambda item: item.area, reverse=True)[: args.max_boxes]:
        nameplate = find_nameplate_for_panel(image, panel)
        name_crop = crop_box(image, nameplate, 4) if nameplate else crop_floating_name_label(image, panel)
        regions.append(
            DialogueRegion(
                kind="visual-novel",
                box=panel,
                text_crop=crop_visual_novel_text(image, panel, nameplate),
                name_crop=name_crop,
            )
        )
    if not regions:
        lower_region = find_lower_third_dialogue(image)
        return ([lower_region], cream_mask) if lower_region else ([], cream_mask)
    return regions, cream_mask


def find_lower_third_dialogue(image: Image.Image) -> DialogueRegion | None:
    width, height = image.size
    panel = find_lower_third_panel(image)
    if panel is None:
        return None

    name_crop = image.crop(
        (
            max(0, panel.x + int(panel.width * 0.05)),
            max(0, panel.y + int(panel.height * 0.13)),
            min(width, panel.x + int(panel.width * 0.36)),
            min(height, panel.y + int(panel.height * 0.36)),
        )
    )
    text_crop = image.crop(
        (
            max(0, panel.x + int(panel.width * 0.04)),
            min(height, panel.y + int(panel.height * 0.31)),
            min(width, panel.x + int(panel.width * 0.90)),
            min(height, panel.y + int(panel.height * 0.84)),
        )
    )
    if text_crop.width <= 0 or text_crop.height <= 0:
        return None
    return DialogueRegion(
        kind="lower-third",
        box=panel,
        text_crop=text_crop,
        name_crop=name_crop,
    )


def find_lower_third_panel(image: Image.Image) -> Box | None:
    rgb = np.array(image)
    height, width = rgb.shape[:2]
    lower_y = int(height * 0.68)
    roi = rgb[lower_y:, :]

    # Large FFXIV-style lower dialogue panels are warm cream, not just any
    # bright low-saturation screen area.
    mask = build_light_dialogue_mask(roi)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Box] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < width * height * 0.08:
            continue
        x, y, panel_width, panel_height = cv2.boundingRect(contour)
        y += lower_y
        if panel_width <= 0 or panel_height <= 0:
            continue
        aspect_ratio = panel_width / panel_height
        fill_ratio = area / float(panel_width * panel_height)
        bottom = y + panel_height
        if x > width * 0.10 or panel_width < width * 0.75:
            continue
        if panel_height < height * 0.16 or panel_height > height * 0.45:
            continue
        if aspect_ratio < 2.4 or fill_ratio < 0.55:
            continue
        if y < height * 0.52 or bottom < height * 0.88:
            continue
        candidate = Box(x=x, y=y, width=panel_width, height=panel_height, area=area)
        if not has_light_dialogue_fill(rgb, candidate, min_ratio=0.50):
            continue
        candidates.append(candidate)

    if not candidates:
        return None
    return max(candidates, key=lambda item: item.area)


def build_light_dialogue_mask(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(int)
    green = rgb[:, :, 1].astype(int)
    blue = rgb[:, :, 2].astype(int)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(int)
    saturation = hsv[:, :, 1].astype(int)
    value = hsv[:, :, 2].astype(int)

    warm_cream = (
        (red >= 198)
        & (green >= 184)
        & (blue >= 150)
        & (green <= 245)
        & (blue <= 230)
    )
    balanced = (
        ((red - green) >= -8)
        & ((red - green) <= 34)
        & ((green - blue) >= 6)
        & ((green - blue) <= 58)
        & ((red - blue) >= 18)
        & ((red - blue) <= 82)
    )
    low_saturation = (saturation >= 8) & (saturation <= 74) & (value >= 172)
    warm_hue = (hue <= 38) | (hue >= 170)
    return (warm_cream & balanced & low_saturation & warm_hue).astype("uint8") * 255


def has_light_dialogue_fill(rgb: np.ndarray, box: Box, min_ratio: float) -> bool:
    height, width = rgb.shape[:2]
    left = max(0, box.x + int(box.width * 0.08))
    right = min(width, box.right - int(box.width * 0.08))
    top = max(0, box.y + int(box.height * 0.16))
    bottom = min(height, box.bottom - int(box.height * 0.12))
    if right <= left or bottom <= top:
        return False

    core = rgb[top:bottom, left:right]
    mask = build_light_dialogue_mask(core)
    ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if ratio < min_ratio:
        LOGGER.debug("Rejected light panel: cream fill ratio %.3f below %.3f", ratio, min_ratio)
        return False

    hsv = cv2.cvtColor(core, cv2.COLOR_RGB2HSV)
    values = hsv[:, :, 2]
    light_pixels = values[mask > 0]
    if light_pixels.size == 0:
        return False
    if float(np.std(light_pixels)) > 28.0:
        LOGGER.debug("Rejected light panel: fill is too textured.")
        return False
    return True


def find_nameplate_for_panel(image: Image.Image, panel: Box) -> Box | None:
    bgr = pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    strip_top = max(0, int(panel.y - panel.height * 0.4))
    strip_bottom = min(image.height, int(panel.y + panel.height * 0.05))
    strip_left = max(0, panel.x)
    strip_right = min(image.width, panel.right)
    if strip_bottom <= strip_top or strip_right <= strip_left:
        return None

    strip = gray[strip_top:strip_bottom, strip_left:strip_right]
    dark_mask = cv2.inRange(strip, 0, 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[Box] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1200:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue
        aspect_ratio = width / height
        if aspect_ratio < 2.5 or width < panel.width * 0.12:
            continue
        boxes.append(
            Box(
                x=strip_left + x,
                y=strip_top + y,
                width=width,
                height=height,
                area=area,
            )
        )
    if not boxes:
        return None
    return max(boxes, key=lambda item: item.area)


def crop_visual_novel_text(image: Image.Image, panel: Box, nameplate: Box | None) -> Image.Image:
    left_margin = 20
    right_margin = max(10, int(panel.width * 0.02))
    top_margin = max(8, int(panel.height * 0.10))
    bottom_ratio = 0.58 if panel.height < 140 else 0.82
    bottom = panel.y + max(top_margin + 1, int(panel.height * bottom_ratio))
    if nameplate is not None:
        left_margin = max(20, int(panel.width * 0.055))
        right_margin = max(36, int(panel.width * 0.06))
        top_margin = max(top_margin, nameplate.bottom - panel.y + 4)
        bottom = panel.y + max(top_margin + 1, int(panel.height * bottom_ratio))

    left = min(image.width, panel.x + left_margin)
    top = min(image.height, panel.y + top_margin)
    right = max(left + 1, min(image.width, panel.right - right_margin))
    bottom = max(top + 1, min(image.height, bottom))
    return image.crop((left, top, right, bottom))


def crop_floating_name_label(image: Image.Image, panel: Box) -> Image.Image | None:
    left = max(0, panel.x + int(panel.width * 0.04))
    top = max(0, panel.y - int(panel.height * 0.22))
    right = min(image.width, panel.x + int(panel.width * 0.55))
    bottom = min(image.height, panel.y + int(panel.height * 0.08))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


def find_dialogue_regions(image: Image.Image, args: Args) -> tuple[list[DialogueRegion], np.ndarray]:
    if args.box_style in {"auto", "visual-novel"}:
        regions, mask = find_visual_novel_dialogues(image, args)
        if regions or args.box_style == "visual-novel":
            return regions, mask

    boxes, mask = find_brown_boxes(image, args)
    regions = [
        DialogueRegion(kind="brown", box=box, text_crop=crop_box(image, box, args.padding))
        for box in boxes
    ]
    return regions, mask


def merge_overlapping_boxes(boxes: Iterable[Box]) -> list[Box]:
    merged: list[Box] = []
    for box in sorted(boxes, key=lambda item: item.area, reverse=True):
        target_index = next(
            (index for index, existing in enumerate(merged) if intersection_over_union(box, existing) > 0.25),
            None,
        )
        if target_index is None:
            merged.append(box)
            continue
        existing = merged[target_index]
        left = min(existing.x, box.x)
        top = min(existing.y, box.y)
        right = max(existing.right, box.right)
        bottom = max(existing.bottom, box.bottom)
        merged[target_index] = Box(left, top, right - left, bottom - top, existing.area + box.area)
    return merged


def intersection_over_union(first: Box, second: Box) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = first.width * first.height
    second_area = second.width * second.height
    return intersection / float(first_area + second_area - intersection)


def crop_box(image: Image.Image, box: Box, padding: int) -> Image.Image:
    left = max(0, box.x - padding)
    top = max(0, box.y - padding)
    right = min(image.width, box.right + padding)
    bottom = min(image.height, box.bottom + padding)
    return image.crop((left, top, right, bottom))


def clean_model_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped.strip("`").strip()
    if stripped.lower() in {"empty", "(empty)", "no text", "none", "null"}:
        return ""
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        stripped = stripped[1:-1].strip()
    return stripped


def read_text_local_ocr(image: Image.Image, scale: int = 1, backend: str = "windows") -> tuple[str, float]:
    if backend == "windows":
        try:
            return read_text_windows_ocr(image, scale=scale)
        except Exception:
            LOGGER.exception("Windows OCR failed; falling back to RapidOCR.")
            return read_text_rapidocr(image, scale=scale)
    return read_text_rapidocr(image, scale=scale)


def read_text_rapidocr(image: Image.Image, scale: int = 1) -> tuple[str, float]:
    from rapidocr_onnxruntime import RapidOCR

    ocr = get_rapid_ocr()
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
    result, _ = ocr(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    if not result:
        return "", 0.0

    lines = sorted(result, key=lambda item: min(point[1] for point in item[0]))
    text_parts: list[str] = []
    confidences: list[float] = []
    for _points, text, confidence in lines:
        cleaned = clean_model_text(str(text))
        if cleaned:
            text_parts.append(cleaned)
            confidences.append(float(confidence))

    if not text_parts:
        return "", 0.0
    average_confidence = sum(confidences) / len(confidences)
    return " ".join(text_parts), average_confidence


def read_text_windows_ocr(image: Image.Image, scale: int = 1) -> tuple[str, float]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(read_text_windows_ocr_async(image, scale=scale))

    result: tuple[str, float] | None = None
    error: BaseException | None = None

    def runner() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(read_text_windows_ocr_async(image, scale=scale))
        except BaseException as exc:
            error = exc

    thread = threading.Thread(target=runner, name="windows-ocr", daemon=True)
    thread.start()
    thread.join()
    if error is not None:
        raise error
    return result or ("", 0.0)


async def read_text_windows_ocr_async(image: Image.Image, scale: int = 1) -> tuple[str, float]:
    from winsdk.windows.globalization import Language
    from winsdk.windows.graphics.imaging import BitmapDecoder
    from winsdk.windows.media.ocr import OcrEngine
    from winsdk.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    stream = InMemoryRandomAccessStream()
    writer = DataWriter(stream)
    writer.write_bytes(buffer.getvalue())
    await writer.store_async()
    stream.seek(0)

    decoder = await BitmapDecoder.create_async(stream)
    bitmap = await decoder.get_software_bitmap_async()
    engine = get_windows_ocr_engine(Language, OcrEngine)
    if engine is None:
        raise RuntimeError("Windows OCR engine is unavailable.")
    result = await engine.recognize_async(bitmap)
    text_parts = [clean_model_text(line.text) for line in result.lines]
    text = " ".join(part for part in text_parts if part)
    return text, 1.0 if text else 0.0


def get_windows_ocr_engine(language_class, engine_class):
    if not hasattr(get_windows_ocr_engine, "_engine"):
        engine = engine_class.try_create_from_language(language_class("en-US"))
        if engine is None:
            engine = engine_class.try_create_from_user_profile_languages()
        setattr(get_windows_ocr_engine, "_engine", engine)
    return getattr(get_windows_ocr_engine, "_engine")


def read_dialogue_body_ocr(image: Image.Image, scale: int = 2, backend: str = "windows") -> tuple[str, float]:
    if backend == "windows":
        return read_text_local_ocr(image, scale=scale, backend=backend)

    line_crops = split_text_line_crops(image)
    if len(line_crops) < 2:
        return read_text_local_ocr(image, scale=scale, backend=backend)

    text_parts: list[str] = []
    confidences: list[float] = []
    for line_crop in line_crops:
        text, confidence = read_text_local_ocr(line_crop, scale=scale, backend=backend)
        cleaned = clean_model_text(text)
        if cleaned:
            text_parts.append(cleaned)
            confidences.append(confidence)

    if not text_parts:
        return read_text_local_ocr(image, scale=scale, backend=backend)
    return " ".join(text_parts), sum(confidences) / len(confidences)


def split_text_line_crops(image: Image.Image) -> list[Image.Image]:
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    dark_mask = gray < 135
    row_counts = dark_mask.sum(axis=1)
    threshold = max(8, int(image.width * 0.01))

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for row_index, count in enumerate(row_counts):
        if count > threshold and start is None:
            start = row_index
        elif (count <= threshold or row_index == len(row_counts) - 1) and start is not None:
            end = row_index if count <= threshold else row_index + 1
            if end - start >= 4:
                runs.append((start, end))
            start = None

    if len(runs) < 2:
        return [image]

    crops: list[Image.Image] = []
    for index, (start_row, end_row) in enumerate(runs):
        top = 0 if index == 0 else max(0, start_row - 5)
        bottom = min(image.height, end_row + 5)
        if bottom - top >= 12:
            crops.append(image.crop((0, top, image.width, bottom)))
    return crops or [image]


def get_rapid_ocr():
    if not hasattr(get_rapid_ocr, "_instance"):
        from rapidocr_onnxruntime import RapidOCR

        setattr(get_rapid_ocr, "_instance", RapidOCR())
    return getattr(get_rapid_ocr, "_instance")


async def speak_pocket_tts(text: str, args: Args) -> None:
    if args.stream_tts:
        play_pocket_tts_stream(
            text,
            args.voice,
            args.tts_language,
            args.tts_device,
            args.tts_stream_max_tokens,
        )
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as media_file:
        media_path = Path(media_file.name)
    try:
        synthesize_pocket_tts(text, args.voice, args.tts_language, args.tts_device, media_path)
        play_audio(media_path)
    finally:
        try:
            media_path.unlink(missing_ok=True)
        except OSError:
            LOGGER.debug("Could not remove temporary audio file: %s", media_path)


def synthesize_pocket_tts(
    text: str,
    voice: str,
    language: str,
    device: str,
    media_path: Path,
) -> None:
    import scipy.io.wavfile

    resolved_device = resolve_tts_device(device)
    model = get_pocket_tts_model(language, resolved_device)
    voice_state = get_pocket_tts_voice_state(model, voice)
    with _POCKET_TTS_LOCK:
        audio = model.generate_audio(voice_state, text)
    audio_array = audio.detach().cpu().numpy()
    if audio_array.ndim == 2 and audio_array.shape[0] <= 2:
        audio_array = audio_array.T
    scipy.io.wavfile.write(str(media_path), model.sample_rate, audio_array)


def play_pocket_tts_stream(
    text: str,
    voice: str,
    language: str,
    device: str,
    max_tokens: int,
) -> None:
    resolved_device = resolve_tts_device(device)
    model = get_pocket_tts_model(language, resolved_device)
    voice_state = get_pocket_tts_voice_state(model, voice)
    pygame = ensure_pygame_mixer(model.sample_rate)
    channel = pygame.mixer.find_channel(force=True)
    mixer_channels = pygame.mixer.get_init()[2]

    with _POCKET_TTS_LOCK:
        for audio_chunk in model.generate_audio_stream(voice_state, text, max_tokens=max_tokens):
            audio_array = tensor_audio_to_int16(audio_chunk, mixer_channels)
            if audio_array.size == 0:
                continue
            sound = pygame.sndarray.make_sound(audio_array)
            while channel.get_busy() and channel.get_queue() is not None:
                time.sleep(0.01)
            if channel.get_busy():
                channel.queue(sound)
            else:
                channel.play(sound)

    while channel.get_busy():
        time.sleep(0.03)


def tensor_audio_to_int16(audio, channels: int) -> np.ndarray:
    audio_array = audio.detach().cpu().numpy()
    audio_array = np.asarray(audio_array)
    audio_array = np.squeeze(audio_array)
    if audio_array.ndim == 0:
        return np.array([], dtype=np.int16)
    if audio_array.ndim == 2:
        if audio_array.shape[0] <= 2:
            audio_array = audio_array.T
        if audio_array.ndim == 2 and audio_array.shape[1] > 1:
            audio_array = audio_array.mean(axis=1)
        else:
            audio_array = audio_array.reshape(-1)
    audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=1.0, neginf=-1.0)
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_array = (audio_array * 32767).astype(np.int16, copy=False)
    if channels > 1:
        audio_array = np.repeat(audio_array.reshape(-1, 1), channels, axis=1)
    return audio_array


def resolve_tts_device(device: str) -> str:
    requested = device.strip().lower()
    if requested in {"", "auto"}:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            LOGGER.warning("Requested Pocket TTS device %s, but CUDA is unavailable. Falling back to CPU.", device)
            return "cpu"
    return requested


def configure_torch_for_tts(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_pocket_tts_model(language: str, device: str):
    with _POCKET_TTS_LOCK:
        models = getattr(get_pocket_tts_model, "_models", {})
        cache_key = (language, device)
        if cache_key not in models:
            from pocket_tts import TTSModel

            LOGGER.info("Loading Pocket TTS model for language %s on %s...", language, device)
            configure_torch_for_tts(device)
            model = TTSModel.load_model(language=language)
            if device != "cpu":
                model.to(device)
            models[cache_key] = model
            setattr(get_pocket_tts_model, "_models", models)
        return models[cache_key]


def get_pocket_tts_voice_state(model, voice: str):
    with _POCKET_TTS_LOCK:
        cache_key = (id(model), voice)
        voice_states = getattr(get_pocket_tts_voice_state, "_voice_states", {})
        if cache_key not in voice_states:
            LOGGER.info("Loading Pocket TTS voice %s...", voice)
            voice_states[cache_key] = model.get_state_for_audio_prompt(voice)
            setattr(get_pocket_tts_voice_state, "_voice_states", voice_states)
        return voice_states[cache_key]


def warm_up_tts(args: Args) -> None:
    resolved_device = resolve_tts_device(args.tts_device)
    model = get_pocket_tts_model(args.tts_language, resolved_device)
    voice_state = get_pocket_tts_voice_state(model, args.voice)
    LOGGER.info("Priming Pocket TTS generation...")
    with _POCKET_TTS_LOCK:
        model.generate_audio(voice_state, "Ready.", max_tokens=50)
    LOGGER.info("Pocket TTS ready.")


def warm_up_tts_in_background(args: Args) -> threading.Thread:
    def runner() -> None:
        try:
            warm_up_tts(args)
        except Exception:
            LOGGER.exception("Pocket TTS warm-up failed")

    thread = threading.Thread(target=runner, name="pocket-tts-warmup", daemon=True)
    thread.start()
    return thread


def play_audio(media_path: Path) -> None:
    pygame = ensure_pygame_mixer()

    pygame.mixer.music.load(str(media_path))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.05)
    pygame.mixer.music.unload()


def ensure_pygame_mixer(sample_rate: int | None = None):
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    if sample_rate is None:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        return pygame

    current = pygame.mixer.get_init()
    if current is not None:
        current_rate, current_size, current_channels = current
        if current_rate != sample_rate or current_size != -16 or current_channels != 1:
            pygame.mixer.quit()

    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=1, buffer=512)
    return pygame


def normalize_for_repeat(text: str) -> str:
    return " ".join(text.casefold().split())


def normalize_dialogue_key(text: str) -> str:
    normalized = text.casefold()
    normalized = normalized.replace("…", "...")
    normalized = normalized.replace("...", " ")
    normalized = normalized.replace("’", "'")
    normalized = normalized.replace("`", "'")
    normalized = re.sub(r"\b(says|said)\b", " says ", normalized)
    normalized = re.sub(r"[^a-z0-9']+", " ", normalized)
    return " ".join(normalized.split())


def normalize_dialogue_repeat_key(text: str) -> str:
    normalized = normalize_dialogue_key(text)
    if not normalized:
        return ""

    body = re.sub(r"^.{1,60}\bsays\b\s+", "", normalized)
    if body:
        normalized = body

    # OCR can change word boundaries between frames, so collapse spacing.
    return re.sub(r"[^a-z0-9]+", "", normalized)


def clean_speaker_name(text: str) -> str:
    cleaned = clean_model_text(text)
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1].strip()
    cleaned = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cleaned)
    cleaned = cleaned.strip(" \t\r\n-—–:;,.")
    return cleaned


def clean_dialogue_text(text: str) -> str:
    cleaned = clean_model_text(text)
    cleaned = re.sub(r"\s+Retur+\w*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+Return to Default Position.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+Scale Window.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\.{4,}", "...", cleaned)
    cleaned = re.sub(r"([,.;?!])(?=[A-Za-z])", r"\1 ", cleaned)
    cleaned = re.sub(r"\bIam\b", "I am", cleaned)
    cleaned = re.sub(r"\bnere\b", "here", cleaned)
    cleaned = re.sub(r"\bhereto\b", "here to", cleaned)
    cleaned = re.sub(r"\bDoyou\b", "Do you", cleaned)
    cleaned = re.sub(r"\bwishan\b", "wish an", cleaned)
    cleaned = re.sub(r"\bofaetherial\b", "of aetherial", cleaned)
    cleaned = re.sub(r"\bexplanationofaetherial\b", "explanation of aetherial", cleaned)
    cleaned = re.sub(r"\banyqueriesyou\b", "any queries you", cleaned)
    cleaned = re.sub(r"\banyqueries\b", "any queries", cleaned)
    cleaned = re.sub(r"\byoumight\b", "you might", cleaned)
    cleaned = re.sub(r"\bmighthave\b", "might have", cleaned)
    cleaned = re.sub(r"\bgreatwealth\b", "great wealth", cleaned)
    cleaned = re.sub(r"\bansweranyqueriesyoumighthave\b", "answer any queries you might have", cleaned)
    return " ".join(cleaned.split())


def format_spoken_dialogue(name: str, body: str) -> str:
    if name and body:
        return f"{name} says {body}"
    if name:
        return name
    return body


def is_plausible_speaker_name(text: str, confidence: float) -> bool:
    if confidence < 0.75:
        return False
    cleaned = clean_speaker_name(text)
    if not cleaned or len(cleaned) > 40:
        return False
    if not re.search(r"[A-Za-z]", cleaned):
        return False
    return not looks_like_menu_text(cleaned)


def is_plausible_dialogue_text(text: str, confidence: float, *, strict: bool = False) -> bool:
    if confidence < (0.90 if strict else 0.82):
        return False
    cleaned = clean_dialogue_text(text)
    words = re.findall(r"[A-Za-z']+", cleaned)
    if len(cleaned) < 8 or len(words) < 2:
        return False
    if looks_like_menu_text(cleaned):
        return False
    if strict and not re.search(r"[.?!…]|\.{2,}", cleaned):
        return False
    return True


def looks_like_menu_text(text: str) -> bool:
    normalized = normalize_for_repeat(text)
    menu_phrases = (
        "return to default position",
        "scale window",
        "close",
        "settings",
        "inventory",
        "map",
        "journal",
        "quest log",
        "system",
        "main menu",
    )
    hits = sum(1 for phrase in menu_phrases if phrase in normalized)
    return hits >= 2 or normalized in menu_phrases


def save_debug(frame: Image.Image, mask: np.ndarray, crops: Sequence[Image.Image], args: Args) -> None:
    if args.debug_dir is None:
        return
    args.debug_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    frame.save(args.debug_dir / f"{stamp}-frame.png")
    Image.fromarray(mask).save(args.debug_dir / f"{stamp}-mask.png")
    for index, crop in enumerate(crops, start=1):
        crop.save(args.debug_dir / f"{stamp}-crop-{index}.png")


def save_dialogue_debug(
    frame: Image.Image,
    mask: np.ndarray,
    regions: Sequence[DialogueRegion],
    args: Args,
) -> None:
    if args.debug_dir is None:
        return
    args.debug_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    frame.save(args.debug_dir / f"{stamp}-frame.png")
    Image.fromarray(mask).save(args.debug_dir / f"{stamp}-mask.png")
    for index, region in enumerate(regions, start=1):
        region.text_crop.save(args.debug_dir / f"{stamp}-{region.kind}-text-{index}.png")
        if region.name_crop is not None:
            region.name_crop.save(args.debug_dir / f"{stamp}-{region.kind}-name-{index}.png")


def read_visual_novel_region(region: DialogueRegion, args: Args, index: int) -> str:
    name = ""
    if region.name_crop is not None:
        raw_name, name_confidence = read_text_local_ocr(
            region.name_crop,
            scale=3,
            backend=args.ocr_backend,
        )
        if is_plausible_speaker_name(raw_name, name_confidence):
            name = clean_speaker_name(raw_name)
        LOGGER.info(
            "Local OCR name %s confidence %.3f: %s",
            index,
            name_confidence,
            json.dumps(name, ensure_ascii=False),
        )

    raw_body, body_confidence = read_dialogue_body_ocr(
        region.text_crop,
        scale=2,
        backend=args.ocr_backend,
    )
    body = clean_dialogue_text(raw_body)
    strict = region.kind == "lower-third"
    LOGGER.info(
        "Local OCR text %s confidence %.3f: %s",
        index,
        body_confidence,
        json.dumps(body, ensure_ascii=False),
    )

    if not is_plausible_dialogue_text(body, body_confidence, strict=strict):
        LOGGER.debug("Rejected OCR text as likely false positive.")
        return ""
    if strict and not name:
        LOGGER.debug("Rejected lower-third candidate because no speaker name was detected.")
        return ""

    return format_spoken_dialogue(name, body)


async def process_frame(args: Args) -> str:
    if args.image is None and args.require_fullscreen and not is_foreground_window_fullscreen(args):
        LOGGER.debug("Foreground window is not fullscreen on the target monitor.")
        return ""

    frame = load_frame(args)
    regions, mask = find_dialogue_regions(frame, args)
    save_dialogue_debug(frame, mask, regions, args)

    if not regions:
        LOGGER.debug("No matching dialogue box found.")
        return ""

    texts: list[str] = []
    for index, region in enumerate(regions, start=1):
        if region.kind in {"visual-novel", "lower-third"}:
            text = read_visual_novel_region(region, args, index)
            if text:
                texts.append(text)
            continue

        text, confidence = read_text_local_ocr(region.text_crop, scale=2, backend=args.ocr_backend)
        LOGGER.info(
            "Local OCR crop %s confidence %.3f: %s",
            index,
            confidence,
            json.dumps(text, ensure_ascii=False),
        )
        if confidence < args.local_ocr_min_confidence:
            text = ""
        if text:
            texts.append(text)

    combined = "\n".join(dict.fromkeys(texts))
    if combined:
        LOGGER.info("Detected text: %s", json.dumps(combined, ensure_ascii=False))
    else:
        LOGGER.info("OCR did not return readable dialogue text.")
    return combined


async def watch(args: Args) -> None:
    recent_spoken: dict[str, float] = {}
    pending_text = ""
    pending_count = 0
    last_spoken_key = ""

    while True:
        try:
            text = await process_frame(args)
        except KeyboardInterrupt:
            raise
        except Exception:
            LOGGER.exception("Frame processing failed")
            text = ""

        normalized = normalize_dialogue_repeat_key(text)
        should_speak = False
        now = time.monotonic()
        if args.repeat_cooldown <= 0:
            recent_spoken = {}
        else:
            recent_spoken = {
                key: seen_at
                for key, seen_at in recent_spoken.items()
                if now - seen_at < args.repeat_cooldown
            }
        if normalized and normalized == last_spoken_key:
            pending_text = ""
            pending_count = 0
            LOGGER.debug("Skipping unchanged last-spoken dialogue: %s", text)
        elif normalized and normalized not in recent_spoken:
            if normalized == pending_text:
                pending_count += 1
            else:
                pending_text = normalized
                pending_count = 1
            if args.once or pending_count >= args.confidence_repeat:
                recent_spoken[normalized] = now
                last_spoken_key = normalized
                pending_count = 0
                should_speak = True
        elif normalized in recent_spoken:
            pending_text = ""
            pending_count = 0
            LOGGER.debug("Skipping repeated dialogue: %s", text)

        if should_speak and args.speak:
            await speak_pocket_tts(text, args)

        if args.once:
            break
        await asyncio.sleep(args.interval)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.warm_up_ocr:
        if args.ocr_backend == "rapidocr":
            LOGGER.info("Warming up RapidOCR engine...")
            get_rapid_ocr()
        else:
            LOGGER.info("Using Windows OCR backend.")
        if args.speak:
            LOGGER.info("Warming up Pocket TTS in the background...")
            warm_up_tts_in_background(args)
    try:
        asyncio.run(watch(args))
    except KeyboardInterrupt:
        LOGGER.info("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
