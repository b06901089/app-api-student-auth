from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

_ocr_instance = None  # lazy singleton


@dataclass
class OCRBox:
    text: str
    score: float
    x: float
    y: float
    x2: float
    y2: float
    cx: float
    cy: float
    w: float
    h: float


def get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR  # deferred import so server starts fast
        _ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            show_log=False,
        )
    return _ocr_instance


def is_ready() -> bool:
    return _ocr_instance is not None


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Prepare a BGR image for OCR:
      1. Upscale if too small
      2. Deskew
      3. CLAHE contrast enhancement
      4. Denoise
    """
    # 1. Upscale very small images
    h, w = img.shape[:2]
    if max(h, w) < 800:
        scale = 800 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # # 2. Deskew — find dominant angle via edges and rotate to correct it
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # points = np.column_stack(np.where(edges > 0))
    # if len(points) >= 5:
    #     angle = cv2.minAreaRect(points)[2]
    #     # minAreaRect returns angles in [-90, 0); normalise to (-45, 45]
    #     if angle < -45:
    #         angle += 90
    #     if abs(angle) > 0.5:  # skip tiny corrections that add noise
    #         h, w = img.shape[:2]
    #         M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    #         img = cv2.warpAffine(img, M, (w, h),
    #                              flags=cv2.INTER_CUBIC,
    #                              borderMode=cv2.BORDER_REPLICATE)

    # 3. CLAHE on luminance channel to fix glare / low contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 4. Denoise — conservative strength to avoid blurring thin strokes
    img = cv2.fastNlMeansDenoisingColored(img, None, h=7, hColor=7,
                                          templateWindowSize=7,
                                          searchWindowSize=21)
    return img


def _ocr_boxes(img: np.ndarray) -> List[OCRBox]:
    """Run OCR on an already-preprocessed BGR image and return sorted boxes."""
    ocr = get_ocr()
    raw = ocr.ocr(img, cls=True)

    boxes: List[OCRBox] = []
    if not raw or not raw[0]:
        return boxes

    for item in raw[0]:
        box_pts, (text, score) = item
        xs = [p[0] for p in box_pts]
        ys = [p[1] for p in box_pts]
        x, y, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        boxes.append(
            OCRBox(
                text=text.strip(),
                score=score,
                x=x, y=y, x2=x2, y2=y2,
                cx=(x + x2) / 2,
                cy=(y + y2) / 2,
                w=x2 - x,
                h=y2 - y,
            )
        )

    if not boxes:
        return boxes

    # Sort into reading order using row bands
    median_h = sorted(b.h for b in boxes)[len(boxes) // 2]
    band = max(median_h * 0.6, 8)
    boxes.sort(key=lambda b: (round(b.cy / band), b.cx))
    return boxes


def run_ocr(img: np.ndarray) -> List[OCRBox]:
    """
    Preprocess then run OCR on a raw BGR image from cv2.imdecode.
    Returns OCRBox list sorted in reading order (top→bottom, left→right).
    """
    return _ocr_boxes(preprocess(img))
