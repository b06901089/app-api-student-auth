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


def run_ocr(img: np.ndarray) -> List[OCRBox]:
    """
    img: BGR numpy array from cv2.imdecode
    Returns OCRBox list sorted in reading order (top→bottom, left→right).
    """
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
