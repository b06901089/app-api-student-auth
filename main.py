from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import extractor
import ocr_engine

# Use uvicorn's logger so output appears in the same stream
logger = logging.getLogger("uvicorn.error")


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{ts} [APP] {msg}", flush=True)

app = FastAPI(title="Student ID Card Extractor", version="1.0")


# ── Pydantic models ───────────────────────────────────────────────────────────

class StudentIDResponse(BaseModel):
    name: Optional[str] = None
    student_id: Optional[str] = None
    department: Optional[str] = None
    expiry_date: Optional[str] = None
    university: Optional[str] = None
    raw_text: List[str] = []
    confidence: float = 0.0


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend() -> HTMLResponse:
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/extract", response_model=StudentIDResponse)
async def extract(file: UploadFile = File(...)) -> StudentIDResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    # Upscale very small images to improve OCR accuracy
    h, w = img.shape[:2]
    if max(h, w) < 800:
        scale = 800 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    boxes = ocr_engine.run_ocr(img)

    _log(f"── OCR results ({len(boxes)} boxes) ──────────────────")
    for i, b in enumerate(boxes):
        _log(f"  [{i}] score={b.score:.2f}  {b.text!r}")

    fields = extractor.extract_fields(boxes)

    _log("── Extracted fields ────────────────────────")
    for key in ("name", "student_id", "department", "expiry_date", "university"):
        _log(f"  {key + ':':<15} {fields.get(key)}")

    return StudentIDResponse(**fields)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "ocr_loaded": ocr_engine.is_ready()}


@app.post("/debug-ocr")
async def debug_ocr(file: UploadFile = File(...)) -> dict:
    """Returns raw OCR boxes with position info — for debugging extraction issues."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")
    h, w = img.shape[:2]
    if max(h, w) < 800:
        scale = 800 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    boxes = ocr_engine.run_ocr(img)
    return {
        "box_count": len(boxes),
        "boxes": [
            {"i": i, "text": b.text, "score": round(b.score, 3),
             "x": round(b.x), "y": round(b.y), "x2": round(b.x2), "y2": round(b.y2),
             "cx": round(b.cx), "cy": round(b.cy)}
            for i, b in enumerate(boxes)
        ]
    }
