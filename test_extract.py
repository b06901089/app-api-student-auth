"""
Test extraction pipeline on all images in example_img/
Run: conda run -n py39 python test_extract.py
"""
import time
from pathlib import Path

import cv2

from ocr_engine import run_ocr
from extractor import extract_fields

EXAMPLE_DIR = Path("example_img")
FIELDS = ("name", "student_id", "department", "expiry_date", "university")


def process(image_path: Path) -> dict:
    img = cv2.imread(str(image_path))
    if img is None:
        return {"error": "could not decode image"}

    t0 = time.perf_counter()
    boxes = run_ocr(img)
    ocr_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    fields = extract_fields(boxes)
    llm_time = time.perf_counter() - t0

    return {**fields, "_ocr_time": ocr_time, "_llm_time": llm_time}


def main():
    images = sorted(EXAMPLE_DIR.iterdir())
    images = [p for p in images if p.suffix.lower() in
              {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}]

    if not images:
        print(f"No images found in {EXAMPLE_DIR}/")
        return

    print(f"Found {len(images)} image(s) in {EXAMPLE_DIR}/\n")

    for img_path in images:
        print("=" * 60)
        print(f"Image : {img_path.name}")
        print("-" * 60)

        result = process(img_path)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        for field in FIELDS:
            value = result.get(field)
            status = value if value else "— not found"
            print(f"  {field + ':':<16} {status}")

        print(f"\n  OCR confidence : {result.get('confidence', 0):.1%}")
        print(f"  OCR time       : {result['_ocr_time']:.2f}s")
        print(f"  LLM time       : {result['_llm_time']:.2f}s")
        print(f"  Total          : {result['_ocr_time'] + result['_llm_time']:.2f}s")

        raw = result.get("raw_text", [])
        if raw:
            print(f"\n  Raw OCR tokens ({len(raw)}):")
            for t in raw:
                print(f"    • {t}")
        print()

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
