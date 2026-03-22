from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import List, Optional

import ollama

from ocr_engine import OCRBox

logger = logging.getLogger(__name__)


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{ts} [APP] {msg}", flush=True)

OLLAMA_MODEL = "llama3.2:3b"

_SYSTEM_PROMPT = """\
You will receive a list of text tokens extracted via OCR from a potential student ID card.
Your job is to map each token to the correct field.

Return ONLY a valid JSON object with exactly these keys:
{
  "name": string or null,
  "student_id": string or null,
  "department": string or null,
  "expiry_date": string or null,
  "university": string or null
}

Rules:
- Token may contain more information than needed, extract the essential parts
- "name" is the student's full personal name
- "student_id" is the alphanumeric ID/matriculation number
- "department" is the academic department, faculty, or program
- "expiry_date" is any validity or expiry date on the card
- "university" is the institution name
- Use null for any field not present in the tokens
- Do not invent values not present in the input
- Return only the JSON, no explanation
"""


def extract_fields(boxes: List[OCRBox]) -> dict:
    tokens = [b.text for b in boxes]
    confidence = _mean_conf(boxes)

    result = {
        "name": None,
        "student_id": None,
        "department": None,
        "expiry_date": None,
        "university": None,
        "raw_text": tokens,
        "confidence": confidence,
    }

    if not tokens:
        return result

    extracted = _extract_via_llm(tokens)
    if extracted:
        result.update(extracted)
    else:
        # Fallback: regex-only for student_id and date if LLM unavailable
        _log("WARN  LLM unavailable, falling back to regex only")
        result.update(_regex_fallback(tokens))

    return result


def _extract_via_llm(tokens: list[str]) -> Optional[dict]:
    user_msg = "OCR tokens:\n" + "\n".join(f"- {t}" for t in tokens)
    try:
        _log(f"── Ollama request (model={OLLAMA_MODEL}) ────────────────")
        for t in tokens:
            _log(f"  token: {t!r}")

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            options={"temperature": 0},
        )
        content = response["message"]["content"].strip()
        _log("── Ollama raw response ──────────────────────")
        _log(f"  {content}")

        # Strip markdown code fences if model wraps output
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
        parsed = json.loads(content)
        return {k: parsed.get(k) for k in ("name", "student_id", "department", "expiry_date", "university")}
    except Exception as e:
        _log(f"ERROR LLM extraction failed: {e}")
        return None


# ── Regex fallback (used only if Ollama is unreachable) ──────────────────────

_STUDENT_ID_RE = re.compile(
    r"\b([A-Z]{0,4}\d{4,12}[A-Z]{0,3}|\d{2,4}[-/]\d{3,8}|[A-Z]{1,3}\d{6,10})\b"
)
_DATE_RE = re.compile(
    r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\s.\-/]*\d{2,4})\b",
    re.IGNORECASE,
)


def _regex_fallback(tokens: list[str]) -> dict:
    result: dict = {}
    for t in tokens:
        if "student_id" not in result:
            m = _STUDENT_ID_RE.search(t)
            if m:
                result["student_id"] = m.group(0)
        if "expiry_date" not in result:
            m = _DATE_RE.search(t)
            if m:
                result["expiry_date"] = m.group(0)
    return result


def _mean_conf(boxes: List[OCRBox]) -> float:
    if not boxes:
        return 0.0
    return round(sum(b.score for b in boxes) / len(boxes), 3)
