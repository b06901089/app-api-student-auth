# Student ID Card Extractor

A local web app that extracts structured information from student ID card images using OCR and an LLM.

## How it works

1. Upload a student ID card image via the web UI (drag & drop or click to browse)
2. **PaddleOCR** reads the text from the image
3. **Ollama** maps the OCR tokens to structured fields using a selectable model
4. The extracted data is displayed: name, student ID, department, expiry date, and university

If Ollama is unavailable, a regex fallback extracts the student ID and expiry date.

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) running locally with at least one model pulled

```bash
brew install ollama
ollama pull llama3.1:8b   # better accuracy (default)
ollama pull llama3.2:3b   # faster
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload
```

Then open `http://localhost:8000` in your browser.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `POST` | `/extract` | Extract fields from an uploaded image (form fields: `file`, `model`) |
| `GET` | `/health` | Health check |
| `POST` | `/debug-ocr` | Return raw OCR boxes (for debugging) |

## Project Structure

```
main.py          # FastAPI app and routes
ocr_engine.py    # PaddleOCR wrapper
extractor.py     # LLM-based field extraction (with regex fallback)
static/
  index.html     # Web UI
requirements.txt
```
