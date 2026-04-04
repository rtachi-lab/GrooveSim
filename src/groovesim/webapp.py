from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from .pipeline import analyze_audio_bytes, analyze_midi_bytes


APP_TITLE = "GrooveSim Web"
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "web" / "index.html"
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
ALLOWED_MIDI_EXTENSIONS = {".mid", ".midi"}
MAX_UPLOAD_BYTES = 40 * 1024 * 1024


def _allowed_origins() -> list[str]:
    raw = os.getenv(
        "GROOVESIM_ALLOWED_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000,https://rtachi-lab.github.io",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, float | str]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS | ALLOWED_MIDI_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    data = await file.read()
    await file.close()

    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file is too large.")

    try:
        if suffix in ALLOWED_MIDI_EXTENSIONS:
            result = analyze_midi_bytes(data)
        else:
            result = analyze_audio_bytes(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {exc}") from exc

    result["filename"] = file.filename or "upload"
    result["stored_on_server"] = "no"
    return result


def main() -> None:
    uvicorn.run("groovesim.webapp:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
