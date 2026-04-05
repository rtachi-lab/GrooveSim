from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
async def analyze(
    file: UploadFile = File(...),
    tempo_hint: float | None = Form(default=None),
    tempo_min: float | None = Form(default=None),
    tempo_max: float | None = Form(default=None),
) -> dict[str, float | str]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS | ALLOWED_MIDI_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    if tempo_hint is not None and (tempo_min is not None or tempo_max is not None):
        raise HTTPException(status_code=400, detail="Provide either tempo_hint or tempo_min/tempo_max, not both.")
    if (tempo_min is None) != (tempo_max is None):
        raise HTTPException(status_code=400, detail="tempo_min and tempo_max must be provided together.")

    data = await file.read()
    await file.close()

    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file is too large.")

    try:
        if tempo_hint is not None:
            tempo_prior = float(tempo_hint)
        elif tempo_min is not None and tempo_max is not None:
            tempo_prior = (float(tempo_min), float(tempo_max))
        else:
            tempo_prior = None
        if suffix in ALLOWED_MIDI_EXTENSIONS:
            result = analyze_midi_bytes(data, tempo_prior=tempo_prior)
        else:
            result = analyze_audio_bytes(data, tempo_prior=tempo_prior)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {exc}") from exc

    result["filename"] = file.filename or "upload"
    result["stored_on_server"] = "no"
    if tempo_hint is not None:
        result["tempo_hint"] = float(tempo_hint)
    if tempo_min is not None and tempo_max is not None:
        result["tempo_prior_min"] = float(tempo_min)
        result["tempo_prior_max"] = float(tempo_max)
    return result


def main() -> None:
    uvicorn.run("groovesim.webapp:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
