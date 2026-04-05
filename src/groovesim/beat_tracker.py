from __future__ import annotations

import os
from threading import Lock

import numpy as np

_TRACKER = None
_TRACKER_LOCK = Lock()


def beat_this_enabled() -> bool:
    raw = os.getenv("GROOVESIM_ENABLE_BEAT_THIS", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _get_tracker():
    if not beat_this_enabled():
        return None

    global _TRACKER
    if _TRACKER is not None:
        return _TRACKER

    with _TRACKER_LOCK:
        if _TRACKER is not None:
            return _TRACKER
        try:
            from beat_this.inference import Audio2Beats
        except ImportError:
            return None
        _TRACKER = Audio2Beats(device="cpu", dbn=False)
        return _TRACKER


def estimate_tempo_with_beat_this(signal: np.ndarray, sr: int) -> dict[str, float] | None:
    tracker = _get_tracker()
    if tracker is None:
        return None

    beats, _downbeats = tracker(np.asarray(signal, dtype=np.float32), sr)
    beats = np.asarray(beats, dtype=float)
    if beats.size < 3:
        return None

    ibis = np.diff(beats)
    ibis = ibis[np.isfinite(ibis)]
    ibis = ibis[(ibis >= 60.0 / 220.0) & (ibis <= 60.0 / 40.0)]
    if ibis.size < 2:
        return None

    median_ibi = float(np.median(ibis))
    mean_ibi = float(np.mean(ibis))
    ibi_stability = float(np.clip(1.0 - np.std(ibis) / max(mean_ibi, 1e-6), 0.0, 1.0))
    coverage = float(np.clip(beats.size / 16.0, 0.0, 1.0))

    return {
        "tempo_bpm": float(60.0 / max(median_ibi, 1e-6)),
        "beat_count": float(beats.size),
        "ibi_stability": ibi_stability,
        "confidence": float(0.65 * ibi_stability + 0.35 * coverage),
    }
