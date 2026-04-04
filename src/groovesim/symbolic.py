from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def load_onset_times(path: str) -> np.ndarray:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            values = payload.get("onsets", [])
        elif isinstance(payload, list):
            values = payload
        else:
            raise ValueError("JSON onset file must contain a list or {'onsets': [...]} structure.")
        return np.asarray(values, dtype=float)

    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            return np.array([], dtype=float)
        for key in ("time", "onset", "onset_time", "seconds"):
            if key in rows[0]:
                return np.asarray([float(row[key]) for row in rows], dtype=float)
        first_column = next(iter(rows[0].keys()))
        return np.asarray([float(row[first_column]) for row in rows], dtype=float)

    raise ValueError(f"Unsupported onset file format: {suffix}")


def onset_times_to_grid(onsets: np.ndarray, resolution: int = 16) -> tuple[np.ndarray, float]:
    if onsets.size < 2:
        return np.zeros(resolution, dtype=float), 0.5

    iois = np.diff(np.sort(onsets))
    positive_iois = iois[iois > 1e-4]
    if positive_iois.size == 0:
        return np.zeros(resolution, dtype=float), 0.5

    tatum = float(np.median(positive_iois))
    start = float(np.min(onsets))
    indices = np.rint((onsets - start) / max(tatum, 1e-4)).astype(int)
    if indices.size == 0:
        return np.zeros(resolution, dtype=float), tatum

    length = max(indices.max() + 1, resolution)
    grid = np.zeros(length, dtype=float)
    grid[np.clip(indices, 0, length - 1)] = 1.0
    return grid, tatum
