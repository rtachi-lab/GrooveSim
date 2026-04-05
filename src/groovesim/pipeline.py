from __future__ import annotations

import json

from .audio import extract_audio_features, extract_audio_features_from_array, extract_audio_features_from_bytes
from .features import TempoPrior
from .features import compute_audio_feature_set, compute_symbolic_feature_set
from .midi import get_midi_reference_tempo, get_midi_reference_tempo_bytes, render_midi_bytes_to_waveform, render_midi_to_waveform
from .scoring import combine_features
from .symbolic import load_onset_times, onset_times_to_grid


def analyze_audio_file(path: str, tempo_prior: TempoPrior | tuple[float, float] | None = None) -> dict[str, float | str]:
    audio = extract_audio_features(path)
    feature_map = compute_audio_feature_set(audio, tempo_prior=tempo_prior)
    result = combine_features(feature_map, source="audio")
    return result.to_dict()


def analyze_audio_bytes(data: bytes, tempo_prior: TempoPrior | tuple[float, float] | None = None) -> dict[str, float | str]:
    audio = extract_audio_features_from_bytes(data)
    feature_map = compute_audio_feature_set(audio, tempo_prior=tempo_prior)
    result = combine_features(feature_map, source="audio")
    return result.to_dict()


def analyze_midi_file(path: str, sr: int = 22050, tempo_prior: TempoPrior | tuple[float, float] | None = None) -> dict[str, float | str]:
    waveform, sr = render_midi_to_waveform(path, sr=sr)
    audio = extract_audio_features_from_array(waveform, sr=sr)
    feature_map = compute_audio_feature_set(audio, tempo_prior=tempo_prior)
    feature_map["tempo_bpm"] = float(get_midi_reference_tempo(path))
    feature_map["tempo_alignment_score"] = max(
        0.0,
        1.0 - ((feature_map["tempo_bpm"] / 60.0 - 2.0) / 1.2) ** 2,
    )
    result = combine_features(feature_map, source="midi")
    return result.to_dict()


def analyze_midi_bytes(
    data: bytes,
    sr: int = 22050,
    tempo_prior: TempoPrior | tuple[float, float] | None = None,
) -> dict[str, float | str]:
    waveform, sr = render_midi_bytes_to_waveform(data, sr=sr)
    audio = extract_audio_features_from_array(waveform, sr=sr)
    feature_map = compute_audio_feature_set(audio, tempo_prior=tempo_prior)
    feature_map["tempo_bpm"] = float(get_midi_reference_tempo_bytes(data))
    feature_map["tempo_alignment_score"] = max(
        0.0,
        1.0 - ((feature_map["tempo_bpm"] / 60.0 - 2.0) / 1.2) ** 2,
    )
    result = combine_features(feature_map, source="midi")
    return result.to_dict()


def analyze_onset_file(path: str) -> dict[str, float | str]:
    onsets = load_onset_times(path)
    grid, tatum = onset_times_to_grid(onsets)
    feature_map = compute_symbolic_feature_set(onsets, grid, tatum)
    result = combine_features(feature_map, source="onsets")
    return result.to_dict()


def save_json(data: dict[str, float | str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
