from __future__ import annotations

import numpy as np

from .features import GrooveFeatureSet


def combine_features(feature_map: dict[str, float], source: str) -> GrooveFeatureSet:
    entrainment_component = (
        0.35 * feature_map["beat_strength"]
        + 0.25 * feature_map["beat_clarity"]
        + 0.20 * feature_map["periodicity_stability"]
        + 0.20 * feature_map["tempo_alignment_score"]
    )
    tension_component = (
        0.40 * feature_map["complexity_balance_score"]
        + 0.35 * np.clip(feature_map["moderate_surprisal_ratio"], 0.0, 1.0)
        + 0.25 * np.clip(feature_map["mean_surprisal"] / 2.0, 0.0, 1.0)
    )
    embodiment_component = (
        0.35 * np.clip(feature_map["event_density"] / 4.0, 0.0, 1.0)
        + 0.35 * feature_map["low_freq_flux"]
        + 0.20 * feature_map["bass_pulse_strength"]
        + 0.10 * feature_map["low_mid_balance"]
    )

    raw_score = 0.45 * entrainment_component + 0.30 * tension_component + 0.25 * embodiment_component
    gated_score = raw_score * (0.4 + 0.6 * feature_map["confidence"])

    return GrooveFeatureSet(
        groove_score=float(np.clip(gated_score, 0.0, 1.0)),
        tempo_bpm=float(feature_map["tempo_bpm"]),
        beat_strength=float(feature_map["beat_strength"]),
        beat_clarity=float(feature_map["beat_clarity"]),
        periodicity_stability=float(feature_map["periodicity_stability"]),
        tempo_alignment_score=float(feature_map["tempo_alignment_score"]),
        syncopation_index=float(feature_map["syncopation_index"]),
        complexity_balance_score=float(feature_map["complexity_balance_score"]),
        event_density=float(feature_map["event_density"]),
        mean_surprisal=float(feature_map["mean_surprisal"]),
        surprisal_variance=float(feature_map["surprisal_variance"]),
        moderate_surprisal_ratio=float(feature_map["moderate_surprisal_ratio"]),
        low_freq_flux=float(feature_map["low_freq_flux"]),
        bass_pulse_strength=float(feature_map["bass_pulse_strength"]),
        low_mid_balance=float(feature_map["low_mid_balance"]),
        confidence=float(feature_map["confidence"]),
        source=source,
    )
