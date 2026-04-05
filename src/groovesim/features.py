from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .audio import AudioFeatures


@dataclass
class GrooveFeatureSet:
    groove_score: float
    tempo_bpm: float
    beat_strength: float
    beat_clarity: float
    periodicity_stability: float
    tempo_alignment_score: float
    syncopation_index: float
    complexity_balance_score: float
    event_density: float
    mean_surprisal: float
    surprisal_variance: float
    moderate_surprisal_ratio: float
    low_freq_flux: float
    bass_pulse_strength: float
    low_mid_balance: float
    confidence: float
    source: str

    def to_dict(self) -> dict[str, float | str]:
        return self.__dict__.copy()


@dataclass
class MeterEstimate:
    grid: np.ndarray
    steps_per_beat: int
    beats_per_bar: int
    beat_phase: int
    bar_phase: int
    tatum_seconds: float
    confidence: float


@dataclass(frozen=True)
class TempoPrior:
    min_bpm: float
    max_bpm: float


def _sigmoid(x: float, center: float, slope: float) -> float:
    return float(1.0 / (1.0 + np.exp(-slope * (x - center))))


def _inverse_u(x: float, center: float, width: float) -> float:
    value = 1.0 - ((x - center) / max(width, 1e-6)) ** 2
    return float(np.clip(value, 0.0, 1.0))


def _tempo_preference_score(tempo_bpm: float) -> float:
    beat_hz = tempo_bpm / 60.0
    return _inverse_u(beat_hz, center=2.0, width=1.2)


def _normalize_tempo_prior(tempo_prior: TempoPrior | tuple[float, float] | None) -> TempoPrior | None:
    if tempo_prior is None:
        return None
    if isinstance(tempo_prior, TempoPrior):
        min_bpm = float(tempo_prior.min_bpm)
        max_bpm = float(tempo_prior.max_bpm)
    else:
        min_bpm = float(tempo_prior[0])
        max_bpm = float(tempo_prior[1])
    if min_bpm <= 0 or max_bpm <= 0:
        raise ValueError("Tempo prior bounds must be positive.")
    if min_bpm > max_bpm:
        min_bpm, max_bpm = max_bpm, min_bpm
    return TempoPrior(min_bpm=min_bpm, max_bpm=max_bpm)


def _tempo_prior_score(tempo_bpm: float, tempo_prior: TempoPrior | None) -> float:
    if tempo_prior is None:
        return 1.0

    if tempo_prior.min_bpm <= tempo_bpm <= tempo_prior.max_bpm:
        span = max(tempo_prior.max_bpm - tempo_prior.min_bpm, 1.0)
        center = 0.5 * (tempo_prior.min_bpm + tempo_prior.max_bpm)
        closeness = 1.0 - abs(tempo_bpm - center) / (0.5 * span + 1e-6)
        return 1.25 + 0.35 * float(np.clip(closeness, 0.0, 1.0))

    if tempo_bpm < tempo_prior.min_bpm:
        distance = tempo_prior.min_bpm - tempo_bpm
    else:
        distance = tempo_bpm - tempo_prior.max_bpm
    return float(np.exp(-distance / 10.0))


def _expand_tempo_hypotheses(base_bpm: float, base_strength: float) -> list[tuple[float, float]]:
    ratios = (
        (0.5, 0.92),
        (2.0 / 3.0, 0.96),
        (1.0, 1.0),
        (1.5, 0.98),
        (2.0, 0.90),
    )
    candidates: list[tuple[float, float]] = []
    for ratio, ratio_weight in ratios:
        bpm = base_bpm * ratio
        if 40.0 <= bpm <= 220.0:
            candidates.append((float(bpm), float(base_strength * ratio_weight)))
    return candidates


def _select_tempo_candidate(
    valid_bpms: np.ndarray,
    valid_strengths: np.ndarray,
    *,
    tempo_prior: TempoPrior | None = None,
    extra_candidates: list[tuple[float, float]] | None = None,
) -> tuple[float, float]:
    peak_idx = int(np.argmax(valid_strengths))
    peak_bpm = float(valid_bpms[peak_idx])
    peak_strength = float(valid_strengths[peak_idx])
    max_strength = float(np.max(valid_strengths) + 1e-6)

    candidate_indices = np.argsort(valid_strengths)[::-1][:12]
    candidate_pool: list[tuple[float, float]] = [(peak_bpm, peak_strength)]
    if extra_candidates:
        candidate_pool.extend(extra_candidates)

    for idx in candidate_indices:
        bpm = float(valid_bpms[idx])
        strength = float(valid_strengths[idx])
        candidate_pool.extend(_expand_tempo_hypotheses(bpm, strength))

    best_bpm = peak_bpm
    best_strength = peak_strength
    best_score = -np.inf
    seen: set[float] = set()

    for bpm, strength in candidate_pool:
        rounded_bpm = round(float(bpm), 3)
        if rounded_bpm in seen:
            continue
        seen.add(rounded_bpm)

        normalized_strength = float(strength / max_strength)
        score = (
            normalized_strength
            * (0.45 + 0.55 * _tempo_preference_score(bpm))
            * _tempo_prior_score(bpm, tempo_prior)
        )
        if score > best_score:
            best_bpm = bpm
            best_strength = strength
            best_score = score

    return best_bpm, best_strength


def estimate_periodicity(
    onset_env: np.ndarray,
    sr: int,
    hop_length: int,
    *,
    low_onset_env: np.ndarray | None = None,
    tempo_prior: TempoPrior | tuple[float, float] | None = None,
) -> dict[str, float]:
    if onset_env.size < 8 or not np.any(onset_env > 0):
        return {
            "tempo_bpm": 0.0,
            "beat_strength": 0.0,
            "beat_clarity": 0.0,
            "periodicity_stability": 0.0,
            "tempo_alignment_score": 0.0,
        }

    normalized_prior = _normalize_tempo_prior(tempo_prior)

    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    mean_tempogram = tempogram.mean(axis=1)
    if low_onset_env is not None and low_onset_env.size:
        low_tempogram = librosa.feature.tempogram(onset_envelope=low_onset_env, sr=sr, hop_length=hop_length)
        mean_low_tempogram = low_tempogram.mean(axis=1)
        mean_tempogram = 0.6 * mean_tempogram + 0.4 * mean_low_tempogram
    bpms = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
    valid = np.isfinite(bpms) & (bpms >= 40.0) & (bpms <= 220.0)
    if not np.any(valid):
        return {
            "tempo_bpm": 0.0,
            "beat_strength": 0.0,
            "beat_clarity": 0.0,
            "periodicity_stability": 0.0,
            "tempo_alignment_score": 0.0,
        }

    valid_strengths = mean_tempogram[valid]
    valid_bpms = bpms[valid]
    tempo_bpm, peak_strength = _select_tempo_candidate(
        valid_bpms,
        valid_strengths,
        tempo_prior=normalized_prior,
        extra_candidates=None,
    )
    median_strength = float(np.median(valid_strengths))
    clarity = peak_strength / (median_strength + 1e-6)

    framewise_peak_idx = np.argmax(tempogram[valid], axis=0)
    framewise_bpms = valid_bpms[framewise_peak_idx]
    tempo_std = float(np.std(framewise_bpms))
    periodicity_stability = float(np.clip(1.0 - tempo_std / 60.0, 0.0, 1.0))

    beat_strength = float(np.clip(peak_strength / (np.max(valid_strengths) + 1e-6), 0.0, 1.0))
    beat_clarity = float(np.clip((clarity - 1.0) / 3.0, 0.0, 1.0))

    return {
        "tempo_bpm": tempo_bpm,
        "beat_strength": beat_strength,
        "beat_clarity": beat_clarity,
        "periodicity_stability": periodicity_stability,
        "tempo_alignment_score": _tempo_preference_score(tempo_bpm),
    }


def compute_event_density_from_audio(onset_env: np.ndarray, sr: int, hop_length: int) -> float:
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, units="frames")
    duration = max(len(onset_env) * hop_length / sr, 1e-6)
    return float(len(onset_frames) / duration)


def compute_event_density_from_onsets(onsets: np.ndarray) -> float:
    if onsets.size < 2:
        return 0.0
    duration = max(float(np.max(onsets) - np.min(onsets)), 1e-6)
    return float(len(onsets) / duration)


def build_binary_onset_sequence_from_audio(onset_env: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, units="frames")
    sequence = np.zeros(len(onset_env), dtype=int)
    sequence[np.clip(onset_frames, 0, len(onset_env) - 1)] = 1
    return sequence


def _resample_binary_sequence(sequence: np.ndarray, source_steps_per_beat: int, target_steps_per_beat: int) -> np.ndarray:
    seq = np.asarray(sequence, dtype=float)
    if seq.size == 0 or source_steps_per_beat == target_steps_per_beat:
        return np.asarray(seq > 0, dtype=float)

    scale = target_steps_per_beat / max(source_steps_per_beat, 1)
    length = max(int(np.ceil(len(seq) * scale)), target_steps_per_beat * 2)
    out = np.zeros(length, dtype=float)
    indices = np.rint(np.where(seq > 0)[0] * scale).astype(int)
    indices = np.clip(indices, 0, length - 1)
    out[indices] = 1.0
    return out


def estimate_meter_from_grid(
    grid: np.ndarray,
    *,
    candidate_steps_per_beat: tuple[int, ...] = (2, 4),
    candidate_beats_per_bar: tuple[int, ...] = (3, 4),
    tatum_seconds: float = 0.125,
) -> MeterEstimate:
    seq = np.asarray(grid > 0, dtype=float)
    if seq.size == 0 or np.sum(seq) == 0:
        return MeterEstimate(
            grid=np.zeros(16, dtype=float),
            steps_per_beat=4,
            beats_per_bar=4,
            beat_phase=0,
            bar_phase=0,
            tatum_seconds=tatum_seconds,
            confidence=0.0,
        )

    if seq.size < 16:
        seq = np.pad(seq, (0, 16 - seq.size))

    best: MeterEstimate | None = None
    best_score = -np.inf

    for steps_per_beat in candidate_steps_per_beat:
        resampled = _resample_binary_sequence(seq, source_steps_per_beat=1, target_steps_per_beat=steps_per_beat)

        for beats_per_bar in candidate_beats_per_bar:
            bar_len = steps_per_beat * beats_per_bar
            if len(resampled) < bar_len:
                padded = np.pad(resampled, (0, bar_len - len(resampled)))
            else:
                padded = resampled

            strong = np.zeros(bar_len, dtype=float)
            medium = np.zeros(bar_len, dtype=float)
            weak = np.zeros(bar_len, dtype=float)

            for idx in range(bar_len):
                if idx % bar_len == 0:
                    strong[idx] = 4.0
                elif idx % steps_per_beat == 0:
                    strong[idx] = 3.0
                elif steps_per_beat >= 4 and idx % max(steps_per_beat // 2, 1) == 0:
                    medium[idx] = 1.5
                else:
                    weak[idx] = 0.5

            weights = strong + medium + weak

            for phase in range(bar_len):
                rotated = np.roll(weights, phase)
                repeated = np.resize(rotated, len(padded))
                metrical_fit = float(np.dot(padded, repeated) / (np.sum(padded) + 1e-6))

                bar_positions = np.arange(phase, len(padded), bar_len)
                beat_positions = np.arange(phase, len(padded), steps_per_beat)
                bar_strength = float(np.mean(padded[np.clip(bar_positions, 0, len(padded) - 1)])) if bar_positions.size else 0.0
                beat_strength = float(np.mean(padded[np.clip(beat_positions, 0, len(padded) - 1)])) if beat_positions.size else 0.0
                confidence = float(np.clip(0.6 * metrical_fit + 0.25 * bar_strength + 0.15 * beat_strength, 0.0, 4.0) / 4.0)
                score = metrical_fit + 0.35 * bar_strength + 0.15 * beat_strength

                if score > best_score:
                    best_score = score
                    best = MeterEstimate(
                        grid=padded,
                        steps_per_beat=steps_per_beat,
                        beats_per_bar=beats_per_bar,
                        beat_phase=phase % steps_per_beat,
                        bar_phase=phase,
                        tatum_seconds=tatum_seconds / steps_per_beat,
                        confidence=confidence,
                    )

    assert best is not None
    return best


def compute_syncopation_index(meter: MeterEstimate) -> float:
    grid = np.asarray(meter.grid > 0, dtype=float)
    if grid.size == 0 or np.sum(grid) == 0:
        return 0.0

    bar_len = meter.steps_per_beat * meter.beats_per_bar
    weights = np.zeros(bar_len, dtype=float)
    for idx in range(bar_len):
        rel = (idx - meter.bar_phase) % bar_len
        if rel == 0:
            weights[idx] = 5.0
        elif rel % meter.steps_per_beat == 0:
            beat_idx = rel // meter.steps_per_beat
            weights[idx] = 4.0 if beat_idx % 2 == 0 else 3.0
        elif meter.steps_per_beat >= 4 and rel % (meter.steps_per_beat // 2) == 0:
            weights[idx] = 2.0
        else:
            weights[idx] = 1.0

    sync = 0.0
    counted = 0
    for idx, onset in enumerate(grid[:-1]):
        if onset <= 0:
            continue

        cur_weight = weights[idx % bar_len]
        next_idx = idx + 1
        next_weight = weights[next_idx % bar_len]

        if grid[next_idx] <= 0 and next_weight > cur_weight:
            sync += next_weight - cur_weight
            counted += 1
            continue

        future_candidates = []
        for lookahead in range(2, meter.steps_per_beat + 1):
            if idx + lookahead >= len(grid):
                break
            if grid[idx + lookahead] > 0:
                break
            future_weight = weights[(idx + lookahead) % bar_len]
            if future_weight > cur_weight:
                future_candidates.append(future_weight - cur_weight)

        if future_candidates:
            sync += max(future_candidates) * 0.5
            counted += 1

    if counted == 0:
        return 0.0

    normalized = sync / (counted * np.max(weights))
    return float(np.clip(normalized * (0.75 + 0.25 * meter.confidence), 0.0, 1.0))


def _context_probability(sequence: np.ndarray, index: int, order: int) -> tuple[float, float]:
    if index <= 0:
        base_prob = (np.sum(sequence[: max(index, 1)]) + 1.0) / (max(index, 1) + 2.0)
        return float(base_prob), 0.1

    if order == 0 or index - order < 0:
        base_prob = (np.sum(sequence[:index]) + 1.0) / (index + 2.0)
        return float(base_prob), 0.2

    context = tuple(sequence[index - order : index].tolist())
    match_total = 0
    next_onsets = 0

    for start in range(0, index - order):
        if tuple(sequence[start : start + order].tolist()) == context:
            match_total += 1
            next_onsets += int(sequence[start + order])

    if match_total == 0:
        return _context_probability(sequence, index, order - 1)

    prob = (next_onsets + 0.5) / (match_total + 1.0)
    reliability = min(1.0, match_total / 6.0)
    return float(prob), float(reliability)


def compute_idyom_like_surprisal(sequence: np.ndarray, meter: MeterEstimate | None = None, max_order: int = 4) -> dict[str, float]:
    seq = np.asarray(sequence > 0, dtype=int)
    if seq.size < 8:
        return {
            "mean_surprisal": 0.0,
            "surprisal_variance": 0.0,
            "moderate_surprisal_ratio": 0.0,
        }

    if meter is None:
        meter = estimate_meter_from_grid(seq)

    bar_len = meter.steps_per_beat * meter.beats_per_bar
    metrical_prior = np.full(bar_len, 0.08, dtype=float)
    for idx in range(bar_len):
        rel = (idx - meter.bar_phase) % bar_len
        if rel == 0:
            metrical_prior[idx] = 0.88
        elif rel % meter.steps_per_beat == 0:
            metrical_prior[idx] = 0.72 if (rel // meter.steps_per_beat) % 2 == 0 else 0.58
        elif meter.steps_per_beat >= 4 and rel % (meter.steps_per_beat // 2) == 0:
            metrical_prior[idx] = 0.32
        else:
            metrical_prior[idx] = 0.12

    surprisals: list[float] = []
    for idx in range(seq.size):
        order_probs = []
        order_weights = []

        for order in range(0, max_order + 1):
            prob, reliability = _context_probability(seq, idx, order)
            order_probs.append(prob)
            order_weights.append(0.15 + reliability)

        metrical_prob = float(metrical_prior[idx % bar_len])
        order_probs.append(metrical_prob)
        order_weights.append(0.35 + 0.65 * meter.confidence)

        combined_prob = float(np.average(order_probs, weights=order_weights))
        event_prob = combined_prob if seq[idx] == 1 else 1.0 - combined_prob
        event_prob = float(np.clip(event_prob, 1e-4, 1.0))
        surprisals.append(float(-np.log2(event_prob)))

    surprisal_array = np.asarray(surprisals, dtype=float)
    moderate_mask = (surprisal_array >= 0.5) & (surprisal_array <= 2.0)
    return {
        "mean_surprisal": float(np.mean(surprisal_array)),
        "surprisal_variance": float(np.var(surprisal_array)),
        "moderate_surprisal_ratio": float(np.mean(moderate_mask)),
    }


def quantize_audio_onsets_to_meter(binary_seq: np.ndarray, meter: MeterEstimate, frames_per_tatum: int) -> np.ndarray:
    if frames_per_tatum <= 0:
        return np.asarray(binary_seq > 0, dtype=float)

    length = max(int(np.ceil(len(binary_seq) / frames_per_tatum)), meter.steps_per_beat * meter.beats_per_bar)
    grid = np.zeros(length, dtype=float)
    onset_indices = np.where(binary_seq > 0)[0]
    quantized = np.rint(onset_indices / frames_per_tatum).astype(int)
    quantized = np.clip(quantized, 0, length - 1)
    grid[quantized] = 1.0
    return grid


def compute_audio_feature_set(
    audio: AudioFeatures,
    tempo_prior: TempoPrior | tuple[float, float] | None = None,
) -> dict[str, float]:
    periodicity = estimate_periodicity(
        audio.onset_env,
        audio.sr,
        audio.hop_length,
        low_onset_env=audio.low_onset_env,
        tempo_prior=tempo_prior,
    )
    event_density = compute_event_density_from_audio(audio.onset_env, audio.sr, audio.hop_length)
    binary_seq = build_binary_onset_sequence_from_audio(audio.onset_env, audio.sr, audio.hop_length)

    tempo_bpm = max(periodicity["tempo_bpm"], 1e-6)
    beat_period_seconds = 60.0 / tempo_bpm
    tatum_seconds = beat_period_seconds / 4.0
    frames_per_tatum = max(int(round(tatum_seconds * audio.sr / audio.hop_length)), 1)

    coarse_grid = quantize_audio_onsets_to_meter(binary_seq, estimate_meter_from_grid(binary_seq, tatum_seconds=tatum_seconds), frames_per_tatum)
    meter = estimate_meter_from_grid(coarse_grid, tatum_seconds=tatum_seconds)

    surprisal = compute_idyom_like_surprisal(coarse_grid.astype(int), meter=meter)
    syncopation = compute_syncopation_index(meter)

    complexity = _inverse_u(syncopation, center=0.35, width=0.30)
    confidence = float(
        np.clip(
            0.30 * periodicity["beat_strength"]
            + 0.25 * periodicity["beat_clarity"]
            + 0.20 * periodicity["periodicity_stability"]
            + 0.25 * meter.confidence,
            0.0,
            1.0,
        )
    )

    return {
        **periodicity,
        **surprisal,
        "syncopation_index": syncopation,
        "complexity_balance_score": complexity,
        "event_density": event_density,
        "low_freq_flux": float(np.clip(audio.low_freq_flux * 3.0, 0.0, 1.0)),
        "bass_pulse_strength": float(np.clip(audio.bass_pulse_strength / 32.0, 0.0, 1.0)),
        "low_mid_balance": float(np.clip(audio.low_mid_balance / 2.0, 0.0, 1.0)),
        "confidence": confidence,
    }


def compute_symbolic_feature_set(onsets: np.ndarray, grid: np.ndarray, tatum: float) -> dict[str, float]:
    if onsets.size < 2:
        return {
            "tempo_bpm": 0.0,
            "beat_strength": 0.0,
            "beat_clarity": 0.0,
            "periodicity_stability": 0.0,
            "tempo_alignment_score": 0.0,
            "syncopation_index": 0.0,
            "complexity_balance_score": 0.0,
            "event_density": 0.0,
            "mean_surprisal": 0.0,
            "surprisal_variance": 0.0,
            "moderate_surprisal_ratio": 0.0,
            "low_freq_flux": 0.0,
            "bass_pulse_strength": 0.0,
            "low_mid_balance": 0.0,
            "confidence": 0.0,
        }

    iois = np.diff(np.sort(onsets))
    positive_iois = iois[iois > 1e-4]
    median_ioi = float(np.median(positive_iois)) if positive_iois.size else 0.5
    tempo_bpm = 60.0 / max(median_ioi, 1e-4)
    periodicity_stability = float(np.clip(1.0 - (np.std(iois) / (np.mean(iois) + 1e-6)), 0.0, 1.0))
    beat_strength = float(_sigmoid(len(onsets) / max(onsets[-1] - onsets[0], 1.0), center=2.0, slope=1.0))
    beat_clarity = periodicity_stability
    tempo_alignment_score = _tempo_preference_score(tempo_bpm)

    steps_per_beat = 4 if tatum <= median_ioi / 2.5 else 2
    meter = estimate_meter_from_grid(grid.astype(int), candidate_steps_per_beat=(steps_per_beat,), tatum_seconds=tatum)
    syncopation_index = compute_syncopation_index(meter)
    complexity_balance_score = _inverse_u(syncopation_index, center=0.35, width=0.30)
    surprisal = compute_idyom_like_surprisal(meter.grid.astype(int), meter=meter)
    confidence = float(
        np.clip(
            0.35 * beat_clarity + 0.25 * beat_strength + 0.15 * tempo_alignment_score + 0.25 * meter.confidence,
            0.0,
            1.0,
        )
    )

    return {
        "tempo_bpm": tempo_bpm,
        "beat_strength": beat_strength,
        "beat_clarity": beat_clarity,
        "periodicity_stability": periodicity_stability,
        "tempo_alignment_score": tempo_alignment_score,
        "syncopation_index": syncopation_index,
        "complexity_balance_score": complexity_balance_score,
        "event_density": compute_event_density_from_onsets(onsets),
        **surprisal,
        "low_freq_flux": 0.0,
        "bass_pulse_strength": 0.0,
        "low_mid_balance": 0.0,
        "confidence": confidence,
    }
