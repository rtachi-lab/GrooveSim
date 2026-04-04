from __future__ import annotations

from io import BytesIO
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf


def _synthesize_drum_pulse(note: int, velocity: int, duration_s: float, sr: int) -> np.ndarray:
    n = max(int(duration_s * sr), 1)
    t = np.arange(n, dtype=np.float32) / sr
    amp = float(np.clip(velocity / 127.0, 0.0, 1.0))

    if note in {35, 36}:  # kick
        env = np.exp(-10.0 * t)
        freq = 110.0 * np.exp(-6.0 * t) + 40.0
        pulse = np.sin(2.0 * np.pi * freq * t) * env
    elif note in {38, 40}:  # snare
        env = np.exp(-18.0 * t)
        rng = np.random.default_rng(note * 1000 + velocity + n)
        pulse = rng.standard_normal(n).astype(np.float32) * env
    elif note in {42, 44, 46}:  # hihat
        env = np.exp(-45.0 * t)
        rng = np.random.default_rng(note * 1000 + velocity + n)
        pulse = rng.standard_normal(n).astype(np.float32) * env
    else:
        env = np.exp(-20.0 * t)
        freq = 180.0 + (note % 24) * 12.0
        pulse = np.sin(2.0 * np.pi * freq * t) * env

    return (0.35 * amp * pulse).astype(np.float32)


def _fallback_render_with_drums(midi: pretty_midi.PrettyMIDI, sr: int) -> np.ndarray:
    end_time = max(midi.get_end_time(), 0.5)
    waveform = np.zeros(int(np.ceil(end_time * sr)) + sr, dtype=np.float32)

    for instrument in midi.instruments:
        for note in instrument.notes:
            start = int(max(note.start, 0.0) * sr)
            duration = max(note.end - note.start, 0.05)
            pulse = _synthesize_drum_pulse(note.pitch, note.velocity, duration, sr)
            end = min(start + len(pulse), len(waveform))
            waveform[start:end] += pulse[: end - start]

    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        waveform /= peak
    return waveform


def render_midi_to_waveform(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    midi = pretty_midi.PrettyMIDI(path)
    waveform = _render_pretty_midi(midi, sr)
    return waveform, sr


def _render_pretty_midi(midi: pretty_midi.PrettyMIDI, sr: int) -> np.ndarray:
    waveform = np.asarray(midi.synthesize(fs=sr), dtype=np.float32)
    if waveform.size == 0 or not np.all(np.isfinite(waveform)) or np.max(np.abs(waveform)) == 0.0:
        waveform = _fallback_render_with_drums(midi, sr)
    return waveform


def render_midi_bytes_to_waveform(data: bytes, sr: int = 22050) -> tuple[np.ndarray, int]:
    midi = pretty_midi.PrettyMIDI(BytesIO(data))
    waveform = _render_pretty_midi(midi, sr)
    return waveform, sr


def get_midi_reference_tempo(path: str) -> float:
    midi = pretty_midi.PrettyMIDI(path)
    return get_pretty_midi_tempo(midi)


def get_midi_reference_tempo_bytes(data: bytes) -> float:
    midi = pretty_midi.PrettyMIDI(BytesIO(data))
    return get_pretty_midi_tempo(midi)


def get_pretty_midi_tempo(midi: pretty_midi.PrettyMIDI) -> float:
    times, tempi = midi.get_tempo_changes()
    if len(tempi) == 0:
        return float(midi.estimate_tempo())
    if len(tempi) == 1:
        return float(tempi[0])

    durations = np.diff(np.append(times, midi.get_end_time()))
    weighted = float(np.average(tempi, weights=np.maximum(durations, 1e-6)))
    return weighted


def render_midi_to_wav_file(path: str, output_path: str | None = None, sr: int = 22050) -> str:
    waveform, sr = render_midi_to_waveform(path, sr=sr)
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = str(Path(temp_dir) / (Path(path).stem + "_rendered.wav"))
    sf.write(output_path, waveform, sr)
    return output_path
