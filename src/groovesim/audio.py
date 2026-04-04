from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
import subprocess

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d


@dataclass
class AudioFeatures:
    y: np.ndarray
    sr: int
    hop_length: int
    times: np.ndarray
    onset_env: np.ndarray
    low_onset_env: np.ndarray
    mid_onset_env: np.ndarray
    spectral_flux: np.ndarray
    low_freq_flux: float
    bass_pulse_strength: float
    low_mid_balance: float


def load_audio(path: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return normalize_audio(y), sr


def load_audio_bytes(data: bytes, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "wav",
        "pipe:1",
    ]
    result = subprocess.run(command, input=data, capture_output=True, check=True)
    y, sr = sf.read(BytesIO(result.stdout), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return normalize_audio(y), sr


def normalize_audio(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        raise ValueError("Audio file is empty.")
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def _band_limited_onset(y: np.ndarray, sr: int, low_hz: float, high_hz: float, hop_length: int) -> np.ndarray:
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return np.zeros(stft.shape[1], dtype=np.float32)
    band_spec = stft[mask]
    flux = np.diff(band_spec, axis=1, prepend=band_spec[:, :1])
    flux = np.maximum(flux, 0.0)
    env = flux.sum(axis=0)
    env = uniform_filter1d(env, size=3, mode="nearest")
    if np.max(env) > 0:
        env = env / np.max(env)
    return env.astype(np.float32)


def extract_audio_features_from_array(y: np.ndarray, sr: int = 22050, hop_length: int = 512) -> AudioFeatures:
    y = normalize_audio(np.asarray(y, dtype=np.float32))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    spectral_flux = onset_env.copy()
    if np.max(onset_env) > 0:
        onset_env = onset_env / np.max(onset_env)

    low_onset_env = _band_limited_onset(y, sr, 20.0, 150.0, hop_length)
    mid_onset_env = _band_limited_onset(y, sr, 150.0, 4000.0, hop_length)

    low_flux_raw = float(np.mean(low_onset_env))
    mid_flux_raw = float(np.mean(mid_onset_env))
    bass_pulse_strength = float(np.max(librosa.autocorrelate(low_onset_env, max_size=min(512, len(low_onset_env)))))
    low_mid_balance = float(low_flux_raw / (mid_flux_raw + 1e-6))

    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    return AudioFeatures(
        y=y,
        sr=sr,
        hop_length=hop_length,
        times=times,
        onset_env=onset_env.astype(np.float32),
        low_onset_env=low_onset_env,
        mid_onset_env=mid_onset_env,
        spectral_flux=spectral_flux.astype(np.float32),
        low_freq_flux=low_flux_raw,
        bass_pulse_strength=bass_pulse_strength,
        low_mid_balance=low_mid_balance,
    )


def extract_audio_features(path: str, sr: int = 22050, hop_length: int = 512) -> AudioFeatures:
    y, sr = load_audio(path, target_sr=sr)
    return extract_audio_features_from_array(y, sr=sr, hop_length=hop_length)


def extract_audio_features_from_bytes(data: bytes, sr: int = 22050, hop_length: int = 512) -> AudioFeatures:
    y, sr = load_audio_bytes(data, target_sr=sr)
    return extract_audio_features_from_array(y, sr=sr, hop_length=hop_length)
