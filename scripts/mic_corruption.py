#!/usr/bin/env python
"""
Mic-local corruption utilities for Claim-2 verification.

Design goals
------------
- Existing WAVs only: noise is sourced from silence windows in the same recording.
- Mic-local: only MicL/MicR are corrupted; LDV remains clean.
- Deterministic: all randomness is driven by a fixed RNG seed and recorded noise centers.
- In-band: SNR is defined and enforced within an analysis band (default 500–2000 Hz).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MicCorruptionConfig:
    snr_db: float
    band_lo_hz: float
    band_hi_hz: float
    preclip_gain: float
    clip_limit: float
    seed: int

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def bandpass_fft(x: np.ndarray, *, fs: int, lo_hz: float, hi_hz: float) -> np.ndarray:
    """
    Deterministic frequency-domain hard-mask bandpass.
    Returns a time-domain signal of the same length.
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n <= 0:
        raise ValueError("Empty signal")
    if not (0.0 <= float(lo_hz) < float(hi_hz) <= float(fs) / 2.0 + 1e-9):
        raise ValueError(f"Invalid bandpass: lo={lo_hz}, hi={hi_hz}, fs={fs}")

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    mask = (freqs >= float(lo_hz)) & (freqs <= float(hi_hz))
    X = X * mask.astype(np.float64)
    y = np.fft.irfft(X, n)
    return y.astype(np.float64, copy=False)


def apply_occlusion_fft(
    x: np.ndarray,
    *,
    fs: int,
    kind: str,
    lowpass_hz: float | None = None,
    tilt_k: float | None = None,
    tilt_pivot_hz: float | None = None,
) -> np.ndarray:
    """
    Zero-phase (magnitude-only) spectral shaping to approximate mic occlusion.

    Notes:
    - This intentionally avoids introducing extra delay (preserves phase).
    - The goal is to reduce usable mic coherence/SNR within the analysis band.
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n <= 0:
        raise ValueError("Empty signal")

    kind = str(kind).lower().strip()
    if kind in ("none", "off", "0"):
        return x.astype(np.float64, copy=False)

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))

    if kind == "lowpass":
        if lowpass_hz is None:
            raise ValueError("lowpass_hz is required for occlusion kind=lowpass")
        fc = float(lowpass_hz)
        if not (0.0 < fc <= float(fs) / 2.0 + 1e-9):
            raise ValueError(f"Invalid lowpass_hz: {fc}")
        mask = freqs <= fc
        X = X * mask.astype(np.float64)
    elif kind == "tilt":
        if tilt_k is None or tilt_pivot_hz is None:
            raise ValueError("tilt_k and tilt_pivot_hz are required for occlusion kind=tilt")
        k = float(tilt_k)
        pivot = float(tilt_pivot_hz)
        if pivot <= 0.0 or pivot > float(fs) / 2.0:
            raise ValueError(f"Invalid tilt_pivot_hz: {pivot}")
        if k < 0.0:
            raise ValueError(f"Invalid tilt_k: {k}")
        gain = np.ones_like(freqs, dtype=np.float64)
        m = freqs >= pivot
        gain[m] = (np.maximum(freqs[m], pivot) / pivot) ** (-k)
        X = X * gain
    else:
        raise ValueError(f"Unknown occlusion kind: {kind}")

    y = np.fft.irfft(X, n)
    return y.astype(np.float64, copy=False)


def compute_inband_power(x: np.ndarray, *, fs: int, lo_hz: float, hi_hz: float) -> float:
    xb = bandpass_fft(x, fs=fs, lo_hz=lo_hz, hi_hz=hi_hz)
    p = float(np.mean(xb * xb))
    if not np.isfinite(p):
        raise ValueError("Non-finite in-band power")
    return p


def select_silence_centers(
    micl: np.ndarray,
    *,
    fs: int,
    centers_grid: np.ndarray,
    window_sec: float,
    silence_percent: float,
    min_windows: int = 3,
) -> list[float]:
    """
    Silence windows are selected from the candidate grid by MicL RMS.
    Returns center seconds (float).
    """
    micl = np.asarray(micl, dtype=np.float64)
    centers_grid = np.asarray(centers_grid, dtype=np.float64)

    win_samples = int(round(float(window_sec) * float(fs)))
    half = win_samples // 2
    if win_samples <= 0:
        raise ValueError("Invalid window_sec")

    rms_vals: list[float] = []
    valid_centers: list[float] = []
    for csec in centers_grid.tolist():
        center_samp = int(round(float(csec) * float(fs)))
        start = int(center_samp - half)
        end = int(start + win_samples)
        if start < 0 or end > micl.size:
            continue
        seg = micl[start:end]
        r = float(np.sqrt(np.mean(seg * seg)))
        rms_vals.append(r)
        valid_centers.append(float(csec))

    if not valid_centers:
        raise ValueError("No valid centers in bounds for silence selection")

    rms_arr = np.asarray(rms_vals, dtype=np.float64)
    n = int(rms_arr.size)
    k = int(np.ceil(float(silence_percent) / 100.0 * float(n)))
    k = max(int(min_windows), int(k))
    if k > n:
        raise ValueError(f"Not enough candidate windows for silence set: need={k}, have={n}")

    idx = np.argsort(rms_arr)[:k]
    centers = [valid_centers[int(i)] for i in idx.tolist()]
    return centers


def choose_noise_center(rng: np.random.Generator, silence_centers: list[float]) -> float:
    if not silence_centers:
        raise ValueError("Empty silence_centers")
    i = int(rng.integers(0, len(silence_centers)))
    return float(silence_centers[i])


def apply_mic_corruption(
    mic_clean: np.ndarray,
    noise_window: np.ndarray,
    *,
    cfg: MicCorruptionConfig,
    fs: int,
    signal_for_alpha: np.ndarray | None = None,
    signal_for_mix: np.ndarray | None = None,
    eps: float = 1e-18,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Corrupt a mic window by adding in-band noise at a target SNR, then applying
    gain→clip→de-gain nonlinearity.

    Returns (mic_corrupt, diag) where diag includes achieved SNR (pre-clip) and clip fraction.
    """
    mic_clean = np.asarray(mic_clean, dtype=np.float64)
    noise_window = np.asarray(noise_window, dtype=np.float64)
    if mic_clean.shape != noise_window.shape:
        raise ValueError(f"Shape mismatch: mic={mic_clean.shape}, noise={noise_window.shape}")

    if signal_for_alpha is None:
        signal_for_alpha = mic_clean
    if signal_for_mix is None:
        signal_for_mix = mic_clean
    signal_for_alpha = np.asarray(signal_for_alpha, dtype=np.float64)
    signal_for_mix = np.asarray(signal_for_mix, dtype=np.float64)
    if signal_for_alpha.shape != mic_clean.shape or signal_for_mix.shape != mic_clean.shape:
        raise ValueError("signal_for_alpha/signal_for_mix must match mic_clean shape")

    lo = float(cfg.band_lo_hz)
    hi = float(cfg.band_hi_hz)
    snr_db = float(cfg.snr_db)

    s_bp = bandpass_fft(signal_for_alpha, fs=fs, lo_hz=lo, hi_hz=hi)
    n_bp = bandpass_fft(noise_window, fs=fs, lo_hz=lo, hi_hz=hi)

    P_s = float(np.mean(s_bp * s_bp))
    P_n = float(np.mean(n_bp * n_bp))
    if not (np.isfinite(P_s) and np.isfinite(P_n)):
        raise ValueError("Non-finite powers in corruption")
    if P_s <= 0.0:
        raise ValueError("Zero in-band signal power")
    if P_n <= eps:
        raise ValueError("Noise in-band power too small")

    alpha = float(np.sqrt(P_s / (P_n * (10.0 ** (snr_db / 10.0)) + eps)))
    mic_noisy = signal_for_mix + alpha * n_bp

    pre = float(cfg.preclip_gain) * mic_noisy
    clip_limit = float(cfg.clip_limit)
    mic_clip = np.clip(pre, -clip_limit, clip_limit)
    mic_corrupt = (mic_clip / float(cfg.preclip_gain)).astype(np.float64, copy=False)

    # Diagnostics
    # Achieved SNR is computed using the mixed signal power (post-occlusion, if any).
    s_mix_bp = bandpass_fft(signal_for_mix, fs=fs, lo_hz=lo, hi_hz=hi)
    P_s_mix = float(np.mean(s_mix_bp * s_mix_bp))
    P_n_scaled = float((alpha * alpha) * P_n)
    snr_achieved_db = 10.0 * float(np.log10(P_s_mix / (P_n_scaled + eps)))
    clip_frac = float(np.mean(np.abs(pre) >= clip_limit))

    diag = {
        "snr_target_db": snr_db,
        "snr_achieved_db_preclip": float(snr_achieved_db),
        "alpha": float(alpha),
        "clip_frac": float(clip_frac),
        "P_s_ref": float(P_s),
        "P_s_mix": float(P_s_mix),
        "band_lo_hz": lo,
        "band_hi_hz": hi,
        "preclip_gain": float(cfg.preclip_gain),
        "clip_limit": clip_limit,
    }
    return mic_corrupt, diag
