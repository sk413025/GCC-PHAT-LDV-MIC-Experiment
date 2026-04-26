#!/usr/bin/env python
"""
Multi-Sensor Near-Field Fusion (MSNF) DoA estimation.

Exploits LDV geometric advantage by treating it as an independent
geometric constraint rather than a microphone replacement.

TDOA measurements:
  tau1 : MicL -> MicR      (GCC-PHAT, guided)
  tau2 : LDV  -> MicL      (GCC-PHAT, guided, large lag)
  tau3 : LDV  -> MicR      (GCC-PHAT, guided, large lag)
  tau4 : pseudo-MicL -> MicR (OMP-aligned LDV, GCC-PHAT)

Methods:
  A : MIC-MIC           tau1 -> arcsin(tau1*c/d)
  B : OMP               tau4 -> arcsin(tau4*c/d)
  C : theta-fusion      PSR-weighted mean of theta_A and theta_B
  D : MSNF-2            WLS(tau1, tau2)
  E : MSNF-3            WLS(tau1, tau2, tau3)
  F : MSNF-4            WLS(tau1, tau2, tau3, tau4)

Outputs per-speaker summary.json with Phase 0 diagnostic and all method results.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.fft import fft, ifft
from scipy.io import wavfile
from scipy.optimize import minimize_scalar
from scipy.signal import butter, filtfilt, istft, stft

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# [A] Constants & geometry
# ─────────────────────────────────────────────────────────────────────

GEOMETRY = {
    "ldv": (0.0, 0.5),
    "mic_left": (-0.7, 2.0),
    "mic_right": (0.7, 2.0),
    "speakers": {
        "18": (0.8, 0.0),
        "19": (0.4, 0.0),
        "20": (0.0, 0.0),
        "21": (-0.4, 0.0),
        "22": (-0.8, 0.0),
    },
}

DEFAULT_CONFIG = {
    "fs": 48000,
    "n_fft": 6144,
    "hop_length": 160,
    "max_lag": 50,
    "max_k": 3,
    "tw": 64,
    "freq_min": 100,
    "freq_max": 8000,
    "speed_of_sound": 343.0,
    "mic_spacing": 1.4,
    # GCC-PHAT for tau1 (MicL-MicR)
    "gcc_max_lag_ms": 10.0,
    "gcc_guided_peak_radius_ms": 0.3,
    "psr_exclude_samples": 50,
    # GCC-PHAT for tau2, tau3 (LDV-Mic)
    "tau2_max_lag_ms": 7.0,
    "tau2_guided_radius_ms": 2.0,
    # Bandpass
    "gcc_bandpass_low": 0.0,
    "gcc_bandpass_high": 0.0,
    # Segment selection
    "analysis_slice_sec": 5.0,
    "eval_window_sec": 5.0,
    "segment_spacing_sec": 50.0,
    "segment_offset_sec": 100.0,
    # OMP prealign
    "ldv_prealign": "gcc_phat",
    # PSR -> weight
    "psr_floor_db": -20.0,
    "psr_ceiling_db": 20.0,
}


def compute_all_ground_truths(speaker_key: str, c: float = 343.0, d: float = 1.4) -> dict:
    """Compute ground-truth tau1/tau2/tau3 and theta for a speaker."""
    if speaker_key not in GEOMETRY["speakers"]:
        raise ValueError(f"Unknown speaker key: {speaker_key}")

    x_s, y_s = GEOMETRY["speakers"][speaker_key]
    x_ldv, y_ldv = GEOMETRY["ldv"]
    x_ml, y_ml = GEOMETRY["mic_left"]
    x_mr, y_mr = GEOMETRY["mic_right"]

    dist_ldv = float(np.hypot(x_s - x_ldv, y_s - y_ldv))
    dist_ml = float(np.hypot(x_s - x_ml, y_s - y_ml))
    dist_mr = float(np.hypot(x_s - x_mr, y_s - y_mr))

    tau1_true = (dist_ml - dist_mr) / c
    tau2_true = (dist_ldv - dist_ml) / c
    tau3_true = (dist_ldv - dist_mr) / c

    sin_theta = tau1_true * c / d
    sin_theta = float(np.clip(sin_theta, -1.0, 1.0))
    theta_true = float(np.degrees(np.arcsin(sin_theta)))

    return {
        "x_s_true": float(x_s),
        "theta_true_deg": theta_true,
        "tau1_true_ms": float(tau1_true * 1000.0),
        "tau2_true_ms": float(tau2_true * 1000.0),
        "tau3_true_ms": float(tau3_true * 1000.0),
        "d_ldv": dist_ldv,
        "d_micl": dist_ml,
        "d_micr": dist_mr,
    }


# ─────────────────────────────────────────────────────────────────────
# [B] I/O and signal utilities
# ─────────────────────────────────────────────────────────────────────


def load_wav(path: str) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    return sr, data


def bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 5
) -> np.ndarray:
    nyq = 0.5 * fs
    low = float(lowcut) / nyq
    high = float(highcut) / nyq
    if not (0.0 < low < high < 1.0):
        raise ValueError(
            f"Invalid bandpass range: lowcut={lowcut}, highcut={highcut}, fs={fs}"
        )
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def extract_centered_slice(
    signals: list[np.ndarray], *, center_sample: int, slice_samples: int
) -> tuple[int, int]:
    if slice_samples <= 0:
        raise ValueError(f"slice_samples must be > 0, got {slice_samples}")
    min_len = min(len(s) for s in signals)
    center_sample = int(np.clip(center_sample, 0, max(0, min_len - 1)))
    slice_start = max(0, center_sample - slice_samples // 2)
    slice_end = slice_start + slice_samples
    if slice_end > min_len:
        slice_end = min_len
        slice_start = max(0, slice_end - slice_samples)
    return slice_start, slice_end


def apply_fractional_delay_fd(
    signal: np.ndarray, fs: int, delay_sec: float
) -> np.ndarray:
    delay_sec = float(delay_sec)
    if abs(delay_sec) < 1e-12:
        return signal.astype(np.float64, copy=True)
    x = signal.astype(np.float64, copy=False)
    n = len(x)
    if n == 0:
        return x.copy()
    n_fft = 1 << int(np.ceil(np.log2(max(2, n * 2))))
    X = np.fft.rfft(x, n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(fs))
    X *= np.exp(-1j * 2 * np.pi * freqs * delay_sec)
    y = np.fft.irfft(X, n_fft)[:n]
    return y.astype(np.float64, copy=False)


# ─────────────────────────────────────────────────────────────────────
# [C] Near-field forward models & WLS solver
# ─────────────────────────────────────────────────────────────────────


def _d_ldv(x_s: float) -> float:
    """Distance from speaker at (x_s, 0) to LDV at (0, 0.5)."""
    return float(np.sqrt(x_s**2 + 0.25))


def _d_micl(x_s: float) -> float:
    """Distance from speaker at (x_s, 0) to MicL at (-0.7, 2.0)."""
    return float(np.sqrt((x_s + 0.7) ** 2 + 4.0))


def _d_micr(x_s: float) -> float:
    """Distance from speaker at (x_s, 0) to MicR at (0.7, 2.0)."""
    return float(np.sqrt((x_s - 0.7) ** 2 + 4.0))


def tau1_model(x_s: float, c: float = 343.0) -> float:
    """Forward model: tau1 = (d_MicL - d_MicR) / c."""
    return (_d_micl(x_s) - _d_micr(x_s)) / c


def tau2_model(x_s: float, c: float = 343.0) -> float:
    """Forward model: tau2 = (d_LDV - d_MicL) / c."""
    return (_d_ldv(x_s) - _d_micl(x_s)) / c


def tau3_model(x_s: float, c: float = 343.0) -> float:
    """Forward model: tau3 = (d_LDV - d_MicR) / c."""
    return (_d_ldv(x_s) - _d_micr(x_s)) / c


def xs_to_theta(x_s: float, c: float = 343.0, d: float = 1.4) -> float:
    """Convert speaker x-position to DoA angle (MIC-MIC convention)."""
    tau1 = tau1_model(x_s, c)
    sin_theta = tau1 * c / d
    sin_theta = float(np.clip(sin_theta, -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_theta)))


def tau_to_doa(tau_ms: float, c: float = 343.0, d: float = 1.4) -> float:
    """Convert tau1 (ms) to DoA angle."""
    sin_theta = (tau_ms / 1000.0) * c / d
    sin_theta = float(np.clip(sin_theta, -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_theta)))


def psr_to_weight(
    psr_db: float, psr_floor_db: float = -20.0, psr_ceiling_db: float = 20.0
) -> float:
    """Convert PSR (dB) to WLS weight."""
    psr_clipped = float(np.clip(psr_db, psr_floor_db, psr_ceiling_db))
    return float(10.0 ** (psr_clipped / 10.0))


def solve_msnf(
    measurements: list[tuple[float, callable]],
    weights: list[float],
    c: float = 343.0,
    d: float = 1.4,
) -> tuple[float, float]:
    """
    Weighted Least Squares solver for near-field source position.

    Parameters
    ----------
    measurements : list of (tau_measured_sec, forward_model_func)
        Each forward model takes x_s and returns predicted tau in seconds.
    weights : list of float
        Weight for each measurement (from PSR).

    Returns
    -------
    x_s_hat, theta_hat_deg
    """
    if not measurements:
        return 0.0, 0.0

    def cost(x_s):
        total = 0.0
        for (tau_meas, model), w in zip(measurements, weights):
            residual = tau_meas - model(x_s)
            total += w * residual**2
        return total

    result = minimize_scalar(cost, bounds=(-3.0, 3.0), method="bounded")
    x_s_hat = float(result.x)
    theta_hat = xs_to_theta(x_s_hat, c, d)
    return x_s_hat, theta_hat


# ─────────────────────────────────────────────────────────────────────
# [D] GCC-PHAT with guided search
# ─────────────────────────────────────────────────────────────────────


def gcc_phat_full_analysis(
    sig1: np.ndarray,
    sig2: np.ndarray,
    fs: int,
    *,
    max_tau: float | None = None,
    bandpass: tuple[float, float] | None = None,
    psr_exclude_samples: int = 50,
    guided_tau: float | None = None,
    guided_radius: float | None = None,
) -> tuple[float, float]:
    if bandpass is not None:
        sig1 = bandpass_filter(sig1, bandpass[0], bandpass[1], fs)
        sig2 = bandpass_filter(sig2, bandpass[0], bandpass[1], fs)

    n = len(sig1) + len(sig2)
    SIG1 = fft(sig1, n)
    SIG2 = fft(sig2, n)
    R = SIG1 * np.conj(SIG2)
    R = R / (np.abs(R) + 1e-10)
    cc = np.real(ifft(R))

    if max_tau is not None:
        max_shift = int(max_tau * fs)
    else:
        max_shift = n // 2

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    abs_cc = np.abs(cc)

    peak_idx = None
    if (
        guided_tau is not None
        and guided_radius is not None
        and float(guided_radius) > 0
    ):
        guided_center_idx = int(round(float(guided_tau) * fs)) + max_shift
        guided_radius_samples = int(round(float(guided_radius) * fs))
        lo = max(0, guided_center_idx - guided_radius_samples)
        hi = min(len(abs_cc) - 1, guided_center_idx + guided_radius_samples)
        if lo <= hi:
            peak_idx = int(np.argmax(abs_cc[lo : hi + 1])) + lo

    if peak_idx is None:
        peak_idx = int(np.argmax(abs_cc))

    if 0 < peak_idx < len(abs_cc) - 1:
        y0 = abs_cc[peak_idx - 1]
        y1 = abs_cc[peak_idx]
        y2 = abs_cc[peak_idx + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            shift = 0.5 * (y0 - y2) / denom
        else:
            shift = 0.0
    else:
        shift = 0.0

    tau = ((peak_idx - max_shift) + shift) / fs

    mask = np.ones_like(abs_cc, dtype=bool)
    lo = max(0, peak_idx - psr_exclude_samples)
    hi = min(len(abs_cc), peak_idx + psr_exclude_samples + 1)
    mask[lo:hi] = False

    sidelobe_max = abs_cc[mask].max() if np.any(mask) else 0.0
    peak_val = abs_cc[peak_idx]
    psr = 20 * np.log10(peak_val / (sidelobe_max + 1e-10))

    return float(tau), float(psr)


# ─────────────────────────────────────────────────────────────────────
# [E] OMP alignment
# ─────────────────────────────────────────────────────────────────────


def normalize_per_freq_maxabs(
    X_stft: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if X_stft.ndim == 1:
        max_abs = np.abs(X_stft).max()
        if max_abs < 1e-10:
            max_abs = 1.0
        return X_stft / max_abs, max_abs
    if X_stft.ndim == 2:
        max_abs = np.abs(X_stft).max(axis=-1)
        max_abs = np.where(max_abs < 1e-10, 1.0, max_abs)
        return X_stft / max_abs[:, np.newaxis], max_abs
    max_abs = np.abs(X_stft).max(axis=(-2, -1))
    max_abs = np.where(max_abs < 1e-10, 1.0, max_abs)
    return X_stft / max_abs[:, np.newaxis, np.newaxis], max_abs


def build_lagged_dictionary(
    X_stft: np.ndarray, max_lag: int, tw: int, start_t: int
) -> np.ndarray:
    n_freq, n_time = X_stft.shape
    n_lags = 2 * max_lag + 1
    Dict_tensor = np.zeros((n_freq, n_lags, tw), dtype=X_stft.dtype)
    for lag_idx, lag in enumerate(range(-max_lag, max_lag + 1)):
        t_start = start_t + lag
        t_end = t_start + tw
        if t_start >= 0 and t_end <= n_time:
            Dict_tensor[:, lag_idx, :] = X_stft[:, t_start:t_end]
    return Dict_tensor


def omp_single_freq(
    Dict_f: np.ndarray, Y_f: np.ndarray, max_k: int
) -> tuple[list[int], np.ndarray, np.ndarray]:
    n_lags, tw = Dict_f.shape
    D = Dict_f.T
    D_norms = np.linalg.norm(np.abs(D), axis=0, keepdims=True) + 1e-10
    D_normalized = D / D_norms
    residual = Y_f.copy()
    selected_lags: list[int] = []
    coeffs = None
    for _ in range(max_k):
        corrs = np.abs(D_normalized.conj().T @ residual)
        for lag in selected_lags:
            corrs[lag] = -np.inf
        best_lag = int(np.argmax(corrs))
        selected_lags.append(best_lag)
        A = D[:, selected_lags]
        coeffs, _, _, _ = np.linalg.lstsq(A, Y_f, rcond=None)
        reconstructed = A @ coeffs
        residual = Y_f - reconstructed
    return selected_lags, coeffs, reconstructed


def apply_omp_alignment(
    Zxx_ldv: np.ndarray, Zxx_mic: np.ndarray, config: dict, start_t: int
) -> np.ndarray:
    max_lag = int(config["max_lag"])
    max_k = int(config["max_k"])
    tw = int(config["tw"])
    freq_min = float(config["freq_min"])
    freq_max = float(config["freq_max"])
    fs = int(config["fs"])
    n_fft = int(config["n_fft"])

    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    freq_indices = np.where(freq_mask)[0]

    Dict_full = build_lagged_dictionary(Zxx_ldv, max_lag, tw, start_t)
    Dict_selected = Dict_full[freq_mask, :, :]

    Zxx_omp = Zxx_ldv.copy()

    for f_idx in range(len(freq_indices)):
        Dict_f = Dict_selected[f_idx]
        Y_f = Zxx_mic[freq_indices[f_idx], start_t : start_t + tw]

        Dict_norm, _ = normalize_per_freq_maxabs(Dict_f)
        Y_norm, _ = normalize_per_freq_maxabs(Y_f)

        selected_lags, _, _ = omp_single_freq(Dict_norm, Y_norm, max_k)

        D_orig = Dict_selected[f_idx].T
        A = D_orig[:, selected_lags]
        Y_orig = Zxx_mic[freq_indices[f_idx], start_t : start_t + tw]
        coeffs_orig, _, _, _ = np.linalg.lstsq(A, Y_orig, rcond=None)
        reconstructed_orig = A @ coeffs_orig

        Zxx_omp[freq_indices[f_idx], start_t : start_t + tw] = reconstructed_orig

    return Zxx_omp


# ─────────────────────────────────────────────────────────────────────
# [F] Segment selection
# ─────────────────────────────────────────────────────────────────────


def scan_segment_centers(
    mic_left_signal: np.ndarray,
    mic_right_signal: np.ndarray,
    *,
    fs: int,
    eval_window_sec: float,
    max_lag_ms: float,
    bandpass: tuple[float, float] | None,
    psr_exclude_samples: int,
    tau_true_ms: float,
    guided_tau_ms: float | None = None,
    guided_radius_ms: float | None = None,
    scan_start_sec: float,
    scan_end_sec: float,
    scan_hop_sec: float,
    scan_psr_min_db: float | None,
    scan_tau_err_max_ms: float | None,
    scan_sort_by: str,
    n_segments: int,
    min_separation_sec: float | None,
) -> tuple[list[float], dict]:
    """Select best segment centers based on MicL-MicR GCC quality."""
    eval_window_samples = int(eval_window_sec * fs)
    max_tau = float(max_lag_ms) / 1000.0
    candidates: list[dict] = []

    center = scan_start_sec
    while center <= scan_end_sec + 1e-9:
        center_sample = int(center * fs)
        t_start = center_sample - eval_window_samples // 2
        t_end = t_start + eval_window_samples
        if (
            t_start < 0
            or t_end > len(mic_left_signal)
            or t_end > len(mic_right_signal)
        ):
            center += scan_hop_sec
            continue

        mic_l_seg = mic_left_signal[t_start:t_end]
        mic_r_seg = mic_right_signal[t_start:t_end]
        tau_sec, psr_db = gcc_phat_full_analysis(
            mic_l_seg.astype(np.float64, copy=False),
            mic_r_seg.astype(np.float64, copy=False),
            fs,
            max_tau=max_tau,
            bandpass=bandpass,
            psr_exclude_samples=psr_exclude_samples,
            guided_tau=(
                None if guided_tau_ms is None else float(guided_tau_ms) / 1000.0
            ),
            guided_radius=(
                None if guided_radius_ms is None else float(guided_radius_ms) / 1000.0
            ),
        )
        tau_ms = float(tau_sec * 1000.0)
        tau_err_ms = float(abs(tau_ms - tau_true_ms))

        candidates.append(
            {
                "center_sec": float(center),
                "tau_ms": tau_ms,
                "psr_db": float(psr_db),
                "tau_err_ms": tau_err_ms,
            }
        )
        center += scan_hop_sec

    # Filter
    filtered = []
    for cand in candidates:
        if scan_psr_min_db is not None and cand["psr_db"] < scan_psr_min_db:
            continue
        if scan_tau_err_max_ms is not None and cand["tau_err_ms"] > scan_tau_err_max_ms:
            continue
        filtered.append(cand)

    if not filtered:
        filtered = list(candidates)  # fallback

    if scan_sort_by.lower() == "psr":
        filtered.sort(key=lambda x: x["psr_db"], reverse=True)
    else:
        filtered.sort(key=lambda x: x["tau_err_ms"])

    if min_separation_sec is None:
        min_separation_sec = eval_window_sec

    selected: list[float] = []
    for cand in filtered:
        if len(selected) >= n_segments:
            break
        if all(
            abs(cand["center_sec"] - prev) >= min_separation_sec for prev in selected
        ):
            selected.append(cand["center_sec"])

    summary = {
        "n_scanned": len(candidates),
        "n_filtered": len(filtered),
        "n_selected": len(selected),
    }
    return selected, summary


# ─────────────────────────────────────────────────────────────────────
# [G] MSNF evaluation pipeline
# ─────────────────────────────────────────────────────────────────────


def _get_bandpass(config: dict) -> tuple[float, float] | None:
    bp_low = float(config.get("gcc_bandpass_low", 0.0))
    bp_high = float(config.get("gcc_bandpass_high", 0.0))
    if bp_low > 0 and bp_high > 0 and bp_high > bp_low:
        return (bp_low, bp_high)
    return None


def _measure_tau1(micl_eval, micr_eval, fs, config, ground_truths):
    """Measure tau1: MicL -> MicR."""
    bp = _get_bandpass(config)
    psr_exclude = int(config.get("psr_exclude_samples", 50))
    tau1_true_ms = ground_truths["tau1_true_ms"]
    guided_radius_ms = config.get("gcc_guided_peak_radius_ms")
    use_guided = guided_radius_ms is not None and float(guided_radius_ms) > 0

    tau_sec, psr_db = gcc_phat_full_analysis(
        micl_eval.astype(np.float64, copy=False),
        micr_eval.astype(np.float64, copy=False),
        fs,
        max_tau=float(config["gcc_max_lag_ms"]) / 1000.0,
        bandpass=bp,
        psr_exclude_samples=psr_exclude,
        guided_tau=float(tau1_true_ms) / 1000.0 if use_guided else None,
        guided_radius=float(guided_radius_ms) / 1000.0 if use_guided else None,
    )
    return float(tau_sec * 1000.0), float(psr_db)


def _measure_tau_ldv_mic(ldv_eval, mic_eval, fs, config, tau_expected_ms):
    """Measure tau2 or tau3: LDV -> Mic with guided search."""
    bp = _get_bandpass(config)
    psr_exclude = int(config.get("psr_exclude_samples", 50))
    max_lag_ms = float(config.get("tau2_max_lag_ms", 7.0))
    guided_radius_ms = float(config.get("tau2_guided_radius_ms", 2.0))

    tau_sec, psr_db = gcc_phat_full_analysis(
        ldv_eval.astype(np.float64, copy=False),
        mic_eval.astype(np.float64, copy=False),
        fs,
        max_tau=max_lag_ms / 1000.0,
        bandpass=bp,
        psr_exclude_samples=psr_exclude,
        guided_tau=float(tau_expected_ms) / 1000.0,
        guided_radius=guided_radius_ms / 1000.0,
    )
    return float(tau_sec * 1000.0), float(psr_db)


def _measure_tau4_omp(
    ldv_slice, micl_slice, micr_slice, t_start, t_end, desired_center_in_slice,
    fs, config, ground_truths,
):
    """Measure tau4: OMP-aligned pseudo-MicL -> MicR."""
    bp = _get_bandpass(config)
    psr_exclude = int(config.get("psr_exclude_samples", 50))
    eval_window_samples = t_end - t_start
    tau1_true_ms = ground_truths["tau1_true_ms"]
    guided_radius_ms = config.get("gcc_guided_peak_radius_ms")
    use_guided = guided_radius_ms is not None and float(guided_radius_ms) > 0

    # Optional prealignment
    ldv_for_omp = ldv_slice
    prealign_info = None
    ldv_prealign = str(config.get("ldv_prealign", "none")).lower()
    if ldv_prealign == "gcc_phat":
        ldv_seg = ldv_slice[t_start:t_end]
        micl_seg = micl_slice[t_start:t_end]
        pre_tau_sec, pre_psr = gcc_phat_full_analysis(
            ldv_seg.astype(np.float64, copy=False),
            micl_seg.astype(np.float64, copy=False),
            fs,
            max_tau=float(config["gcc_max_lag_ms"]) / 1000.0,
            bandpass=bp,
            psr_exclude_samples=psr_exclude,
        )
        delay_sec = -float(pre_tau_sec)
        ldv_for_omp = apply_fractional_delay_fd(ldv_slice, fs, delay_sec)
        prealign_info = {
            "tau_ldv_to_micl_ms": float(pre_tau_sec * 1000.0),
            "psr_db": float(pre_psr),
            "applied_delay_ms": float(delay_sec * 1000.0),
        }

    # STFT
    _, _, Zxx_ldv = stft(
        ldv_for_omp,
        fs=fs,
        nperseg=config["n_fft"],
        noverlap=config["n_fft"] - config["hop_length"],
        window="hann",
    )
    _, _, Zxx_micl = stft(
        micl_slice,
        fs=fs,
        nperseg=config["n_fft"],
        noverlap=config["n_fft"] - config["hop_length"],
        window="hann",
    )

    n_time = min(Zxx_ldv.shape[1], Zxx_micl.shape[1])
    tw = int(config["tw"])
    max_lag = int(config["max_lag"])
    desired_frame = int(
        round(
            (desired_center_in_slice - int(config["n_fft"]) // 2)
            / int(config["hop_length"])
        )
    )
    start_t_omp = desired_frame - tw // 2
    start_t_omp = max(start_t_omp, max_lag + 1)
    start_t_omp = min(start_t_omp, n_time - tw - max_lag - 1)

    if start_t_omp < max_lag + 1:
        return None, None, prealign_info

    Zxx_omp = apply_omp_alignment(Zxx_ldv, Zxx_micl, config, start_t_omp)
    _, ldv_omp_td = istft(
        Zxx_omp,
        fs=fs,
        nperseg=config["n_fft"],
        noverlap=config["n_fft"] - config["hop_length"],
        window="hann",
    )

    max_len_omp = min(len(ldv_omp_td), len(micr_slice))
    t_end_omp = min(t_end, max_len_omp)
    t_start_omp = max(0, t_end_omp - eval_window_samples)

    omp_eval = ldv_omp_td[t_start_omp:t_end_omp]
    micr_eval = micr_slice[t_start_omp:t_end_omp]

    tau4_sec, tau4_psr = gcc_phat_full_analysis(
        omp_eval.astype(np.float64, copy=False),
        micr_eval.astype(np.float64, copy=False),
        fs,
        max_tau=float(config["gcc_max_lag_ms"]) / 1000.0,
        bandpass=bp,
        psr_exclude_samples=psr_exclude,
        guided_tau=float(tau1_true_ms) / 1000.0 if use_guided else None,
        guided_radius=float(guided_radius_ms) / 1000.0 if use_guided else None,
    )
    return float(tau4_sec * 1000.0), float(tau4_psr), prealign_info


def evaluate_segment(
    ldv_signal: np.ndarray,
    mic_left_signal: np.ndarray,
    mic_right_signal: np.ndarray,
    center_sec: float,
    config: dict,
    ground_truths: dict,
) -> dict:
    """Evaluate all 6 methods for a single segment."""
    fs = int(config["fs"])
    c = float(config["speed_of_sound"])
    d = float(config["mic_spacing"])

    slice_samples = int(float(config["analysis_slice_sec"]) * fs)
    eval_window_samples = int(float(config["eval_window_sec"]) * fs)
    center_sample = int(center_sec * fs)

    # Extract analysis slice
    slice_start, slice_end = extract_centered_slice(
        [ldv_signal, mic_left_signal, mic_right_signal],
        center_sample=center_sample,
        slice_samples=slice_samples,
    )

    ldv_slice = ldv_signal[slice_start:slice_end]
    micl_slice = mic_left_signal[slice_start:slice_end]
    micr_slice = mic_right_signal[slice_start:slice_end]

    # Eval window within slice
    desired_center_in_slice = center_sample - slice_start
    eval_center = int(
        np.clip(desired_center_in_slice, 0, max(0, len(micl_slice) - 1))
    )
    t_start = eval_center - eval_window_samples // 2
    t_end = t_start + eval_window_samples
    max_len = min(len(ldv_slice), len(micl_slice), len(micr_slice))
    if t_start < 0:
        t_start = 0
        t_end = min(max_len, eval_window_samples)
    if t_end > max_len:
        t_end = max_len
        t_start = max(0, t_end - eval_window_samples)

    ldv_eval = ldv_slice[t_start:t_end]
    micl_eval = micl_slice[t_start:t_end]
    micr_eval = micr_slice[t_start:t_end]

    tau1_true_ms = ground_truths["tau1_true_ms"]
    tau2_true_ms = ground_truths["tau2_true_ms"]
    tau3_true_ms = ground_truths["tau3_true_ms"]
    theta_true = ground_truths["theta_true_deg"]

    # --- Measure all TDOAs ---
    tau1_ms, tau1_psr = _measure_tau1(micl_eval, micr_eval, fs, config, ground_truths)
    tau2_ms, tau2_psr = _measure_tau_ldv_mic(
        ldv_eval, micl_eval, fs, config, tau2_true_ms
    )
    tau3_ms, tau3_psr = _measure_tau_ldv_mic(
        ldv_eval, micr_eval, fs, config, tau3_true_ms
    )

    tau4_ms, tau4_psr, omp_prealign = None, None, None
    try:
        tau4_ms, tau4_psr, omp_prealign = _measure_tau4_omp(
            ldv_slice, micl_slice, micr_slice,
            t_start, t_end, desired_center_in_slice,
            fs, config, ground_truths,
        )
    except Exception as exc:
        logger.warning("OMP alignment failed at %.2fs: %s", center_sec, exc)

    # --- Phase 0 diagnostic ---
    tau2_radius = float(config.get("tau2_guided_radius_ms", 2.0))
    phase0 = {
        "tau2_guided": {
            "tau_ms": tau2_ms,
            "psr_db": tau2_psr,
            "expected_ms": tau2_true_ms,
            "in_range": bool(abs(tau2_ms - tau2_true_ms) < tau2_radius),
        },
        "tau3_guided": {
            "tau_ms": tau3_ms,
            "psr_db": tau3_psr,
            "expected_ms": tau3_true_ms,
            "in_range": bool(abs(tau3_ms - tau3_true_ms) < tau2_radius),
        },
    }

    # --- PSR weights ---
    psr_floor = float(config.get("psr_floor_db", -20.0))
    psr_ceiling = float(config.get("psr_ceiling_db", 20.0))
    w1 = psr_to_weight(tau1_psr, psr_floor, psr_ceiling)
    w2 = psr_to_weight(tau2_psr, psr_floor, psr_ceiling)
    w3 = psr_to_weight(tau3_psr, psr_floor, psr_ceiling)
    w4 = psr_to_weight(tau4_psr, psr_floor, psr_ceiling) if tau4_psr is not None else 0.0

    # --- Method A: MIC-MIC ---
    theta_A = tau_to_doa(tau1_ms, c, d)

    # --- Method B: OMP ---
    theta_B = tau_to_doa(tau4_ms, c, d) if tau4_ms is not None else None

    # --- Method C: theta-fusion ---
    if theta_B is not None and (w1 + w4) > 0:
        theta_C = float((w1 * theta_A + w4 * theta_B) / (w1 + w4))
    else:
        theta_C = theta_A

    # --- Forward model closures ---
    def _tau1_m(x):
        return tau1_model(x, c)

    def _tau2_m(x):
        return tau2_model(x, c)

    def _tau3_m(x):
        return tau3_model(x, c)

    # --- Method D: MSNF-2 (tau1, tau2) ---
    x_D, theta_D = solve_msnf(
        [(tau1_ms / 1000.0, _tau1_m), (tau2_ms / 1000.0, _tau2_m)],
        [w1, w2],
        c,
        d,
    )

    # --- Method E: MSNF-3 (tau1, tau2, tau3) ---
    x_E, theta_E = solve_msnf(
        [
            (tau1_ms / 1000.0, _tau1_m),
            (tau2_ms / 1000.0, _tau2_m),
            (tau3_ms / 1000.0, _tau3_m),
        ],
        [w1, w2, w3],
        c,
        d,
    )

    # --- Method F: MSNF-4 (tau1, tau2, tau3, tau4) ---
    if tau4_ms is not None:
        x_F, theta_F = solve_msnf(
            [
                (tau1_ms / 1000.0, _tau1_m),
                (tau2_ms / 1000.0, _tau2_m),
                (tau3_ms / 1000.0, _tau3_m),
                (tau4_ms / 1000.0, _tau1_m),  # tau4 uses tau1's forward model
            ],
            [w1, w2, w3, w4],
            c,
            d,
        )
    else:
        x_F, theta_F = x_E, theta_E

    return {
        "center_sec": float(center_sec),
        "phase0_diagnostic": phase0,
        "omp_prealign": omp_prealign,
        "tdoas": {
            "tau1_ms": tau1_ms,
            "tau1_psr_db": tau1_psr,
            "tau2_ms": tau2_ms,
            "tau2_psr_db": tau2_psr,
            "tau3_ms": tau3_ms,
            "tau3_psr_db": tau3_psr,
            "tau4_ms": tau4_ms,
            "tau4_psr_db": tau4_psr,
        },
        "weights": {"w1": w1, "w2": w2, "w3": w3, "w4": w4},
        "A_mic_mic": {
            "tau1_ms": tau1_ms,
            "theta_deg": theta_A,
            "theta_error_deg": float(abs(theta_A - theta_true)),
            "psr_db": tau1_psr,
        },
        "B_omp": {
            "tau4_ms": tau4_ms,
            "theta_deg": theta_B,
            "theta_error_deg": (
                float(abs(theta_B - theta_true)) if theta_B is not None else None
            ),
            "psr_db": tau4_psr,
        },
        "C_theta_fusion": {
            "theta_deg": theta_C,
            "theta_error_deg": float(abs(theta_C - theta_true)),
            "weights": [w1, w4],
        },
        "D_msnf_2": {
            "x_s": x_D,
            "theta_deg": theta_D,
            "theta_error_deg": float(abs(theta_D - theta_true)),
            "weights": [w1, w2],
        },
        "E_msnf_3": {
            "x_s": x_E,
            "theta_deg": theta_E,
            "theta_error_deg": float(abs(theta_E - theta_true)),
            "weights": [w1, w2, w3],
        },
        "F_msnf_4": {
            "x_s": x_F,
            "theta_deg": theta_F,
            "theta_error_deg": float(abs(theta_F - theta_true)),
            "weights": [w1, w2, w3, w4],
        },
    }


def _median_or_none(values: list) -> float | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(np.median(valid))


def _aggregate_method(per_segment: list[dict], method_key: str) -> dict:
    """Aggregate a method's results across segments."""
    entries = [seg[method_key] for seg in per_segment]
    theta_errors = [e["theta_error_deg"] for e in entries]
    thetas = [e["theta_deg"] for e in entries]

    result = {
        "theta_median_deg": _median_or_none(thetas),
        "theta_error_median_deg": _median_or_none(theta_errors),
    }

    if "psr_db" in entries[0]:
        psrs = [e["psr_db"] for e in entries]
        result["psr_median_db"] = _median_or_none(psrs)

    if "tau1_ms" in entries[0]:
        taus = [e["tau1_ms"] for e in entries]
        result["tau_median_ms"] = _median_or_none(taus)
    elif "tau4_ms" in entries[0]:
        taus = [e["tau4_ms"] for e in entries]
        result["tau_median_ms"] = _median_or_none(taus)

    if "x_s" in entries[0]:
        xs = [e["x_s"] for e in entries]
        result["x_s_median"] = _median_or_none(xs)

    return result


def _aggregate_phase0(per_segment: list[dict]) -> dict:
    """Aggregate Phase 0 diagnostics across segments."""
    tau2_psrs = [seg["phase0_diagnostic"]["tau2_guided"]["psr_db"] for seg in per_segment]
    tau3_psrs = [seg["phase0_diagnostic"]["tau3_guided"]["psr_db"] for seg in per_segment]
    tau2_in_range = [seg["phase0_diagnostic"]["tau2_guided"]["in_range"] for seg in per_segment]
    tau3_in_range = [seg["phase0_diagnostic"]["tau3_guided"]["in_range"] for seg in per_segment]

    tau2_go_count = sum(1 for p in tau2_psrs if p > 3.0)
    tau3_go_count = sum(1 for p in tau3_psrs if p > 3.0)
    n = len(per_segment)

    return {
        "n_segments": n,
        "tau2_psr_median_db": float(np.median(tau2_psrs)),
        "tau2_go_count": tau2_go_count,
        "tau2_in_range_count": sum(tau2_in_range),
        "tau3_psr_median_db": float(np.median(tau3_psrs)),
        "tau3_go_count": tau3_go_count,
        "tau3_in_range_count": sum(tau3_in_range),
        "tau2_go": bool(tau2_go_count >= 3),
        "tau3_go": bool(tau3_go_count >= 3),
    }


# ─────────────────────────────────────────────────────────────────────
# [H] Main evaluation & output
# ─────────────────────────────────────────────────────────────────────


def run_msnf_evaluation(
    ldv_path: str,
    mic_left_path: str,
    mic_right_path: str,
    config: dict,
    output_dir: str,
    *,
    speaker_key: str,
    n_segments: int,
    segment_mode: str,
    scan_start_sec: float | None,
    scan_end_sec: float | None,
    scan_hop_sec: float,
    scan_psr_min_db: float | None,
    scan_tau_err_max_ms: float | None,
    scan_sort_by: str,
    scan_min_separation_sec: float | None,
) -> dict:
    logger.info("=" * 70)
    logger.info("MSNF Evaluation")
    logger.info("=" * 70)

    c = float(config["speed_of_sound"])
    d = float(config["mic_spacing"])
    fs = int(config["fs"])

    speaker_id = Path(ldv_path).parent.name
    spk = speaker_key or speaker_id.split("-")[0]
    logger.info("Speaker: %s (key=%s)", speaker_id, spk)

    ground_truths = compute_all_ground_truths(spk, c, d)
    logger.info(
        "Ground truth: x_s=%.2f, theta=%.2f deg, tau1=%.3f ms, tau2=%.3f ms, tau3=%.3f ms",
        ground_truths["x_s_true"],
        ground_truths["theta_true_deg"],
        ground_truths["tau1_true_ms"],
        ground_truths["tau2_true_ms"],
        ground_truths["tau3_true_ms"],
    )

    # Load signals
    sr_ldv, ldv_signal = load_wav(ldv_path)
    sr_ml, mic_left_signal = load_wav(mic_left_path)
    sr_mr, mic_right_signal = load_wav(mic_right_path)
    if not (sr_ldv == sr_ml == sr_mr == fs):
        raise ValueError("Sample rates do not match expected fs")

    duration_s = min(len(ldv_signal), len(mic_left_signal), len(mic_right_signal)) / fs
    logger.info("Duration: %.2f s", duration_s)

    # Segment selection
    bp = _get_bandpass(config)
    psr_exclude = int(config.get("psr_exclude_samples", 50))
    tau1_true_ms = ground_truths["tau1_true_ms"]

    slice_samples = int(float(config["analysis_slice_sec"]) * fs)
    half_slice_sec = (slice_samples / fs) / 2
    min_center = half_slice_sec
    max_center = max(min_center, duration_s - half_slice_sec)
    segment_offset_sec = float(config.get("segment_offset_sec", 100.0))
    segment_spacing_sec = float(config.get("segment_spacing_sec", 50.0))

    if scan_start_sec is None:
        scan_start_sec = segment_offset_sec
    if scan_end_sec is None:
        scan_end_sec = min(max_center, 600.0) if duration_s > 600.0 else max_center

    guided_radius_ms = config.get("gcc_guided_peak_radius_ms")
    use_guided = guided_radius_ms is not None and float(guided_radius_ms) > 0

    segment_mode = (segment_mode or "scan").lower()
    scan_summary = None

    if segment_mode == "scan":
        if scan_min_separation_sec is None:
            scan_min_separation_sec = float(config.get("eval_window_sec", 5.0))
        segment_centers_sec, scan_summary = scan_segment_centers(
            mic_left_signal,
            mic_right_signal,
            fs=fs,
            eval_window_sec=float(config.get("eval_window_sec", 5.0)),
            max_lag_ms=float(config["gcc_max_lag_ms"]),
            bandpass=bp,
            psr_exclude_samples=psr_exclude,
            tau_true_ms=tau1_true_ms,
            guided_tau_ms=float(tau1_true_ms) if use_guided else None,
            guided_radius_ms=float(guided_radius_ms) if use_guided else None,
            scan_start_sec=float(scan_start_sec),
            scan_end_sec=float(scan_end_sec),
            scan_hop_sec=float(scan_hop_sec),
            scan_psr_min_db=scan_psr_min_db,
            scan_tau_err_max_ms=scan_tau_err_max_ms,
            scan_sort_by=scan_sort_by,
            n_segments=n_segments,
            min_separation_sec=scan_min_separation_sec,
        )
        if not segment_centers_sec:
            raise ValueError("scan selected 0 segments")
    else:
        start_center = max(min_center, segment_offset_sec)
        if n_segments <= 1:
            segment_centers_sec = [min(max_center, max(min_center, duration_s / 2))]
        else:
            segment_centers_sec = [
                start_center + i * segment_spacing_sec for i in range(n_segments)
            ]
            segment_centers_sec = [t for t in segment_centers_sec if t <= max_center]

    n_segments_used = len(segment_centers_sec)
    logger.info("Evaluating %d segments", n_segments_used)

    # Evaluate each segment
    per_segment = []
    for seg_idx, center_sec in enumerate(segment_centers_sec):
        logger.info(
            "Segment %d/%d: center_t=%.2fs", seg_idx + 1, n_segments_used, center_sec
        )
        seg_result = evaluate_segment(
            ldv_signal,
            mic_left_signal,
            mic_right_signal,
            center_sec,
            config,
            ground_truths,
        )
        per_segment.append(seg_result)

    # Aggregate
    phase0_agg = _aggregate_phase0(per_segment)
    logger.info(
        "Phase0: tau2_go=%s (PSR>3dB: %d/%d), tau3_go=%s (%d/%d)",
        phase0_agg["tau2_go"],
        phase0_agg["tau2_go_count"],
        phase0_agg["n_segments"],
        phase0_agg["tau3_go"],
        phase0_agg["tau3_go_count"],
        phase0_agg["n_segments"],
    )

    method_keys = [
        "A_mic_mic",
        "B_omp",
        "C_theta_fusion",
        "D_msnf_2",
        "E_msnf_3",
        "F_msnf_4",
    ]
    aggregated_results = {}
    for mk in method_keys:
        aggregated_results[mk] = _aggregate_method(per_segment, mk)
        err = aggregated_results[mk]["theta_error_median_deg"]
        logger.info("  %s: median theta_error = %s deg", mk, err)

    # Success criteria
    theta_A_err = aggregated_results["A_mic_mic"]["theta_error_median_deg"]
    criteria = {}
    for mk in method_keys:
        mk_err = aggregated_results[mk]["theta_error_median_deg"]
        if mk_err is not None and theta_A_err is not None:
            criteria[mk] = {
                "beats_mic_mic": bool(mk_err < theta_A_err),
                "error_under_2deg": bool(mk_err < 2.0),
                "error_under_5deg": bool(mk_err < 5.0),
            }

    # Build summary
    summary = {
        "speaker_id": speaker_id,
        "experiment": "msnf_fusion",
        "ground_truth": ground_truths,
        "phase0_diagnostic": phase0_agg,
        "n_segments": n_segments_used,
        "segment_mode": segment_mode,
        "scan_summary": scan_summary,
        "segment_centers_sec": segment_centers_sec,
        "config": config,
        "result": aggregated_results,
        "criteria": criteria,
        "per_segment": per_segment,
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "summary.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved to: %s", output_file)

    return summary


# ─────────────────────────────────────────────────────────────────────
# [I] CLI
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Sensor Near-Field Fusion (MSNF) DoA estimation"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Root directory of dataset"
    )
    parser.add_argument(
        "--speaker", type=str, required=True, help="Speaker folder (e.g., 20-0.1V)"
    )
    parser.add_argument(
        "--speaker_key",
        type=str,
        default=None,
        help="Override geometry speaker key (e.g., 20)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )

    # Segment selection
    parser.add_argument("--n_segments", type=int, default=5)
    parser.add_argument(
        "--segment_mode", type=str, default="scan", choices=["fixed", "scan"]
    )
    parser.add_argument("--scan_start_sec", type=float, default=None)
    parser.add_argument("--scan_end_sec", type=float, default=None)
    parser.add_argument("--scan_hop_sec", type=float, default=1.0)
    parser.add_argument("--scan_psr_min_db", type=float, default=-20.0)
    parser.add_argument("--scan_tau_err_max_ms", type=float, default=2.0)
    parser.add_argument(
        "--scan_sort_by", type=str, default="tau_err", choices=["tau_err", "psr"]
    )
    parser.add_argument("--scan_min_separation_sec", type=float, default=None)

    # GCC-PHAT config
    parser.add_argument("--gcc_bandpass_low", type=float, default=None)
    parser.add_argument("--gcc_bandpass_high", type=float, default=None)
    parser.add_argument("--gcc_guided_peak_radius_ms", type=float, default=None)
    parser.add_argument("--tau2_max_lag_ms", type=float, default=None)
    parser.add_argument("--tau2_guided_radius_ms", type=float, default=None)
    parser.add_argument("--psr_exclude_samples", type=int, default=None)

    # Slice / eval window
    parser.add_argument("--analysis_slice_sec", type=float, default=None)
    parser.add_argument("--eval_window_sec", type=float, default=None)
    parser.add_argument("--segment_offset_sec", type=float, default=None)
    parser.add_argument("--segment_spacing_sec", type=float, default=None)

    # OMP config
    parser.add_argument(
        "--ldv_prealign", type=str, choices=["none", "gcc_phat"], default=None
    )
    parser.add_argument("--max_k", type=int, default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.speaker
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    config = DEFAULT_CONFIG.copy()
    config_overrides = {
        "gcc_bandpass_low": args.gcc_bandpass_low,
        "gcc_bandpass_high": args.gcc_bandpass_high,
        "gcc_guided_peak_radius_ms": args.gcc_guided_peak_radius_ms,
        "tau2_max_lag_ms": args.tau2_max_lag_ms,
        "tau2_guided_radius_ms": args.tau2_guided_radius_ms,
        "psr_exclude_samples": args.psr_exclude_samples,
        "analysis_slice_sec": args.analysis_slice_sec,
        "eval_window_sec": args.eval_window_sec,
        "segment_offset_sec": args.segment_offset_sec,
        "segment_spacing_sec": args.segment_spacing_sec,
        "ldv_prealign": args.ldv_prealign,
        "max_k": args.max_k,
    }
    for key, val in config_overrides.items():
        if val is not None:
            config[key] = val

    data_dir = Path(args.data_root) / args.speaker
    ldv_files = list(data_dir.glob("*LDV*.wav"))
    left_mic_files = list(data_dir.glob("*LEFT*.wav"))
    right_mic_files = list(data_dir.glob("*RIGHT*.wav"))

    if not ldv_files or not left_mic_files or not right_mic_files:
        raise FileNotFoundError(f"Missing audio files in {data_dir}")

    try:
        run_msnf_evaluation(
            ldv_path=str(ldv_files[0]),
            mic_left_path=str(left_mic_files[0]),
            mic_right_path=str(right_mic_files[0]),
            config=config,
            output_dir=str(output_dir),
            speaker_key=args.speaker_key,
            n_segments=args.n_segments,
            segment_mode=args.segment_mode,
            scan_start_sec=args.scan_start_sec,
            scan_end_sec=args.scan_end_sec,
            scan_hop_sec=args.scan_hop_sec,
            scan_psr_min_db=args.scan_psr_min_db,
            scan_tau_err_max_ms=args.scan_tau_err_max_ms,
            scan_sort_by=args.scan_sort_by,
            scan_min_separation_sec=args.scan_min_separation_sec,
        )
    except Exception as exc:
        logger.exception("Run failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
