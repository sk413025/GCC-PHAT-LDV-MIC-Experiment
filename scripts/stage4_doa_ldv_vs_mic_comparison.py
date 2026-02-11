#!/usr/bin/env python
"""
Stage 4 DoA: LDV-vs-Mic comparison (GCC-PHAT, guided search)

Signal pairs:
- micl_micr: MicL-MicR baseline
- ldv_micl: LDV aligned to MicL via OMP, then paired with MicR

Truth reference:
- Geometry truth (default)
- Chirp truth override via --truth_tau_ms / --truth_theta_deg

Outputs per-speaker summary.json with median tau/theta/PSR and pass/fail.
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
from scipy.signal import butter, filtfilt, istft, stft

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "fs": 48000,
    "n_fft": 6144,
    "hop_length": 160,
    "max_lag": 50,
    "max_k": 3,
    "tw": 64,
    "freq_min": 100,
    "freq_max": 8000,
    "gcc_max_lag_ms": 10.0,
    "gcc_bandpass_low": 0.0,
    "gcc_bandpass_high": 0.0,
    "psr_exclude_samples": 50,
    "gcc_guided_peak_radius_ms": None,
    "analysis_slice_sec": 5.0,
    "eval_window_sec": 5.0,
    "segment_spacing_sec": 50.0,
    "segment_offset_sec": 100.0,
    "ldv_prealign": "none",
    "alignment_mode": "omp",
    "dtmin_model_path": None,
    "speed_of_sound": 343.0,
    "mic_spacing": 1.4,
}

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


# -----------------------------------------------------------------------------
# Geometry and conversions
# -----------------------------------------------------------------------------


def compute_ground_truth(speaker_id: str, config: dict) -> dict:
    speaker_key = speaker_id.split("-")[0]
    if speaker_key not in GEOMETRY["speakers"]:
        logger.warning("Unknown speaker %s, using speaker 20", speaker_key)
        speaker_key = "20"

    speaker_pos = GEOMETRY["speakers"][speaker_key]
    mic_left = GEOMETRY["mic_left"]
    mic_right = GEOMETRY["mic_right"]
    c = config["speed_of_sound"]
    d = config["mic_spacing"]

    d_left = float(np.hypot(speaker_pos[0] - mic_left[0], speaker_pos[1] - mic_left[1]))
    d_right = float(np.hypot(speaker_pos[0] - mic_right[0], speaker_pos[1] - mic_right[1]))

    # Convention: tau = (d_left - d_right) / c
    tau_true = (d_left - d_right) / c

    sin_theta = tau_true * c / d
    sin_theta = float(np.clip(sin_theta, -1.0, 1.0))
    theta_true = float(np.degrees(np.arcsin(sin_theta)))

    return {
        "tau_true_ms": tau_true * 1000.0,
        "theta_true_deg": theta_true,
        "d_left": d_left,
        "d_right": d_right,
        "speaker_pos": speaker_pos,
    }


def tau_to_doa(tau_ms: float, config: dict) -> float:
    c = config["speed_of_sound"]
    d = config["mic_spacing"]
    sin_theta = (tau_ms / 1000.0) * c / d
    sin_theta = float(np.clip(sin_theta, -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_theta)))


# -----------------------------------------------------------------------------
# IO and signal utilities
# -----------------------------------------------------------------------------


def load_wav(path: str) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    return sr, data


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


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 5) -> np.ndarray:
    nyq = 0.5 * fs
    low = float(lowcut) / nyq
    high = float(highcut) / nyq
    if not (0.0 < low < high < 1.0):
        raise ValueError(f"Invalid bandpass range: lowcut={lowcut}, highcut={highcut}, fs={fs}")
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def apply_fractional_delay_fd(signal: np.ndarray, fs: int, delay_sec: float) -> np.ndarray:
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


# -----------------------------------------------------------------------------
# GCC-PHAT
# -----------------------------------------------------------------------------


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
    if guided_tau is not None and guided_radius is not None and float(guided_radius) > 0:
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


def estimate_tdoa_gcc_phat(
    sig1: np.ndarray,
    sig2: np.ndarray,
    fs: int,
    *,
    max_lag_samples: int | None,
    bandpass: tuple[float, float] | None,
    psr_exclude_samples: int,
    guided_tau_ms: float | None,
    guided_radius_ms: float | None,
) -> dict:
    max_tau = None
    if max_lag_samples is not None:
        max_tau = float(max_lag_samples) / float(fs)

    tau_sec, psr_db = gcc_phat_full_analysis(
        sig1.astype(np.float64, copy=False),
        sig2.astype(np.float64, copy=False),
        fs,
        max_tau=max_tau,
        bandpass=bandpass,
        psr_exclude_samples=int(psr_exclude_samples),
        guided_tau=None if guided_tau_ms is None else float(guided_tau_ms) / 1000.0,
        guided_radius=None if guided_radius_ms is None else float(guided_radius_ms) / 1000.0,
    )
    return {"tau_ms": float(tau_sec * 1000.0), "psr_db": float(psr_db)}


# -----------------------------------------------------------------------------
# OMP alignment (Stage 1-2)
# -----------------------------------------------------------------------------


def normalize_per_freq_maxabs(X_stft: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if X_stft.ndim == 1:
        max_abs = np.abs(X_stft).max()
        if max_abs < 1e-10:
            max_abs = 1.0
        X_norm = X_stft / max_abs
        return X_norm, max_abs
    if X_stft.ndim == 2:
        max_abs = np.abs(X_stft).max(axis=-1)
        max_abs = np.where(max_abs < 1e-10, 1.0, max_abs)
        X_norm = X_stft / max_abs[:, np.newaxis]
        return X_norm, max_abs
    max_abs = np.abs(X_stft).max(axis=(-2, -1))
    max_abs = np.where(max_abs < 1e-10, 1.0, max_abs)
    X_norm = X_stft / max_abs[:, np.newaxis, np.newaxis]
    return X_norm, max_abs


def build_lagged_dictionary(X_stft: np.ndarray, max_lag: int, tw: int, start_t: int) -> np.ndarray:
    n_freq, n_time = X_stft.shape
    n_lags = 2 * max_lag + 1
    Dict_tensor = np.zeros((n_freq, n_lags, tw), dtype=X_stft.dtype)
    for lag_idx, lag in enumerate(range(-max_lag, max_lag + 1)):
        t_start = start_t + lag
        t_end = t_start + tw
        if t_start >= 0 and t_end <= n_time:
            Dict_tensor[:, lag_idx, :] = X_stft[:, t_start:t_end]
    return Dict_tensor


def omp_single_freq(Dict_f: np.ndarray, Y_f: np.ndarray, max_k: int) -> tuple[list[int], np.ndarray, np.ndarray]:
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


def load_dtmin_policy(model_path: str, *, expected_k: int, expected_n_lags: int) -> dict:
    payload = np.load(model_path, allow_pickle=False)
    centroids = payload["centroids"]
    action_valid = payload["action_valid"]
    metadata = {}
    if "metadata_json" in payload:
        metadata = json.loads(str(payload["metadata_json"]))

    if centroids.ndim != 3:
        raise ValueError(f"Invalid centroids shape {centroids.shape}; expected (K, n_lags, n_lags)")
    if action_valid.ndim != 2:
        raise ValueError(f"Invalid action_valid shape {action_valid.shape}; expected (K, n_lags)")
    if centroids.shape[0] < expected_k:
        raise ValueError(f"Model K={centroids.shape[0]} < required max_k={expected_k}")
    if centroids.shape[1] != expected_n_lags or centroids.shape[2] != expected_n_lags:
        raise ValueError(
            "Model lag shape mismatch: "
            f"centroids={centroids.shape}, expected_n_lags={expected_n_lags}"
        )
    if action_valid.shape[0] < expected_k or action_valid.shape[1] != expected_n_lags:
        raise ValueError(
            "Model action mask mismatch: "
            f"action_valid={action_valid.shape}, expected=({expected_k}, {expected_n_lags})"
        )
    return {"centroids": centroids, "action_valid": action_valid, "metadata": metadata}


def select_lag_dtmin(
    obs: np.ndarray,
    *,
    step_idx: int,
    selected_lags: list[int],
    policy: dict | None,
) -> int:
    # Fallback behavior when no policy is available for this step.
    def fallback() -> int:
        scores = obs.copy()
        for lag in selected_lags:
            scores[lag] = -np.inf
        return int(np.argmax(scores))

    if policy is None:
        return fallback()
    if step_idx >= policy["centroids"].shape[0]:
        return fallback()

    centroids = policy["centroids"][step_idx]
    action_valid = policy["action_valid"][step_idx]
    if not np.any(action_valid):
        return fallback()

    dists = np.linalg.norm(centroids - obs[np.newaxis, :], axis=1)
    dists = np.where(action_valid, dists, np.inf)
    for lag in selected_lags:
        dists[lag] = np.inf
    if not np.isfinite(dists).any():
        return fallback()
    return int(np.argmin(dists))


def dtmin_single_freq(
    Dict_f: np.ndarray,
    Y_f: np.ndarray,
    max_k: int,
    policy: dict | None,
) -> tuple[list[int], np.ndarray | None, np.ndarray]:
    n_lags, _ = Dict_f.shape
    D = Dict_f.T
    D_norms = np.linalg.norm(np.abs(D), axis=0, keepdims=True) + 1e-10
    D_normalized = D / D_norms
    residual = Y_f.copy()
    selected_lags: list[int] = []
    coeffs = None
    reconstructed = np.zeros_like(Y_f)
    for step_idx in range(max_k):
        corrs = np.abs(D_normalized.conj().T @ residual).astype(np.float64, copy=False)
        best_lag = select_lag_dtmin(corrs, step_idx=step_idx, selected_lags=selected_lags, policy=policy)
        if best_lag in selected_lags:
            # Safety guard; fallback chooser should already avoid duplicates.
            fallback = np.argsort(-corrs)
            for cand in fallback:
                if int(cand) not in selected_lags:
                    best_lag = int(cand)
                    break
        selected_lags.append(best_lag)
        if len(selected_lags) > n_lags:
            break
        A = D[:, selected_lags]
        coeffs, _, _, _ = np.linalg.lstsq(A, Y_f, rcond=None)
        reconstructed = A @ coeffs
        residual = Y_f - reconstructed
    return selected_lags, coeffs, reconstructed


def apply_omp_alignment(Zxx_ldv: np.ndarray, Zxx_mic: np.ndarray, config: dict, start_t: int) -> np.ndarray:
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


def apply_dtmin_alignment(
    Zxx_ldv: np.ndarray,
    Zxx_mic: np.ndarray,
    config: dict,
    start_t: int,
    policy: dict,
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

    Zxx_aligned = Zxx_ldv.copy()
    for f_idx in range(len(freq_indices)):
        Dict_f = Dict_selected[f_idx]
        Y_f = Zxx_mic[freq_indices[f_idx], start_t : start_t + tw]

        Dict_norm, _ = normalize_per_freq_maxabs(Dict_f)
        Y_norm, _ = normalize_per_freq_maxabs(Y_f)
        selected_lags, _, _ = dtmin_single_freq(Dict_norm, Y_norm, max_k, policy)

        D_orig = Dict_selected[f_idx].T
        A = D_orig[:, selected_lags]
        Y_orig = Zxx_mic[freq_indices[f_idx], start_t : start_t + tw]
        coeffs_orig, _, _, _ = np.linalg.lstsq(A, Y_orig, rcond=None)
        reconstructed_orig = A @ coeffs_orig
        Zxx_aligned[freq_indices[f_idx], start_t : start_t + tw] = reconstructed_orig
    return Zxx_aligned


# -----------------------------------------------------------------------------
# Scan window selection
# -----------------------------------------------------------------------------


def scan_segment_centers_by_mic_mic(
    ldv_signal: np.ndarray | None,
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
    scan_ldv_micl_psr_min_db: float | None,
    scan_tau_err_max_ms: float | None,
    scan_sort_by: str,
    n_segments: int,
    min_separation_sec: float | None,
    allow_fallback: bool,
) -> tuple[list[float], dict]:
    eval_window_samples = int(eval_window_sec * fs)
    if eval_window_samples <= 0:
        raise ValueError(f"eval_window_sec must be > 0, got {eval_window_sec}")

    duration_s = min(len(mic_left_signal), len(mic_right_signal)) / fs
    half_win_sec = (eval_window_samples / fs) / 2
    min_center = half_win_sec
    max_center = max(min_center, duration_s - half_win_sec)

    start_center = max(min_center, scan_start_sec)
    end_center = min(max_center, scan_end_sec if scan_end_sec is not None else max_center)

    if scan_hop_sec <= 0:
        raise ValueError(f"scan_hop_sec must be > 0, got {scan_hop_sec}")

    max_tau = float(max_lag_ms) / 1000.0
    candidates: list[dict] = []

    center = start_center
    n_scanned = 0
    while center <= end_center + 1e-9:
        center_sample = int(center * fs)
        t_start = center_sample - eval_window_samples // 2
        t_end = t_start + eval_window_samples
        if t_start < 0 or t_end > len(mic_left_signal) or t_end > len(mic_right_signal):
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
            guided_tau=None if guided_tau_ms is None else float(guided_tau_ms) / 1000.0,
            guided_radius=None if guided_radius_ms is None else float(guided_radius_ms) / 1000.0,
        )
        tau_ms = float(tau_sec * 1000.0)
        tau_err_ms = float(abs(tau_ms - tau_true_ms))

        n_scanned += 1
        cand = {
            "center_sec": float(center),
            "tau_ms": tau_ms,
            "psr_db": float(psr_db),
            "tau_err_ms": tau_err_ms,
        }

        if ldv_signal is not None:
            if t_end <= len(ldv_signal):
                ldv_seg = ldv_signal[t_start:t_end]
                tau_ldv_micl_sec, psr_ldv_micl_db = gcc_phat_full_analysis(
                    ldv_seg.astype(np.float64, copy=False),
                    mic_l_seg.astype(np.float64, copy=False),
                    fs,
                    max_tau=max_tau,
                    bandpass=bandpass,
                    psr_exclude_samples=psr_exclude_samples,
                )
                cand["ldv_micl_tau_ms"] = float(tau_ldv_micl_sec * 1000.0)
                cand["ldv_micl_psr_db"] = float(psr_ldv_micl_db)
            else:
                cand["ldv_micl_tau_ms"] = None
                cand["ldv_micl_psr_db"] = None

        candidates.append(cand)
        center += scan_hop_sec

    filtered = []
    for cand in candidates:
        if scan_psr_min_db is not None and cand["psr_db"] < scan_psr_min_db:
            continue
        if scan_tau_err_max_ms is not None and cand["tau_err_ms"] > scan_tau_err_max_ms:
            continue
        if scan_ldv_micl_psr_min_db is not None:
            ldv_psr = cand.get("ldv_micl_psr_db")
            if ldv_psr is None or ldv_psr < scan_ldv_micl_psr_min_db:
                continue
        filtered.append(cand)

    if not filtered and allow_fallback:
        filtered = candidates

    sort_by = scan_sort_by.lower()
    if sort_by == "psr":
        filtered.sort(key=lambda x: x["psr_db"], reverse=True)
    else:
        filtered.sort(key=lambda x: x["tau_err_ms"])

    if min_separation_sec is None:
        min_separation_sec = eval_window_sec

    selected = []
    for cand in filtered:
        if len(selected) >= n_segments:
            break
        if all(abs(cand["center_sec"] - prev) >= min_separation_sec for prev in selected):
            selected.append(cand["center_sec"])

    summary = {
        "n_scanned": int(n_scanned),
        "n_candidates": int(len(candidates)),
        "n_filtered": int(len(filtered)),
        "n_selected": int(len(selected)),
        "scan_sort_by": scan_sort_by,
        "scan_psr_min_db": scan_psr_min_db,
        "scan_tau_err_max_ms": scan_tau_err_max_ms,
        "scan_ldv_micl_psr_min_db": scan_ldv_micl_psr_min_db,
    }

    return selected, summary


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------


def run_stage4_evaluation(
    ldv_path: str,
    mic_left_path: str,
    mic_right_path: str,
    config: dict,
    output_dir: str,
    *,
    signal_pair: str,
    n_segments: int,
    segment_mode: str,
    scan_start_sec: float | None,
    scan_end_sec: float | None,
    scan_hop_sec: float,
    scan_psr_min_db: float | None,
    scan_ldv_micl_psr_min_db: float | None,
    scan_tau_err_max_ms: float | None,
    scan_sort_by: str,
    scan_min_separation_sec: float | None,
    scan_allow_fallback: bool,
    truth_tau_ms: float | None,
    truth_theta_deg: float | None,
    truth_label: str | None,
    pass_theta_max_deg: float,
    pass_psr_min_db: float | None,
    pass_mode: str,
    speaker_key_override: str | None,
    alignment_mode: str,
    dtmin_model_path: str | None,
) -> dict:
    logger.info("=" * 70)
    logger.info("Stage 4: LDV-vs-Mic DoA (GCC-PHAT)")
    logger.info("=" * 70)

    speaker_id = Path(ldv_path).parent.name
    logger.info("Speaker: %s", speaker_id)
    logger.info("Signal pair: %s", signal_pair)
    logger.info("Alignment mode: %s", alignment_mode)

    ground_truth = compute_ground_truth(speaker_key_override or speaker_id, config)
    tau_geom_ms = float(ground_truth["tau_true_ms"])
    theta_geom_deg = float(ground_truth["theta_true_deg"])

    truth_mode = "geometry"
    if truth_tau_ms is not None:
        truth_mode = "override"
        tau_ref_ms = float(truth_tau_ms)
        if truth_theta_deg is not None:
            theta_ref_deg = float(truth_theta_deg)
        else:
            theta_ref_deg = float(tau_to_doa(tau_ref_ms, config))
    else:
        tau_ref_ms = tau_geom_ms
        theta_ref_deg = theta_geom_deg

    logger.info("Geometry truth: tau=%.3f ms, theta=%.2f deg", tau_geom_ms, theta_geom_deg)
    logger.info("Reference(%s): tau=%.3f ms, theta=%.2f deg", truth_mode, tau_ref_ms, theta_ref_deg)

    effective_pass_mode = pass_mode
    if pass_mode == "auto":
        effective_pass_mode = "omp_vs_raw" if signal_pair == "ldv_micl" else "theta_only"
    if effective_pass_mode not in {"omp_vs_raw", "theta_only"}:
        raise ValueError(f"Invalid pass_mode: {pass_mode}")
    logger.info("Pass mode: %s", effective_pass_mode)

    if alignment_mode not in {"omp", "dtmin"}:
        raise ValueError(f"Invalid alignment_mode={alignment_mode!r} (expected: omp|dtmin)")
    dtmin_policy = None
    if signal_pair == "ldv_micl" and alignment_mode == "dtmin":
        if not dtmin_model_path:
            raise ValueError("alignment_mode=dtmin requires dtmin_model_path")
        n_lags = 2 * int(config["max_lag"]) + 1
        dtmin_policy = load_dtmin_policy(
            dtmin_model_path,
            expected_k=int(config["max_k"]),
            expected_n_lags=n_lags,
        )

    sr_ldv, ldv_signal = load_wav(ldv_path)
    sr_mic_l, mic_left_signal = load_wav(mic_left_path)
    sr_mic_r, mic_right_signal = load_wav(mic_right_path)

    if not (sr_ldv == sr_mic_l == sr_mic_r == config["fs"]):
        raise ValueError("Sample rates do not match expected fs")

    duration_s = min(len(ldv_signal), len(mic_left_signal), len(mic_right_signal)) / config["fs"]
    logger.info("Duration: %.2f s", duration_s)

    slice_samples = int(float(config.get("analysis_slice_sec", 5.0)) * config["fs"])
    half_slice_sec = (slice_samples / config["fs"]) / 2
    segment_spacing_sec = float(config.get("segment_spacing_sec", 50.0))
    segment_offset_sec = float(config.get("segment_offset_sec", 100.0))

    min_center = half_slice_sec
    max_center = max(min_center, duration_s - half_slice_sec)
    start_center = max(min_center, segment_offset_sec)

    bp_low = float(config.get("gcc_bandpass_low", 0.0))
    bp_high = float(config.get("gcc_bandpass_high", 0.0))
    bp = (bp_low, bp_high) if (bp_low > 0 and bp_high > 0 and bp_high > bp_low) else None
    psr_exclude_samples = int(config.get("psr_exclude_samples", 50))

    if scan_start_sec is None:
        scan_start_sec = float(segment_offset_sec)
    if scan_end_sec is None:
        scan_end_sec = float(min(max_center, 600.0)) if duration_s > 600.0 else float(max_center)

    segment_mode = (segment_mode or "fixed").lower()
    if segment_mode not in {"fixed", "scan"}:
        raise ValueError(f"Invalid segment_mode={segment_mode!r} (expected: fixed|scan)")

    if segment_mode == "scan":
        if scan_min_separation_sec is None:
            scan_min_separation_sec = float(config.get("eval_window_sec", 1.0))
        scan_use_ldv = scan_ldv_micl_psr_min_db is not None or signal_pair == "ldv_micl"
        segment_centers_sec, scan_summary = scan_segment_centers_by_mic_mic(
            ldv_signal if scan_use_ldv else None,
            mic_left_signal,
            mic_right_signal,
            fs=config["fs"],
            eval_window_sec=float(config.get("eval_window_sec", 1.0)),
            max_lag_ms=float(config["gcc_max_lag_ms"]),
            bandpass=bp,
            psr_exclude_samples=psr_exclude_samples,
            tau_true_ms=float(tau_ref_ms),
            guided_tau_ms=float(tau_ref_ms)
            if (config.get("gcc_guided_peak_radius_ms") is not None and float(config.get("gcc_guided_peak_radius_ms")) > 0)
            else None,
            guided_radius_ms=None
            if config.get("gcc_guided_peak_radius_ms") is None
            else float(config.get("gcc_guided_peak_radius_ms")),
            scan_start_sec=float(scan_start_sec),
            scan_end_sec=float(scan_end_sec),
            scan_hop_sec=float(scan_hop_sec),
            scan_psr_min_db=scan_psr_min_db,
            scan_ldv_micl_psr_min_db=scan_ldv_micl_psr_min_db,
            scan_tau_err_max_ms=scan_tau_err_max_ms,
            scan_sort_by=scan_sort_by,
            n_segments=int(n_segments),
            min_separation_sec=scan_min_separation_sec,
            allow_fallback=bool(scan_allow_fallback),
        )
        if not segment_centers_sec:
            raise ValueError(
                "segment_mode=scan selected 0 segments. "
                "Try relaxing scan thresholds or allow fallback."
            )
    else:
        scan_summary = None
        if n_segments <= 1:
            segment_centers_sec = [min(max_center, max(min_center, duration_s / 2))]
        else:
            segment_centers_sec = [start_center + i * segment_spacing_sec for i in range(n_segments)]
            segment_centers_sec = [t for t in segment_centers_sec if t <= max_center]
            if len(segment_centers_sec) < n_segments:
                logger.warning(
                    "Requested %d segments but only %d fit in [%.2fs, %.2fs]",
                    n_segments,
                    len(segment_centers_sec),
                    min_center,
                    max_center,
                )

    n_segments_used = len(segment_centers_sec)
    logger.info("Evaluating %d segments", n_segments_used)

    max_lag_samples = int(config["gcc_max_lag_ms"] * config["fs"] / 1000.0)
    eval_window_samples = int(float(config.get("eval_window_sec", 1.0)) * config["fs"])
    ldv_prealign = str(config.get("ldv_prealign", "none")).lower()
    if ldv_prealign not in {"none", "gcc_phat"}:
        raise ValueError(f"Invalid ldv_prealign={ldv_prealign!r} (expected: none|gcc_phat)")

    gcc_guided_radius_ms = config.get("gcc_guided_peak_radius_ms", None)
    use_guided_gcc = gcc_guided_radius_ms is not None and float(gcc_guided_radius_ms) > 0

    per_segment = []
    per_segment_raw = [] if signal_pair == "ldv_micl" else None

    for seg_idx, center_sec in enumerate(segment_centers_sec):
        logger.info("Segment %d/%d: center_t=%.2fs", seg_idx + 1, n_segments_used, center_sec)
        center_sample_global = int(center_sec * config["fs"])

        slice_start, slice_end = extract_centered_slice(
            [ldv_signal, mic_left_signal, mic_right_signal],
            center_sample=center_sample_global,
            slice_samples=slice_samples,
        )

        ldv_slice = ldv_signal[slice_start:slice_end]
        mic_left_slice = mic_left_signal[slice_start:slice_end]
        mic_right_slice = mic_right_signal[slice_start:slice_end]
        desired_center_in_slice = int(center_sample_global - slice_start)

        eval_center_sample = int(np.clip(desired_center_in_slice, 0, max(0, len(mic_left_slice) - 1)))
        t_start = eval_center_sample - eval_window_samples // 2
        t_end = t_start + eval_window_samples
        max_len_pre = min(len(ldv_slice), len(mic_left_slice), len(mic_right_slice))
        if t_start < 0:
            t_start = 0
            t_end = min(max_len_pre, eval_window_samples)
        if t_end > max_len_pre:
            t_end = max_len_pre
            t_start = max(0, t_end - eval_window_samples)

        prealign_info = None
        ldv_for_alignment_slice = ldv_slice
        if signal_pair == "ldv_micl" and ldv_prealign == "gcc_phat":
            ldv_seg_for_tau = ldv_slice[t_start:t_end]
            mic_l_seg_for_tau = mic_left_slice[t_start:t_end]
            tau_ldv_to_micl_sec, psr_ldv_to_micl_db = gcc_phat_full_analysis(
                ldv_seg_for_tau.astype(np.float64, copy=False),
                mic_l_seg_for_tau.astype(np.float64, copy=False),
                config["fs"],
                max_tau=float(config["gcc_max_lag_ms"]) / 1000.0,
                bandpass=bp,
                psr_exclude_samples=psr_exclude_samples,
            )
            delay_sec = -float(tau_ldv_to_micl_sec)
            ldv_for_alignment_slice = apply_fractional_delay_fd(ldv_slice, config["fs"], delay_sec)
            prealign_info = {
                "mode": "gcc_phat",
                "tau_ldv_to_micl_ms": float(tau_ldv_to_micl_sec * 1000.0),
                "psr_ldv_to_micl_db": float(psr_ldv_to_micl_db),
                "applied_delay_ms": float(delay_sec * 1000.0),
            }

        raw_result = None
        if signal_pair == "ldv_micl":
            _, _, Zxx_ldv = stft(
                ldv_for_alignment_slice,
                fs=config["fs"],
                nperseg=config["n_fft"],
                noverlap=config["n_fft"] - config["hop_length"],
                window="hann",
            )
            _, _, Zxx_mic_left = stft(
                mic_left_slice,
                fs=config["fs"],
                nperseg=config["n_fft"],
                noverlap=config["n_fft"] - config["hop_length"],
                window="hann",
            )

            n_time = min(Zxx_ldv.shape[1], Zxx_mic_left.shape[1])
            tw = int(config["tw"])
            max_lag = int(config["max_lag"])
            desired_frame = int(
                round((desired_center_in_slice - int(config["n_fft"]) // 2) / int(config["hop_length"]))
            )
            start_t = desired_frame - tw // 2
            start_t = max(start_t, max_lag + 1)
            start_t = min(start_t, n_time - tw - max_lag - 1)
            if start_t < max_lag + 1:
                raise ValueError(
                    "analysis_slice_sec too short for OMP window; "
                    f"need > {(tw + 2 * max_lag) * config['hop_length'] / config['fs']:.3f}s of STFT support"
                )

            if alignment_mode == "omp":
                Zxx_omp = apply_omp_alignment(Zxx_ldv, Zxx_mic_left, config, start_t)
            else:
                Zxx_omp = apply_dtmin_alignment(Zxx_ldv, Zxx_mic_left, config, start_t, dtmin_policy)
            _, ldv_omp_td = istft(
                Zxx_omp,
                fs=config["fs"],
                nperseg=config["n_fft"],
                noverlap=config["n_fft"] - config["hop_length"],
                window="hann",
            )

            max_len = min(len(ldv_omp_td), len(mic_right_slice))
            if t_end > max_len:
                t_end = max_len
                t_start = max(0, t_end - eval_window_samples)

            ldv_omp_seg = ldv_omp_td[t_start:t_end]
            mic_right_seg = mic_right_slice[t_start:t_end]

            result = estimate_tdoa_gcc_phat(
                ldv_omp_seg,
                mic_right_seg,
                config["fs"],
                max_lag_samples=max_lag_samples,
                bandpass=bp,
                psr_exclude_samples=psr_exclude_samples,
                guided_tau_ms=float(tau_ref_ms) if use_guided_gcc else None,
                guided_radius_ms=float(gcc_guided_radius_ms) if use_guided_gcc else None,
            )
            raw_result = estimate_tdoa_gcc_phat(
                ldv_slice[t_start:t_end],
                mic_right_seg,
                config["fs"],
                max_lag_samples=max_lag_samples,
                bandpass=bp,
                psr_exclude_samples=psr_exclude_samples,
                guided_tau_ms=float(tau_ref_ms) if use_guided_gcc else None,
                guided_radius_ms=float(gcc_guided_radius_ms) if use_guided_gcc else None,
            )
        else:
            mic_left_seg = mic_left_slice[t_start:t_end]
            mic_right_seg = mic_right_slice[t_start:t_end]
            result = estimate_tdoa_gcc_phat(
                mic_left_seg,
                mic_right_seg,
                config["fs"],
                max_lag_samples=max_lag_samples,
                bandpass=bp,
                psr_exclude_samples=psr_exclude_samples,
                guided_tau_ms=float(tau_ref_ms) if use_guided_gcc else None,
                guided_radius_ms=float(gcc_guided_radius_ms) if use_guided_gcc else None,
            )

        result["theta_deg"] = float(tau_to_doa(result["tau_ms"], config))
        result["theta_error_deg"] = float(abs(result["theta_deg"] - theta_ref_deg))
        result["center_sec"] = float(center_sec)
        if prealign_info is not None:
            result["prealign"] = prealign_info

        per_segment.append(result)
        if raw_result is not None and per_segment_raw is not None:
            raw_result["theta_deg"] = float(tau_to_doa(raw_result["tau_ms"], config))
            raw_result["theta_error_deg"] = float(abs(raw_result["theta_deg"] - theta_ref_deg))
            raw_result["center_sec"] = float(center_sec)
            per_segment_raw.append(raw_result)

    tau_median = float(np.median([r["tau_ms"] for r in per_segment]))
    theta_median = float(np.median([r["theta_deg"] for r in per_segment]))
    theta_err_median = float(np.median([r["theta_error_deg"] for r in per_segment]))
    psr_median = float(np.median([r["psr_db"] for r in per_segment]))

    raw_summary = None
    if per_segment_raw:
        raw_tau_median = float(np.median([r["tau_ms"] for r in per_segment_raw]))
        raw_theta_median = float(np.median([r["theta_deg"] for r in per_segment_raw]))
        raw_theta_err_median = float(np.median([r["theta_error_deg"] for r in per_segment_raw]))
        raw_psr_median = float(np.median([r["psr_db"] for r in per_segment_raw]))
        raw_summary = {
            "tau_median_ms": raw_tau_median,
            "theta_median_deg": raw_theta_median,
            "theta_error_median_deg": raw_theta_err_median,
            "psr_median_db": raw_psr_median,
            "per_segment": per_segment_raw,
        }

    psr_min_ok = True if pass_psr_min_db is None else bool(psr_median >= pass_psr_min_db)
    if effective_pass_mode == "omp_vs_raw":
        if raw_summary is None:
            raise ValueError("pass_mode=omp_vs_raw requires raw LDV evaluation")
        omp_better_than_raw = bool(theta_err_median < raw_summary["theta_error_median_deg"])
        omp_error_small = bool(theta_err_median < pass_theta_max_deg)
        passed = bool(omp_better_than_raw and omp_error_small and psr_min_ok)
        pass_conditions = {
            "pass_mode": "omp_vs_raw",
            "alignment_mode": alignment_mode,
            "omp_better_than_raw": omp_better_than_raw,
            "aligned_better_than_raw": omp_better_than_raw,
            "omp_error_small": omp_error_small,
            "aligned_error_small": omp_error_small,
            "omp_psr_improved": bool(psr_median > raw_summary["psr_median_db"]),
            "aligned_psr_improved": bool(psr_median > raw_summary["psr_median_db"]),
            "psr_min_db": None if pass_psr_min_db is None else float(pass_psr_min_db),
            "psr_min_ok": psr_min_ok,
            "passed": passed,
        }
    else:
        theta_error_small = bool(theta_err_median < pass_theta_max_deg)
        passed = bool(theta_error_small and psr_min_ok)
        pass_conditions = {
            "pass_mode": "theta_only",
            "alignment_mode": alignment_mode,
            "theta_error_small": theta_error_small,
            "psr_min_db": None if pass_psr_min_db is None else float(pass_psr_min_db),
            "psr_min_ok": psr_min_ok,
            "passed": passed,
        }

    summary = {
        "speaker_id": speaker_id,
        "signal_pair": signal_pair,
        "pair_description": (
            f"LDV aligned to MicL ({alignment_mode.upper()}) paired with MicR"
            if signal_pair == "ldv_micl"
            else "MicL-MicR"
        ),
        "ground_truth": ground_truth,
        "truth_reference": {
            "mode": truth_mode,
            "label": None if truth_label is None else str(truth_label),
            "tau_ref_ms": float(tau_ref_ms),
            "theta_ref_deg": float(theta_ref_deg),
        },
        "n_segments": int(n_segments_used),
        "segment_mode": segment_mode,
        "scan_summary": scan_summary,
        "segment_centers_sec": segment_centers_sec,
        "ldv_prealign": ldv_prealign,
        "alignment_mode": alignment_mode,
        "dtmin_model_path": None if dtmin_model_path is None else str(dtmin_model_path),
        "dtmin_model_metadata": None if dtmin_policy is None else dtmin_policy.get("metadata"),
        "config": config,
        "result": {
            "tau_median_ms": tau_median,
            "theta_median_deg": theta_median,
            "theta_error_median_deg": theta_err_median,
            "psr_median_db": psr_median,
            "per_segment": per_segment,
        },
        "result_raw": raw_summary,
        "pass_thresholds": {
            "theta_error_max_deg": float(pass_theta_max_deg),
            "psr_min_db": None if pass_psr_min_db is None else float(pass_psr_min_db),
        },
        "passed": bool(passed),
        "pass_conditions": pass_conditions,
        "pass_mode": effective_pass_mode,
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "summary.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved to: %s", output_file)

    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: LDV-vs-Mic DoA (GCC-PHAT)")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of dataset")
    parser.add_argument("--speaker", type=str, required=True, help="Speaker folder (e.g., 20-0.1V)")
    parser.add_argument("--speaker_key", type=str, default=None, help="Override geometry speaker key")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--signal_pair", type=str, choices=["ldv_micl", "micl_micr"], default="ldv_micl")

    parser.add_argument("--n_segments", type=int, default=5)
    parser.add_argument("--segment_mode", type=str, default="scan", choices=["fixed", "scan"])
    parser.add_argument("--segment_spacing_sec", type=float, default=None)
    parser.add_argument("--segment_offset_sec", type=float, default=None)
    parser.add_argument("--analysis_slice_sec", type=float, default=None)
    parser.add_argument("--eval_window_sec", type=float, default=None)

    parser.add_argument("--scan_start_sec", type=float, default=None)
    parser.add_argument("--scan_end_sec", type=float, default=None)
    parser.add_argument("--scan_hop_sec", type=float, default=1.0)
    parser.add_argument("--scan_psr_min_db", type=float, default=None)
    parser.add_argument("--scan_ldv_micl_psr_min_db", type=float, default=None)
    parser.add_argument("--scan_tau_err_max_ms", type=float, default=2.0)
    parser.add_argument("--scan_sort_by", type=str, default="tau_err", choices=["tau_err", "psr"])
    parser.add_argument("--scan_min_separation_sec", type=float, default=None)
    parser.add_argument("--scan_allow_fallback", action="store_true")

    parser.add_argument("--gcc_bandpass_low", type=float, default=None)
    parser.add_argument("--gcc_bandpass_high", type=float, default=None)
    parser.add_argument("--gcc_guided_peak_radius_ms", type=float, default=None)
    parser.add_argument("--psr_exclude_samples", type=int, default=None)
    parser.add_argument("--ldv_prealign", type=str, choices=["none", "gcc_phat"], default=None)
    parser.add_argument("--alignment_mode", type=str, choices=["omp", "dtmin"], default=None)
    parser.add_argument("--dtmin_model_path", type=str, default=None)
    parser.add_argument("--max_k", type=int, default=None)

    parser.add_argument("--truth_tau_ms", type=float, default=None)
    parser.add_argument("--truth_theta_deg", type=float, default=None)
    parser.add_argument("--truth_label", type=str, default=None)
    parser.add_argument("--use_geometry_truth", action="store_true")

    parser.add_argument("--pass_theta_max_deg", type=float, default=5.0)
    parser.add_argument("--pass_psr_min_db", type=float, default=None)
    parser.add_argument(
        "--pass_mode",
        type=str,
        default="auto",
        choices=["auto", "theta_only", "omp_vs_raw"],
        help="Pass criteria: auto=omp_vs_raw for ldv_micl, theta_only for micl_micr.",
    )

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
    if args.segment_spacing_sec is not None:
        config["segment_spacing_sec"] = float(args.segment_spacing_sec)
    if args.segment_offset_sec is not None:
        config["segment_offset_sec"] = float(args.segment_offset_sec)
    if args.analysis_slice_sec is not None:
        config["analysis_slice_sec"] = float(args.analysis_slice_sec)
    if args.eval_window_sec is not None:
        config["eval_window_sec"] = float(args.eval_window_sec)
    if args.gcc_bandpass_low is not None:
        config["gcc_bandpass_low"] = float(args.gcc_bandpass_low)
    if args.gcc_bandpass_high is not None:
        config["gcc_bandpass_high"] = float(args.gcc_bandpass_high)
    if args.gcc_guided_peak_radius_ms is not None:
        config["gcc_guided_peak_radius_ms"] = float(args.gcc_guided_peak_radius_ms)
    if args.psr_exclude_samples is not None:
        config["psr_exclude_samples"] = int(args.psr_exclude_samples)
    if args.ldv_prealign is not None:
        config["ldv_prealign"] = str(args.ldv_prealign)
    if args.alignment_mode is not None:
        config["alignment_mode"] = str(args.alignment_mode)
    if args.dtmin_model_path is not None:
        config["dtmin_model_path"] = str(args.dtmin_model_path)
    if args.max_k is not None:
        config["max_k"] = int(args.max_k)

    truth_tau_ms = None if args.use_geometry_truth else args.truth_tau_ms
    truth_theta_deg = None if args.use_geometry_truth else args.truth_theta_deg
    truth_label = None if args.use_geometry_truth else args.truth_label

    data_dir = Path(args.data_root) / args.speaker
    ldv_files = list(data_dir.glob("*LDV*.wav"))
    left_mic_files = list(data_dir.glob("*LEFT*.wav"))
    right_mic_files = list(data_dir.glob("*RIGHT*.wav"))

    if not ldv_files or not left_mic_files or not right_mic_files:
        raise FileNotFoundError(f"Missing audio files in {data_dir}")

    try:
        run_stage4_evaluation(
            ldv_path=str(ldv_files[0]),
            mic_left_path=str(left_mic_files[0]),
            mic_right_path=str(right_mic_files[0]),
            config=config,
            output_dir=str(output_dir),
            signal_pair=args.signal_pair,
            n_segments=args.n_segments,
            segment_mode=args.segment_mode,
            scan_start_sec=args.scan_start_sec,
            scan_end_sec=args.scan_end_sec,
            scan_hop_sec=args.scan_hop_sec,
            scan_psr_min_db=args.scan_psr_min_db,
            scan_ldv_micl_psr_min_db=args.scan_ldv_micl_psr_min_db,
            scan_tau_err_max_ms=args.scan_tau_err_max_ms,
            scan_sort_by=args.scan_sort_by,
            scan_min_separation_sec=args.scan_min_separation_sec,
            scan_allow_fallback=args.scan_allow_fallback,
            truth_tau_ms=truth_tau_ms,
            truth_theta_deg=truth_theta_deg,
            truth_label=truth_label,
            pass_theta_max_deg=args.pass_theta_max_deg,
            pass_psr_min_db=args.pass_psr_min_db,
            pass_mode=args.pass_mode,
            speaker_key_override=args.speaker_key,
            alignment_mode=str(config.get("alignment_mode", "omp")),
            dtmin_model_path=config.get("dtmin_model_path"),
        )
    except Exception as exc:
        logger.exception("Run failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
