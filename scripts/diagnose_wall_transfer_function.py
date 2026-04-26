#!/usr/bin/env python
"""
Wall Panel Transfer Function Diagnostic.

Validates the cardboard wall-panel modal hypothesis by analysing
cross-spectral phase, coherence, and narrow-band GCC-PHAT behaviour
across three sensor pairs (LDV-MicL, LDV-MicR, MicL-MicR).

Outputs structured JSON results, a human-readable markdown report,
a hypothesis verdict, and diagnostic plots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.fft import fft, ifft
from scipy.io import wavfile
from scipy.signal import coherence, csd, welch

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Reproducibility helpers
# ─────────────────────────────────────────────────────────────────────


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_and_dirty() -> tuple[str | None, bool | None]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() != ""
        return head, dirty
    except Exception:
        return None, None


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _dataset_fingerprint_sha256(entries: list[tuple[str, str]]) -> str:
    """
    entries: list of (sha256, relpath). Order-independent.
    """
    lines = [f"{sha} {rel}" for sha, rel in sorted(entries, key=lambda t: t[1].lower())]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()

# ─────────────────────────────────────────────────────────────────────
# [A] Constants & geometry  (reused from multi_sensor_fusion_doa.py)
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

FS = 48000
NPERSEG = 8192
NOVERLAP = 4096
ANALYSIS_BAND = (500, 2000)
SWEEP_BW = 50
SWEEP_STEP = 25
SWEEP_F_START = 200
SWEEP_F_STOP = 4000
MAX_LAG_MS = 7.0


# ─────────────────────────────────────────────────────────────────────
# [B] Reused I/O & signal utilities
# ─────────────────────────────────────────────────────────────────────


def compute_all_ground_truths(speaker_key: str, c: float = 343.0, d: float = 1.4) -> dict:
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
    sin_theta = float(np.clip(tau1_true * c / d, -1.0, 1.0))
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
    sig1 = np.asarray(sig1, dtype=np.float64)
    sig2 = np.asarray(sig2, dtype=np.float64)
    if not (np.all(np.isfinite(sig1)) and np.all(np.isfinite(sig2))):
        raise FloatingPointError("Non-finite input samples to gcc_phat_full_analysis")
    n = len(sig1) + len(sig2)
    SIG1 = fft(sig1, n)
    SIG2 = fft(sig2, n)
    R = SIG1 * np.conj(SIG2)
    if bandpass is not None:
        f_lo, f_hi = float(bandpass[0]), float(bandpass[1])
        if not (0.0 < f_lo < f_hi < fs / 2):
            raise ValueError(
                f"Invalid bandpass range: lowcut={bandpass[0]}, highcut={bandpass[1]}, fs={fs}"
            )
        freqs = np.fft.fftfreq(n, d=1.0 / fs)
        mask = (np.abs(freqs) >= f_lo) & (np.abs(freqs) <= f_hi)
        R = np.where(mask, R, 0.0)
    R = R / (np.abs(R) + 1e-10)
    cc = np.real(ifft(R))
    if max_tau is not None:
        max_shift = int(max_tau * fs)
    else:
        max_shift = n // 2
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    abs_cc = np.abs(cc)
    if not np.all(np.isfinite(abs_cc)):
        raise FloatingPointError("Non-finite GCC magnitude (abs_cc)")
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
        shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
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
    if not (np.isfinite(tau) and np.isfinite(psr)):
        raise FloatingPointError(f"Non-finite GCC result: tau={tau}, psr={psr}")
    return float(tau), float(psr)


# ─────────────────────────────────────────────────────────────────────
# [C] New diagnostic functions
# ─────────────────────────────────────────────────────────────────────


def compute_csd_phase(
    sig_a: np.ndarray, sig_b: np.ndarray, fs: int,
    nperseg: int = NPERSEG, noverlap: int = NOVERLAP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cross-spectral density phase (wrapped and unwrapped)."""
    freqs, S12 = csd(sig_a, sig_b, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann")
    phi_wrapped = np.angle(S12)
    phi_unwrapped = np.unwrap(phi_wrapped)
    return freqs, phi_wrapped, phi_unwrapped, S12


def compute_msc(
    sig_a: np.ndarray, sig_b: np.ndarray, fs: int,
    nperseg: int = NPERSEG, noverlap: int = NOVERLAP,
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude-squared coherence."""
    freqs, gamma2 = coherence(sig_a, sig_b, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann")
    return freqs, gamma2


def fit_phase_linearity(
    freqs: np.ndarray, phi_unwrapped: np.ndarray, gamma2: np.ndarray,
    f_low: float, f_high: float,
) -> dict:
    """Weighted least-squares linear fit to unwrapped phase in [f_low, f_high]."""
    mask = (freqs >= f_low) & (freqs <= f_high)
    f_sel = freqs[mask]
    phi_sel = phi_unwrapped[mask]
    w = gamma2[mask].copy()
    w = np.maximum(w, 0.0)
    w_sum = w.sum()
    if w_sum < 1e-12 or len(f_sel) < 3:
        return {
            "tau_fit_ms": 0.0, "phi0": 0.0, "R2_weighted": 0.0,
            "residual_rms_rad": float("inf"), "residual_max_rad": float("inf"),
        }
    # Weighted LS: phi = slope * f + intercept
    f_mean = np.average(f_sel, weights=w)
    phi_mean = np.average(phi_sel, weights=w)
    num = np.sum(w * (f_sel - f_mean) * (phi_sel - phi_mean))
    den = np.sum(w * (f_sel - f_mean) ** 2)
    slope = num / (den + 1e-30)
    intercept = phi_mean - slope * f_mean
    phi_pred = slope * f_sel + intercept
    residual = phi_sel - phi_pred
    SS_res = np.sum(w * residual**2)
    SS_tot = np.sum(w * (phi_sel - phi_mean) ** 2)
    R2 = 1.0 - SS_res / (SS_tot + 1e-30) if SS_tot > 1e-30 else 0.0
    tau_fit_ms = float(-slope / (2 * np.pi) * 1000.0)
    return {
        "tau_fit_ms": tau_fit_ms,
        "phi0": float(intercept),
        "R2_weighted": float(np.clip(R2, 0.0, 1.0)),
        "residual_rms_rad": float(np.sqrt(np.mean(residual**2))),
        "residual_max_rad": float(np.max(np.abs(residual))),
    }


def count_phase_jumps(
    phi_wrapped: np.ndarray, freqs: np.ndarray,
    f_low: float, f_high: float, threshold: float = np.pi / 2,
) -> tuple[int, list[float]]:
    """Count phase jumps exceeding threshold in [f_low, f_high]."""
    mask = (freqs >= f_low) & (freqs <= f_high)
    phi_sel = phi_wrapped[mask]
    f_sel = freqs[mask]
    if len(phi_sel) < 2:
        return 0, []
    dphi = np.diff(phi_sel)
    # Wrap differences to [-pi, pi]
    dphi_wrapped = (dphi + np.pi) % (2 * np.pi) - np.pi
    jumps = np.abs(dphi_wrapped) > threshold
    jump_count = int(np.sum(jumps))
    jump_freqs = f_sel[1:][jumps].tolist()
    return jump_count, jump_freqs


def compute_phase_roughness(
    phi_unwrapped: np.ndarray, freqs: np.ndarray,
    f_low: float, f_high: float,
) -> float:
    """RMS of first-order phase differences (detrended) in [f_low, f_high]."""
    mask = (freqs >= f_low) & (freqs <= f_high)
    phi_sel = phi_unwrapped[mask]
    if len(phi_sel) < 3:
        return 0.0
    # Remove linear trend
    x = np.arange(len(phi_sel), dtype=np.float64)
    coeffs = np.polyfit(x, phi_sel, 1)
    phi_detrended = phi_sel - np.polyval(coeffs, x)
    dphi = np.diff(phi_detrended)
    return float(np.sqrt(np.mean(dphi**2)))


def narrowband_gcc_sweep(
    sig_a: np.ndarray, sig_b: np.ndarray, fs: int,
    f_start: int = SWEEP_F_START, f_stop: int = SWEEP_F_STOP,
    bw: int = SWEEP_BW, step: int = SWEEP_STEP,
    max_lag_ms: float = MAX_LAG_MS,
    tau_expected_ms: float = 0.0, tau_tol_ms: float = 0.5,
) -> list[dict]:
    """Sweep narrow-band GCC-PHAT across frequency."""
    results = []
    max_tau = max_lag_ms / 1000.0
    for f_c in range(f_start, f_stop, step):
        f_lo = max(20, f_c - bw // 2)
        f_hi = f_c + bw // 2
        if f_hi >= fs // 2:
            break
        try:
            tau_sec, psr_db = gcc_phat_full_analysis(
                sig_a.astype(np.float64, copy=False),
                sig_b.astype(np.float64, copy=False),
                fs,
                max_tau=max_tau,
                bandpass=(float(f_lo), float(f_hi)),
                psr_exclude_samples=50,
            )
            tau_ms = tau_sec * 1000.0
            is_correct = abs(tau_ms - tau_expected_ms) < tau_tol_ms
            is_zero_peak = abs(tau_ms) < 0.3
            results.append({
                "f_center": f_c,
                "tau_ms": float(tau_ms),
                "psr_db": float(psr_db),
                "is_correct": bool(is_correct),
                "is_zero_peak": bool(is_zero_peak),
            })
        except Exception:
            results.append({
                "f_center": f_c,
                "tau_ms": None,
                "psr_db": None,
                "is_correct": False,
                "is_zero_peak": False,
            })
    return results


# ─────────────────────────────────────────────────────────────────────
# [D] Checkpoints
# ─────────────────────────────────────────────────────────────────────


def checkpoint_0_data_sanity(
    ldv: np.ndarray, micl: np.ndarray, micr: np.ndarray, fs: int,
) -> dict:
    """Data sanity: fs, equal length (within tolerance), RMS, clipping."""
    results = {"checkpoint": 0, "name": "data_sanity", "checks": {}}
    checks = results["checks"]
    checks["fs_correct"] = fs == FS
    min_len = min(len(ldv), len(micl), len(micr))
    max_len = max(len(ldv), len(micl), len(micr))
    checks["lengths_match"] = (max_len - min_len) < fs  # within 1s
    for name, sig in [("ldv", ldv), ("micl", micl), ("micr", micr)]:
        rms = float(np.sqrt(np.mean(sig[:min_len].astype(np.float64) ** 2)))
        max_abs = float(np.max(np.abs(sig[:min_len])))
        checks[f"{name}_rms"] = rms
        checks[f"{name}_rms_ok"] = rms > 1e-6
        checks[f"{name}_max_abs"] = max_abs
        checks[f"{name}_no_clip"] = max_abs < 0.95
    all_pass = all(v for k, v in checks.items() if k.endswith("_ok") or k.endswith("_correct")
                   or k == "lengths_match" or k.endswith("_no_clip"))
    results["pass"] = bool(all_pass)
    return results


def checkpoint_1_control(
    micl: np.ndarray, micr: np.ndarray, fs: int,
    tau1_expected_ms: float,
) -> dict:
    """MicL-MicR control verification."""
    results = {"checkpoint": 1, "name": "micl_micr_control", "checks": {}}
    checks = results["checks"]

    freqs, gamma2 = compute_msc(micl, micr, fs)
    mask_mid = (freqs >= ANALYSIS_BAND[0]) & (freqs <= ANALYSIS_BAND[1])
    gamma2_median = float(np.median(gamma2[mask_mid]))
    checks["gamma2_median_500_2000"] = gamma2_median
    checks["gamma2_ok"] = gamma2_median > 0.3

    _, phi_w, phi_u, _ = compute_csd_phase(micl, micr, fs)
    lin = fit_phase_linearity(freqs, phi_u, gamma2, *ANALYSIS_BAND)
    checks["R2_weighted"] = lin["R2_weighted"]
    checks["R2_ok"] = lin["R2_weighted"] > 0.8

    checks["tau_fit_ms"] = lin["tau_fit_ms"]
    checks["tau_expected_ms"] = tau1_expected_ms
    tau_err = abs(lin["tau_fit_ms"] - tau1_expected_ms)
    checks["tau_err_ms"] = float(tau_err)
    checks["tau_ok"] = tau_err < 0.5

    roughness = compute_phase_roughness(phi_u, freqs, *ANALYSIS_BAND)
    checks["phase_roughness_rad"] = roughness
    checks["roughness_ok"] = roughness < 0.5

    results["pass"] = all([
        checks["gamma2_ok"], checks["R2_ok"],
        checks["tau_ok"], checks["roughness_ok"],
    ])
    results["phase_linearity"] = lin
    results["gamma2_median"] = gamma2_median
    return results


def checkpoint_2_ldv_quality(
    ldv: np.ndarray, micl: np.ndarray, fs: int,
) -> dict:
    """LDV signal quality check."""
    results = {"checkpoint": 2, "name": "ldv_quality", "checks": {}}
    checks = results["checks"]

    # LDV auto-power in band vs out-of-band
    freqs_psd, psd_ldv = welch(ldv, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP, window="hann")
    mask_in = (freqs_psd >= ANALYSIS_BAND[0]) & (freqs_psd <= ANALYSIS_BAND[1])
    mask_out = ~mask_in & (freqs_psd > 0)
    power_in = float(np.mean(psd_ldv[mask_in])) if np.any(mask_in) else 0.0
    power_out = float(np.mean(psd_ldv[mask_out])) if np.any(mask_out) else 1.0
    checks["band_power_ratio"] = float(power_in / (power_out + 1e-30))
    checks["band_power_ok"] = power_in > power_out * 0.01

    # LDV-MicL coherence above noise floor
    freqs_c, gamma2 = compute_msc(ldv, micl, fs)
    mask_mid = (freqs_c >= ANALYSIS_BAND[0]) & (freqs_c <= ANALYSIS_BAND[1])
    gamma2_mid = gamma2[mask_mid]
    noise_floor = 1.0 / 57  # ~K segments for 5s data
    n_above = int(np.sum(gamma2_mid > noise_floor))
    checks["n_bins_above_noise_floor"] = n_above
    checks["total_bins_in_band"] = int(np.sum(mask_mid))
    checks["frac_above_noise"] = float(n_above / max(1, int(np.sum(mask_mid))))
    checks["coherence_ok"] = n_above > 0

    results["pass"] = checks["band_power_ok"] and checks["coherence_ok"]
    return results


# ─────────────────────────────────────────────────────────────────────
# [E] Per-segment and per-speaker diagnostics
# ─────────────────────────────────────────────────────────────────────


def _band_median(values: np.ndarray, freqs: np.ndarray, f_lo: float, f_hi: float) -> float:
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return 0.0
    return float(np.median(values[mask]))


def diagnose_single_segment(
    ldv: np.ndarray, micl: np.ndarray, micr: np.ndarray,
    fs: int, center_sec: float, ground_truths: dict,
) -> dict:
    """Run all diagnostics on a single 5-second segment."""
    tau1_ms = ground_truths["tau1_true_ms"]
    tau2_ms = ground_truths["tau2_true_ms"]
    tau3_ms = ground_truths["tau3_true_ms"]

    seg = {"center_sec": center_sec}

    # --- Coherence for 3 pairs ---
    pairs = [
        ("LDV_MicL", ldv, micl, tau2_ms),
        ("LDV_MicR", ldv, micr, tau3_ms),
        ("MicL_MicR", micl, micr, tau1_ms),
    ]
    coherence_data = {}
    phase_data = {}
    for label, sa, sb, tau_exp in pairs:
        freqs_c, gamma2 = compute_msc(sa, sb, fs)
        freqs_p, phi_w, phi_u, S12 = compute_csd_phase(sa, sb, fs)

        gamma2_low = _band_median(gamma2, freqs_c, 100, 500)
        gamma2_mid = _band_median(gamma2, freqs_c, *ANALYSIS_BAND)

        lin = fit_phase_linearity(freqs_p, phi_u, gamma2, *ANALYSIS_BAND)
        jump_count, jump_freqs = count_phase_jumps(phi_w, freqs_p, *ANALYSIS_BAND)
        roughness = compute_phase_roughness(phi_u, freqs_p, *ANALYSIS_BAND)

        coherence_data[label] = {
            "gamma2_low_100_500": gamma2_low,
            "gamma2_mid_500_2000": gamma2_mid,
            "gamma2_ratio_low_mid": float(gamma2_low / (gamma2_mid + 1e-10)),
            "gamma2_freqs": freqs_c.tolist(),
            "gamma2_values": gamma2.tolist(),
        }
        phase_data[label] = {
            "phase_linearity": lin,
            "jump_count": jump_count,
            "jump_freqs": jump_freqs[:20],  # cap for JSON size
            "phase_roughness_rad": roughness,
            "phi_wrapped": phi_w.tolist(),
            "phi_unwrapped": phi_u.tolist(),
            "freqs": freqs_p.tolist(),
        }

    # --- Narrowband GCC sweep for 3 pairs ---
    sweep_data = {}
    for label, sa, sb, tau_exp in pairs:
        sweep = narrowband_gcc_sweep(
            sa, sb, fs,
            tau_expected_ms=tau_exp,
            tau_tol_ms=0.5,
        )
        n_valid = sum(1 for r in sweep if r["tau_ms"] is not None)
        n_correct = sum(1 for r in sweep if r["is_correct"])
        n_zero = sum(1 for r in sweep if r["is_zero_peak"])
        sweep_data[label] = {
            "bands": sweep,
            "n_valid": n_valid,
            "n_correct": n_correct,
            "n_zero_peak": n_zero,
            "frac_correct": float(n_correct / max(1, n_valid)),
            "frac_zero_peak": float(n_zero / max(1, n_valid)),
        }

    # --- Wideband GCC-PHAT in the analysis band (post-hoc anchor) ---
    wideband = {}
    for label, sa, sb, _tau_exp in pairs:
        tau_sec, psr_db = gcc_phat_full_analysis(
            sa,
            sb,
            fs,
            max_tau=float(MAX_LAG_MS) / 1000.0,
            bandpass=(float(ANALYSIS_BAND[0]), float(ANALYSIS_BAND[1])),
            psr_exclude_samples=50,
        )
        wideband[label] = {"tau_ms": float(tau_sec * 1000.0), "psr_db": float(psr_db)}

    seg["coherence"] = coherence_data
    seg["phase"] = phase_data
    seg["narrowband_sweep"] = sweep_data
    seg["wideband_gcc"] = wideband
    return seg


def diagnose_speaker(
    data_dir: str, speaker_key: str,
    segment_centers: list[float], output_dir: str,
) -> dict:
    """Run diagnostics for one speaker across all segments."""
    speaker_dir = Path(data_dir)
    ldv_files = sorted(speaker_dir.glob("*LDV*.wav"))
    left_files = sorted(speaker_dir.glob("*LEFT*.wav"))
    right_files = sorted(speaker_dir.glob("*RIGHT*.wav"))
    if not ldv_files or not left_files or not right_files:
        raise FileNotFoundError(f"Missing WAV files in {speaker_dir}")

    logger.info("Loading WAV files for speaker %s ...", speaker_key)
    sr_ldv, ldv_full = load_wav(str(ldv_files[0]))
    sr_ml, micl_full = load_wav(str(left_files[0]))
    sr_mr, micr_full = load_wav(str(right_files[0]))
    fs = sr_ldv
    assert sr_ldv == sr_ml == sr_mr == FS, f"Sample rate mismatch: {sr_ldv},{sr_ml},{sr_mr}"

    ground_truths = compute_all_ground_truths(speaker_key)
    logger.info("  Ground truths: tau1=%.3f ms, tau2=%.3f ms, tau3=%.3f ms",
                ground_truths["tau1_true_ms"], ground_truths["tau2_true_ms"],
                ground_truths["tau3_true_ms"])

    slice_samples = int(5.0 * fs)  # 5s analysis window
    min_len = min(len(ldv_full), len(micl_full), len(micr_full))

    # Checkpoint 0
    ck0 = checkpoint_0_data_sanity(ldv_full, micl_full, micr_full, fs)
    logger.info("  Checkpoint 0 (data sanity): %s", "PASS" if ck0["pass"] else "FAIL")

    per_segment = []
    ck1_results = []
    ck2_results = []

    for seg_idx, center_sec in enumerate(segment_centers):
        center_sample = int(center_sec * fs)
        s_start, s_end = extract_centered_slice(
            [ldv_full, micl_full, micr_full],
            center_sample=center_sample, slice_samples=slice_samples,
        )
        ldv_seg = ldv_full[s_start:s_end].astype(np.float64, copy=False)
        micl_seg = micl_full[s_start:s_end].astype(np.float64, copy=False)
        micr_seg = micr_full[s_start:s_end].astype(np.float64, copy=False)

        # Checkpoint 1
        ck1 = checkpoint_1_control(micl_seg, micr_seg, fs, ground_truths["tau1_true_ms"])
        ck1_results.append(ck1)
        logger.info("  Seg %d (%.1fs) CK1: %s (R²=%.3f, γ²=%.3f, τ_err=%.3fms)",
                     seg_idx, center_sec, "PASS" if ck1["pass"] else "FAIL",
                     ck1["checks"]["R2_weighted"], ck1["checks"]["gamma2_median_500_2000"],
                     ck1["checks"]["tau_err_ms"])

        # Checkpoint 2
        ck2 = checkpoint_2_ldv_quality(ldv_seg, micl_seg, fs)
        ck2_results.append(ck2)
        logger.info("  Seg %d CK2: %s (band_ratio=%.3f, frac_above=%.3f)",
                     seg_idx, "PASS" if ck2["pass"] else "FAIL",
                     ck2["checks"]["band_power_ratio"], ck2["checks"]["frac_above_noise"])

        # Full diagnostic
        logger.info("  Seg %d: running full diagnostic ...", seg_idx)
        seg_result = diagnose_single_segment(
            ldv_seg, micl_seg, micr_seg, fs, center_sec, ground_truths,
        )
        seg_result["checkpoint_1"] = ck1
        seg_result["checkpoint_2"] = ck2
        per_segment.append(seg_result)

    # --- Aggregate across segments ---
    agg = _aggregate_speaker(per_segment, ground_truths)

    result = {
        "speaker_key": speaker_key,
        "speaker_dir": str(speaker_dir),
        "ground_truths": ground_truths,
        "n_segments": len(segment_centers),
        "segment_centers_sec": segment_centers,
        "checkpoint_0": ck0,
        "aggregated": agg,
        "per_segment": per_segment,
    }

    # Save per-speaker results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "segment_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


def _aggregate_speaker(per_segment: list[dict], ground_truths: dict) -> dict:
    """Aggregate diagnostic metrics across segments for one speaker."""
    agg = {}
    for pair_label in ["LDV_MicL", "LDV_MicR", "MicL_MicR"]:
        gamma2_mids = [s["coherence"][pair_label]["gamma2_mid_500_2000"] for s in per_segment]
        gamma2_lows = [s["coherence"][pair_label]["gamma2_low_100_500"] for s in per_segment]
        gamma2_ratios = [s["coherence"][pair_label]["gamma2_ratio_low_mid"] for s in per_segment]
        R2s = [s["phase"][pair_label]["phase_linearity"]["R2_weighted"] for s in per_segment]
        jumps = [s["phase"][pair_label]["jump_count"] for s in per_segment]
        roughness = [s["phase"][pair_label]["phase_roughness_rad"] for s in per_segment]
        tau_fits = [s["phase"][pair_label]["phase_linearity"]["tau_fit_ms"] for s in per_segment]
        frac_correct = [s["narrowband_sweep"][pair_label]["frac_correct"] for s in per_segment]
        frac_zero = [s["narrowband_sweep"][pair_label]["frac_zero_peak"] for s in per_segment]

        agg[pair_label] = {
            "gamma2_mid_median": float(np.median(gamma2_mids)),
            "gamma2_low_median": float(np.median(gamma2_lows)),
            "gamma2_ratio_median": float(np.median(gamma2_ratios)),
            "R2_median": float(np.median(R2s)),
            "jump_count_median": float(np.median(jumps)),
            "phase_roughness_median": float(np.median(roughness)),
            "tau_fit_ms_median": float(np.median(tau_fits)),
            "frac_correct_median": float(np.median(frac_correct)),
            "frac_zero_peak_median": float(np.median(frac_zero)),
        }

    # Checkpoint 3: γ² frequency profile
    ratio = agg["LDV_MicL"]["gamma2_ratio_median"]
    ck3 = {
        "checkpoint": 3, "name": "gamma2_frequency_profile",
        "gamma2_ratio_low_mid": ratio,
        "wall_modal_prediction": ratio > 2,
        "pass": True,  # always pass (informational)
    }
    agg["checkpoint_3"] = ck3

    # Checkpoint 4: phase behaviour classification
    ldv_micl_jumps = agg["LDV_MicL"]["jump_count_median"]
    micl_micr_roughness = agg["MicL_MicR"]["phase_roughness_median"]
    ldv_micl_roughness = agg["LDV_MicL"]["phase_roughness_median"]
    roughness_ratio = ldv_micl_roughness / (micl_micr_roughness + 1e-10)
    ldv_micl_R2 = agg["LDV_MicL"]["R2_median"]
    ck4 = {
        "checkpoint": 4, "name": "phase_behaviour",
        "ldv_micl_jump_count": ldv_micl_jumps,
        "roughness_ratio_ldv_vs_mic": roughness_ratio,
        "ldv_micl_R2": ldv_micl_R2,
        "wall_modal_prediction_jumps": 30 <= ldv_micl_jumps <= 60,
        "wall_modal_prediction_roughness": roughness_ratio > 5,
        "wall_modal_prediction_R2": ldv_micl_R2 < 0.2,
        "pass": True,
    }
    agg["checkpoint_4"] = ck4

    return agg


# ─────────────────────────────────────────────────────────────────────
# [F] Cross-speaker independence analysis (Checkpoint 5)
# ─────────────────────────────────────────────────────────────────────


def compute_speaker_independence(all_speaker_results: dict) -> dict:
    """Compare γ²(f) profiles and phase-jump positions across speakers."""
    speaker_keys = sorted(all_speaker_results.keys())
    if len(speaker_keys) < 2:
        return {"n_speakers": len(speaker_keys), "skip": True}

    # Collect per-speaker γ² profiles (average across segments)
    gamma2_profiles = {}
    jump_freq_sets = {}
    for spk in speaker_keys:
        spk_data = all_speaker_results[spk]
        segs = spk_data["per_segment"]
        # Average γ² across segments for LDV_MicL
        all_g2 = []
        all_jumps = set()
        for seg in segs:
            g2_vals = seg["coherence"]["LDV_MicL"]["gamma2_values"]
            all_g2.append(np.array(g2_vals))
            jf = seg["phase"]["LDV_MicL"]["jump_freqs"]
            all_jumps.update(jf)
        if all_g2:
            gamma2_profiles[spk] = np.mean(all_g2, axis=0)
        jump_freq_sets[spk] = all_jumps

    # γ² profile correlation matrix
    n = len(speaker_keys)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            gi = gamma2_profiles.get(speaker_keys[i])
            gj = gamma2_profiles.get(speaker_keys[j])
            if gi is not None and gj is not None and len(gi) == len(gj):
                # Focus on analysis band
                # freqs from first segment
                segs_i = all_speaker_results[speaker_keys[i]]["per_segment"]
                freqs = np.array(segs_i[0]["coherence"]["LDV_MicL"]["gamma2_freqs"])
                mask = (freqs >= ANALYSIS_BAND[0]) & (freqs <= ANALYSIS_BAND[1])
                gi_band = gi[mask]
                gj_band = gj[mask]
                if len(gi_band) > 2:
                    cc = np.corrcoef(gi_band, gj_band)[0, 1]
                    corr_matrix[i, j] = float(cc) if np.isfinite(cc) else 0.0
                else:
                    corr_matrix[i, j] = 0.0
            else:
                corr_matrix[i, j] = 0.0 if i != j else 1.0

    # Off-diagonal mean
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append(corr_matrix[i, j])
    gamma2_corr_mean = float(np.mean(off_diag)) if off_diag else 0.0

    # Phase-jump overlap
    overlap_ratios = []
    for i in range(n):
        for j in range(i + 1, n):
            si = jump_freq_sets[speaker_keys[i]]
            sj = jump_freq_sets[speaker_keys[j]]
            if not si and not sj:
                continue
            # Quantize to nearest 10 Hz for overlap comparison
            si_q = {round(f / 10) * 10 for f in si}
            sj_q = {round(f / 10) * 10 for f in sj}
            union = si_q | sj_q
            intersection = si_q & sj_q
            if union:
                overlap_ratios.append(len(intersection) / len(union))
    jump_overlap_mean = float(np.mean(overlap_ratios)) if overlap_ratios else 0.0

    ck5 = {
        "checkpoint": 5,
        "name": "speaker_independence",
        "n_speakers": n,
        "speaker_keys": speaker_keys,
        "gamma2_corr_matrix": corr_matrix.tolist(),
        "gamma2_corr_mean": gamma2_corr_mean,
        "jump_overlap_mean": jump_overlap_mean,
        "wall_modal_prediction_corr": gamma2_corr_mean > 0.5,
        "wall_modal_prediction_overlap": jump_overlap_mean > 0.6,
        "pass": True,
    }
    return ck5


# ─────────────────────────────────────────────────────────────────────
# [G] Hypothesis discrimination
# ─────────────────────────────────────────────────────────────────────


def evaluate_hypotheses(agg: dict, ck5: dict) -> dict:
    """Score each hypothesis against the discrimination matrix."""
    ldv = agg.get("LDV_MicL", {})
    jump_count = ldv.get("jump_count_median", 0)
    gamma2_mid = ldv.get("gamma2_mid_median", 0)
    gamma2_ratio = ldv.get("gamma2_ratio_median", 1)
    R2 = ldv.get("R2_median", 0)
    frac_correct = ldv.get("frac_correct_median", 0)
    roughness = ldv.get("phase_roughness_median", 0)
    gamma2_corr = ck5.get("gamma2_corr_mean", 0)

    hypotheses = {}

    # Wall modal
    scores_wall = []
    scores_wall.append(("jump_count_30_60", 30 <= jump_count <= 60))
    scores_wall.append(("gamma2_mid_0.03_0.12", 0.03 <= gamma2_mid <= 0.12))
    scores_wall.append(("gamma2_ratio_gt2", gamma2_ratio > 2))
    scores_wall.append(("R2_lt0.2", R2 < 0.2))
    scores_wall.append(("frac_correct_10_30pct", 0.10 <= frac_correct <= 0.30))
    scores_wall.append(("speaker_corr_gt0.5", gamma2_corr > 0.5))
    scores_wall.append(("roughness_gt1.5", roughness > 1.5))
    hypotheses["wall_modal"] = {
        "criteria": {k: bool(v) for k, v in scores_wall},
        "score": sum(v for _, v in scores_wall),
        "total": len(scores_wall),
    }

    # Broadband noise
    scores_noise = []
    scores_noise.append(("gamma2_ratio_approx1", 0.5 <= gamma2_ratio <= 1.5))
    scores_noise.append(("R2_lt0.1", R2 < 0.1))
    scores_noise.append(("frac_correct_approx0", frac_correct < 0.05))
    scores_noise.append(("speaker_corr_approx0", abs(gamma2_corr) < 0.2))
    scores_noise.append(("roughness_approx1.8", roughness > 1.5))
    hypotheses["broadband_noise"] = {
        "criteria": {k: bool(v) for k, v in scores_noise},
        "score": sum(v for _, v in scores_noise),
        "total": len(scores_noise),
    }

    # Room modal
    scores_room = []
    scores_room.append(("jump_count_3_5", 3 <= jump_count <= 5))
    scores_room.append(("gamma2_mid_gt0.3", gamma2_mid > 0.3))
    scores_room.append(("R2_0.5_0.8", 0.5 <= R2 <= 0.8))
    scores_room.append(("frac_correct_80_95", 0.80 <= frac_correct <= 0.95))
    hypotheses["room_modal"] = {
        "criteria": {k: bool(v) for k, v in scores_room},
        "score": sum(v for _, v in scores_room),
        "total": len(scores_room),
    }

    # Clock drift
    scores_clock = []
    scores_clock.append(("jump_count_0", jump_count == 0))
    scores_clock.append(("gamma2_mid_gt0.5", gamma2_mid > 0.5))
    scores_clock.append(("R2_gt0.95", R2 > 0.95))
    scores_clock.append(("frac_correct_approx100", frac_correct > 0.95))
    hypotheses["clock_drift"] = {
        "criteria": {k: bool(v) for k, v in scores_clock},
        "score": sum(v for _, v in scores_clock),
        "total": len(scores_clock),
    }

    # Non-linear LDV
    scores_nlldv = []
    scores_nlldv.append(("jump_count_5_15", 5 <= jump_count <= 15))
    scores_nlldv.append(("gamma2_mid_0.1_0.3", 0.1 <= gamma2_mid <= 0.3))
    scores_nlldv.append(("R2_0.1_0.4", 0.1 <= R2 <= 0.4))
    scores_nlldv.append(("frac_correct_50_70", 0.50 <= frac_correct <= 0.70))
    hypotheses["nonlinear_ldv"] = {
        "criteria": {k: bool(v) for k, v in scores_nlldv},
        "score": sum(v for _, v in scores_nlldv),
        "total": len(scores_nlldv),
    }

    # Determine best match
    best = max(hypotheses, key=lambda h: hypotheses[h]["score"] / hypotheses[h]["total"])
    return {
        "hypotheses": hypotheses,
        "best_match": best,
        "best_score_frac": hypotheses[best]["score"] / hypotheses[best]["total"],
        "observed_values": {
            "jump_count": jump_count,
            "gamma2_mid": gamma2_mid,
            "gamma2_ratio": gamma2_ratio,
            "R2": R2,
            "frac_correct": frac_correct,
            "roughness": roughness,
            "speaker_gamma2_corr": gamma2_corr,
        },
    }


# ─────────────────────────────────────────────────────────────────────
# [H] Report generation
# ─────────────────────────────────────────────────────────────────────


def generate_report(
    all_results: dict, ck5: dict, verdict: dict, output_dir: str,
) -> str:
    """Generate human-readable markdown report."""
    lines = ["# Wall Panel Transfer Function Diagnostic Report", ""]
    lines.append(f"**Generated**: {datetime.now().isoformat()}")
    lines.append(f"**Speakers**: {', '.join(sorted(all_results.keys()))}")
    lines.append("")

    # Checkpoint summary
    lines.append("## Checkpoint Summary")
    lines.append("")
    lines.append("| CK | Name | Status |")
    lines.append("|----|------|--------|")

    # Aggregate checkpoint pass/fail across speakers
    for spk_key in sorted(all_results.keys()):
        r = all_results[spk_key]
        ck0_pass = r["checkpoint_0"]["pass"]
        segs = r["per_segment"]
        ck1_passes = sum(1 for s in segs if s["checkpoint_1"]["pass"])
        ck2_passes = sum(1 for s in segs if s["checkpoint_2"]["pass"])
        n = len(segs)
        lines.append(f"| 0 | Data sanity ({spk_key}) | {'PASS' if ck0_pass else 'FAIL'} |")
        lines.append(f"| 1 | MicL-MicR control ({spk_key}) | {ck1_passes}/{n} PASS |")
        lines.append(f"| 2 | LDV quality ({spk_key}) | {ck2_passes}/{n} PASS |")

    # Checkpoint 3 & 4 from first speaker aggregated
    first_spk = sorted(all_results.keys())[0]
    agg = all_results[first_spk]["aggregated"]
    ck3 = agg["checkpoint_3"]
    ck4 = agg["checkpoint_4"]
    lines.append(f"| 3 | γ² freq profile | ratio={ck3['gamma2_ratio_low_mid']:.2f} ({'supports wall modal' if ck3['wall_modal_prediction'] else 'does NOT support wall modal'}) |")
    lines.append(f"| 4 | Phase behaviour | jumps={ck4['ldv_micl_jump_count']:.0f}, roughness_ratio={ck4['roughness_ratio_ldv_vs_mic']:.1f}, R²={ck4['ldv_micl_R2']:.3f} |")
    lines.append(f"| 5 | Speaker independence | γ² corr={ck5.get('gamma2_corr_mean', 0):.3f}, jump overlap={ck5.get('jump_overlap_mean', 0):.3f} |")
    lines.append("")

    # Per-speaker coherence & phase summary
    lines.append("## Per-Speaker Diagnostic Summary")
    lines.append("")
    lines.append("| Speaker | Pair | γ²_mid | γ²_ratio | R² | Jumps | Roughness | frac_correct |")
    lines.append("|---------|------|--------|----------|-----|-------|-----------|-------------|")
    for spk_key in sorted(all_results.keys()):
        agg = all_results[spk_key]["aggregated"]
        for pair in ["LDV_MicL", "LDV_MicR", "MicL_MicR"]:
            p = agg[pair]
            lines.append(
                f"| {spk_key} | {pair} | {p['gamma2_mid_median']:.4f} | "
                f"{p['gamma2_ratio_median']:.2f} | {p['R2_median']:.3f} | "
                f"{p['jump_count_median']:.0f} | {p['phase_roughness_median']:.3f} | "
                f"{p['frac_correct_median']:.3f} |"
            )
    lines.append("")

    # Hypothesis verdict
    lines.append("## Hypothesis Verdict")
    lines.append("")
    lines.append(f"**Best match**: `{verdict['best_match']}` "
                 f"(score: {verdict['best_score_frac']:.0%})")
    lines.append("")
    lines.append("| Hypothesis | Score | Criteria met |")
    lines.append("|------------|-------|-------------|")
    for h_name, h_data in verdict["hypotheses"].items():
        met = [k for k, v in h_data["criteria"].items() if v]
        lines.append(f"| {h_name} | {h_data['score']}/{h_data['total']} | {', '.join(met)} |")
    lines.append("")

    lines.append("### Observed values vs. predictions")
    lines.append("")
    obs = verdict["observed_values"]
    lines.append(f"- Phase jumps [500-2000 Hz]: **{obs['jump_count']:.0f}** (wall modal predicts 30-60)")
    lines.append(f"- γ² median [500-2000 Hz]: **{obs['gamma2_mid']:.4f}** (wall modal predicts 0.03-0.12)")
    lines.append(f"- γ² ratio (low/mid): **{obs['gamma2_ratio']:.2f}** (wall modal predicts >2)")
    lines.append(f"- Phase R² [500-2000 Hz]: **{obs['R2']:.3f}** (wall modal predicts <0.2)")
    lines.append(f"- Narrowband correct τ₂ fraction: **{obs['frac_correct']:.1%}** (wall modal predicts 10-30%)")
    lines.append(f"- Phase roughness: **{obs['roughness']:.3f}** rad (wall modal predicts >1.5)")
    lines.append(f"- Speaker γ² correlation: **{obs['speaker_gamma2_corr']:.3f}** (wall modal predicts >0.5)")
    lines.append("")

    # Discrimination summary
    distinct = []
    if obs["gamma2_ratio"] > 2:
        distinct.append("γ² ratio > 2 (frequency-selective, not uniform noise)")
    if obs["speaker_gamma2_corr"] > 0.5:
        distinct.append("Speaker γ² correlation > 0.5 (structural, not random)")
    if 0.10 <= obs["frac_correct"] <= 0.30:
        distinct.append("Narrowband correct τ₂ in 10-30% bands (clustered, not total failure)")
    if distinct:
        lines.append("### Wall-modal distinctive signatures found:")
        for d in distinct:
            lines.append(f"- {d}")
    else:
        lines.append("### No wall-modal distinctive signatures found.")
    lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# [I] Plotting
# ─────────────────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def generate_plots(all_results: dict, ck5: dict, output_dir: str) -> None:
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plots")
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    speaker_keys = sorted(all_results.keys())

    # --- 1. Coherence 3 pairs ---
    fig, axes = plt.subplots(len(speaker_keys), 1, figsize=(12, 3 * len(speaker_keys)),
                             squeeze=False)
    for idx, spk in enumerate(speaker_keys):
        ax = axes[idx, 0]
        seg0 = all_results[spk]["per_segment"][0]  # representative segment
        for pair_label, color in [("MicL_MicR", "green"), ("LDV_MicL", "blue"), ("LDV_MicR", "red")]:
            freqs = np.array(seg0["coherence"][pair_label]["gamma2_freqs"])
            g2 = np.array(seg0["coherence"][pair_label]["gamma2_values"])
            mask = freqs <= 4000
            ax.semilogy(freqs[mask], g2[mask] + 1e-4, label=pair_label, color=color, alpha=0.8, linewidth=0.7)
        ax.axhline(1.0 / 57, color="gray", ls="--", lw=0.8, label="noise floor")
        ax.axvspan(ANALYSIS_BAND[0], ANALYSIS_BAND[1], alpha=0.1, color="yellow")
        ax.set_ylabel("γ²(f)")
        ax.set_title(f"Speaker {spk}")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 4000)
        ax.set_ylim(1e-4, 1.1)
    axes[-1, 0].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "coherence_3pairs.png"), dpi=150)
    plt.close(fig)

    # --- 2. Phase LDV-MicL (unwrapped) ---
    fig, axes = plt.subplots(len(speaker_keys), 1, figsize=(12, 3 * len(speaker_keys)),
                             squeeze=False)
    for idx, spk in enumerate(speaker_keys):
        ax = axes[idx, 0]
        for seg_idx, seg in enumerate(all_results[spk]["per_segment"]):
            freqs = np.array(seg["phase"]["LDV_MicL"]["freqs"])
            phi_u = np.array(seg["phase"]["LDV_MicL"]["phi_unwrapped"])
            mask = (freqs >= 100) & (freqs <= 3000)
            ax.plot(freqs[mask], phi_u[mask], alpha=0.6, linewidth=0.6,
                    label=f"seg {seg_idx}" if seg_idx < 3 else None)
        ax.axvspan(ANALYSIS_BAND[0], ANALYSIS_BAND[1], alpha=0.1, color="yellow")
        ax.set_ylabel("φ (rad)")
        ax.set_title(f"LDV-MicL unwrapped phase — Speaker {spk}")
        if idx == 0:
            ax.legend(fontsize=7)
    axes[-1, 0].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "phase_ldv_micl.png"), dpi=150)
    plt.close(fig)

    # --- 3. Narrowband sweep heatmap ---
    fig, axes = plt.subplots(len(speaker_keys), 2, figsize=(14, 3 * len(speaker_keys)),
                             squeeze=False)
    for idx, spk in enumerate(speaker_keys):
        seg0 = all_results[spk]["per_segment"][0]
        for col, pair_label in enumerate(["LDV_MicL", "MicL_MicR"]):
            ax = axes[idx, col]
            bands = seg0["narrowband_sweep"][pair_label]["bands"]
            fc = [b["f_center"] for b in bands]
            taus = [b["tau_ms"] if b["tau_ms"] is not None else 0 for b in bands]
            psrs = [b["psr_db"] if b["psr_db"] is not None else -30 for b in bands]
            correct = [b["is_correct"] for b in bands]

            colors = ["green" if c else "red" for c in correct]
            ax.bar(fc, taus, width=SWEEP_STEP * 0.8, color=colors, alpha=0.7)
            gt = all_results[spk]["ground_truths"]
            tau_exp = gt["tau2_true_ms"] if pair_label == "LDV_MicL" else gt["tau1_true_ms"]
            ax.axhline(tau_exp, color="blue", ls="--", lw=1, label=f"expected τ={tau_exp:.2f}ms")
            ax.set_ylabel("τ (ms)")
            ax.set_title(f"{pair_label} — {spk}")
            ax.legend(fontsize=7)
    axes[-1, 0].set_xlabel("Center freq (Hz)")
    axes[-1, 1].set_xlabel("Center freq (Hz)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "narrowband_sweep.png"), dpi=150)
    plt.close(fig)

    # --- 4. Speaker independence ---
    if not ck5.get("skip", False) and "gamma2_corr_matrix" in ck5:
        fig, ax = plt.subplots(figsize=(6, 5))
        mat = np.array(ck5["gamma2_corr_matrix"])
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(speaker_keys)))
        ax.set_xticklabels(speaker_keys, fontsize=8)
        ax.set_yticks(range(len(speaker_keys)))
        ax.set_yticklabels(speaker_keys, fontsize=8)
        for i in range(len(speaker_keys)):
            for j in range(len(speaker_keys)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="γ² profile correlation")
        ax.set_title(f"Speaker independence (mean off-diag={ck5['gamma2_corr_mean']:.3f})")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "speaker_independence.png"), dpi=150)
        plt.close(fig)

    # --- 5. Checkpoint summary ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ck_labels = []
    ck_status = []
    for spk in speaker_keys:
        r = all_results[spk]
        ck_labels.append(f"CK0 {spk}")
        ck_status.append(1 if r["checkpoint_0"]["pass"] else 0)
        segs = r["per_segment"]
        ck1_frac = sum(1 for s in segs if s["checkpoint_1"]["pass"]) / max(1, len(segs))
        ck_labels.append(f"CK1 {spk}")
        ck_status.append(ck1_frac)
        ck2_frac = sum(1 for s in segs if s["checkpoint_2"]["pass"]) / max(1, len(segs))
        ck_labels.append(f"CK2 {spk}")
        ck_status.append(ck2_frac)
    colors = ["green" if s >= 0.8 else ("orange" if s >= 0.5 else "red") for s in ck_status]
    ax.barh(range(len(ck_labels)), ck_status, color=colors, alpha=0.8)
    ax.set_yticks(range(len(ck_labels)))
    ax.set_yticklabels(ck_labels, fontsize=7)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Pass rate")
    ax.set_title("Checkpoint Summary")
    ax.axvline(1.0, color="gray", ls="--", lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "checkpoint_summary.png"), dpi=150)
    plt.close(fig)

    logger.info("Plots saved to %s", plots_dir)


def write_posthoc_validation(all_results: dict, *, output_dir: str, run_config: dict) -> None:
    """
    Write a lightweight post-hoc validation memo.

    Purpose:
    - Confirm the narrowband sweep is numerically sane (no NaNs / no boundary pinning dominance).
    - Quantify per-pair τ(f) correlation (to rule out pair-independent artifacts).
    - Summarize wideband GCC-PHAT τ (500–2000 Hz) to diagnose constant offsets / stable paths.
    """
    pairs = ["LDV_MicL", "LDV_MicR", "MicL_MicR"]

    total = 0
    n_nan_psr = 0
    n_boundary = 0
    n_near_zero = 0
    psr_all: list[float] = []

    # Correlations of τ(f) across pairs, pooled over all speaker/segment instances.
    corr_pairs = {
        ("LDV_MicL", "MicL_MicR"): [],
        ("LDV_MicL", "LDV_MicR"): [],
        ("LDV_MicR", "MicL_MicR"): [],
    }

    # Wideband GCC τ summary (ms).
    wideband_tau: dict[str, list[float]] = {p: [] for p in pairs}

    for spk_key, spk in all_results.items():
        _ = spk_key
        for seg in spk.get("per_segment", []):
            # Wideband
            wb = seg.get("wideband_gcc", {})
            for p in pairs:
                v = wb.get(p, {})
                if isinstance(v, dict) and np.isfinite(v.get("tau_ms", np.nan)):
                    wideband_tau[p].append(float(v["tau_ms"]))

            # Narrowband stats + τ(f) vectors
            tau_vecs: dict[str, np.ndarray] = {}
            for p in pairs:
                bands = seg["narrowband_sweep"][p]["bands"]
                taus = []
                psrs = []
                for b in bands:
                    tau_ms = b.get("tau_ms", None)
                    psr_db = b.get("psr_db", np.nan)
                    if psr_db is None:
                        psr_db = np.nan
                    if not np.isfinite(psr_db):
                        n_nan_psr += 1
                    else:
                        psr_all.append(float(psr_db))
                    if tau_ms is None or not np.isfinite(tau_ms):
                        taus.append(np.nan)
                    else:
                        tau_ms_f = float(tau_ms)
                        taus.append(tau_ms_f)
                        if abs(abs(tau_ms_f) - float(MAX_LAG_MS)) < 1e-9:
                            n_boundary += 1
                        if abs(tau_ms_f) < 0.3:
                            n_near_zero += 1
                    psrs.append(psr_db)
                    total += 1
                tau_vecs[p] = np.asarray(taus, dtype=np.float64)

            # Pairwise τ(f) correlations (ignore NaNs)
            for (a, b), store in corr_pairs.items():
                ta = tau_vecs[a]
                tb = tau_vecs[b]
                m = np.isfinite(ta) & np.isfinite(tb)
                if int(np.sum(m)) < 3:
                    continue
                ca = ta[m]
                cb = tb[m]
                if float(np.std(ca)) < 1e-12 or float(np.std(cb)) < 1e-12:
                    continue
                r = float(np.corrcoef(ca, cb)[0, 1])
                if np.isfinite(r):
                    store.append(r)

    def _median(xs: list[float]) -> float:
        if not xs:
            return float("nan")
        return float(np.median(np.asarray(xs, dtype=np.float64)))

    posthoc = {
        "generated": datetime.now().isoformat(),
        "counts": {
            "total_band_estimates": int(total),
            "psr_nan": int(n_nan_psr),
            "tau_boundary_pinned": int(n_boundary),
            "tau_near_zero_lt_0p3ms": int(n_near_zero),
        },
        "psr_db_median": _median(psr_all),
        "tau_corr_median": {f"{a}__vs__{b}": _median(v) for (a, b), v in corr_pairs.items()},
        "wideband_tau_ms_median": {p: _median(v) for p, v in wideband_tau.items()},
    }
    _write_json(os.path.join(output_dir, "posthoc_summary.json"), posthoc)

    # Human-readable memo
    lines = []
    lines.append("# Post-hoc validation (auto-generated)")
    lines.append("")
    lines.append(f"**Run dir**: `{output_dir}/`  ")
    lines.append(f"**Generated**: {posthoc['generated']}  ")
    lines.append("**Purpose**: Sanity-check narrowband sweep outputs and summarize wideband GCC behavior.")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("python -u scripts/diagnose_wall_transfer_function.py \\")
    lines.append(f"  --data_root {os.path.abspath(run_config.get('data_root', ''))} \\")
    lines.append("  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V \\")
    lines.append(f"  --segment_source {os.path.abspath(run_config.get('segment_source', ''))} \\")
    lines.append(
        f"  --output_dir {os.path.abspath(run_config.get('output_dir_prefix', 'results/wall_tf_diagnostic_v2'))}"
    )
    lines.append("```")
    lines.append("")
    lines.append("## 1) Narrowband sweep sanity")
    lines.append("")
    c = posthoc["counts"]
    lines.append(f"- Total band estimates: **{c['total_band_estimates']}**")
    lines.append(f"- `psr_db` NaN: **{c['psr_nan']}**")
    lines.append(f"- `|tau_ms| == MAX_LAG_MS ({MAX_LAG_MS} ms)`: **{c['tau_boundary_pinned']}**")
    lines.append(f"- `|tau_ms| < 0.3 ms`: **{c['tau_near_zero_lt_0p3ms']}**")
    lines.append("")
    lines.append(f"Median narrowband PSR: **{posthoc['psr_db_median']:.3f} dB**")
    lines.append("")
    lines.append("## 2) Pair-to-pair τ(f) correlation (median)")
    lines.append("")
    for k, v in posthoc["tau_corr_median"].items():
        lines.append(f"- `{k}`: **{v:.3f}**")
    lines.append("")
    lines.append("## 3) Wideband GCC-PHAT τ (500–2000 Hz) median")
    lines.append("")
    for p, v in posthoc["wideband_tau_ms_median"].items():
        lines.append(f"- `{p}`: **{v:+.3f} ms**")
    lines.append("")
    lines.append("## Note")
    lines.append("")
    lines.append(
        "Per-band τ(f) estimates are typically low-confidence when PSR is near 0 dB. "
        "Interpret τ physically only after accounting for constant per-channel offsets."
    )
    lines.append("")

    with open(os.path.join(output_dir, "posthoc_validation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────
# [J] Main orchestration
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wall panel transfer function diagnostic"
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of WAV data")
    parser.add_argument("--speakers", type=str, nargs="+", required=True,
                        help="Speaker folder names (e.g., 18-0.1V 19-0.1V)")
    parser.add_argument("--segment_source", type=str, required=True,
                        help="Path to stage4 results dir with summary.json per speaker")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (timestamp appended automatically)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger.info("=" * 70)
    logger.info("Wall Panel Transfer Function Diagnostic")
    logger.info("=" * 70)
    logger.info("Output: %s", output_dir)

    # Reproducibility snapshot (written before running diagnostics)
    script_path = os.path.abspath(__file__)
    git_head, git_dirty = _git_head_and_dirty()
    code_state = {
        "script_path": script_path,
        "script_sha256": _sha256_file(script_path),
        "git_head": git_head,
        "dirty": git_dirty,
        "timestamp": datetime.now().isoformat(),
    }
    _write_json(os.path.join(output_dir, "code_state.json"), code_state)

    run_config = {
        "timestamp": datetime.now().isoformat(),
        "argv": sys.argv,
        "data_root": os.path.abspath(args.data_root),
        "segment_source": os.path.abspath(args.segment_source),
        "speakers": list(args.speakers),
        "output_dir_prefix": os.path.abspath(args.output_dir),
        "output_dir": output_dir,
        "constants": {
            "fs_expected": int(FS),
            "window_sec": 5.0,
            "band_low_hz": float(ANALYSIS_BAND[0]),
            "band_high_hz": float(ANALYSIS_BAND[1]),
            "welch_nperseg": int(NPERSEG),
            "welch_noverlap": int(NOVERLAP),
            "narrowband_step_hz": float(SWEEP_STEP),
            "narrowband_bw_hz": float(SWEEP_BW),
            "gcc_max_lag_ms": float(MAX_LAG_MS),
        },
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    _write_json(os.path.join(output_dir, "run_config.json"), run_config)

    all_speaker_results = {}
    subset_manifest: dict[str, dict] = {}
    fingerprint_entries: list[tuple[str, str]] = []

    for speaker_folder in args.speakers:
        speaker_key = speaker_folder.split("-")[0]
        logger.info("=" * 50)
        logger.info("Processing speaker %s (key=%s)", speaker_folder, speaker_key)

        # Load segment centers from stage4 summary
        summary_path = os.path.join(args.segment_source, speaker_folder, "summary.json")
        if not os.path.exists(summary_path):
            logger.warning("Summary not found: %s — skipping", summary_path)
            continue
        with open(summary_path) as f:
            stage4 = json.load(f)
        segment_centers = stage4.get("segment_centers_sec", [])
        if not segment_centers:
            # fallback to scan_summary
            segment_centers = stage4.get("scan_summary", {}).get("selected_centers_sec", [])
        if not segment_centers:
            logger.warning("No segment centers found for %s — skipping", speaker_folder)
            continue
        logger.info("  Using %d segments: %s", len(segment_centers), segment_centers)

        data_dir = os.path.join(args.data_root, speaker_folder)
        spk_output = os.path.join(output_dir, "per_speaker", speaker_folder)

        # Record exact input files (first match policy must match diagnose_speaker()).
        speaker_dir = Path(data_dir)
        ldv_files = sorted(speaker_dir.glob("*LDV*.wav"))
        left_files = sorted(speaker_dir.glob("*LEFT*.wav"))
        right_files = sorted(speaker_dir.glob("*RIGHT*.wav"))
        if not ldv_files or not left_files or not right_files:
            logger.warning("Missing WAV files in %s — skipping", speaker_dir)
            continue
        ldv_path = str(ldv_files[0].resolve())
        micl_path = str(left_files[0].resolve())
        micr_path = str(right_files[0].resolve())

        subset_manifest[speaker_folder] = {
            "speaker_key": speaker_key,
            "summary_path": os.path.abspath(summary_path),
            "segment_centers_sec": segment_centers,
            "wav": {"ldv": ldv_path, "mic_left": micl_path, "mic_right": micr_path},
        }

        # Fingerprint inputs (sha256 over the exact subset of files).
        for p in [summary_path, ldv_path, micl_path, micr_path]:
            ap = os.path.abspath(p)
            try:
                sha = _sha256_file(ap)
            except Exception as e:
                raise RuntimeError(f"Failed to sha256 input file: {ap}") from e
            rel = os.path.relpath(ap, start=os.getcwd())
            fingerprint_entries.append((sha, rel))

        spk_result = diagnose_speaker(data_dir, speaker_key, segment_centers, spk_output)
        all_speaker_results[speaker_key] = spk_result

    if not all_speaker_results:
        logger.error("No speakers processed successfully.")
        return

    # Checkpoint 5: speaker independence
    logger.info("Computing speaker independence (Checkpoint 5) ...")
    ck5 = compute_speaker_independence(all_speaker_results)
    logger.info("  γ² corr mean=%.3f, jump overlap=%.3f",
                ck5.get("gamma2_corr_mean", 0), ck5.get("jump_overlap_mean", 0))

    # Hypothesis verdict (use median across speakers)
    # Aggregate across all speakers
    agg_all = {}
    for pair in ["LDV_MicL", "LDV_MicR", "MicL_MicR"]:
        metrics = {}
        for metric_key in ["gamma2_mid_median", "gamma2_low_median", "gamma2_ratio_median",
                           "R2_median", "jump_count_median", "phase_roughness_median",
                           "frac_correct_median", "frac_zero_peak_median", "tau_fit_ms_median"]:
            vals = [all_speaker_results[spk]["aggregated"][pair][metric_key]
                    for spk in all_speaker_results]
            metrics[metric_key] = float(np.median(vals))
        agg_all[pair] = metrics

    verdict = evaluate_hypotheses(agg_all, ck5)
    logger.info("Hypothesis verdict: best_match=%s (%.0f%%)",
                verdict["best_match"], verdict["best_score_frac"] * 100)

    # Generate report
    report_md = generate_report(all_speaker_results, ck5, verdict, output_dir)
    report_path = os.path.join(output_dir, "diagnostic_report.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    logger.info("Report saved: %s", report_path)

    # Save structured outputs
    # Strip large arrays from summary to keep file manageable
    summary_slim = {}
    for spk_key, spk_data in all_speaker_results.items():
        summary_slim[spk_key] = {
            "speaker_key": spk_data["speaker_key"],
            "ground_truths": spk_data["ground_truths"],
            "n_segments": spk_data["n_segments"],
            "segment_centers_sec": spk_data["segment_centers_sec"],
            "checkpoint_0": spk_data["checkpoint_0"],
            "aggregated": spk_data["aggregated"],
        }

    diagnostic_summary = {
        "timestamp": datetime.now().isoformat(),
        "output_dir": output_dir,
        "speakers": summary_slim,
        "checkpoint_5": ck5,
    }
    with open(os.path.join(output_dir, "diagnostic_summary.json"), "w") as f:
        json.dump(diagnostic_summary, f, indent=2)

    with open(os.path.join(output_dir, "hypothesis_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)

    subset_manifest_payload = {
        "timestamp": datetime.now().isoformat(),
        "data_root": os.path.abspath(args.data_root),
        "segment_source": os.path.abspath(args.segment_source),
        "speakers": subset_manifest,
        "fingerprint_sha256": _dataset_fingerprint_sha256(fingerprint_entries),
        "files": [{"sha256": sha, "path": rel} for sha, rel in sorted(fingerprint_entries, key=lambda t: t[1].lower())],
    }
    _write_json(os.path.join(output_dir, "subset_manifest.json"), subset_manifest_payload)

    write_posthoc_validation(all_speaker_results, output_dir=output_dir, run_config=run_config)

    # Generate plots
    generate_plots(all_speaker_results, ck5, output_dir)

    logger.info("=" * 70)
    logger.info("DONE. All results in: %s", output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
