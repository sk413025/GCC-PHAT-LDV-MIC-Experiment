#!/usr/bin/env python3
"""
Extended TDoA Methods: GCC-SCOT, Subband Analysis, and POC

This script implements additional TDoA estimation methods for comparison:
1. GCC-SCOT (Smoothed Coherence Transform)
2. Subband Analysis (per-band GCC-PHAT with robust aggregation)
3. POC (Phase-Only Correlation)

These methods are compared with/without DTmin frequency compensation.

Usage:
    python run_extended_tdoa_methods.py \
        --mic_root "path/to/MIC" \
        --ldv_root "path/to/LDV" \
        --out_dir "results/extended_methods_xxx" \
        --mode scale

Author: exp-tdoa-cross-correlation
Date: 2026-01-28
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from scipy import signal
from scipy.io import wavfile

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Utility Functions (reused from run_dtmin_comparison.py)
# =============================================================================

def next_pow2(x: int) -> int:
    """Return the smallest power of 2 >= x."""
    return 1 << (int(x) - 1).bit_length()


def bandpass_filter(fs: int, freq_min: float, freq_max: float, order: int = 4):
    """Design a Butterworth bandpass filter."""
    nyq = fs / 2.0
    low = max(freq_min / nyq, 0.01)
    high = min(freq_max / nyq, 0.99)
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def load_wav(path: str, target_fs: int = 16000) -> np.ndarray:
    """Load WAV file and resample if necessary."""
    fs, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float64:
        data = data.astype(np.float32)

    if fs != target_fs:
        num_samples = int(len(data) * target_fs / fs)
        data = signal.resample(data, num_samples)

    return data


def discover_pairs(mic_root: str, ldv_root: str) -> List[Tuple[str, str]]:
    """Discover MIC-LDV pairs from directories."""
    mic_path = Path(mic_root)
    ldv_path = Path(ldv_root)

    pairs = []
    for mic_file in sorted(mic_path.glob("*.wav")):
        ldv_name = mic_file.name.replace("_MIC_", "_LDV_")
        ldv_file = ldv_path / ldv_name

        if ldv_file.exists():
            pairs.append((str(mic_file), str(ldv_file)))

    return pairs


# =============================================================================
# DTmin: Per-Frequency Delay Estimation and Compensation
# =============================================================================

def estimate_per_frequency_delay(
    X_mic: np.ndarray,
    X_ldv: np.ndarray,
    freqs: np.ndarray,
    fs: int,
    freq_min: float = 300.0,
    freq_max: float = 3000.0,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-frequency delay tau(f) using phase difference.
    """
    n_freq, n_frames = X_mic.shape
    tau_f = np.zeros(n_freq)
    confidence = np.zeros(n_freq)

    for f_idx, f in enumerate(freqs):
        if f < freq_min or f > freq_max or f < 1:
            tau_f[f_idx] = 0.0
            confidence[f_idx] = 0.0
            continue

        x_f = X_mic[f_idx, :]
        y_f = X_ldv[f_idx, :]

        cross = np.conj(x_f) * y_f
        cross_phat = cross / (np.abs(cross) + eps)
        phase_diff = np.angle(cross_phat)

        weights = np.abs(cross) + eps
        mean_phase = np.sum(phase_diff * weights) / np.sum(weights)

        if f > 1:
            tau_f[f_idx] = -mean_phase / (2 * np.pi * f)
        else:
            tau_f[f_idx] = 0.0

        coherence = np.abs(np.mean(cross_phat))
        confidence[f_idx] = coherence

    return tau_f, confidence


def apply_dtmin_compensation(
    X_ldv: np.ndarray,
    tau_f: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """Apply DTmin phase compensation to LDV STFT."""
    phase_comp = np.exp(1j * 2 * np.pi * freqs * tau_f)
    X_ldv_comp = X_ldv * phase_comp[:, np.newaxis]
    return X_ldv_comp


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class CCResult:
    """Result of cross-correlation delay estimation."""
    method: str
    tau_samples: Optional[float]
    tau_ms: Optional[float]
    peak_value: Optional[float]
    psr: Optional[float]
    boundary_hit: Optional[bool]
    undefined_reason: Optional[str] = None


@dataclass
class SubbandResult:
    """Result of subband analysis."""
    method: str
    tau_samples: Optional[float]
    tau_ms: Optional[float]
    peak_value: Optional[float]
    psr: Optional[float]
    boundary_hit: Optional[bool]
    # Subband-specific metrics
    num_subbands: Optional[int] = None
    subband_taus_ms: Optional[List[float]] = None
    subband_consistency: Optional[float] = None  # IQR / median
    aggregation_method: Optional[str] = None
    undefined_reason: Optional[str] = None


@dataclass
class MUSICResult:
    """Result of MUSIC DoA estimation."""
    method: str
    doa_deg: Optional[float]
    tau_ms: Optional[float]
    snr_eigen: Optional[float]
    peak_value: Optional[float]
    spectrum_psr: Optional[float]
    undefined_reason: Optional[str] = None


# =============================================================================
# GCC-SCOT Implementation
# =============================================================================

def compute_gcc_scot(
    X_mic: np.ndarray,
    X_ldv: np.ndarray,
    freqs: np.ndarray,
    fs: int,
    hop_length: int = 256,
    search_radius_ms: float = 10.0,
    exclusion_radius_samples: int = 16,
    eps: float = 1e-12,
) -> CCResult:
    """
    GCC-SCOT (Smoothed Coherence Transform).

    Weighting: W(f) = 1 / sqrt(S_xx(f) * S_yy(f))

    This is a middle ground between CC (amplitude weighted) and PHAT (equal weighted).
    SCOT uses the geometric mean of auto-spectra as normalization.

    Args:
        X_mic, X_ldv: STFT arrays (n_freq, n_frames)
        freqs: Frequency bins
        fs: Sampling rate
        hop_length: STFT hop length
        search_radius_ms: Search window in ms
        exclusion_radius_samples: PSR exclusion zone
        eps: Small value for numerical stability

    Returns:
        CCResult
    """
    n_freq, n_frames = X_mic.shape

    # Compute auto-spectra (averaged across frames)
    S_xx = np.mean(np.abs(X_mic) ** 2, axis=1)  # (n_freq,)
    S_yy = np.mean(np.abs(X_ldv) ** 2, axis=1)  # (n_freq,)

    # Cross-spectrum
    cross = np.conj(X_mic) * X_ldv  # (n_freq, n_frames)
    avg_cross = np.mean(cross, axis=1)  # (n_freq,)

    # SCOT weighting: normalize by sqrt(S_xx * S_yy)
    scot_weight = 1.0 / (np.sqrt(S_xx * S_yy) + eps)
    weighted_cross = avg_cross * scot_weight

    # IFFT to get cross-correlation
    nfft = 2 * n_freq
    full_cross = np.zeros(nfft, dtype=np.complex128)
    full_cross[:n_freq] = weighted_cross
    full_cross[n_freq:] = np.conj(weighted_cross[::-1])

    cc = np.fft.ifft(full_cross).real
    cc = np.fft.fftshift(cc)
    lags = np.arange(-nfft // 2, nfft // 2)

    # Search window
    search_radius_samples = int(search_radius_ms * fs / 1000)
    search_radius_bins = max(search_radius_samples // hop_length, nfft // 4)

    center = nfft // 2
    lo = max(0, center - search_radius_bins)
    hi = min(nfft, center + search_radius_bins)

    cc_sel = cc[lo:hi]
    lags_sel = lags[lo:hi]

    if len(cc_sel) == 0:
        return CCResult("gcc_scot", None, None, None, None, None, "no_valid_lags")

    # Find peak
    abs_cc = np.abs(cc_sel)
    peak_i = int(np.argmax(abs_cc))
    peak_lag = int(lags_sel[peak_i])
    peak_value = float(abs_cc[peak_i])

    tau_samples = float(peak_lag * hop_length)
    tau_ms = tau_samples * 1000.0 / fs

    # PSR
    exc = int(max(1, exclusion_radius_samples // hop_length))
    mask = np.ones(len(abs_cc), dtype=bool)
    mask[max(0, peak_i - exc):min(len(abs_cc), peak_i + exc + 1)] = False
    sidelobe = abs_cc[mask]
    psr = None
    if len(sidelobe) > 0:
        median_sidelobe = float(np.median(sidelobe))
        if median_sidelobe > eps:
            psr = float(peak_value / median_sidelobe)

    boundary_hit = bool(peak_i == 0 or peak_i == len(cc_sel) - 1)

    return CCResult("gcc_scot", tau_samples, tau_ms, peak_value, psr, boundary_hit)


# =============================================================================
# POC (Phase-Only Correlation) Implementation
# =============================================================================

def compute_poc(
    X_mic: np.ndarray,
    X_ldv: np.ndarray,
    freqs: np.ndarray,
    fs: int,
    hop_length: int = 256,
    search_radius_ms: float = 10.0,
    exclusion_radius_samples: int = 16,
    eps: float = 1e-12,
) -> CCResult:
    """
    POC (Phase-Only Correlation).

    Unlike GCC-PHAT which normalizes the cross-spectrum,
    POC works purely with phase information:

    POC = IFFT{ exp(j * (angle(X) - angle(Y))) }

    This is mathematically equivalent to:
    POC = IFFT{ (X/|X|) * conj(Y/|Y|) }

    The key difference from GCC-PHAT:
    - GCC-PHAT: normalizes cross-spectrum |X*Y|
    - POC: normalizes each spectrum individually |X| and |Y|

    Args:
        X_mic, X_ldv: STFT arrays (n_freq, n_frames)
        freqs: Frequency bins
        fs: Sampling rate
        hop_length: STFT hop length
        search_radius_ms: Search window in ms
        exclusion_radius_samples: PSR exclusion zone
        eps: Small value for numerical stability

    Returns:
        CCResult
    """
    n_freq, n_frames = X_mic.shape

    # Average across frames first (or use instantaneous - let's average for stability)
    X_mic_avg = np.mean(X_mic, axis=1)  # (n_freq,)
    X_ldv_avg = np.mean(X_ldv, axis=1)  # (n_freq,)

    # Normalize each spectrum individually (phase-only)
    X_mic_phase = X_mic_avg / (np.abs(X_mic_avg) + eps)  # unit magnitude
    X_ldv_phase = X_ldv_avg / (np.abs(X_ldv_avg) + eps)  # unit magnitude

    # POC cross-spectrum: just the phase difference
    poc_cross = X_mic_phase * np.conj(X_ldv_phase)
    # This equals: exp(j * (angle(X_mic) - angle(X_ldv)))

    # IFFT to get POC
    nfft = 2 * n_freq
    full_cross = np.zeros(nfft, dtype=np.complex128)
    full_cross[:n_freq] = poc_cross
    full_cross[n_freq:] = np.conj(poc_cross[::-1])

    cc = np.fft.ifft(full_cross).real
    cc = np.fft.fftshift(cc)
    lags = np.arange(-nfft // 2, nfft // 2)

    # Search window
    search_radius_samples = int(search_radius_ms * fs / 1000)
    search_radius_bins = max(search_radius_samples // hop_length, nfft // 4)

    center = nfft // 2
    lo = max(0, center - search_radius_bins)
    hi = min(nfft, center + search_radius_bins)

    cc_sel = cc[lo:hi]
    lags_sel = lags[lo:hi]

    if len(cc_sel) == 0:
        return CCResult("poc", None, None, None, None, None, "no_valid_lags")

    # Find peak
    abs_cc = np.abs(cc_sel)
    peak_i = int(np.argmax(abs_cc))
    peak_lag = int(lags_sel[peak_i])
    peak_value = float(abs_cc[peak_i])

    tau_samples = float(peak_lag * hop_length)
    tau_ms = tau_samples * 1000.0 / fs

    # PSR
    exc = int(max(1, exclusion_radius_samples // hop_length))
    mask = np.ones(len(abs_cc), dtype=bool)
    mask[max(0, peak_i - exc):min(len(abs_cc), peak_i + exc + 1)] = False
    sidelobe = abs_cc[mask]
    psr = None
    if len(sidelobe) > 0:
        median_sidelobe = float(np.median(sidelobe))
        if median_sidelobe > eps:
            psr = float(peak_value / median_sidelobe)

    boundary_hit = bool(peak_i == 0 or peak_i == len(cc_sel) - 1)

    return CCResult("poc", tau_samples, tau_ms, peak_value, psr, boundary_hit)


# =============================================================================
# Subband Analysis Implementation
# =============================================================================

def compute_subband_analysis(
    X_mic: np.ndarray,
    X_ldv: np.ndarray,
    freqs: np.ndarray,
    fs: int,
    hop_length: int = 256,
    num_subbands: int = 8,
    freq_min: float = 300.0,
    freq_max: float = 3000.0,
    search_radius_ms: float = 10.0,
    exclusion_radius_samples: int = 16,
    aggregation: str = "median",  # "median", "mode", "weighted_mean"
    eps: float = 1e-12,
) -> SubbandResult:
    """
    Subband Analysis for TDoA estimation.

    Splits the frequency band into N subbands, performs GCC-PHAT on each,
    then aggregates using a robust estimator (median/mode).

    This approach:
    1. Quantifies frequency dispersion (subband_consistency)
    2. Uses robust aggregation to handle outliers
    3. Provides per-band estimates for analysis

    Args:
        X_mic, X_ldv: STFT arrays (n_freq, n_frames)
        freqs: Frequency bins (n_freq,)
        fs: Sampling rate
        hop_length: STFT hop length
        num_subbands: Number of frequency subbands
        freq_min, freq_max: Frequency range
        search_radius_ms: Search window in ms
        exclusion_radius_samples: PSR exclusion zone
        aggregation: "median", "mode", or "weighted_mean"
        eps: Small value for numerical stability

    Returns:
        SubbandResult with per-band estimates and consistency metric
    """
    n_freq, n_frames = X_mic.shape

    # Find valid frequency range
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    valid_indices = np.where(freq_mask)[0]

    if len(valid_indices) < num_subbands:
        return SubbandResult(
            "subband", None, None, None, None, None,
            num_subbands=num_subbands, undefined_reason="insufficient_freq_bins"
        )

    # Divide into subbands
    subband_edges = np.linspace(valid_indices[0], valid_indices[-1], num_subbands + 1).astype(int)

    subband_taus_ms = []
    subband_psrs = []
    subband_peaks = []

    for sb in range(num_subbands):
        sb_start = subband_edges[sb]
        sb_end = subband_edges[sb + 1]

        if sb_end <= sb_start:
            continue

        # Extract subband
        X_mic_sb = X_mic[sb_start:sb_end, :]
        X_ldv_sb = X_ldv[sb_start:sb_end, :]
        freqs_sb = freqs[sb_start:sb_end]

        # GCC-PHAT on subband
        cross = np.conj(X_mic_sb) * X_ldv_sb
        cross_phat = cross / (np.abs(cross) + eps)
        avg_cross = np.mean(cross_phat, axis=1)

        # Simple IFFT approach for subband
        n_sb = len(avg_cross)
        nfft_sb = max(2 * n_sb, 64)  # Minimum FFT size for resolution

        full_cross = np.zeros(nfft_sb, dtype=np.complex128)
        full_cross[:n_sb] = avg_cross
        if n_sb > 1:
            full_cross[nfft_sb - n_sb + 1:] = np.conj(avg_cross[1:][::-1])

        cc = np.fft.ifft(full_cross).real
        cc = np.fft.fftshift(cc)
        lags = np.arange(-nfft_sb // 2, nfft_sb // 2)

        # Search window
        search_radius_samples = int(search_radius_ms * fs / 1000)
        search_radius_bins = max(search_radius_samples // hop_length, nfft_sb // 4)

        center = nfft_sb // 2
        lo = max(0, center - search_radius_bins)
        hi = min(nfft_sb, center + search_radius_bins)

        cc_sel = cc[lo:hi]
        lags_sel = lags[lo:hi]

        if len(cc_sel) == 0:
            continue

        # Find peak
        abs_cc = np.abs(cc_sel)
        peak_i = int(np.argmax(abs_cc))
        peak_lag = int(lags_sel[peak_i])
        peak_value = float(abs_cc[peak_i])

        tau_samples = float(peak_lag * hop_length)
        tau_ms = tau_samples * 1000.0 / fs

        # PSR for this subband
        exc = max(1, int(exclusion_radius_samples // hop_length))
        mask = np.ones(len(abs_cc), dtype=bool)
        mask[max(0, peak_i - exc):min(len(abs_cc), peak_i + exc + 1)] = False
        sidelobe = abs_cc[mask]
        psr = peak_value / (np.median(sidelobe) + eps) if len(sidelobe) > 0 else 1.0

        subband_taus_ms.append(tau_ms)
        subband_psrs.append(psr)
        subband_peaks.append(peak_value)

    if len(subband_taus_ms) == 0:
        return SubbandResult(
            "subband", None, None, None, None, None,
            num_subbands=num_subbands, undefined_reason="no_valid_subbands"
        )

    subband_taus_ms = np.array(subband_taus_ms)
    subband_psrs = np.array(subband_psrs)
    subband_peaks = np.array(subband_peaks)

    # Aggregate estimates
    if aggregation == "median":
        tau_ms_final = float(np.median(subband_taus_ms))
    elif aggregation == "mode":
        # Approximate mode using histogram
        hist, bin_edges = np.histogram(subband_taus_ms, bins='auto')
        mode_idx = np.argmax(hist)
        tau_ms_final = float((bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2)
    elif aggregation == "weighted_mean":
        # Weight by PSR
        weights = subband_psrs / (np.sum(subband_psrs) + eps)
        tau_ms_final = float(np.sum(subband_taus_ms * weights))
    else:
        tau_ms_final = float(np.median(subband_taus_ms))

    tau_samples_final = tau_ms_final * fs / 1000.0

    # Overall PSR (from aggregated peak)
    psr_final = float(np.median(subband_psrs))
    peak_final = float(np.mean(subband_peaks))

    # Consistency metric: IQR / median (lower = more consistent)
    if len(subband_taus_ms) >= 4:
        q75, q25 = np.percentile(subband_taus_ms, [75, 25])
        iqr = q75 - q25
        median_abs = np.abs(np.median(subband_taus_ms)) + eps
        consistency = float(iqr / median_abs) if median_abs > eps else float(iqr)
    else:
        consistency = float(np.std(subband_taus_ms))

    return SubbandResult(
        method="subband",
        tau_samples=tau_samples_final,
        tau_ms=tau_ms_final,
        peak_value=peak_final,
        psr=psr_final,
        boundary_hit=False,
        num_subbands=len(subband_taus_ms),
        subband_taus_ms=subband_taus_ms.tolist(),
        subband_consistency=consistency,
        aggregation_method=aggregation,
    )


# =============================================================================
# Standard Methods (for comparison)
# =============================================================================

def compute_cc_from_stft(
    X_mic: np.ndarray,
    X_ldv: np.ndarray,
    freqs: np.ndarray,
    fs: int,
    method: str,
    hop_length: int = 256,
    search_radius_ms: float = 10.0,
    exclusion_radius_samples: int = 16,
    eps: float = 1e-12,
) -> CCResult:
    """Standard CC/NCC/GCC-PHAT from STFT."""
    n_freq, n_frames = X_mic.shape

    cross = np.conj(X_mic) * X_ldv

    if method == 'cc':
        weighted_cross = cross
    elif method == 'ncc':
        energy_mic = np.sum(np.abs(X_mic) ** 2)
        energy_ldv = np.sum(np.abs(X_ldv) ** 2)
        norm_factor = np.sqrt(energy_mic * energy_ldv) + eps
        weighted_cross = cross / norm_factor
    elif method == 'gcc_phat':
        weighted_cross = cross / (np.abs(cross) + eps)
    else:
        return CCResult(method, None, None, None, None, None, f"unknown_method_{method}")

    avg_cross = np.mean(weighted_cross, axis=1)

    nfft = 2 * n_freq
    full_cross = np.zeros(nfft, dtype=np.complex128)
    full_cross[:n_freq] = avg_cross
    full_cross[n_freq:] = np.conj(avg_cross[::-1])

    cc = np.fft.ifft(full_cross).real
    cc = np.fft.fftshift(cc)
    lags = np.arange(-nfft // 2, nfft // 2)

    search_radius_samples = int(search_radius_ms * fs / 1000)
    search_radius_bins = max(search_radius_samples // hop_length, nfft // 4)

    center = nfft // 2
    lo = max(0, center - search_radius_bins)
    hi = min(nfft, center + search_radius_bins)

    cc_sel = cc[lo:hi]
    lags_sel = lags[lo:hi]

    if len(cc_sel) == 0:
        return CCResult(method, None, None, None, None, None, "no_valid_lags")

    abs_cc = np.abs(cc_sel)
    peak_i = int(np.argmax(abs_cc))
    peak_lag = int(lags_sel[peak_i])
    peak_value = float(abs_cc[peak_i])

    tau_samples = float(peak_lag * hop_length)
    tau_ms = tau_samples * 1000.0 / fs

    exc = int(max(1, exclusion_radius_samples // hop_length))
    mask = np.ones(len(abs_cc), dtype=bool)
    mask[max(0, peak_i - exc):min(len(abs_cc), peak_i + exc + 1)] = False
    sidelobe = abs_cc[mask]
    psr = None
    if len(sidelobe) > 0:
        median_sidelobe = float(np.median(sidelobe))
        if median_sidelobe > eps:
            psr = float(peak_value / median_sidelobe)

    boundary_hit = bool(peak_i == 0 or peak_i == len(cc_sel) - 1)

    return CCResult(method, tau_samples, tau_ms, peak_value, psr, boundary_hit)


# =============================================================================
# MUSIC Method
# =============================================================================

def steering_vector(freq: float, theta_rad: float, d: float, c: float = 343.0) -> np.ndarray:
    """Compute steering vector for 2-element ULA."""
    tau = d * np.sin(theta_rad) / c
    return np.array([[1.0], [np.exp(-1j * 2 * np.pi * freq * tau)]], dtype=np.complex128)


def compute_music(
    X_mic: np.ndarray,
    X_ldv: np.ndarray,
    freqs: np.ndarray,
    spacing_m: float,
    freq_min: float = 300.0,
    freq_max: float = 3000.0,
    angle_step: float = 1.0,
    c: float = 343.0,
    eps: float = 1e-12,
) -> Tuple[MUSICResult, np.ndarray, np.ndarray]:
    """2-element MUSIC from STFT."""
    n_freq, n_frames = X_mic.shape

    angles_deg = np.arange(-90, 90 + angle_step, angle_step)
    angles_rad = np.deg2rad(angles_deg)
    n_angles = len(angles_deg)

    freq_mask = (freqs >= freq_min) & (freqs <= freq_max) & (freqs > 1)

    if not np.any(freq_mask):
        return MUSICResult("music", None, None, None, None, None, "no_valid_freqs"), angles_deg, np.zeros(n_angles)

    P_music = np.zeros(n_angles, dtype=np.float64)
    eigenvalue_ratios = []

    for f_idx in np.where(freq_mask)[0]:
        f = freqs[f_idx]

        X = np.vstack([X_mic[f_idx, :], X_ldv[f_idx, :]])
        R = (X @ X.conj().T) / n_frames

        eigvals, eigvecs = np.linalg.eigh(R)
        Un = eigvecs[:, 0:1]

        if eigvals[0] > eps:
            eigenvalue_ratios.append(eigvals[1] / eigvals[0])

        for i, theta_rad in enumerate(angles_rad):
            a_theta = steering_vector(f, theta_rad, spacing_m, c)
            denom = np.abs(a_theta.conj().T @ Un @ Un.conj().T @ a_theta).item()
            P_music[i] += 1.0 / (denom + eps)

    if np.max(P_music) > eps:
        P_music_norm = P_music / np.max(P_music)
    else:
        P_music_norm = P_music

    peak_idx = int(np.argmax(P_music))
    doa_deg = float(angles_deg[peak_idx])
    peak_value = float(P_music[peak_idx])

    tau_sec = spacing_m * np.sin(np.deg2rad(doa_deg)) / c
    tau_ms = tau_sec * 1000.0

    snr_eigen = None
    if eigenvalue_ratios:
        mean_ratio = float(np.mean(eigenvalue_ratios))
        if mean_ratio > 1:
            snr_eigen = 10.0 * np.log10(mean_ratio - 1)

    exc_radius = max(1, int(10 / angle_step))
    mask = np.ones(n_angles, dtype=bool)
    mask[max(0, peak_idx - exc_radius):min(n_angles, peak_idx + exc_radius + 1)] = False
    sidelobe = P_music[mask]
    spectrum_psr = None
    if len(sidelobe) > 0:
        median_sidelobe = float(np.median(sidelobe))
        if median_sidelobe > eps:
            spectrum_psr = float(peak_value / median_sidelobe)

    result = MUSICResult(
        method="music",
        doa_deg=doa_deg,
        tau_ms=tau_ms,
        snr_eigen=snr_eigen,
        peak_value=peak_value,
        spectrum_psr=spectrum_psr,
    )

    return result, angles_deg, P_music_norm


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_pair(
    mic_path: str,
    ldv_path: str,
    *,
    fs: int,
    n_fft: int,
    hop_length: int,
    freq_min: float,
    freq_max: float,
    spacing_m: float,
    c: float,
    use_dtmin: bool,
    num_subbands: int = 8,
) -> Dict[str, Any]:
    """Evaluate all methods on a single MIC-LDV pair."""

    # Load signals
    mic_signal = load_wav(mic_path, target_fs=fs)
    ldv_signal = load_wav(ldv_path, target_fs=fs)

    min_len = min(len(mic_signal), len(ldv_signal))
    mic_signal = mic_signal[:min_len]
    ldv_signal = ldv_signal[:min_len]

    # Bandpass filter
    b, a = bandpass_filter(fs, freq_min, freq_max)
    mic_filtered = signal.filtfilt(b, a, (mic_signal - np.mean(mic_signal)).astype(np.float32))
    ldv_filtered = signal.filtfilt(b, a, (ldv_signal - np.mean(ldv_signal)).astype(np.float32))

    # STFT
    freqs, times, X_mic = signal.stft(mic_filtered, fs, nperseg=n_fft, noverlap=n_fft - hop_length)
    _, _, X_ldv = signal.stft(ldv_filtered, fs, nperseg=n_fft, noverlap=n_fft - hop_length)

    # DTmin compensation if enabled
    dtmin_info = {}
    if use_dtmin:
        tau_f, confidence = estimate_per_frequency_delay(
            X_mic, X_ldv, freqs, fs,
            freq_min=freq_min, freq_max=freq_max
        )
        X_ldv_processed = apply_dtmin_compensation(X_ldv, tau_f, freqs)

        dtmin_info = {
            "tau_f_mean_ms": float(np.mean(tau_f[freqs > freq_min]) * 1000) if np.any(freqs > freq_min) else None,
            "tau_f_std_ms": float(np.std(tau_f[freqs > freq_min]) * 1000) if np.any(freqs > freq_min) else None,
            "confidence_mean": float(np.mean(confidence[freqs > freq_min])) if np.any(freqs > freq_min) else None,
        }
    else:
        X_ldv_processed = X_ldv

    # Initialize results
    results = {
        "mic_path": mic_path,
        "ldv_path": ldv_path,
        "use_dtmin": use_dtmin,
        "dtmin_info": dtmin_info,
    }

    # ===== Standard CC methods =====
    for method in ['cc', 'ncc', 'gcc_phat']:
        cc_result = compute_cc_from_stft(
            X_mic, X_ldv_processed, freqs, fs, method,
            hop_length=hop_length, search_radius_ms=10.0
        )
        results[method] = asdict(cc_result)

    # ===== GCC-SCOT =====
    scot_result = compute_gcc_scot(
        X_mic, X_ldv_processed, freqs, fs,
        hop_length=hop_length, search_radius_ms=10.0
    )
    results["gcc_scot"] = asdict(scot_result)

    # ===== POC =====
    poc_result = compute_poc(
        X_mic, X_ldv_processed, freqs, fs,
        hop_length=hop_length, search_radius_ms=10.0
    )
    results["poc"] = asdict(poc_result)

    # ===== Subband Analysis =====
    subband_result = compute_subband_analysis(
        X_mic, X_ldv_processed, freqs, fs,
        hop_length=hop_length,
        num_subbands=num_subbands,
        freq_min=freq_min, freq_max=freq_max,
        search_radius_ms=10.0,
        aggregation="median"
    )
    results["subband"] = asdict(subband_result)

    # ===== MUSIC =====
    music_result, angles, spectrum = compute_music(
        X_mic, X_ldv_processed, freqs,
        spacing_m=spacing_m,
        freq_min=freq_min, freq_max=freq_max,
        c=c
    )
    results["music"] = asdict(music_result)
    results["music_spectrum"] = spectrum.tolist()
    results["music_angles"] = angles.tolist()

    return results


def compute_summary(all_results: List[Dict], condition: str) -> Dict[str, Any]:
    """Compute summary statistics for a condition."""

    methods = ['cc', 'ncc', 'gcc_phat', 'gcc_scot', 'poc', 'subband', 'music']
    summary = {"condition": condition}

    for method in methods:
        tau_values = []
        psr_values = []
        snr_values = []
        doa_values = []
        consistency_values = []

        for r in all_results:
            m = r.get(method, {})
            if m.get("tau_ms") is not None:
                tau_values.append(m["tau_ms"])
            if m.get("psr") is not None:
                psr_values.append(m["psr"])
            if m.get("spectrum_psr") is not None:
                psr_values.append(m["spectrum_psr"])
            if m.get("snr_eigen") is not None:
                snr_values.append(m["snr_eigen"])
            if m.get("doa_deg") is not None:
                doa_values.append(m["doa_deg"])
            if m.get("subband_consistency") is not None:
                consistency_values.append(m["subband_consistency"])

        def stats(arr, name):
            if len(arr) == 0:
                return {f"{name}_count": 0}
            arr = np.array(arr)
            return {
                f"{name}_count": len(arr),
                f"{name}_mean": float(np.mean(arr)),
                f"{name}_std": float(np.std(arr)),
                f"{name}_median": float(np.median(arr)),
                f"{name}_p10": float(np.percentile(arr, 10)),
                f"{name}_p90": float(np.percentile(arr, 90)),
            }

        method_stats = {}
        method_stats.update(stats(tau_values, "tau_ms"))
        method_stats.update(stats(psr_values, "psr"))
        if snr_values:
            method_stats.update(stats(snr_values, "snr_db"))
        if doa_values:
            method_stats.update(stats(doa_values, "doa_deg"))
        if consistency_values:
            method_stats.update(stats(consistency_values, "consistency"))

        summary[method] = method_stats

    # DTmin info aggregation
    if all_results and all_results[0].get("use_dtmin"):
        tau_f_means = [r["dtmin_info"]["tau_f_mean_ms"] for r in all_results
                       if r["dtmin_info"].get("tau_f_mean_ms") is not None]
        if tau_f_means:
            summary["dtmin_tau_f_mean_ms"] = float(np.mean(tau_f_means))
            summary["dtmin_tau_f_std_ms"] = float(np.std(tau_f_means))

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extended TDoA Methods: GCC-SCOT, Subband, POC")

    parser.add_argument("--mic_root", type=str, required=True)
    parser.add_argument("--ldv_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke", "scale", "full"])
    parser.add_argument("--num_pairs", type=int, default=None)

    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--freq_min", type=float, default=300.0)
    parser.add_argument("--freq_max", type=float, default=3000.0)
    parser.add_argument("--spacing_m", type=float, default=0.05)
    parser.add_argument("--c", type=float, default=343.0)
    parser.add_argument("--num_subbands", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 80)
    print("Extended TDoA Methods: CC / NCC / GCC-PHAT / GCC-SCOT / POC / Subband / MUSIC")
    print("=" * 80)
    print(f"MIC root: {args.mic_root}")
    print(f"LDV root: {args.ldv_root}")
    print(f"Output: {args.out_dir}")
    print(f"Mode: {args.mode}")
    print(f"Subbands: {args.num_subbands}")
    print("=" * 80)

    # Discover pairs
    all_pairs = discover_pairs(args.mic_root, args.ldv_root)
    print(f"Found {len(all_pairs)} MIC-LDV pairs")

    if len(all_pairs) == 0:
        print("ERROR: No pairs found!")
        sys.exit(1)

    # Select subset
    if args.num_pairs is not None:
        num_pairs = min(args.num_pairs, len(all_pairs))
    elif args.mode == "smoke":
        num_pairs = 1
    elif args.mode == "scale":
        num_pairs = min(48, len(all_pairs))
    else:
        num_pairs = len(all_pairs)

    selected_pairs = all_pairs[:num_pairs]
    print(f"Selected {len(selected_pairs)} pairs")

    config = {
        "fs": args.fs,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "freq_min": args.freq_min,
        "freq_max": args.freq_max,
        "spacing_m": args.spacing_m,
        "c": args.c,
        "num_subbands": args.num_subbands,
        "seed": args.seed,
    }

    # Run both conditions
    results_no_dtmin = []
    results_with_dtmin = []

    for i, (mic_path, ldv_path) in enumerate(selected_pairs):
        print(f"\nProcessing pair {i+1}/{len(selected_pairs)}: {os.path.basename(mic_path)}")

        # Without DTmin
        result_no = evaluate_pair(
            mic_path, ldv_path,
            fs=args.fs, n_fft=args.n_fft, hop_length=args.hop_length,
            freq_min=args.freq_min, freq_max=args.freq_max,
            spacing_m=args.spacing_m, c=args.c,
            use_dtmin=False,
            num_subbands=args.num_subbands,
        )
        results_no_dtmin.append(result_no)

        # With DTmin
        result_dt = evaluate_pair(
            mic_path, ldv_path,
            fs=args.fs, n_fft=args.n_fft, hop_length=args.hop_length,
            freq_min=args.freq_min, freq_max=args.freq_max,
            spacing_m=args.spacing_m, c=args.c,
            use_dtmin=True,
            num_subbands=args.num_subbands,
        )
        results_with_dtmin.append(result_dt)

        # Print quick comparison for new methods
        print(f"  [No DTmin]   GCC-SCOT tau={result_no['gcc_scot']['tau_ms']:.2f}ms, "
              f"POC tau={result_no['poc']['tau_ms']:.2f}ms, "
              f"Subband tau={result_no['subband']['tau_ms']:.2f}ms")
        print(f"  [With DTmin] GCC-SCOT tau={result_dt['gcc_scot']['tau_ms']:.2f}ms, "
              f"POC tau={result_dt['poc']['tau_ms']:.2f}ms, "
              f"Subband tau={result_dt['subband']['tau_ms']:.2f}ms")

        # Subband consistency
        cons_no = result_no['subband'].get('subband_consistency', 0)
        cons_dt = result_dt['subband'].get('subband_consistency', 0)
        print(f"  Subband consistency: No DTmin={cons_no:.3f}, With DTmin={cons_dt:.3f}")

    # Compute summaries
    summary_no_dtmin = compute_summary(results_no_dtmin, "no_dtmin")
    summary_with_dtmin = compute_summary(results_with_dtmin, "with_dtmin")

    # Save results
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "num_pairs": len(selected_pairs),
        "config": config,
        "methods": ["cc", "ncc", "gcc_phat", "gcc_scot", "poc", "subband", "music"],
    }

    with open(os.path.join(args.out_dir, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(args.out_dir, "results_no_dtmin.json"), 'w') as f:
        json.dump(results_no_dtmin, f, indent=2)

    with open(os.path.join(args.out_dir, "results_with_dtmin.json"), 'w') as f:
        json.dump(results_with_dtmin, f, indent=2)

    with open(os.path.join(args.out_dir, "summary_no_dtmin.json"), 'w') as f:
        json.dump(summary_no_dtmin, f, indent=2)

    with open(os.path.join(args.out_dir, "summary_with_dtmin.json"), 'w') as f:
        json.dump(summary_with_dtmin, f, indent=2)

    # Comparison summary
    comparison = {
        "num_pairs": len(selected_pairs),
        "methods": {}
    }

    for method in ['cc', 'ncc', 'gcc_phat', 'gcc_scot', 'poc', 'subband', 'music']:
        no_dt = summary_no_dtmin.get(method, {})
        with_dt = summary_with_dtmin.get(method, {})

        comparison["methods"][method] = {
            "no_dtmin": {
                "tau_ms_median": no_dt.get("tau_ms_median"),
                "tau_ms_std": no_dt.get("tau_ms_std"),
                "psr_median": no_dt.get("psr_median"),
                "snr_db_median": no_dt.get("snr_db_median"),
                "consistency_median": no_dt.get("consistency_median"),
            },
            "with_dtmin": {
                "tau_ms_median": with_dt.get("tau_ms_median"),
                "tau_ms_std": with_dt.get("tau_ms_std"),
                "psr_median": with_dt.get("psr_median"),
                "snr_db_median": with_dt.get("snr_db_median"),
                "consistency_median": with_dt.get("consistency_median"),
            },
        }

        # Compute improvement
        if no_dt.get("tau_ms_std") and with_dt.get("tau_ms_std") and no_dt["tau_ms_std"] > 0:
            improvement = (no_dt["tau_ms_std"] - with_dt["tau_ms_std"]) / no_dt["tau_ms_std"] * 100
            comparison["methods"][method]["tau_std_reduction_pct"] = improvement

        if no_dt.get("psr_median") and with_dt.get("psr_median") and no_dt["psr_median"] > 0:
            improvement = (with_dt["psr_median"] - no_dt["psr_median"]) / no_dt["psr_median"] * 100
            comparison["methods"][method]["psr_improvement_pct"] = improvement

        # Consistency improvement for subband
        if no_dt.get("consistency_median") and with_dt.get("consistency_median") and no_dt["consistency_median"] > 0:
            improvement = (no_dt["consistency_median"] - with_dt["consistency_median"]) / no_dt["consistency_median"] * 100
            comparison["methods"][method]["consistency_improvement_pct"] = improvement

    with open(os.path.join(args.out_dir, "comparison_summary.json"), 'w') as f:
        json.dump(comparison, f, indent=2)

    # Print summary
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Method':<12} {'Condition':<12} {'tau_ms(med)':<12} {'tau_ms(std)':<12} {'PSR(med)':<10} {'Consist.':<10}")
    print("-" * 100)

    for method in ['cc', 'ncc', 'gcc_phat', 'gcc_scot', 'poc', 'subband', 'music']:
        no_dt = summary_no_dtmin.get(method, {})
        with_dt = summary_with_dtmin.get(method, {})

        def fmt(v, d=2):
            return f"{v:.{d}f}" if v is not None else "N/A"

        print(f"{method:<12} {'No DTmin':<12} {fmt(no_dt.get('tau_ms_median')):<12} "
              f"{fmt(no_dt.get('tau_ms_std')):<12} {fmt(no_dt.get('psr_median'), 1):<10} "
              f"{fmt(no_dt.get('consistency_median'), 3):<10}")
        print(f"{'':<12} {'With DTmin':<12} {fmt(with_dt.get('tau_ms_median')):<12} "
              f"{fmt(with_dt.get('tau_ms_std')):<12} {fmt(with_dt.get('psr_median'), 1):<10} "
              f"{fmt(with_dt.get('consistency_median'), 3):<10}")
        print("-" * 100)

    print("\nResults saved to:", args.out_dir)
    print("=" * 100)
    print("DONE")


if __name__ == "__main__":
    main()
