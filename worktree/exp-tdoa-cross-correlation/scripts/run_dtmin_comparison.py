#!/usr/bin/env python3
"""
DTmin Comparison: All TDoA Methods With and Without Frequency-Dependent Delay Compensation

This script compares CC, NCC, GCC-PHAT, and MUSIC methods under two conditions:
1. Baseline (no DTmin): Raw LDV signal
2. With DTmin: LDV signal with per-frequency phase compensation

DTmin Implementation:
- Estimates per-frequency delay tau(f) using GCC-PHAT on each frequency bin
- Applies phase compensation: Y_comp(f) = Y(f) * exp(+j*2*pi*f*tau(f))

Usage:
    python run_dtmin_comparison.py \
        --mic_root "path/to/MIC" \
        --ldv_root "path/to/LDV" \
        --out_dir "results/dtmin_comparison_xxx" \
        --mode scale

Author: Auto-generated for exp-tdoa-cross-correlation
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
# Utility Functions
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
    max_lag_ms: float = 10.0,
    freq_min: float = 300.0,
    freq_max: float = 3000.0,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-frequency delay tau(f) using GCC-PHAT on each frequency bin.

    Args:
        X_mic: MIC STFT (n_freq, n_frames)
        X_ldv: LDV STFT (n_freq, n_frames)
        freqs: Frequency bins (n_freq,)
        fs: Sampling rate
        max_lag_ms: Maximum lag to search in ms
        freq_min, freq_max: Frequency band for estimation
        eps: Small value to avoid division by zero

    Returns:
        tau_f: Per-frequency delay estimates in seconds (n_freq,)
        confidence: Confidence score for each estimate (n_freq,)
    """
    n_freq, n_frames = X_mic.shape
    tau_f = np.zeros(n_freq)
    confidence = np.zeros(n_freq)

    # Convert max_lag to samples (for STFT frame resolution)
    hop_length = 256  # Assume standard hop
    max_lag_frames = int(max_lag_ms * fs / 1000 / hop_length)
    max_lag_frames = max(max_lag_frames, 1)

    for f_idx, f in enumerate(freqs):
        # Skip frequencies outside band
        if f < freq_min or f > freq_max or f < 1:
            tau_f[f_idx] = 0.0
            confidence[f_idx] = 0.0
            continue

        # Get time series for this frequency bin
        x_f = X_mic[f_idx, :]  # (n_frames,)
        y_f = X_ldv[f_idx, :]  # (n_frames,)

        # GCC-PHAT for this frequency bin
        # Cross-spectrum
        cross = np.conj(x_f) * y_f

        # Phase transform (normalize by magnitude)
        cross_phat = cross / (np.abs(cross) + eps)

        # The delay is encoded in the phase progression across frames
        # For a single frequency bin, we estimate delay from phase difference
        phase_diff = np.angle(cross_phat)

        # Weighted average phase (weight by magnitude)
        weights = np.abs(cross) + eps
        mean_phase = np.sum(phase_diff * weights) / np.sum(weights)

        # Convert phase to time delay
        # phase = -2*pi*f*tau => tau = -phase / (2*pi*f)
        if f > 1:
            tau_f[f_idx] = -mean_phase / (2 * np.pi * f)
        else:
            tau_f[f_idx] = 0.0

        # Confidence based on coherence
        coherence = np.abs(np.mean(cross_phat))
        confidence[f_idx] = coherence

    return tau_f, confidence


def apply_dtmin_compensation(
    X_ldv: np.ndarray,
    tau_f: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """
    Apply DTmin phase compensation to LDV STFT.

    Y_compensated(f) = Y(f) * exp(+j * 2 * pi * f * tau(f))

    This shifts each frequency bin to align with MIC.

    Args:
        X_ldv: LDV STFT (n_freq, n_frames)
        tau_f: Per-frequency delay estimates in seconds (n_freq,)
        freqs: Frequency bins (n_freq,)

    Returns:
        X_ldv_comp: Compensated LDV STFT (n_freq, n_frames)
    """
    n_freq, n_frames = X_ldv.shape

    # Build phase compensation vector
    phase_comp = np.exp(1j * 2 * np.pi * freqs * tau_f)  # (n_freq,)

    # Apply to all frames
    X_ldv_comp = X_ldv * phase_comp[:, np.newaxis]

    return X_ldv_comp


# =============================================================================
# Cross-Correlation Methods
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


def compute_cc_from_stft(
    X_mic: np.ndarray,
    X_ldv: np.ndarray,
    freqs: np.ndarray,
    fs: int,
    method: str,
    search_radius_ms: float = 10.0,
    exclusion_radius_samples: int = 16,
    eps: float = 1e-12,
) -> CCResult:
    """
    Compute cross-correlation based TDoA from STFT representations.

    Args:
        X_mic, X_ldv: STFT arrays (n_freq, n_frames)
        freqs: Frequency bins
        fs: Sampling rate
        method: 'cc', 'ncc', or 'gcc_phat'
        search_radius_ms: Search window in ms

    Returns:
        CCResult
    """
    n_freq, n_frames = X_mic.shape

    # Compute cross-spectrum
    cross = np.conj(X_mic) * X_ldv  # (n_freq, n_frames)

    # Apply weighting based on method
    if method == 'cc':
        # Standard CC: use cross-spectrum as-is (amplitude weighted)
        weighted_cross = cross
    elif method == 'ncc':
        # NCC: normalize by energy
        energy_mic = np.sum(np.abs(X_mic) ** 2)
        energy_ldv = np.sum(np.abs(X_ldv) ** 2)
        norm_factor = np.sqrt(energy_mic * energy_ldv) + eps
        weighted_cross = cross / norm_factor
    elif method == 'gcc_phat':
        # GCC-PHAT: phase transform
        weighted_cross = cross / (np.abs(cross) + eps)
    else:
        return CCResult(method, None, None, None, None, None, f"unknown_method_{method}")

    # Average across frames to get cross-spectrum
    avg_cross = np.mean(weighted_cross, axis=1)  # (n_freq,)

    # Inverse FFT to get cross-correlation
    nfft = 2 * n_freq
    # Reconstruct full spectrum (assuming real signals)
    full_cross = np.zeros(nfft, dtype=np.complex128)
    full_cross[:n_freq] = avg_cross
    full_cross[n_freq:] = np.conj(avg_cross[::-1])

    cc = np.fft.ifft(full_cross).real

    # Rearrange to center zero lag
    cc = np.fft.fftshift(cc)
    lags = np.arange(-nfft // 2, nfft // 2)

    # Convert lag limits to STFT bin indices
    hop_length = 256  # Assume standard
    search_radius_samples = int(search_radius_ms * fs / 1000)
    search_radius_bins = search_radius_samples // hop_length
    search_radius_bins = max(search_radius_bins, nfft // 4)  # Ensure reasonable range

    # Search window
    center = nfft // 2
    lo = max(0, center - search_radius_bins)
    hi = min(nfft, center + search_radius_bins)

    cc_sel = cc[lo:hi]
    lags_sel = lags[lo:hi]

    if len(cc_sel) == 0:
        return CCResult(method, None, None, None, None, None, "no_valid_lags")

    # Find peak
    abs_cc = np.abs(cc_sel)
    peak_i = int(np.argmax(abs_cc))
    peak_lag = int(lags_sel[peak_i])
    peak_value = float(abs_cc[peak_i])

    # Convert to samples (approximate)
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

    return CCResult(method, tau_samples, tau_ms, peak_value, psr, boundary_hit)


# =============================================================================
# MUSIC Method
# =============================================================================

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
    angle_min: float = -90.0,
    angle_max: float = 90.0,
    angle_step: float = 1.0,
    c: float = 343.0,
    eps: float = 1e-12,
) -> Tuple[MUSICResult, np.ndarray, np.ndarray]:
    """
    2-element MUSIC from STFT representations.

    Returns:
        (MUSICResult, angles_deg, music_spectrum)
    """
    n_freq, n_frames = X_mic.shape

    # Angle scan
    angles_deg = np.arange(angle_min, angle_max + angle_step, angle_step)
    angles_rad = np.deg2rad(angles_deg)
    n_angles = len(angles_deg)

    # Frequency mask
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max) & (freqs > 1)

    if not np.any(freq_mask):
        return MUSICResult("music", None, None, None, None, None, "no_valid_freqs"), angles_deg, np.zeros(n_angles)

    # Accumulate MUSIC spectrum
    P_music = np.zeros(n_angles, dtype=np.float64)
    eigenvalue_ratios = []

    for f_idx in np.where(freq_mask)[0]:
        f = freqs[f_idx]

        # Build 2x2 covariance matrix
        X = np.vstack([X_mic[f_idx, :], X_ldv[f_idx, :]])  # 2 x n_frames
        R = (X @ X.conj().T) / n_frames

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(R)
        Un = eigvecs[:, 0:1]  # Noise subspace

        if eigvals[0] > eps:
            eigenvalue_ratios.append(eigvals[1] / eigvals[0])

        # MUSIC spectrum
        for i, theta_rad in enumerate(angles_rad):
            a_theta = steering_vector(f, theta_rad, spacing_m, c)
            denom = np.abs(a_theta.conj().T @ Un @ Un.conj().T @ a_theta).item()
            P_music[i] += 1.0 / (denom + eps)

    # Normalize
    if np.max(P_music) > eps:
        P_music_norm = P_music / np.max(P_music)
    else:
        P_music_norm = P_music

    # Find peak
    peak_idx = int(np.argmax(P_music))
    doa_deg = float(angles_deg[peak_idx])
    peak_value = float(P_music[peak_idx])

    # Convert to TDoA
    tau_sec = spacing_m * np.sin(np.deg2rad(doa_deg)) / c
    tau_ms = tau_sec * 1000.0

    # SNR estimation
    snr_eigen = None
    if eigenvalue_ratios:
        mean_ratio = float(np.mean(eigenvalue_ratios))
        if mean_ratio > 1:
            snr_eigen = 10.0 * np.log10(mean_ratio - 1)

    # Spectrum PSR
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
) -> Dict[str, Any]:
    """Evaluate all methods on a single MIC-LDV pair."""

    # Load signals
    mic_signal = load_wav(mic_path, target_fs=fs)
    ldv_signal = load_wav(ldv_path, target_fs=fs)

    # Ensure same length
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

        # Store DTmin info
        dtmin_info = {
            "tau_f_mean_ms": float(np.mean(tau_f[freqs > freq_min]) * 1000) if np.any(freqs > freq_min) else None,
            "tau_f_std_ms": float(np.std(tau_f[freqs > freq_min]) * 1000) if np.any(freqs > freq_min) else None,
            "confidence_mean": float(np.mean(confidence[freqs > freq_min])) if np.any(freqs > freq_min) else None,
        }
    else:
        X_ldv_processed = X_ldv

    # Run all methods
    results = {
        "mic_path": mic_path,
        "ldv_path": ldv_path,
        "use_dtmin": use_dtmin,
        "dtmin_info": dtmin_info,
    }

    # CC methods
    for method in ['cc', 'ncc', 'gcc_phat']:
        cc_result = compute_cc_from_stft(
            X_mic, X_ldv_processed, freqs, fs, method,
            search_radius_ms=10.0
        )
        results[method] = asdict(cc_result)

    # MUSIC
    music_result, angles, spectrum = compute_music(
        X_mic, X_ldv_processed, freqs,
        spacing_m=spacing_m,
        freq_min=freq_min,
        freq_max=freq_max,
        c=c
    )
    results["music"] = asdict(music_result)
    results["music_spectrum"] = spectrum.tolist()
    results["music_angles"] = angles.tolist()

    return results


def compute_summary(all_results: List[Dict], condition: str) -> Dict[str, Any]:
    """Compute summary statistics for a condition."""

    methods = ['cc', 'ncc', 'gcc_phat', 'music']
    summary = {"condition": condition}

    for method in methods:
        tau_values = []
        psr_values = []
        snr_values = []
        doa_values = []

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
    parser = argparse.ArgumentParser(description="DTmin Comparison: All TDoA Methods")

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
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 70)
    print("DTmin Comparison: CC / NCC / GCC-PHAT / MUSIC")
    print("=" * 70)
    print(f"MIC root: {args.mic_root}")
    print(f"LDV root: {args.ldv_root}")
    print(f"Output: {args.out_dir}")
    print(f"Mode: {args.mode}")
    print("=" * 70)

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
            use_dtmin=False
        )
        results_no_dtmin.append(result_no)

        # With DTmin
        result_dt = evaluate_pair(
            mic_path, ldv_path,
            fs=args.fs, n_fft=args.n_fft, hop_length=args.hop_length,
            freq_min=args.freq_min, freq_max=args.freq_max,
            spacing_m=args.spacing_m, c=args.c,
            use_dtmin=True
        )
        results_with_dtmin.append(result_dt)

        # Print quick comparison
        print(f"  [No DTmin]   GCC-PHAT tau={result_no['gcc_phat']['tau_ms']:.3f}ms, "
              f"MUSIC SNR={result_no['music']['snr_eigen']:.1f}dB"
              if result_no['music']['snr_eigen'] else f"  [No DTmin]   GCC-PHAT tau={result_no['gcc_phat']['tau_ms']:.3f}ms")
        print(f"  [With DTmin] GCC-PHAT tau={result_dt['gcc_phat']['tau_ms']:.3f}ms, "
              f"MUSIC SNR={result_dt['music']['snr_eigen']:.1f}dB"
              if result_dt['music']['snr_eigen'] else f"  [With DTmin] GCC-PHAT tau={result_dt['gcc_phat']['tau_ms']:.3f}ms")

    # Compute summaries
    summary_no_dtmin = compute_summary(results_no_dtmin, "no_dtmin")
    summary_with_dtmin = compute_summary(results_with_dtmin, "with_dtmin")

    # Save results
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "num_pairs": len(selected_pairs),
        "config": config,
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

    for method in ['cc', 'ncc', 'gcc_phat', 'music']:
        no_dt = summary_no_dtmin.get(method, {})
        with_dt = summary_with_dtmin.get(method, {})

        comparison["methods"][method] = {
            "no_dtmin": {
                "tau_ms_median": no_dt.get("tau_ms_median"),
                "tau_ms_std": no_dt.get("tau_ms_std"),
                "psr_median": no_dt.get("psr_median"),
                "snr_db_median": no_dt.get("snr_db_median"),
            },
            "with_dtmin": {
                "tau_ms_median": with_dt.get("tau_ms_median"),
                "tau_ms_std": with_dt.get("tau_ms_std"),
                "psr_median": with_dt.get("psr_median"),
                "snr_db_median": with_dt.get("snr_db_median"),
            },
        }

        # Compute improvement
        if no_dt.get("tau_ms_std") and with_dt.get("tau_ms_std"):
            improvement = (no_dt["tau_ms_std"] - with_dt["tau_ms_std"]) / no_dt["tau_ms_std"] * 100
            comparison["methods"][method]["tau_std_reduction_pct"] = improvement

        if no_dt.get("psr_median") and with_dt.get("psr_median"):
            improvement = (with_dt["psr_median"] - no_dt["psr_median"]) / no_dt["psr_median"] * 100
            comparison["methods"][method]["psr_improvement_pct"] = improvement

    with open(os.path.join(args.out_dir, "comparison_summary.json"), 'w') as f:
        json.dump(comparison, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<12} {'Condition':<12} {'tau_ms (med)':<14} {'tau_ms (std)':<14} {'PSR (med)':<12} {'SNR (med)':<12}")
    print("-" * 70)

    for method in ['cc', 'ncc', 'gcc_phat', 'music']:
        no_dt = summary_no_dtmin.get(method, {})
        with_dt = summary_with_dtmin.get(method, {})

        tau_no = f"{no_dt.get('tau_ms_median', 0):.3f}" if no_dt.get('tau_ms_median') is not None else "N/A"
        tau_dt = f"{with_dt.get('tau_ms_median', 0):.3f}" if with_dt.get('tau_ms_median') is not None else "N/A"
        std_no = f"{no_dt.get('tau_ms_std', 0):.3f}" if no_dt.get('tau_ms_std') is not None else "N/A"
        std_dt = f"{with_dt.get('tau_ms_std', 0):.3f}" if with_dt.get('tau_ms_std') is not None else "N/A"
        psr_no = f"{no_dt.get('psr_median', 0):.1f}" if no_dt.get('psr_median') is not None else "N/A"
        psr_dt = f"{with_dt.get('psr_median', 0):.1f}" if with_dt.get('psr_median') is not None else "N/A"
        snr_no = f"{no_dt.get('snr_db_median', 0):.1f}" if no_dt.get('snr_db_median') is not None else "N/A"
        snr_dt = f"{with_dt.get('snr_db_median', 0):.1f}" if with_dt.get('snr_db_median') is not None else "N/A"

        print(f"{method:<12} {'No DTmin':<12} {tau_no:<14} {std_no:<14} {psr_no:<12} {snr_no:<12}")
        print(f"{'':<12} {'With DTmin':<12} {tau_dt:<14} {std_dt:<14} {psr_dt:<12} {snr_dt:<12}")
        print("-" * 70)

    print("\nResults saved to:", args.out_dir)
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
