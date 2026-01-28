#!/usr/bin/env python3
"""
MUSIC-based DoA/TDoA Estimation for 2-Element MIC-LDV Array

This script implements the MUSIC (Multiple Signal Classification) algorithm
for a 2-element "virtual array" formed by MIC and LDV sensors.

Outputs:
1. DoA (Direction of Arrival) estimation
2. Spatial spectrum P(θ) visualization
3. SNR estimation from eigenvalue ratio (λ1/λ2)

Note: For accurate DoA, the sensor spacing 'd' must be calibrated for your setup.
      The default value is for demonstration purposes.

Usage:
    python run_music_doa.py \
        --mic_root "path/to/MIC" \
        --ldv_root "path/to/LDV" \
        --out_dir "results/music_xxx" \
        --spacing_m 0.05 \
        --mode smoke

Author: Auto-generated for exp-tdoa-cross-correlation
Date: 2026-01-28
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from scipy import signal
from scipy.io import wavfile

# Optional: matplotlib for spectrum plots
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Utility Functions (shared with run_cross_correlation_tdoa.py)
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
# MUSIC Algorithm
# =============================================================================

@dataclass
class MUSICResult:
    """Result of MUSIC DoA estimation."""
    method: str
    doa_deg: Optional[float]           # Estimated DoA in degrees
    tau_ms: Optional[float]            # Equivalent TDoA in ms
    snr_eigen: Optional[float]         # SNR from eigenvalue ratio (λ1/λ2 in dB)
    peak_value: Optional[float]        # Peak value of MUSIC spectrum
    spectrum_psr: Optional[float]      # Peak-to-sidelobe ratio of spectrum
    undefined_reason: Optional[str] = None


def steering_vector(freq: float, theta_rad: float, d: float, c: float = 343.0) -> np.ndarray:
    """
    Compute steering vector for 2-element ULA.

    a(θ) = [1, exp(-j * 2π * f * d * sin(θ) / c)]^T

    Args:
        freq: Frequency in Hz
        theta_rad: Angle in radians
        d: Sensor spacing in meters
        c: Speed of sound in m/s

    Returns:
        2x1 complex steering vector
    """
    tau = d * np.sin(theta_rad) / c
    return np.array([[1.0], [np.exp(-1j * 2 * np.pi * freq * tau)]], dtype=np.complex128)


def music_2element(
    x_mic: np.ndarray,
    x_ldv: np.ndarray,
    *,
    fs: int,
    spacing_m: float,
    freq_min: float,
    freq_max: float,
    angle_min: float = -90.0,
    angle_max: float = 90.0,
    angle_step: float = 1.0,
    n_fft: int = 1024,
    c: float = 343.0,
    b: np.ndarray,
    a: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[MUSICResult, np.ndarray, np.ndarray]:
    """
    2-element MUSIC for MIC + LDV.

    Args:
        x_mic: MIC signal
        x_ldv: LDV signal
        fs: Sampling rate
        spacing_m: Sensor spacing in meters
        freq_min, freq_max: Frequency band for analysis
        angle_min, angle_max, angle_step: Angle scan range and resolution
        n_fft: FFT size for STFT
        c: Speed of sound
        b, a: Bandpass filter coefficients
        eps: Small value to avoid division by zero

    Returns:
        (MUSICResult, angles_deg, music_spectrum)
    """
    if x_mic.size == 0 or x_ldv.size == 0:
        angles = np.arange(angle_min, angle_max + angle_step, angle_step)
        return MUSICResult("music", None, None, None, None, None, "empty_input"), angles, np.zeros_like(angles)

    if x_mic.size != x_ldv.size:
        angles = np.arange(angle_min, angle_max + angle_step, angle_step)
        return MUSICResult("music", None, None, None, None, None, "size_mismatch"), angles, np.zeros_like(angles)

    # Bandpass filter
    x_mic_f = signal.filtfilt(b, a, (x_mic - np.mean(x_mic)).astype(np.float32))
    x_ldv_f = signal.filtfilt(b, a, (x_ldv - np.mean(x_ldv)).astype(np.float32))

    # STFT
    hop_length = n_fft // 4
    freqs, times, Zmic = signal.stft(x_mic_f, fs, nperseg=n_fft, noverlap=n_fft - hop_length)
    _, _, Zldv = signal.stft(x_ldv_f, fs, nperseg=n_fft, noverlap=n_fft - hop_length)

    n_freqs, n_frames = Zmic.shape

    # Frequency bin selection
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    valid_freqs = freqs[freq_mask]

    if len(valid_freqs) == 0:
        angles = np.arange(angle_min, angle_max + angle_step, angle_step)
        return MUSICResult("music", None, None, None, None, None, "no_valid_freqs"), angles, np.zeros_like(angles)

    # Angle scan range
    angles_deg = np.arange(angle_min, angle_max + angle_step, angle_step)
    angles_rad = np.deg2rad(angles_deg)
    n_angles = len(angles_deg)

    # Accumulate MUSIC spectrum across frequencies (incoherent wideband MUSIC)
    P_music = np.zeros(n_angles, dtype=np.float64)
    eigenvalue_ratios = []

    for f_idx in np.where(freq_mask)[0]:
        f = freqs[f_idx]
        if f < 1:  # Skip DC
            continue

        # Build 2x2 covariance matrix
        # X = [Zmic[f_idx, :], Zldv[f_idx, :]]  shape: 2 x n_frames
        X = np.vstack([Zmic[f_idx, :], Zldv[f_idx, :]])  # 2 x n_frames

        # Covariance matrix: R = X @ X^H / n_frames
        R = (X @ X.conj().T) / n_frames

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(R)
        # eigvals are in ascending order: [λ_small, λ_large]

        # For 2 elements, 1 source:
        # - Signal subspace: eigvec corresponding to λ_large (index 1)
        # - Noise subspace: eigvec corresponding to λ_small (index 0)
        Un = eigvecs[:, 0:1]  # Noise subspace (2x1)

        # Store eigenvalue ratio for SNR estimation
        if eigvals[0] > eps:
            eigenvalue_ratios.append(eigvals[1] / eigvals[0])

        # MUSIC spectrum for this frequency
        for i, theta_rad in enumerate(angles_rad):
            a_theta = steering_vector(f, theta_rad, spacing_m, c)

            # P(θ) = 1 / |a^H Un Un^H a|
            denom = np.abs(a_theta.conj().T @ Un @ Un.conj().T @ a_theta).item()
            P_music[i] += 1.0 / (denom + eps)

    # Normalize spectrum
    if np.max(P_music) > eps:
        P_music_norm = P_music / np.max(P_music)
    else:
        P_music_norm = P_music

    # Find peak = DoA estimate
    peak_idx = int(np.argmax(P_music))
    doa_deg = float(angles_deg[peak_idx])
    peak_value = float(P_music[peak_idx])

    # Convert DoA to TDoA
    # τ = d * sin(θ) / c
    tau_sec = spacing_m * np.sin(np.deg2rad(doa_deg)) / c
    tau_ms = tau_sec * 1000.0

    # SNR estimation from eigenvalue ratio (in dB)
    snr_eigen = None
    if eigenvalue_ratios:
        mean_ratio = float(np.mean(eigenvalue_ratios))
        if mean_ratio > 1:
            snr_eigen = 10.0 * np.log10(mean_ratio - 1)  # λ1/λ2 - 1 ≈ SNR for single source

    # Spectrum PSR (peak-to-sidelobe ratio)
    spectrum_psr = None
    exc_radius = max(1, int(10 / angle_step))  # Exclude ±10° around peak
    mask = np.ones(n_angles, dtype=bool)
    mask[max(0, peak_idx - exc_radius):min(n_angles, peak_idx + exc_radius + 1)] = False
    sidelobe = P_music[mask]
    if sidelobe.size > 0:
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
# GCC-PHAT for comparison (simplified)
# =============================================================================

def gcc_phat_simple(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fs: int,
    b: np.ndarray,
    a: np.ndarray,
    max_lag_samples: int,
    eps: float = 1e-12,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Simple GCC-PHAT for TDoA estimation.

    Returns:
        (tau_samples, tau_ms) or (None, None) on failure
    """
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return None, None

    x_f = signal.filtfilt(b, a, (x - np.mean(x)).astype(np.float32))
    y_f = signal.filtfilt(b, a, (y - np.mean(y)).astype(np.float32))

    nfft = next_pow2(2 * x_f.size)
    X = np.fft.rfft(x_f, n=nfft)
    Y = np.fft.rfft(y_f, n=nfft)

    R = np.conj(X) * Y
    R = R / (np.abs(R) + eps)
    cc = np.fft.irfft(R, n=nfft)

    cc = np.concatenate([cc[-(nfft // 2):], cc[:nfft // 2]])
    lags = np.arange(-(nfft // 2), nfft // 2, dtype=np.int64)

    # Search within max_lag
    sel = np.abs(lags) <= max_lag_samples
    cc_sel = cc[sel]
    lags_sel = lags[sel]

    abs_cc = np.abs(cc_sel)
    peak_i = int(np.argmax(abs_cc))
    tau_samples = float(lags_sel[peak_i])
    tau_ms = tau_samples * 1000.0 / fs

    return tau_samples, tau_ms


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_pair(
    mic_path: str,
    ldv_path: str,
    *,
    fs: int,
    hop_length: int,
    n_fft: int,
    spacing_m: float,
    freq_min: float,
    freq_max: float,
    angle_min: float,
    angle_max: float,
    angle_step: float,
    c: float,
) -> Dict[str, Any]:
    """Evaluate MUSIC on a single MIC-LDV pair."""

    # Load signals
    mic_signal = load_wav(mic_path, target_fs=fs)
    ldv_signal = load_wav(ldv_path, target_fs=fs)

    # Ensure same length
    min_len = min(len(mic_signal), len(ldv_signal))
    mic_signal = mic_signal[:min_len]
    ldv_signal = ldv_signal[:min_len]

    # Design bandpass filter
    b, a = bandpass_filter(fs, freq_min, freq_max)

    # Window parameters
    win_samples = n_fft * 4  # Use larger window for better frequency resolution

    # Process windows
    results = []
    all_spectra = []
    num_windows = (min_len - win_samples) // hop_length + 1

    for win_idx in range(num_windows):
        start = win_idx * hop_length
        end = start + win_samples

        if end > min_len:
            break

        x_mic = mic_signal[start:end]
        x_ldv = ldv_signal[start:end]

        # MUSIC
        music_result, angles, spectrum = music_2element(
            x_mic, x_ldv,
            fs=fs,
            spacing_m=spacing_m,
            freq_min=freq_min,
            freq_max=freq_max,
            angle_min=angle_min,
            angle_max=angle_max,
            angle_step=angle_step,
            n_fft=n_fft,
            c=c,
            b=b, a=a,
        )

        # GCC-PHAT for comparison (use larger search range for robustness)
        max_lag_samples = int(0.01 * fs)  # ±10ms search range
        gcc_tau_samples, gcc_tau_ms = gcc_phat_simple(
            x_mic, x_ldv, fs=fs, b=b, a=a, max_lag_samples=max_lag_samples
        )

        results.append({
            "window_idx": win_idx,
            "music": asdict(music_result),
            "gcc_phat": {
                "tau_samples": gcc_tau_samples,
                "tau_ms": gcc_tau_ms,
            },
        })

        all_spectra.append(spectrum)

    # Average spectrum across windows
    if all_spectra:
        avg_spectrum = np.mean(all_spectra, axis=0).tolist()
    else:
        avg_spectrum = []

    return {
        "mic_path": mic_path,
        "ldv_path": ldv_path,
        "num_windows": len(results),
        "windows": results,
        "angles_deg": angles.tolist() if len(results) > 0 else [],
        "avg_spectrum": avg_spectrum,
    }


def compute_summary(pair_results: List[Dict], config: Dict) -> Dict[str, Any]:
    """Compute summary statistics."""

    doa_values = []
    tau_values = []
    snr_values = []
    psr_values = []
    gcc_tau_values = []

    for pair in pair_results:
        for win in pair["windows"]:
            music = win["music"]
            if music["doa_deg"] is not None:
                doa_values.append(music["doa_deg"])
            if music["tau_ms"] is not None:
                tau_values.append(music["tau_ms"])
            if music["snr_eigen"] is not None:
                snr_values.append(music["snr_eigen"])
            if music["spectrum_psr"] is not None:
                psr_values.append(music["spectrum_psr"])

            gcc = win["gcc_phat"]
            if gcc["tau_ms"] is not None:
                gcc_tau_values.append(gcc["tau_ms"])

    def stats(arr):
        if len(arr) == 0:
            return {"count": 0, "mean": None, "std": None, "median": None, "p10": None, "p90": None}
        arr = np.array(arr)
        return {
            "count": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }

    summary = {
        "music": {
            "doa_deg": stats(doa_values),
            "tau_ms": stats(tau_values),
            "snr_eigen_db": stats(snr_values),
            "spectrum_psr": stats(psr_values),
        },
        "gcc_phat": {
            "tau_ms": stats(gcc_tau_values),
        },
    }

    # Cross-method comparison
    if tau_values and gcc_tau_values and len(tau_values) == len(gcc_tau_values):
        diff = np.abs(np.array(tau_values) - np.array(gcc_tau_values))
        summary["music_vs_gcc_phat"] = {
            "abs_diff_ms_mean": float(np.mean(diff)),
            "abs_diff_ms_median": float(np.median(diff)),
            "correlation": float(np.corrcoef(tau_values, gcc_tau_values)[0, 1]) if len(tau_values) > 1 else None,
        }

    summary["config"] = config

    return summary


def plot_spectrum(angles_deg: np.ndarray, spectrum: np.ndarray,
                  doa_est: float, out_path: str, title: str = "MUSIC Spatial Spectrum"):
    """Plot and save MUSIC spatial spectrum."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(angles_deg, spectrum, 'b-', linewidth=1.5, label='MUSIC Spectrum')
    ax.axvline(x=doa_est, color='r', linestyle='--', linewidth=1.5,
               label=f'DoA = {doa_est:.1f} deg')

    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Normalized MUSIC Spectrum', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim([angles_deg[0], angles_deg[-1]])
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MUSIC DoA/TDoA Estimation for MIC-LDV")

    # Required arguments
    parser.add_argument("--mic_root", type=str, required=True, help="Path to MIC WAV directory")
    parser.add_argument("--ldv_root", type=str, required=True, help="Path to LDV WAV directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")

    # Mode
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke", "scale", "full"],
                        help="Run mode: smoke (1 pair), scale (48 pairs), full (all pairs)")
    parser.add_argument("--num_pairs", type=int, default=None, help="Override number of pairs")

    # Array geometry
    parser.add_argument("--spacing_m", type=float, default=0.05,
                        help="Sensor spacing in meters (calibrate for your setup!)")
    parser.add_argument("--c", type=float, default=343.0, help="Speed of sound in m/s")

    # Signal processing parameters
    parser.add_argument("--fs", type=int, default=16000, help="Sampling rate")
    parser.add_argument("--hop_length", type=int, default=8000, help="Hop length in samples (500ms default)")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT size for STFT")
    parser.add_argument("--freq_min", type=float, default=300.0, help="Minimum frequency (Hz)")
    parser.add_argument("--freq_max", type=float, default=3000.0, help="Maximum frequency (Hz)")

    # Angle scan parameters
    parser.add_argument("--angle_min", type=float, default=-90.0, help="Minimum scan angle (deg)")
    parser.add_argument("--angle_max", type=float, default=90.0, help="Maximum scan angle (deg)")
    parser.add_argument("--angle_step", type=float, default=1.0, help="Angle step (deg)")

    # Random seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Set random seed
    np.random.seed(args.seed)

    # Print configuration
    print("=" * 60)
    print("MUSIC DoA/TDoA Estimation for MIC-LDV")
    print("=" * 60)
    print(f"MIC root: {args.mic_root}")
    print(f"LDV root: {args.ldv_root}")
    print(f"Output: {args.out_dir}")
    print(f"Mode: {args.mode}")
    print(f"Sensor spacing: {args.spacing_m} m (CALIBRATE FOR YOUR SETUP!)")
    print(f"Speed of sound: {args.c} m/s")
    print(f"fs={args.fs}, hop={args.hop_length}, n_fft={args.n_fft}")
    print(f"freq_band=[{args.freq_min}, {args.freq_max}] Hz")
    print(f"angle_scan=[{args.angle_min}, {args.angle_max}]deg step={args.angle_step}deg")
    print("=" * 60)

    # Discover pairs
    all_pairs = discover_pairs(args.mic_root, args.ldv_root)
    print(f"Found {len(all_pairs)} MIC-LDV pairs")

    if len(all_pairs) == 0:
        print("ERROR: No pairs found!")
        sys.exit(1)

    # Select subset based on mode
    if args.num_pairs is not None:
        num_pairs = min(args.num_pairs, len(all_pairs))
    elif args.mode == "smoke":
        num_pairs = 1
    elif args.mode == "scale":
        num_pairs = min(48, len(all_pairs))
    else:
        num_pairs = len(all_pairs)

    selected_pairs = all_pairs[:num_pairs]
    print(f"Selected {len(selected_pairs)} pairs for evaluation")

    # Config for saving
    config = {
        "fs": args.fs,
        "hop_length": args.hop_length,
        "n_fft": args.n_fft,
        "spacing_m": args.spacing_m,
        "c": args.c,
        "freq_min": args.freq_min,
        "freq_max": args.freq_max,
        "angle_min": args.angle_min,
        "angle_max": args.angle_max,
        "angle_step": args.angle_step,
        "seed": args.seed,
    }

    # Save manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "num_pairs": len(selected_pairs),
        "pairs": [{"mic": p[0], "ldv": p[1]} for p in selected_pairs],
        "config": config,
    }

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")

    # Evaluate all pairs
    all_results = []
    for i, (mic_path, ldv_path) in enumerate(selected_pairs):
        print(f"\nProcessing pair {i+1}/{len(selected_pairs)}: {os.path.basename(mic_path)}")

        result = evaluate_pair(
            mic_path, ldv_path,
            fs=args.fs,
            hop_length=args.hop_length,
            n_fft=args.n_fft,
            spacing_m=args.spacing_m,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            angle_min=args.angle_min,
            angle_max=args.angle_max,
            angle_step=args.angle_step,
            c=args.c,
        )

        all_results.append(result)

        # Print quick summary
        if result["num_windows"] > 0:
            first_win = result["windows"][0]
            music = first_win["music"]
            gcc = first_win["gcc_phat"]
            print(f"  Windows: {result['num_windows']}")
            print(f"  First window:")
            print(f"    MUSIC: DoA={music['doa_deg']:.1f}deg, tau={music['tau_ms']:.3f}ms, SNR={music['snr_eigen']:.1f}dB"
                  if music['snr_eigen'] else f"    MUSIC: DoA={music['doa_deg']:.1f}deg, tau={music['tau_ms']:.3f}ms")
            print(f"    GCC-PHAT: tau={gcc['tau_ms']:.3f}ms" if gcc['tau_ms'] else "    GCC-PHAT: undefined")

        # Plot average spectrum for first pair
        if i == 0 and result["avg_spectrum"] and HAS_MATPLOTLIB:
            angles = np.array(result["angles_deg"])
            spectrum = np.array(result["avg_spectrum"])
            if len(result["windows"]) > 0:
                doa_est = result["windows"][0]["music"]["doa_deg"] or 0.0
                plot_path = os.path.join(args.out_dir, "music_spectrum_sample.png")
                plot_spectrum(angles, spectrum, doa_est, plot_path,
                            title=f"MUSIC Spectrum - {os.path.basename(mic_path)}")
                print(f"  Saved spectrum plot to {plot_path}")

    # Save detailed results
    results_path = os.path.join(args.out_dir, "detailed_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved detailed results to {results_path}")

    # Compute and save summary
    summary = compute_summary(all_results, config)
    summary["num_pairs"] = len(selected_pairs)
    summary["total_windows"] = sum(r["num_windows"] for r in all_results)

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total pairs: {summary['num_pairs']}")
    print(f"Total windows: {summary['total_windows']}")

    m = summary["music"]
    print(f"\nMUSIC:")
    if m["doa_deg"]["count"] > 0:
        print(f"  DoA: median={m['doa_deg']['median']:.1f}deg, mean={m['doa_deg']['mean']:.1f}deg, std={m['doa_deg']['std']:.1f}deg")
        print(f"  tau: median={m['tau_ms']['median']:.3f}ms, mean={m['tau_ms']['mean']:.3f}ms")
    if m["snr_eigen_db"]["count"] > 0:
        print(f"  SNR(eigen): median={m['snr_eigen_db']['median']:.1f}dB, mean={m['snr_eigen_db']['mean']:.1f}dB")
    if m["spectrum_psr"]["count"] > 0:
        print(f"  Spectrum PSR: median={m['spectrum_psr']['median']:.1f}")

    g = summary["gcc_phat"]
    print(f"\nGCC-PHAT (comparison):")
    if g["tau_ms"]["count"] > 0:
        print(f"  tau: median={g['tau_ms']['median']:.3f}ms, mean={g['tau_ms']['mean']:.3f}ms")

    if "music_vs_gcc_phat" in summary:
        comp = summary["music_vs_gcc_phat"]
        print(f"\nMUSIC vs GCC-PHAT:")
        print(f"  |diff_tau|: median={comp['abs_diff_ms_median']:.3f}ms, mean={comp['abs_diff_ms_mean']:.3f}ms")
        if comp["correlation"] is not None:
            print(f"  Correlation: {comp['correlation']:.4f}")

    print("\n" + "=" * 60)
    print("NOTES:")
    print("- DoA accuracy depends on correct 'spacing_m' calibration")
    print("- SNR(eigen) = 10*log10(lambda1/lambda2 - 1), estimates signal-to-noise ratio")
    print("- +/-theta ambiguity exists for linear array (cannot distinguish front/back)")
    print("=" * 60)
    print("DONE")


if __name__ == "__main__":
    main()
