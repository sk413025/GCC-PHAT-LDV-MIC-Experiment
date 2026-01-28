#!/usr/bin/env python3
"""
Cross-Correlation TDoA Estimation for MIC-LDV Speech Data

This script compares different cross-correlation methods for Time Difference of Arrival
(TDoA) estimation between MIC and LDV signals.

Methods implemented:
1. Standard Cross-Correlation (CC)
2. Normalized Cross-Correlation (NCC)
3. GCC-PHAT (baseline from E4m)

Usage:
    python run_cross_correlation_tdoa.py \
        --mic_root "path/to/MIC" \
        --ldv_root "path/to/LDV" \
        --out_dir "results/smoke_xxx" \
        --mode smoke \
        --num_pairs 1

Author: Auto-generated for exp-tdoa-cross-correlation
Date: 2026-01-27
"""

import argparse
import json
import os
import sys
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from scipy import signal
from scipy.io import wavfile


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


def md5_file(path: str) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Cross-Correlation Methods
# =============================================================================

@dataclass
class CCResult:
    """Result of a cross-correlation delay estimation."""
    method: str
    tau_samples: Optional[float]
    tau_ms: Optional[float]
    peak_value: Optional[float]
    psr: Optional[float]  # Peak-to-sidelobe ratio
    boundary_hit: Optional[bool]
    undefined_reason: Optional[str] = None


def standard_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fs: int,
    tau_center_samples: int,
    search_radius_samples: int,
    b: np.ndarray,
    a: np.ndarray,
    exclusion_radius_samples: int = 16,
) -> CCResult:
    """
    Standard Cross-Correlation for delay estimation.

    R_xy[tau] = sum_n { x[n] * y[n + tau] }

    In frequency domain: R_xy = IFFT( conj(FFT(x)) * FFT(y) )

    Sign convention: tau > 0 means y lags x (y[t] ~= x[t - tau])
    """
    if x.size == 0 or y.size == 0:
        return CCResult("cc", None, None, None, None, None, "empty_input")
    if x.size != y.size:
        return CCResult("cc", None, None, None, None, None, f"size_mismatch_{x.size}_{y.size}")

    # Bandpass filter
    x_f = signal.filtfilt(b, a, (x - np.mean(x)).astype(np.float32))
    y_f = signal.filtfilt(b, a, (y - np.mean(y)).astype(np.float32))

    # FFT
    nfft = next_pow2(2 * x_f.size)
    X = np.fft.rfft(x_f, n=nfft)
    Y = np.fft.rfft(y_f, n=nfft)

    # Cross-correlation in frequency domain
    # conj(X) * Y gives positive tau when y lags x
    R = np.conj(X) * Y
    cc = np.fft.irfft(R, n=nfft)

    # Rearrange to center zero lag
    cc = np.concatenate([cc[-(nfft // 2):], cc[:nfft // 2]])
    lags = np.arange(-(nfft // 2), nfft // 2, dtype=np.int64)

    # Search window
    lo = int(tau_center_samples) - int(search_radius_samples)
    hi = int(tau_center_samples) + int(search_radius_samples)
    sel = (lags >= lo) & (lags <= hi)
    if not np.any(sel):
        return CCResult("cc", None, None, None, None, None, "no_valid_lags")

    cc_sel = cc[sel]
    lags_sel = lags[sel]

    # Find peak (use absolute value for robustness)
    abs_cc = np.abs(cc_sel)
    peak_i = int(np.argmax(abs_cc))
    peak_lag = int(lags_sel[peak_i])
    peak_value = float(abs_cc[peak_i])
    boundary_hit = bool(peak_lag == lo or peak_lag == hi)

    # Parabolic interpolation for sub-sample precision
    tau_hat = float(peak_lag)
    if 0 < peak_i < abs_cc.size - 1:
        y1 = float(abs_cc[peak_i - 1])
        y2 = float(abs_cc[peak_i])
        y3 = float(abs_cc[peak_i + 1])
        denom = y1 - 2.0 * y2 + y3
        if abs(denom) > 1e-12:
            offset = 0.5 * (y1 - y3) / denom
            if -1.0 <= offset <= 1.0:
                tau_hat += offset

    # PSR calculation
    exc = int(max(1, exclusion_radius_samples))
    mask = np.ones(abs_cc.size, dtype=bool)
    mask[max(0, peak_i - exc):min(abs_cc.size, peak_i + exc + 1)] = False
    sidelobe = abs_cc[mask]
    psr = None
    if sidelobe.size > 0:
        median_sidelobe = float(np.median(sidelobe))
        if median_sidelobe > 1e-12:
            psr = float(peak_value / median_sidelobe)

    tau_ms = float(tau_hat) * 1000.0 / float(fs)

    return CCResult("cc", float(tau_hat), tau_ms, peak_value, psr, boundary_hit)


def normalized_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fs: int,
    tau_center_samples: int,
    search_radius_samples: int,
    b: np.ndarray,
    a: np.ndarray,
    exclusion_radius_samples: int = 16,
) -> CCResult:
    """
    Normalized Cross-Correlation (NCC).

    NCC[tau] = R_xy[tau] / sqrt(R_xx[0] * R_yy[0])

    Range: [-1, +1]
    """
    if x.size == 0 or y.size == 0:
        return CCResult("ncc", None, None, None, None, None, "empty_input")
    if x.size != y.size:
        return CCResult("ncc", None, None, None, None, None, f"size_mismatch_{x.size}_{y.size}")

    # Bandpass filter
    x_f = signal.filtfilt(b, a, (x - np.mean(x)).astype(np.float32))
    y_f = signal.filtfilt(b, a, (y - np.mean(y)).astype(np.float32))

    # Normalization factor
    energy_x = float(np.sum(x_f ** 2))
    energy_y = float(np.sum(y_f ** 2))
    if energy_x < 1e-12 or energy_y < 1e-12:
        return CCResult("ncc", None, None, None, None, None, "zero_energy")
    norm_factor = np.sqrt(energy_x * energy_y)

    # FFT
    nfft = next_pow2(2 * x_f.size)
    X = np.fft.rfft(x_f, n=nfft)
    Y = np.fft.rfft(y_f, n=nfft)

    # Cross-correlation
    R = np.conj(X) * Y
    cc = np.fft.irfft(R, n=nfft)

    # Normalize
    cc = cc / norm_factor

    # Rearrange
    cc = np.concatenate([cc[-(nfft // 2):], cc[:nfft // 2]])
    lags = np.arange(-(nfft // 2), nfft // 2, dtype=np.int64)

    # Search window
    lo = int(tau_center_samples) - int(search_radius_samples)
    hi = int(tau_center_samples) + int(search_radius_samples)
    sel = (lags >= lo) & (lags <= hi)
    if not np.any(sel):
        return CCResult("ncc", None, None, None, None, None, "no_valid_lags")

    cc_sel = cc[sel]
    lags_sel = lags[sel]

    # Find peak
    abs_cc = np.abs(cc_sel)
    peak_i = int(np.argmax(abs_cc))
    peak_lag = int(lags_sel[peak_i])
    peak_value = float(abs_cc[peak_i])
    boundary_hit = bool(peak_lag == lo or peak_lag == hi)

    # Parabolic interpolation
    tau_hat = float(peak_lag)
    if 0 < peak_i < abs_cc.size - 1:
        y1 = float(abs_cc[peak_i - 1])
        y2 = float(abs_cc[peak_i])
        y3 = float(abs_cc[peak_i + 1])
        denom = y1 - 2.0 * y2 + y3
        if abs(denom) > 1e-12:
            offset = 0.5 * (y1 - y3) / denom
            if -1.0 <= offset <= 1.0:
                tau_hat += offset

    # PSR calculation
    exc = int(max(1, exclusion_radius_samples))
    mask = np.ones(abs_cc.size, dtype=bool)
    mask[max(0, peak_i - exc):min(abs_cc.size, peak_i + exc + 1)] = False
    sidelobe = abs_cc[mask]
    psr = None
    if sidelobe.size > 0:
        median_sidelobe = float(np.median(sidelobe))
        if median_sidelobe > 1e-12:
            psr = float(peak_value / median_sidelobe)

    tau_ms = float(tau_hat) * 1000.0 / float(fs)

    return CCResult("ncc", float(tau_hat), tau_ms, peak_value, psr, boundary_hit)


def gcc_phat(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fs: int,
    tau_center_samples: int,
    search_radius_samples: int,
    b: np.ndarray,
    a: np.ndarray,
    exclusion_radius_samples: int = 16,
    eps: float = 1e-12,
) -> CCResult:
    """
    Generalized Cross-Correlation with Phase Transform (GCC-PHAT).

    R_phat = IFFT( conj(X) * Y / |conj(X) * Y| )

    This is the baseline method from E4m.
    """
    if x.size == 0 or y.size == 0:
        return CCResult("gcc_phat", None, None, None, None, None, "empty_input")
    if x.size != y.size:
        return CCResult("gcc_phat", None, None, None, None, None, f"size_mismatch_{x.size}_{y.size}")

    # Bandpass filter
    x_f = signal.filtfilt(b, a, (x - np.mean(x)).astype(np.float32))
    y_f = signal.filtfilt(b, a, (y - np.mean(y)).astype(np.float32))

    # FFT
    nfft = next_pow2(2 * x_f.size)
    X = np.fft.rfft(x_f, n=nfft)
    Y = np.fft.rfft(y_f, n=nfft)

    # GCC-PHAT: normalize by magnitude (phase transform)
    R = np.conj(X) * Y
    R = R / (np.abs(R) + eps)
    cc = np.fft.irfft(R, n=nfft)

    # Rearrange
    cc = np.concatenate([cc[-(nfft // 2):], cc[:nfft // 2]])
    lags = np.arange(-(nfft // 2), nfft // 2, dtype=np.int64)

    # Search window
    lo = int(tau_center_samples) - int(search_radius_samples)
    hi = int(tau_center_samples) + int(search_radius_samples)
    sel = (lags >= lo) & (lags <= hi)
    if not np.any(sel):
        return CCResult("gcc_phat", None, None, None, None, None, "no_valid_lags")

    cc_sel = cc[sel]
    lags_sel = lags[sel]

    # Find peak
    abs_cc = np.abs(cc_sel)
    peak_i = int(np.argmax(abs_cc))
    peak_lag = int(lags_sel[peak_i])
    peak_value = float(abs_cc[peak_i])
    boundary_hit = bool(peak_lag == lo or peak_lag == hi)

    # Parabolic interpolation
    tau_hat = float(peak_lag)
    if 0 < peak_i < abs_cc.size - 1:
        y1 = float(abs_cc[peak_i - 1])
        y2 = float(abs_cc[peak_i])
        y3 = float(abs_cc[peak_i + 1])
        denom = y1 - 2.0 * y2 + y3
        if abs(denom) > 1e-12:
            offset = 0.5 * (y1 - y3) / denom
            if -1.0 <= offset <= 1.0:
                tau_hat += offset

    # PSR calculation
    exc = int(max(1, exclusion_radius_samples))
    mask = np.ones(abs_cc.size, dtype=bool)
    mask[max(0, peak_i - exc):min(abs_cc.size, peak_i + exc + 1)] = False
    sidelobe = abs_cc[mask]
    psr = None
    if sidelobe.size > 0:
        median_sidelobe = float(np.median(sidelobe))
        if median_sidelobe > 1e-12:
            psr = float(peak_value / median_sidelobe)

    tau_ms = float(tau_hat) * 1000.0 / float(fs)

    return CCResult("gcc_phat", float(tau_hat), tau_ms, peak_value, psr, boundary_hit)


# =============================================================================
# Dataset Handling
# =============================================================================

def discover_pairs(mic_root: str, ldv_root: str, require_wav_only: bool = True) -> List[Tuple[str, str]]:
    """Discover MIC-LDV pairs from directories."""
    mic_path = Path(mic_root)
    ldv_path = Path(ldv_root)

    pairs = []
    for mic_file in sorted(mic_path.glob("*.wav")):
        # Construct expected LDV filename
        ldv_name = mic_file.name.replace("_MIC_", "_LDV_")
        ldv_file = ldv_path / ldv_name

        if ldv_file.exists():
            pairs.append((str(mic_file), str(ldv_file)))

    return pairs


# =============================================================================
# Main Evaluation Loop
# =============================================================================

def evaluate_pair(
    mic_path: str,
    ldv_path: str,
    *,
    fs: int,
    hop_length: int,
    n_fft: int,
    freq_min: float,
    freq_max: float,
    search_radius_frames: int,
    tau_center_samples: int = 0,
) -> Dict[str, Any]:
    """Evaluate all CC methods on a single MIC-LDV pair."""

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
    win_samples = n_fft
    search_radius_samples = search_radius_frames * hop_length

    # Process windows
    results = []
    num_windows = (min_len - win_samples) // hop_length + 1

    for win_idx in range(num_windows):
        start = win_idx * hop_length
        end = start + win_samples

        if end > min_len:
            break

        x = mic_signal[start:end]
        y = ldv_signal[start:end]

        # Run all methods
        cc_result = standard_cross_correlation(
            x, y, fs=fs, tau_center_samples=tau_center_samples,
            search_radius_samples=search_radius_samples, b=b, a=a
        )

        ncc_result = normalized_cross_correlation(
            x, y, fs=fs, tau_center_samples=tau_center_samples,
            search_radius_samples=search_radius_samples, b=b, a=a
        )

        phat_result = gcc_phat(
            x, y, fs=fs, tau_center_samples=tau_center_samples,
            search_radius_samples=search_radius_samples, b=b, a=a
        )

        results.append({
            "window_idx": win_idx,
            "cc": asdict(cc_result),
            "ncc": asdict(ncc_result),
            "gcc_phat": asdict(phat_result),
        })

    return {
        "mic_path": mic_path,
        "ldv_path": ldv_path,
        "num_windows": len(results),
        "windows": results,
    }


def compute_summary(pair_results: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics across all pairs and windows."""

    methods = ["cc", "ncc", "gcc_phat"]
    summary = {}

    for method in methods:
        tau_values = []
        psr_values = []

        for pair in pair_results:
            for win in pair["windows"]:
                result = win[method]
                if result["tau_ms"] is not None:
                    tau_values.append(result["tau_ms"])
                if result["psr"] is not None:
                    psr_values.append(result["psr"])

        tau_arr = np.array(tau_values) if tau_values else np.array([])
        psr_arr = np.array(psr_values) if psr_values else np.array([])

        summary[method] = {
            "num_defined": len(tau_values),
            "tau_ms_mean": float(np.mean(tau_arr)) if len(tau_arr) > 0 else None,
            "tau_ms_std": float(np.std(tau_arr)) if len(tau_arr) > 0 else None,
            "tau_ms_median": float(np.median(tau_arr)) if len(tau_arr) > 0 else None,
            "tau_ms_p10": float(np.percentile(tau_arr, 10)) if len(tau_arr) > 0 else None,
            "tau_ms_p90": float(np.percentile(tau_arr, 90)) if len(tau_arr) > 0 else None,
            "psr_mean": float(np.mean(psr_arr)) if len(psr_arr) > 0 else None,
            "psr_median": float(np.median(psr_arr)) if len(psr_arr) > 0 else None,
            "psr_p50": float(np.percentile(psr_arr, 50)) if len(psr_arr) > 0 else None,
            "psr_p90": float(np.percentile(psr_arr, 90)) if len(psr_arr) > 0 else None,
        }

    # Cross-method comparison
    cc_taus = []
    phat_taus = []
    for pair in pair_results:
        for win in pair["windows"]:
            if win["cc"]["tau_ms"] is not None and win["gcc_phat"]["tau_ms"] is not None:
                cc_taus.append(win["cc"]["tau_ms"])
                phat_taus.append(win["gcc_phat"]["tau_ms"])

    if cc_taus and phat_taus:
        cc_arr = np.array(cc_taus)
        phat_arr = np.array(phat_taus)
        diff = np.abs(cc_arr - phat_arr)
        summary["cc_vs_gcc_phat"] = {
            "num_compared": len(cc_taus),
            "abs_diff_mean_ms": float(np.mean(diff)),
            "abs_diff_median_ms": float(np.median(diff)),
            "abs_diff_p90_ms": float(np.percentile(diff, 90)),
            "correlation": float(np.corrcoef(cc_arr, phat_arr)[0, 1]) if len(cc_arr) > 1 else None,
        }

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-Correlation TDoA Evaluation")

    # Required arguments
    parser.add_argument("--mic_root", type=str, required=True, help="Path to MIC WAV directory")
    parser.add_argument("--ldv_root", type=str, required=True, help="Path to LDV WAV directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")

    # Mode
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke", "scale", "full"],
                        help="Run mode: smoke (1 pair), scale (48 pairs), full (all pairs)")
    parser.add_argument("--num_pairs", type=int, default=None, help="Override number of pairs")

    # Signal processing parameters
    parser.add_argument("--fs", type=int, default=16000, help="Sampling rate")
    parser.add_argument("--hop_length", type=int, default=160, help="Hop length in samples")
    parser.add_argument("--n_fft", type=int, default=2048, help="FFT size")
    parser.add_argument("--freq_min", type=float, default=300.0, help="Minimum frequency (Hz)")
    parser.add_argument("--freq_max", type=float, default=3000.0, help="Maximum frequency (Hz)")
    parser.add_argument("--search_radius_frames", type=int, default=50, help="Search radius in frames")

    # Random seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subset selection")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Set random seed
    np.random.seed(args.seed)

    # Print configuration
    print("=" * 60)
    print("Cross-Correlation TDoA Evaluation")
    print("=" * 60)
    print(f"MIC root: {args.mic_root}")
    print(f"LDV root: {args.ldv_root}")
    print(f"Output: {args.out_dir}")
    print(f"Mode: {args.mode}")
    print(f"fs={args.fs}, hop={args.hop_length}, n_fft={args.n_fft}")
    print(f"freq_band=[{args.freq_min}, {args.freq_max}] Hz")
    print(f"search_radius_frames={args.search_radius_frames}")
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
    else:  # full
        num_pairs = len(all_pairs)

    selected_pairs = all_pairs[:num_pairs]
    print(f"Selected {len(selected_pairs)} pairs for evaluation")

    # Save manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "num_pairs": len(selected_pairs),
        "pairs": [{"mic": p[0], "ldv": p[1]} for p in selected_pairs],
        "config": {
            "fs": args.fs,
            "hop_length": args.hop_length,
            "n_fft": args.n_fft,
            "freq_min": args.freq_min,
            "freq_max": args.freq_max,
            "search_radius_frames": args.search_radius_frames,
            "seed": args.seed,
        }
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
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            search_radius_frames=args.search_radius_frames,
        )

        all_results.append(result)

        # Print quick summary for this pair
        if result["num_windows"] > 0:
            first_win = result["windows"][0]
            print(f"  Windows: {result['num_windows']}")
            print(f"  First window tau_ms: CC={first_win['cc']['tau_ms']:.3f}, "
                  f"NCC={first_win['ncc']['tau_ms']:.3f}, "
                  f"GCC-PHAT={first_win['gcc_phat']['tau_ms']:.3f}")

    # Save detailed results
    results_path = os.path.join(args.out_dir, "detailed_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved detailed results to {results_path}")

    # Compute and save summary
    summary = compute_summary(all_results)
    summary["config"] = manifest["config"]
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

    for method in ["cc", "ncc", "gcc_phat"]:
        m = summary[method]
        print(f"\n{method.upper()}:")
        print(f"  tau_ms: median={m['tau_ms_median']:.4f}, mean={m['tau_ms_mean']:.4f}, std={m['tau_ms_std']:.4f}")
        print(f"  psr: median={m['psr_median']:.2f}, p90={m['psr_p90']:.2f}")

    if "cc_vs_gcc_phat" in summary:
        comp = summary["cc_vs_gcc_phat"]
        print(f"\nCC vs GCC-PHAT:")
        print(f"  abs_diff_ms: median={comp['abs_diff_median_ms']:.4f}, mean={comp['abs_diff_mean_ms']:.4f}")
        print(f"  correlation: {comp['correlation']:.4f}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
