#!/usr/bin/env python3
"""
Ablation Study: Baseline vs Direct Phase Estimation vs DTmin (Resampled)

This script compares three delay compensation methods on GCC-PHAT LDV-MIC experiment data.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy import signal
from scipy.io import wavfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Audio I/O
# ============================================================================

def load_wav(path: str) -> tuple:
    """Load WAV file and return (sample_rate, data)."""
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    return sr, data


def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return data
    num_samples = int(len(data) * target_sr / orig_sr)
    return signal.resample(data, num_samples)


# ============================================================================
# STFT
# ============================================================================

def compute_stft(data: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Compute STFT of audio signal."""
    f, t, Zxx = signal.stft(data, nperseg=n_fft, noverlap=n_fft - hop_length)
    return Zxx  # (n_freq, n_frames)


def compute_istft(Zxx: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Compute inverse STFT."""
    _, data = signal.istft(Zxx, nperseg=n_fft, noverlap=n_fft - hop_length)
    return data


# ============================================================================
# GCC-PHAT
# ============================================================================

def gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: int, max_delay_ms: float = 10.0) -> dict:
    """
    Compute GCC-PHAT between two signals.

    Returns dict with tau_ms, peak, psr, correlation.
    """
    n = len(sig1) + len(sig2)
    nfft = 2 ** int(np.ceil(np.log2(n)))

    # FFT
    X1 = np.fft.rfft(sig1, n=nfft)
    X2 = np.fft.rfft(sig2, n=nfft)

    # Cross-spectrum with PHAT weighting
    R = X1 * np.conj(X2)
    R_phat = R / (np.abs(R) + 1e-12)

    # IFFT
    cc = np.fft.irfft(R_phat, n=nfft)
    cc = np.fft.fftshift(cc)

    # Find peak within max delay
    max_delay_samples = int(max_delay_ms * fs / 1000)
    center = len(cc) // 2
    search_start = max(0, center - max_delay_samples)
    search_end = min(len(cc), center + max_delay_samples)

    search_region = cc[search_start:search_end]
    peak_idx_local = np.argmax(np.abs(search_region))
    peak_idx = search_start + peak_idx_local

    tau_samples = peak_idx - center
    tau_ms = tau_samples * 1000 / fs
    peak = np.abs(cc[peak_idx])

    # PSR calculation (exclude main peak region)
    sidelobe_mask = np.ones(len(cc), dtype=bool)
    exclude_half = 50  # samples
    sidelobe_mask[max(0, peak_idx - exclude_half):min(len(cc), peak_idx + exclude_half)] = False
    sidelobes = np.abs(cc[sidelobe_mask])
    max_sidelobe = np.max(sidelobes) if len(sidelobes) > 0 else 1e-12
    psr = 20 * np.log10(peak / (max_sidelobe + 1e-12))

    return {
        'tau_ms': tau_ms,
        'peak': float(peak),
        'psr': float(psr),
    }


# ============================================================================
# Method A: Baseline (No Compensation)
# ============================================================================

def method_baseline(mic_data: np.ndarray, ldv_data: np.ndarray, fs: int) -> dict:
    """Method A: Direct GCC-PHAT without compensation."""
    return gcc_phat(mic_data, ldv_data, fs)


# ============================================================================
# Method B: Direct Phase Estimation
# ============================================================================

def method_direct_phase(mic_data: np.ndarray, ldv_data: np.ndarray, fs: int,
                        n_fft: int = 2048, hop_length: int = 512,
                        freq_min: float = 300, freq_max: float = 3000) -> dict:
    """Method B: Per-frequency phase difference estimation and compensation."""

    # Compute STFT
    X_mic = compute_stft(mic_data, n_fft, hop_length)  # (n_freq, n_frames)
    X_ldv = compute_stft(ldv_data, n_fft, hop_length)

    n_freq, n_frames = X_mic.shape
    freqs = np.fft.rfftfreq(n_fft, 1/fs)[:n_freq]

    # Frequency band mask
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)

    # Estimate per-frequency delay
    # Average phase difference across frames
    cross_spec = np.conj(X_mic) * X_ldv  # (n_freq, n_frames)
    avg_cross_spec = np.mean(cross_spec, axis=1)  # (n_freq,)

    phase_diff = np.angle(avg_cross_spec)  # (n_freq,)

    # Convert phase to delay (avoiding division by zero for DC)
    tau_f = np.zeros(n_freq)
    valid_freq = freqs > 1  # Avoid DC
    tau_f[valid_freq] = -phase_diff[valid_freq] / (2 * np.pi * freqs[valid_freq])

    # Apply phase compensation to LDV
    compensation = np.exp(1j * 2 * np.pi * freqs[:, np.newaxis] * tau_f[:, np.newaxis])
    X_ldv_comp = X_ldv * compensation

    # Convert back to time domain
    ldv_comp = compute_istft(X_ldv_comp, n_fft, hop_length)

    # Truncate to original length
    min_len = min(len(mic_data), len(ldv_comp))

    # GCC-PHAT on compensated signals
    result = gcc_phat(mic_data[:min_len], ldv_comp[:min_len], fs)
    result['tau_f_mean'] = float(np.mean(tau_f[freq_mask]) * 1000)  # ms
    result['tau_f_std'] = float(np.std(tau_f[freq_mask]) * 1000)  # ms

    return result


# ============================================================================
# Method C: DTmin (Resampled) - Simplified version without full DT model
# ============================================================================

def method_dtmin_simplified(mic_data: np.ndarray, ldv_data: np.ndarray, fs: int,
                            target_fs: int = 16000, n_fft: int = 2048,
                            hop_length: int = 256, max_lag: int = 32) -> dict:
    """
    Method C: DTmin-style processing with resampling.

    This is a simplified version that uses OMP-style lag selection
    without the full Decision Transformer model.
    """

    # Resample to target fs
    mic_16k = resample_audio(mic_data, fs, target_fs)
    ldv_16k = resample_audio(ldv_data, fs, target_fs)

    # Compute STFT at 16kHz
    X_mic = compute_stft(mic_16k, n_fft, hop_length)
    X_ldv = compute_stft(ldv_16k, n_fft, hop_length)

    n_freq, n_frames = X_mic.shape
    freqs = np.fft.rfftfreq(n_fft, 1/target_fs)[:n_freq]

    # Simplified OMP-style: find best lag per frequency
    # Build lag dictionary
    lags = np.arange(-max_lag, max_lag + 1)

    best_lags = np.zeros(n_freq, dtype=int)

    for f_idx in range(n_freq):
        best_corr = -1
        best_lag = 0

        for lag in lags:
            if lag >= 0:
                mic_shifted = X_mic[f_idx, lag:]
                ldv_slice = X_ldv[f_idx, :len(mic_shifted)]
            else:
                ldv_shifted = X_ldv[f_idx, -lag:]
                mic_shifted = X_mic[f_idx, :len(ldv_shifted)]
                ldv_slice = ldv_shifted

            if len(mic_shifted) < 10:
                continue

            corr = np.abs(np.vdot(mic_shifted, ldv_slice))
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        best_lags[f_idx] = best_lag

    # Convert lag to time delay
    frame_time = hop_length / target_fs
    tau_f = best_lags * frame_time

    # Apply phase compensation
    compensation = np.exp(1j * 2 * np.pi * freqs[:, np.newaxis] * tau_f[:, np.newaxis])
    X_ldv_comp = X_ldv * compensation

    # Convert back to time domain
    ldv_comp = compute_istft(X_ldv_comp, n_fft, hop_length)

    # Truncate to original length
    min_len = min(len(mic_16k), len(ldv_comp))

    # GCC-PHAT on compensated signals (at 16kHz)
    result = gcc_phat(mic_16k[:min_len], ldv_comp[:min_len], target_fs)

    # Scale tau back to original time scale (no change needed for time)
    result['resampled_fs'] = target_fs
    result['original_fs'] = fs

    return result


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(data_dir: str, out_dir: str):
    """Run the ablation study on all data folders."""

    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Folders with LDV + MIC data (speech)
    folders = ['18-0.1V', '19-0.1V', '20-0.1V', '21-0.1V', '22-0.1V']

    # Theoretical TDOA values (ms)
    theory_tdoa = {
        '18-0.1V': {'MIC-MIC': 1.450, 'LDV-LEFT': -4.538, 'LDV-RIGHT': -3.088},
        '19-0.1V': {'MIC-MIC': 0.759, 'LDV-LEFT': -4.788, 'LDV-RIGHT': -4.029},
        '20-0.1V': {'MIC-MIC': 0.000, 'LDV-LEFT': -4.720, 'LDV-RIGHT': -4.720},
        '21-0.1V': {'MIC-MIC': -0.759, 'LDV-LEFT': -4.029, 'LDV-RIGHT': -4.788},
        '22-0.1V': {'MIC-MIC': -1.450, 'LDV-LEFT': -3.088, 'LDV-RIGHT': -4.538},
    }

    results = {
        'baseline': [],
        'direct_phase': [],
        'dtmin': [],
    }

    for folder in folders:
        folder_path = data_path / folder
        if not folder_path.exists():
            logger.warning(f"Folder not found: {folder_path}")
            continue

        logger.info(f"Processing {folder}...")

        # Find WAV files
        wav_files = list(folder_path.glob("*.wav"))

        ldv_file = None
        left_file = None
        right_file = None

        for f in wav_files:
            fname = f.name.upper()
            if 'LDV' in fname:
                ldv_file = f
            elif 'LEFT' in fname:
                left_file = f
            elif 'RIGHT' in fname:
                right_file = f

        if not all([ldv_file, left_file, right_file]):
            logger.warning(f"Missing files in {folder}")
            continue

        # Load audio
        logger.info(f"  Loading audio files...")
        fs_ldv, ldv_data = load_wav(str(ldv_file))
        fs_left, left_data = load_wav(str(left_file))
        fs_right, right_data = load_wav(str(right_file))

        assert fs_ldv == fs_left == fs_right, "Sample rates must match"
        fs = fs_ldv
        logger.info(f"  Sample rate: {fs} Hz, Duration: {len(ldv_data)/fs:.1f}s")

        # Truncate to same length
        min_len = min(len(ldv_data), len(left_data), len(right_data))
        ldv_data = ldv_data[:min_len]
        left_data = left_data[:min_len]
        right_data = right_data[:min_len]

        # Process each pair
        pairs = [
            ('LDV-LEFT', ldv_data, left_data),
            ('LDV-RIGHT', ldv_data, right_data),
            ('MIC-MIC', left_data, right_data),
        ]

        for pair_name, sig1, sig2 in pairs:
            logger.info(f"  Processing {pair_name}...")

            theory = theory_tdoa.get(folder, {}).get(pair_name, 0)

            # Method A: Baseline
            logger.info(f"    Method A: Baseline...")
            res_a = method_baseline(sig1, sig2, fs)
            res_a['folder'] = folder
            res_a['pair'] = pair_name
            res_a['theory_tau_ms'] = theory
            res_a['error_ms'] = abs(res_a['tau_ms'] - theory)
            results['baseline'].append(res_a)

            # Method B: Direct Phase
            logger.info(f"    Method B: Direct Phase Estimation...")
            res_b = method_direct_phase(sig1, sig2, fs)
            res_b['folder'] = folder
            res_b['pair'] = pair_name
            res_b['theory_tau_ms'] = theory
            res_b['error_ms'] = abs(res_b['tau_ms'] - theory)
            results['direct_phase'].append(res_b)

            # Method C: DTmin (simplified)
            logger.info(f"    Method C: DTmin (Resampled)...")
            res_c = method_dtmin_simplified(sig1, sig2, fs)
            res_c['folder'] = folder
            res_c['pair'] = pair_name
            res_c['theory_tau_ms'] = theory
            res_c['error_ms'] = abs(res_c['tau_ms'] - theory)
            results['dtmin'].append(res_c)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for method, data in results.items():
        out_file = out_path / f"results_{method}_{timestamp}.json"
        with open(out_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {out_file}")

    # Generate summary
    summary = generate_summary(results)
    summary_file = out_path / f"summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved {summary_file}")

    # Print summary
    print_summary(summary)

    return results, summary


def generate_summary(results: dict) -> dict:
    """Generate summary statistics."""
    summary = {}

    for method, data in results.items():
        if not data:
            continue

        taus = [d['tau_ms'] for d in data]
        peaks = [d['peak'] for d in data]
        psrs = [d['psr'] for d in data]
        errors = [d['error_ms'] for d in data]

        summary[method] = {
            'count': len(data),
            'tau_ms': {
                'mean': float(np.mean(taus)),
                'std': float(np.std(taus)),
                'median': float(np.median(taus)),
            },
            'peak': {
                'mean': float(np.mean(peaks)),
                'std': float(np.std(peaks)),
                'median': float(np.median(peaks)),
            },
            'psr': {
                'mean': float(np.mean(psrs)),
                'std': float(np.std(psrs)),
                'median': float(np.median(psrs)),
            },
            'error_ms': {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'median': float(np.median(errors)),
            },
        }

    return summary


def print_summary(summary: dict):
    """Print summary to console."""
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)

    for method, stats in summary.items():
        print(f"\n{method.upper()}")
        print("-"*40)
        print(f"  Count: {stats['count']}")
        print(f"  PSR (dB):  mean={stats['psr']['mean']:.2f}, std={stats['psr']['std']:.2f}")
        print(f"  Peak:      mean={stats['peak']['mean']:.4f}, std={stats['peak']['std']:.4f}")
        print(f"  Error (ms): mean={stats['error_ms']['mean']:.2f}, std={stats['error_ms']['std']:.2f}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Ablation Study: TDOA Methods")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/GCC-PHAT-LDV-MIC-Experiment",
        help="Path to experiment data directory"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="worktree/exp-tdoa-methods-validation/results/ablation_study",
        help="Path to output directory"
    )
    args = parser.parse_args()

    logger.info("Starting Ablation Study...")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.out_dir}")

    run_experiment(args.data_dir, args.out_dir)

    logger.info("Ablation Study completed!")


if __name__ == "__main__":
    main()
