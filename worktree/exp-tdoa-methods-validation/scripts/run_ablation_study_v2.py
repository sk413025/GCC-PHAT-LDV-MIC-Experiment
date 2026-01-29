#!/usr/bin/env python3
"""
Ablation Study V2: Baseline vs Direct Phase Estimation (Correct DTmin)

This script compares delay compensation methods using the CORRECT implementation:
- Method A: Baseline (no compensation)
- Method B: Direct Phase Estimation (correct DTmin - continuous phase)
- Method C: Direct Phase at 48kHz (no resampling)

Key fix: Use continuous phase difference estimation, not discrete lag selection.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
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

def compute_stft(data: np.ndarray, fs: int, n_fft: int = 2048, hop_length: int = 512) -> tuple:
    """Compute STFT of audio signal. Returns (Zxx, freqs)."""
    f, t, Zxx = signal.stft(data, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length)
    return Zxx, f  # (n_freq, n_frames), (n_freq,)


def compute_istft(Zxx: np.ndarray, fs: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Compute inverse STFT."""
    _, data = signal.istft(Zxx, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length)
    return data


# ============================================================================
# GCC-PHAT
# ============================================================================

def gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: int, max_delay_ms: float = 10.0) -> dict:
    """
    Compute GCC-PHAT between two signals.
    Returns dict with tau_ms, peak, psr.
    """
    n = len(sig1) + len(sig2)
    nfft = 2 ** int(np.ceil(np.log2(n)))

    X1 = np.fft.rfft(sig1, n=nfft)
    X2 = np.fft.rfft(sig2, n=nfft)

    R = X1 * np.conj(X2)
    R_phat = R / (np.abs(R) + 1e-12)

    cc = np.fft.irfft(R_phat, n=nfft)
    cc = np.fft.fftshift(cc)

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

    sidelobe_mask = np.ones(len(cc), dtype=bool)
    exclude_half = 50
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
# Method B: Direct Phase Estimation (CORRECT DTmin - at 16kHz for comparison)
# ============================================================================

def estimate_per_frequency_delay(
    X_mic: np.ndarray,
    X_ldv: np.ndarray,
    freqs: np.ndarray,
    freq_min: float = 300.0,
    freq_max: float = 3000.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Estimate per-frequency delay tau(f) using phase difference.

    This is the CORRECT DTmin implementation:
    tau(f) = -phase_diff(f) / (2 * pi * f)

    Returns:
        tau_f: Per-frequency delay in seconds (n_freq,)
    """
    n_freq, n_frames = X_mic.shape
    tau_f = np.zeros(n_freq)

    for f_idx, f in enumerate(freqs):
        if f < freq_min or f > freq_max or f < 1:
            tau_f[f_idx] = 0.0
            continue

        # Cross-spectrum for this frequency
        cross = np.conj(X_mic[f_idx, :]) * X_ldv[f_idx, :]

        # Phase transform (PHAT weighting)
        cross_phat = cross / (np.abs(cross) + eps)

        # Weighted average phase
        weights = np.abs(cross) + eps
        phase_diff = np.angle(cross_phat)
        mean_phase = np.sum(phase_diff * weights) / np.sum(weights)

        # Convert phase to time delay: tau = -phase / (2*pi*f)
        tau_f[f_idx] = -mean_phase / (2 * np.pi * f)

    return tau_f


def apply_phase_compensation(
    X_ldv: np.ndarray,
    tau_f: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """
    Apply phase compensation: Y_comp(f) = Y(f) * exp(+j * 2 * pi * f * tau(f))
    """
    phase_comp = np.exp(1j * 2 * np.pi * freqs * tau_f)
    X_ldv_comp = X_ldv * phase_comp[:, np.newaxis]
    return X_ldv_comp


def method_dtmin_correct(mic_data: np.ndarray, ldv_data: np.ndarray, fs: int,
                         n_fft: int = 2048, hop_length: int = 512,
                         target_fs: int = 16000) -> dict:
    """
    Method B: Correct DTmin implementation with resampling to 16kHz.
    Uses continuous phase difference estimation.
    """
    # Resample to target fs
    mic_resampled = resample_audio(mic_data, fs, target_fs)
    ldv_resampled = resample_audio(ldv_data, fs, target_fs)

    # STFT
    X_mic, freqs = compute_stft(mic_resampled, target_fs, n_fft, hop_length)
    X_ldv, _ = compute_stft(ldv_resampled, target_fs, n_fft, hop_length)

    # Estimate per-frequency delay (CORRECT method)
    tau_f = estimate_per_frequency_delay(X_mic, X_ldv, freqs)

    # Apply phase compensation
    X_ldv_comp = apply_phase_compensation(X_ldv, tau_f, freqs)

    # Convert back to time domain
    ldv_comp = compute_istft(X_ldv_comp, target_fs, n_fft, hop_length)

    # GCC-PHAT on compensated signals
    min_len = min(len(mic_resampled), len(ldv_comp))
    result = gcc_phat(mic_resampled[:min_len], ldv_comp[:min_len], target_fs)

    # Add extra info
    result['resampled_fs'] = target_fs
    result['tau_f_mean_ms'] = float(np.mean(tau_f[(freqs >= 300) & (freqs <= 3000)]) * 1000)
    result['tau_f_std_ms'] = float(np.std(tau_f[(freqs >= 300) & (freqs <= 3000)]) * 1000)

    return result


# ============================================================================
# Method C: Direct Phase at Native 48kHz (No Resampling)
# ============================================================================

def method_dtmin_native(mic_data: np.ndarray, ldv_data: np.ndarray, fs: int,
                        n_fft: int = 4096, hop_length: int = 1024) -> dict:
    """
    Method C: Correct DTmin at native sample rate (no resampling).
    """
    # STFT at native fs
    X_mic, freqs = compute_stft(mic_data, fs, n_fft, hop_length)
    X_ldv, _ = compute_stft(ldv_data, fs, n_fft, hop_length)

    # Estimate per-frequency delay
    tau_f = estimate_per_frequency_delay(X_mic, X_ldv, freqs)

    # Apply phase compensation
    X_ldv_comp = apply_phase_compensation(X_ldv, tau_f, freqs)

    # Convert back to time domain
    ldv_comp = compute_istft(X_ldv_comp, fs, n_fft, hop_length)

    # GCC-PHAT on compensated signals
    min_len = min(len(mic_data), len(ldv_comp))
    result = gcc_phat(mic_data[:min_len], ldv_comp[:min_len], fs)

    # Add extra info
    result['native_fs'] = fs
    result['tau_f_mean_ms'] = float(np.mean(tau_f[(freqs >= 300) & (freqs <= 3000)]) * 1000)
    result['tau_f_std_ms'] = float(np.std(tau_f[(freqs >= 300) & (freqs <= 3000)]) * 1000)

    return result


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(data_dir: str, out_dir: str):
    """Run the ablation study V2 on all data folders."""

    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    folders = ['18-0.1V', '19-0.1V', '20-0.1V', '21-0.1V', '22-0.1V']

    theory_tdoa = {
        '18-0.1V': {'MIC-MIC': 1.450, 'LDV-LEFT': -4.538, 'LDV-RIGHT': -3.088},
        '19-0.1V': {'MIC-MIC': 0.759, 'LDV-LEFT': -4.788, 'LDV-RIGHT': -4.029},
        '20-0.1V': {'MIC-MIC': 0.000, 'LDV-LEFT': -4.720, 'LDV-RIGHT': -4.720},
        '21-0.1V': {'MIC-MIC': -0.759, 'LDV-LEFT': -4.029, 'LDV-RIGHT': -4.788},
        '22-0.1V': {'MIC-MIC': -1.450, 'LDV-LEFT': -3.088, 'LDV-RIGHT': -4.538},
    }

    results = {
        'baseline': [],
        'dtmin_16k': [],
        'dtmin_native': [],
    }

    for folder in folders:
        folder_path = data_path / folder
        if not folder_path.exists():
            logger.warning(f"Folder not found: {folder_path}")
            continue

        logger.info(f"Processing {folder}...")

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

        logger.info(f"  Loading audio files...")
        fs_ldv, ldv_data = load_wav(str(ldv_file))
        fs_left, left_data = load_wav(str(left_file))
        fs_right, right_data = load_wav(str(right_file))

        assert fs_ldv == fs_left == fs_right, "Sample rates must match"
        fs = fs_ldv
        logger.info(f"  Sample rate: {fs} Hz, Duration: {len(ldv_data)/fs:.1f}s")

        min_len = min(len(ldv_data), len(left_data), len(right_data))
        ldv_data = ldv_data[:min_len]
        left_data = left_data[:min_len]
        right_data = right_data[:min_len]

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

            # Method B: DTmin at 16kHz (correct implementation)
            logger.info(f"    Method B: DTmin @ 16kHz (correct)...")
            res_b = method_dtmin_correct(sig1, sig2, fs)
            res_b['folder'] = folder
            res_b['pair'] = pair_name
            res_b['theory_tau_ms'] = theory
            res_b['error_ms'] = abs(res_b['tau_ms'] - theory)
            results['dtmin_16k'].append(res_b)

            # Method C: DTmin at native 48kHz
            logger.info(f"    Method C: DTmin @ 48kHz (native)...")
            res_c = method_dtmin_native(sig1, sig2, fs)
            res_c['folder'] = folder
            res_c['pair'] = pair_name
            res_c['theory_tau_ms'] = theory
            res_c['error_ms'] = abs(res_c['tau_ms'] - theory)
            results['dtmin_native'].append(res_c)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for method, data in results.items():
        out_file = out_path / f"results_v2_{method}_{timestamp}.json"
        with open(out_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {out_file}")

    # Generate summary
    summary = generate_summary(results)
    summary_file = out_path / f"summary_v2_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved {summary_file}")

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
    print("ABLATION STUDY V2 SUMMARY (Correct DTmin Implementation)")
    print("="*70)

    for method, stats in summary.items():
        print(f"\n{method.upper()}")
        print("-"*40)
        print(f"  Count: {stats['count']}")
        print(f"  tau_ms:    mean={stats['tau_ms']['mean']:.2f}, std={stats['tau_ms']['std']:.2f}")
        print(f"  PSR (dB):  mean={stats['psr']['mean']:.2f}, std={stats['psr']['std']:.2f}")
        print(f"  Peak:      mean={stats['peak']['mean']:.4f}, std={stats['peak']['std']:.4f}")
        print(f"  Error (ms): mean={stats['error_ms']['mean']:.2f}, std={stats['error_ms']['std']:.2f}")

    # Comparison
    if 'baseline' in summary and 'dtmin_16k' in summary:
        psr_improve_16k = (summary['dtmin_16k']['psr']['mean'] - summary['baseline']['psr']['mean']) / summary['baseline']['psr']['mean'] * 100
        print(f"\nDTmin@16kHz PSR improvement: {psr_improve_16k:+.1f}%")

    if 'baseline' in summary and 'dtmin_native' in summary:
        psr_improve_native = (summary['dtmin_native']['psr']['mean'] - summary['baseline']['psr']['mean']) / summary['baseline']['psr']['mean'] * 100
        print(f"DTmin@48kHz PSR improvement: {psr_improve_native:+.1f}%")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Ablation Study V2: Correct DTmin")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/GCC-PHAT-LDV-MIC-Experiment",
        help="Path to experiment data directory"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="worktree/exp-tdoa-methods-validation/results/ablation_study_v2",
        help="Path to output directory"
    )
    args = parser.parse_args()

    logger.info("Starting Ablation Study V2 (Correct DTmin)...")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.out_dir}")

    run_experiment(args.data_dir, args.out_dir)

    logger.info("Ablation Study V2 completed!")


if __name__ == "__main__":
    main()
