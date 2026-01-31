#!/usr/bin/env python3
"""
Stage 3 Multi-Segment: Cross-Mic TDoA Evaluation with Multiple Segments

Evaluates TDoA across multiple time segments to ensure robustness.

Author: Stage validation for LDV-to-Mic alignment
Date: 2026-01-30
"""

import numpy as np
import argparse
import json
import os
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import stft, istft
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================
DEFAULT_CONFIG = {
    'fs': 48000,
    'n_fft': 6144,
    'hop_length': 160,
    'max_lag': 50,
    'max_k': 3,
    'tw': 64,
    'freq_min': 100,
    'freq_max': 8000,
    'gcc_max_lag_ms': 10.0,
    'speed_of_sound': 343.0,
    # Multi-segment config
    'n_segments': 10,
    'segment_spacing_sec': 50.0,  # seconds between segment centers
}

GEOMETRY = {
    'ldv': (0.0, 0.5),
    'mic_left': (-0.7, 2.0),
    'mic_right': (0.7, 2.0),
    'speakers': {
        '18': (0.8, 0.0),
        '19': (0.4, 0.0),
        '20': (0.0, 0.0),
        '21': (-0.4, 0.0),
        '22': (-0.8, 0.0),
    }
}


# ==============================================================================
# Utility Functions (same as stage3)
# ==============================================================================
def load_wav(path: str) -> tuple:
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    return sr, data


def normalize_per_freq_maxabs(X_stft: np.ndarray) -> tuple:
    if X_stft.ndim == 2:
        max_abs = np.abs(X_stft).max(axis=-1)
    else:
        max_abs = np.abs(X_stft).max(axis=(-2, -1))
    max_abs = np.where(max_abs < 1e-10, 1.0, max_abs)
    if X_stft.ndim == 2:
        X_norm = X_stft / max_abs[:, np.newaxis]
    else:
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


def omp_single_freq(Dict_f: np.ndarray, Y_f: np.ndarray, max_k: int) -> tuple:
    n_lags, tw = Dict_f.shape
    D = Dict_f.T
    D_norms = np.linalg.norm(np.abs(D), axis=0, keepdims=True) + 1e-10
    D_normalized = D / D_norms
    residual = Y_f.copy()
    selected_lags = []
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


def gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: int, max_lag_samples: int = None) -> tuple:
    n = len(sig1) + len(sig2) - 1
    n_fft = 2 ** int(np.ceil(np.log2(n)))
    S1 = np.fft.fft(sig1, n_fft)
    S2 = np.fft.fft(sig2, n_fft)
    cross = S1 * np.conj(S2)
    cross_phat = cross / (np.abs(cross) + 1e-10)
    gcc = np.fft.ifft(cross_phat).real
    gcc = np.fft.fftshift(gcc)
    lags = np.arange(-n_fft // 2, n_fft // 2)

    if max_lag_samples is not None:
        center = n_fft // 2
        search_start = max(0, center - max_lag_samples)
        search_end = min(len(gcc), center + max_lag_samples + 1)
    else:
        search_start = 0
        search_end = len(gcc)

    search_gcc = gcc[search_start:search_end]
    search_lags = lags[search_start:search_end]

    peak_idx = np.argmax(np.abs(search_gcc))
    peak_val = np.abs(search_gcc[peak_idx])
    tau_samples = search_lags[peak_idx]
    tau = tau_samples / fs

    sidelobe_mask = np.ones(len(search_gcc), dtype=bool)
    peak_region = range(max(0, peak_idx - 5), min(len(search_gcc), peak_idx + 6))
    sidelobe_mask[list(peak_region)] = False

    if sidelobe_mask.sum() > 0:
        sidelobe_max = np.abs(search_gcc[sidelobe_mask]).max()
        psr = 20 * np.log10(peak_val / (sidelobe_max + 1e-10))
    else:
        psr = 0.0

    return tau, psr


def apply_omp_alignment_segment(
    Zxx_ldv: np.ndarray,
    Zxx_mic: np.ndarray,
    config: dict,
    start_t: int
) -> np.ndarray:
    """Apply OMP alignment for a specific segment."""
    n_freq, n_time = Zxx_ldv.shape
    tw = config['tw']
    max_lag = config['max_lag']
    max_k = config['max_k']
    freq_min = config['freq_min']
    freq_max = config['freq_max']
    fs = config['fs']
    n_fft = config['n_fft']

    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    freq_indices = np.where(freq_mask)[0]

    Y_chunk = Zxx_mic[freq_mask, start_t:start_t + tw]
    Y_norm, _ = normalize_per_freq_maxabs(Y_chunk)

    Dict_full = build_lagged_dictionary(Zxx_ldv, max_lag, tw, start_t)
    Dict_selected = Dict_full[freq_mask, :, :]
    Dict_norm, _ = normalize_per_freq_maxabs(Dict_selected)

    n_freq_selected = len(freq_indices)
    Zxx_omp = Zxx_ldv.copy()

    for f_idx in range(n_freq_selected):
        Dict_f = Dict_norm[f_idx]
        Y_f = Y_norm[f_idx]

        selected_lags, _, _ = omp_single_freq(Dict_f, Y_f, max_k)

        D_orig = Dict_selected[f_idx].T
        A = D_orig[:, selected_lags]
        Y_orig = Zxx_mic[freq_indices[f_idx], start_t:start_t + tw]
        coeffs_orig, _, _, _ = np.linalg.lstsq(A, Y_orig, rcond=None)
        reconstructed_orig = A @ coeffs_orig

        Zxx_omp[freq_indices[f_idx], start_t:start_t + tw] = reconstructed_orig

    return Zxx_omp


# ==============================================================================
# Main Multi-Segment Evaluation
# ==============================================================================
def run_multi_segment_evaluation(
    ldv_path: str,
    mic_left_path: str,
    mic_right_path: str,
    config: dict,
    output_dir: str
) -> dict:
    """Run Stage 3 evaluation across multiple segments."""

    logger.info("Loading audio files...")
    sr_ldv, ldv_signal = load_wav(ldv_path)
    sr_left, mic_left_signal = load_wav(mic_left_path)
    sr_right, mic_right_signal = load_wav(mic_right_path)

    assert sr_ldv == sr_left == sr_right == config['fs']

    duration = len(ldv_signal) / sr_ldv
    logger.info(f"Sample rate: {sr_ldv} Hz, Duration: {duration:.2f} s")

    # Compute STFT for all signals
    logger.info("Computing STFT...")

    _, _, Zxx_ldv = stft(ldv_signal, fs=config['fs'],
                         nperseg=config['n_fft'],
                         noverlap=config['n_fft'] - config['hop_length'],
                         window='hann')

    _, _, Zxx_left = stft(mic_left_signal, fs=config['fs'],
                          nperseg=config['n_fft'],
                          noverlap=config['n_fft'] - config['hop_length'],
                          window='hann')

    _, _, Zxx_right = stft(mic_right_signal, fs=config['fs'],
                           nperseg=config['n_fft'],
                           noverlap=config['n_fft'] - config['hop_length'],
                           window='hann')

    n_time = min(Zxx_ldv.shape[1], Zxx_left.shape[1], Zxx_right.shape[1])
    logger.info(f"STFT shape: {Zxx_ldv.shape}")

    # Determine segment positions
    tw = config['tw']
    max_lag = config['max_lag']
    hop = config['hop_length']
    n_segments = config['n_segments']
    segment_spacing = config['segment_spacing_sec']

    # Convert segment spacing to frames
    frames_per_sec = config['fs'] / hop
    segment_spacing_frames = int(segment_spacing * frames_per_sec)

    # Calculate segment start positions (evenly distributed)
    min_start = max_lag + 1
    max_start = n_time - tw - max_lag - 1

    if n_segments == 1:
        segment_starts = [n_time // 2 - tw // 2]
    else:
        # Distribute segments evenly
        total_range = max_start - min_start
        if segment_spacing_frames * (n_segments - 1) > total_range:
            # Not enough space, compress spacing
            segment_spacing_frames = total_range // (n_segments - 1)

        segment_starts = [min_start + i * segment_spacing_frames for i in range(n_segments)]
        segment_starts = [s for s in segment_starts if s < max_start]

    logger.info(f"Evaluating {len(segment_starts)} segments...")

    # Results storage
    results_baseline = []
    results_raw = []
    results_omp = []

    max_lag_samples = int(config['gcc_max_lag_ms'] * config['fs'] / 1000)

    for seg_idx, start_t in enumerate(segment_starts):
        logger.info(f"  Segment {seg_idx+1}/{len(segment_starts)}: start_t={start_t}")

        # Apply OMP alignment for this segment
        Zxx_ldv_as_left = apply_omp_alignment_segment(Zxx_ldv, Zxx_left, config, start_t)

        # Convert to time domain
        _, ldv_raw_td = istft(Zxx_ldv, fs=config['fs'],
                              nperseg=config['n_fft'],
                              noverlap=config['n_fft'] - config['hop_length'],
                              window='hann')

        _, ldv_as_left_td = istft(Zxx_ldv_as_left, fs=config['fs'],
                                   nperseg=config['n_fft'],
                                   noverlap=config['n_fft'] - config['hop_length'],
                                   window='hann')

        # Extract 1-second segment
        center_sample = start_t * hop + config['n_fft'] // 2
        segment_samples = int(1.0 * config['fs'])

        t_start = max(0, center_sample - segment_samples // 2)
        t_end = min(len(ldv_raw_td), t_start + segment_samples)
        t_end = min(len(mic_left_signal), t_end)
        t_end = min(len(mic_right_signal), t_end)

        ldv_raw_seg = ldv_raw_td[t_start:t_end]
        ldv_as_left_seg = ldv_as_left_td[t_start:t_end]
        mic_left_seg = mic_left_signal[t_start:t_end]
        mic_right_seg = mic_right_signal[t_start:t_end]

        # Compute GCC-PHAT
        tau_baseline, psr_baseline = gcc_phat(mic_left_seg, mic_right_seg, config['fs'], max_lag_samples)
        tau_raw, psr_raw = gcc_phat(ldv_raw_seg, mic_right_seg, config['fs'], max_lag_samples)
        tau_omp, psr_omp = gcc_phat(ldv_as_left_seg, mic_right_seg, config['fs'], max_lag_samples)

        results_baseline.append({'tau': float(tau_baseline), 'psr': float(psr_baseline)})
        results_raw.append({'tau': float(tau_raw), 'psr': float(psr_raw)})
        results_omp.append({'tau': float(tau_omp), 'psr': float(psr_omp)})

    # Compute statistics
    tau_baseline_arr = np.array([r['tau'] for r in results_baseline])
    tau_raw_arr = np.array([r['tau'] for r in results_raw])
    tau_omp_arr = np.array([r['tau'] for r in results_omp])

    psr_baseline_arr = np.array([r['psr'] for r in results_baseline])
    psr_raw_arr = np.array([r['psr'] for r in results_raw])
    psr_omp_arr = np.array([r['psr'] for r in results_omp])

    # Error relative to baseline
    error_raw_arr = np.abs(tau_raw_arr - tau_baseline_arr)
    error_omp_arr = np.abs(tau_omp_arr - tau_baseline_arr)

    # Statistics
    stats = {
        'baseline': {
            'tau_median_ms': float(np.median(tau_baseline_arr) * 1000),
            'tau_std_ms': float(np.std(tau_baseline_arr) * 1000),
            'psr_median_db': float(np.median(psr_baseline_arr)),
            'psr_std_db': float(np.std(psr_baseline_arr)),
        },
        'raw_ldv': {
            'tau_median_ms': float(np.median(tau_raw_arr) * 1000),
            'tau_std_ms': float(np.std(tau_raw_arr) * 1000),
            'psr_median_db': float(np.median(psr_raw_arr)),
            'error_median_ms': float(np.median(error_raw_arr) * 1000),
            'error_std_ms': float(np.std(error_raw_arr) * 1000),
        },
        'omp_ldv': {
            'tau_median_ms': float(np.median(tau_omp_arr) * 1000),
            'tau_std_ms': float(np.std(tau_omp_arr) * 1000),
            'psr_median_db': float(np.median(psr_omp_arr)),
            'error_median_ms': float(np.median(error_omp_arr) * 1000),
            'error_std_ms': float(np.std(error_omp_arr) * 1000),
        }
    }

    # Pass conditions
    error_improved = np.median(error_omp_arr) < np.median(error_raw_arr)
    psr_improved = np.median(psr_omp_arr) > np.median(psr_raw_arr)
    error_small = np.median(error_omp_arr) < 0.5e-3

    passed = error_improved and error_small

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'speaker_id': Path(ldv_path).parent.name,
        'n_segments': len(segment_starts),
        'statistics': stats,
        'per_segment': {
            'baseline': results_baseline,
            'raw_ldv': results_raw,
            'omp_ldv': results_omp,
        },
        'pass_conditions': {
            'error_improved': bool(error_improved),
            'psr_improved': bool(psr_improved),
            'error_small': bool(error_small),
            'passed': bool(passed)
        }
    }

    # Print summary
    logger.info("=" * 70)
    logger.info(f"Stage 3 Multi-Segment Results ({len(segment_starts)} segments)")
    logger.info("=" * 70)
    logger.info("GCC-PHAT TDoA (median ± std):")
    logger.info(f"  Baseline (MicL,MicR):    τ = {stats['baseline']['tau_median_ms']:+.3f} ± {stats['baseline']['tau_std_ms']:.3f} ms")
    logger.info(f"  Raw LDV:                 τ = {stats['raw_ldv']['tau_median_ms']:+.3f} ± {stats['raw_ldv']['tau_std_ms']:.3f} ms, error = {stats['raw_ldv']['error_median_ms']:.3f} ms")
    logger.info(f"  OMP LDV:                 τ = {stats['omp_ldv']['tau_median_ms']:+.3f} ± {stats['omp_ldv']['tau_std_ms']:.3f} ms, error = {stats['omp_ldv']['error_median_ms']:.3f} ms")
    logger.info("-" * 70)
    logger.info("PSR (median):")
    logger.info(f"  Baseline: {stats['baseline']['psr_median_db']:.1f} dB")
    logger.info(f"  Raw LDV:  {stats['raw_ldv']['psr_median_db']:.1f} dB")
    logger.info(f"  OMP LDV:  {stats['omp_ldv']['psr_median_db']:.1f} dB")
    logger.info("-" * 70)
    logger.info(f"Error improved:  {'✓ PASS' if error_improved else '✗ FAIL'} ({stats['raw_ldv']['error_median_ms']:.3f} → {stats['omp_ldv']['error_median_ms']:.3f} ms)")
    logger.info(f"PSR improved:    {'✓ PASS' if psr_improved else '✗ FAIL'}")
    logger.info(f"Error < 0.5 ms:  {'✓ PASS' if error_small else '✗ FAIL'} ({stats['omp_ldv']['error_median_ms']:.3f} ms)")
    logger.info("-" * 70)
    logger.info(f"STAGE 3:         {'✓ PASSED' if passed else '✗ FAILED'}")
    logger.info("=" * 70)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'summary.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Stage 3 Multi-Segment TDoA Evaluation')
    parser.add_argument('--data_root', type=str,
                        default='dataset/GCC-PHAT-LDV-MIC-Experiment')
    parser.add_argument('--speaker', type=str, default='20-0.1V')
    parser.add_argument('--output_dir', type=str,
                        default='results/stage3_multi_segment')
    parser.add_argument('--n_segments', type=int, default=10)
    parser.add_argument('--segment_spacing', type=float, default=50.0,
                        help='Spacing between segments in seconds')

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config['n_segments'] = args.n_segments
    config['segment_spacing_sec'] = args.segment_spacing

    data_dir = Path(args.data_root) / args.speaker

    ldv_files = list(data_dir.glob('*LDV*.wav'))
    left_mic_files = list(data_dir.glob('*LEFT*.wav'))
    right_mic_files = list(data_dir.glob('*RIGHT*.wav'))

    ldv_path = str(ldv_files[0])
    mic_left_path = str(left_mic_files[0])
    mic_right_path = str(right_mic_files[0])

    output_dir = Path(args.output_dir) / args.speaker

    results = run_multi_segment_evaluation(
        ldv_path=ldv_path,
        mic_left_path=mic_left_path,
        mic_right_path=mic_right_path,
        config=config,
        output_dir=str(output_dir)
    )

    return results


if __name__ == '__main__':
    main()
