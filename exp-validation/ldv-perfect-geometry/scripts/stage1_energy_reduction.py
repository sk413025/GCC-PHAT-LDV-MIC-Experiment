#!/usr/bin/env python3
"""
Stage 1: Energy Reduction Validation

Goal: Verify OMP alignment (LDV → Mic) is learning something, not just random guessing.

Comparison:
- OMP: Select best lags via Orthogonal Matching Pursuit
- Random: Randomly select same number of lags as baseline

Pass Condition:
- OMP Energy Reduction > Random Energy Reduction
- Improvement > 10%

Author: Stage validation for LDV-to-Mic alignment
Date: 2026-01-30
"""

import numpy as np
import torch
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
# Configuration (from STAGE_VALIDATION_PLAN.md)
# ==============================================================================
DEFAULT_CONFIG = {
    'fs': 48000,
    'n_fft': 6144,
    'hop_length': 160,
    'max_lag': 50,      # ±50 samples @ 48kHz = ±1.04 ms
    'max_k': 3,         # OMP sparsity (reduced from 16 to avoid overfitting)
    'tw': 64,           # Time window for OMP (increased for harder task)
    'segment_length': 50,   # seconds
    'segment_overlap': 1,   # seconds
    'freq_min': 100,    # Hz (avoid DC)
    'freq_max': 8000,   # Hz
}


# ==============================================================================
# Utility Functions
# ==============================================================================
def load_wav(path: str) -> tuple:
    """Load WAV file and return (sample_rate, data)."""
    sr, data = wavfile.read(path)
    # Convert to float if needed
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    return sr, data


def normalize_per_freq_maxabs(X_stft: np.ndarray) -> tuple:
    """
    Per-frequency max-abs normalization.

    Args:
        X_stft: complex array, shape (n_freq, n_time) or (n_freq, n_lags, n_time)

    Returns:
        X_norm: normalized array, same shape
        scale: normalization scale per frequency, shape (n_freq,)
    """
    if X_stft.ndim == 2:
        # Shape: (n_freq, n_time)
        max_abs = np.abs(X_stft).max(axis=-1)  # shape (n_freq,)
    else:
        # Shape: (n_freq, n_lags, n_time)
        max_abs = np.abs(X_stft).max(axis=(-2, -1))  # shape (n_freq,)

    max_abs = np.where(max_abs < 1e-10, 1.0, max_abs)

    if X_stft.ndim == 2:
        X_norm = X_stft / max_abs[:, np.newaxis]
    else:
        X_norm = X_stft / max_abs[:, np.newaxis, np.newaxis]

    return X_norm, max_abs


def compute_energy_reduction(Y_target: np.ndarray, Y_reconstructed: np.ndarray) -> float:
    """
    Compute energy reduction ratio.

    Args:
        Y_target: target signal (freq domain), shape (n_freq, n_time) or flat
        Y_reconstructed: reconstructed signal, same shape

    Returns:
        reduction: scalar, energy reduction ratio [0, 1]
    """
    residual = Y_target - Y_reconstructed
    E_target = np.sum(np.abs(Y_target)**2)
    E_residual = np.sum(np.abs(residual)**2)

    if E_target < 1e-12:
        return 0.0

    return (E_target - E_residual) / E_target


def build_lagged_dictionary(X_stft: np.ndarray, max_lag: int, tw: int, start_t: int) -> np.ndarray:
    """
    Build lagged dictionary for OMP.

    Args:
        X_stft: LDV STFT, shape (n_freq, n_time)
        max_lag: maximum lag (both positive and negative)
        tw: time window size
        start_t: starting time frame

    Returns:
        Dict_tensor: shape (n_freq, 2*max_lag+1, tw)
    """
    n_freq, n_time = X_stft.shape
    n_lags = 2 * max_lag + 1

    Dict_tensor = np.zeros((n_freq, n_lags, tw), dtype=X_stft.dtype)

    for lag_idx, lag in enumerate(range(-max_lag, max_lag + 1)):
        t_start = start_t + lag
        t_end = t_start + tw

        if t_start >= 0 and t_end <= n_time:
            Dict_tensor[:, lag_idx, :] = X_stft[:, t_start:t_end]
        # else: leave as zeros (edge case handling)

    return Dict_tensor


# ==============================================================================
# OMP Implementation
# ==============================================================================
def omp_single_freq(Dict_f: np.ndarray, Y_f: np.ndarray, max_k: int) -> tuple:
    """
    OMP for a single frequency bin.

    Args:
        Dict_f: dictionary, shape (n_lags, tw)
        Y_f: target, shape (tw,)
        max_k: maximum sparsity

    Returns:
        selected_lags: list of selected lag indices
        coefficients: corresponding coefficients
        reconstructed: reconstructed signal, shape (tw,)
    """
    n_lags, tw = Dict_f.shape

    # D: (tw, n_lags) for standard OMP formulation
    D = Dict_f.T  # shape (tw, n_lags)

    # Normalize dictionary columns (along time axis)
    D_norms = np.linalg.norm(np.abs(D), axis=0, keepdims=True) + 1e-10  # shape (1, n_lags)
    D_normalized = D / D_norms  # shape (tw, n_lags)

    residual = Y_f.copy()
    selected_lags = []
    coeffs = None

    for _ in range(max_k):
        # Compute correlations: (n_lags,)
        corrs = np.abs(D_normalized.conj().T @ residual)

        # Mask already selected lags
        for lag in selected_lags:
            corrs[lag] = -np.inf

        # Select best lag
        best_lag = int(np.argmax(corrs))
        selected_lags.append(best_lag)

        # Solve least squares for all selected lags
        A = D[:, selected_lags]
        coeffs, _, _, _ = np.linalg.lstsq(A, Y_f, rcond=None)

        # Update residual
        reconstructed = A @ coeffs
        residual = Y_f - reconstructed

    return selected_lags, coeffs, reconstructed


def random_single_freq(Dict_f: np.ndarray, Y_f: np.ndarray, max_k: int, rng: np.random.Generator) -> tuple:
    """
    Random baseline for a single frequency bin.

    Args:
        Dict_f: dictionary, shape (n_lags, tw)
        Y_f: target, shape (tw,)
        max_k: maximum sparsity
        rng: random number generator

    Returns:
        selected_lags: list of randomly selected lag indices
        coefficients: corresponding coefficients (via least squares)
        reconstructed: reconstructed signal, shape (tw,)
    """
    n_lags, tw = Dict_f.shape
    D = Dict_f.T  # shape (tw, n_lags)

    # Randomly select lags
    selected_lags = rng.choice(n_lags, size=min(max_k, n_lags), replace=False).tolist()

    # Solve least squares
    A = D[:, selected_lags]
    coeffs, _, _, _ = np.linalg.lstsq(A, Y_f, rcond=None)

    reconstructed = A @ coeffs

    return selected_lags, coeffs, reconstructed


# ==============================================================================
# Main Stage 1 Evaluation
# ==============================================================================
def run_stage1_evaluation(
    ldv_path: str,
    mic_path: str,
    config: dict,
    output_dir: str,
    target_name: str = "MicL"
) -> dict:
    """
    Run Stage 1 energy reduction evaluation.

    Args:
        ldv_path: path to LDV WAV file
        mic_path: path to target mic WAV file
        config: configuration dictionary
        output_dir: output directory
        target_name: name of target mic (for logging)

    Returns:
        results: dictionary with evaluation results
    """
    logger.info(f"Loading audio files...")
    logger.info(f"  LDV: {ldv_path}")
    logger.info(f"  Target Mic ({target_name}): {mic_path}")

    # Load audio
    sr_ldv, ldv_signal = load_wav(ldv_path)
    sr_mic, mic_signal = load_wav(mic_path)

    assert sr_ldv == sr_mic == config['fs'], f"Sample rate mismatch: LDV={sr_ldv}, Mic={sr_mic}, expected={config['fs']}"

    logger.info(f"Sample rate: {sr_ldv} Hz")
    logger.info(f"LDV duration: {len(ldv_signal)/sr_ldv:.2f} s")
    logger.info(f"Mic duration: {len(mic_signal)/sr_mic:.2f} s")

    # Compute STFT
    logger.info(f"Computing STFT (n_fft={config['n_fft']}, hop={config['hop_length']})...")

    freqs_ldv, times_ldv, Zxx_ldv = stft(
        ldv_signal, fs=config['fs'],
        nperseg=config['n_fft'],
        noverlap=config['n_fft'] - config['hop_length'],
        window='hann'
    )

    freqs_mic, times_mic, Zxx_mic = stft(
        mic_signal, fs=config['fs'],
        nperseg=config['n_fft'],
        noverlap=config['n_fft'] - config['hop_length'],
        window='hann'
    )

    logger.info(f"STFT shape: LDV={Zxx_ldv.shape}, Mic={Zxx_mic.shape}")

    # Frequency bin selection
    freq_mask = (freqs_ldv >= config['freq_min']) & (freqs_ldv <= config['freq_max'])
    freq_indices = np.where(freq_mask)[0]
    logger.info(f"Using frequency bins {freq_indices[0]}-{freq_indices[-1]} ({config['freq_min']}-{config['freq_max']} Hz)")

    # Select a segment (middle portion for stability)
    n_time = min(Zxx_ldv.shape[1], Zxx_mic.shape[1])
    tw = config['tw']
    max_lag = config['max_lag']
    max_k = config['max_k']

    # Ensure we have enough frames
    min_frames_needed = 2 * max_lag + tw + 10
    if n_time < min_frames_needed:
        raise ValueError(f"Not enough time frames: {n_time} < {min_frames_needed}")

    # Select center chunk
    start_t = n_time // 2 - tw // 2
    start_t = max(start_t, max_lag + 1)  # Ensure we have room for lags

    logger.info(f"Using time window: start_t={start_t}, tw={tw}")

    # Extract target chunk from Mic
    Y_chunk = Zxx_mic[freq_mask, start_t:start_t + tw]  # shape (n_freq_selected, tw)

    # Normalize target per frequency
    Y_norm, Y_scale = normalize_per_freq_maxabs(Y_chunk)

    # Build lagged dictionary from LDV
    logger.info(f"Building lagged dictionary (max_lag={max_lag})...")
    Dict_full = build_lagged_dictionary(Zxx_ldv, max_lag, tw, start_t)
    Dict_selected = Dict_full[freq_mask, :, :]  # shape (n_freq_selected, 2*max_lag+1, tw)

    # Normalize dictionary per frequency
    Dict_norm, Dict_scale = normalize_per_freq_maxabs(Dict_selected)

    logger.info(f"Dictionary shape: {Dict_norm.shape}")

    # Run OMP and Random for each frequency bin
    n_freq_selected = len(freq_indices)

    omp_reductions = []
    random_reductions = []
    omp_lags_all = []

    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    logger.info(f"Running OMP and Random baseline for {n_freq_selected} frequency bins...")

    for f_idx in range(n_freq_selected):
        Dict_f = Dict_norm[f_idx]  # shape (2*max_lag+1, tw)
        Y_f = Y_norm[f_idx]        # shape (tw,)

        # OMP
        omp_lags, omp_coeffs, omp_recon = omp_single_freq(Dict_f, Y_f, max_k)
        omp_reduction = compute_energy_reduction(Y_f, omp_recon)
        omp_reductions.append(omp_reduction)
        omp_lags_all.append(omp_lags)

        # Random (run multiple times and take mean)
        random_reductions_f = []
        for _ in range(10):
            _, _, random_recon = random_single_freq(Dict_f, Y_f, max_k, rng)
            random_reduction = compute_energy_reduction(Y_f, random_recon)
            random_reductions_f.append(random_reduction)
        random_reductions.append(np.mean(random_reductions_f))

    # Aggregate results
    omp_mean = np.mean(omp_reductions)
    omp_std = np.std(omp_reductions)
    random_mean = np.mean(random_reductions)
    random_std = np.std(random_reductions)

    improvement = omp_mean - random_mean
    improvement_pct = (improvement / random_mean) * 100 if random_mean > 0 else 0

    # Check pass conditions
    passed = (omp_mean > random_mean) and (improvement > 0.10)

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'files': {
            'ldv': ldv_path,
            'mic': mic_path,
            'target_name': target_name
        },
        'stft_info': {
            'n_freq_total': Zxx_ldv.shape[0],
            'n_time_total': n_time,
            'n_freq_selected': n_freq_selected,
            'freq_range': [int(config['freq_min']), int(config['freq_max'])],
            'time_window': {
                'start_t': int(start_t),
                'tw': int(tw)
            }
        },
        'metrics': {
            'omp': {
                'energy_reduction_mean': float(omp_mean),
                'energy_reduction_std': float(omp_std),
                'energy_reduction_min': float(np.min(omp_reductions)),
                'energy_reduction_max': float(np.max(omp_reductions))
            },
            'random': {
                'energy_reduction_mean': float(random_mean),
                'energy_reduction_std': float(random_std),
                'energy_reduction_min': float(np.min(random_reductions)),
                'energy_reduction_max': float(np.max(random_reductions))
            },
            'comparison': {
                'improvement_absolute': float(improvement),
                'improvement_percent': float(improvement_pct)
            }
        },
        'pass_conditions': {
            'omp_gt_random': bool(omp_mean > random_mean),
            'improvement_gt_10pct': bool(improvement > 0.10),
            'passed': bool(passed)
        }
    }

    # Print summary
    logger.info("=" * 60)
    logger.info("Stage 1: Energy Reduction Results")
    logger.info("=" * 60)
    logger.info(f"OMP Energy Reduction:    {omp_mean:.4f} ± {omp_std:.4f}")
    logger.info(f"Random Energy Reduction: {random_mean:.4f} ± {random_std:.4f}")
    logger.info(f"Improvement:             {improvement:.4f} ({improvement_pct:.1f}%)")
    logger.info("-" * 60)
    logger.info(f"OMP > Random:            {'✓ PASS' if results['pass_conditions']['omp_gt_random'] else '✗ FAIL'}")
    logger.info(f"Improvement > 10%:       {'✓ PASS' if results['pass_conditions']['improvement_gt_10pct'] else '✗ FAIL'}")
    logger.info("-" * 60)
    logger.info(f"STAGE 1:                 {'✓ PASSED' if passed else '✗ FAILED'}")
    logger.info("=" * 60)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'summary.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_file}")

    # Save detailed per-frequency data
    detail_file = os.path.join(output_dir, 'per_freq_results.npz')
    np.savez(
        detail_file,
        freq_indices=freq_indices,
        freqs=freqs_ldv[freq_indices],
        omp_reductions=np.array(omp_reductions),
        random_reductions=np.array(random_reductions),
        omp_lags=np.array(omp_lags_all, dtype=object)
    )
    logger.info(f"Per-frequency data saved to: {detail_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Energy Reduction Validation')
    parser.add_argument('--data_root', type=str,
                        default='dataset/GCC-PHAT-LDV-MIC-Experiment',
                        help='Root directory of dataset')
    parser.add_argument('--speaker', type=str, default='20-0.1V',
                        help='Speaker position folder (e.g., 20-0.1V)')
    parser.add_argument('--target', type=str, default='left', choices=['left', 'right'],
                        help='Target mic (left or right)')
    parser.add_argument('--output_dir', type=str,
                        default='results/stage1_energy_reduction',
                        help='Output directory')

    # Override config parameters
    parser.add_argument('--n_fft', type=int, default=None)
    parser.add_argument('--hop_length', type=int, default=None)
    parser.add_argument('--max_lag', type=int, default=None)
    parser.add_argument('--max_k', type=int, default=None)
    parser.add_argument('--tw', type=int, default=None)

    args = parser.parse_args()

    # Build config
    config = DEFAULT_CONFIG.copy()
    if args.n_fft is not None:
        config['n_fft'] = args.n_fft
    if args.hop_length is not None:
        config['hop_length'] = args.hop_length
    if args.max_lag is not None:
        config['max_lag'] = args.max_lag
    if args.max_k is not None:
        config['max_k'] = args.max_k
    if args.tw is not None:
        config['tw'] = args.tw

    # Find files
    data_dir = Path(args.data_root) / args.speaker

    # Find LDV and Mic files
    ldv_files = list(data_dir.glob('*LDV*.wav'))
    left_mic_files = list(data_dir.glob('*LEFT*.wav'))
    right_mic_files = list(data_dir.glob('*RIGHT*.wav'))

    if not ldv_files:
        raise FileNotFoundError(f"No LDV WAV file found in {data_dir}")
    if args.target == 'left' and not left_mic_files:
        raise FileNotFoundError(f"No LEFT MIC WAV file found in {data_dir}")
    if args.target == 'right' and not right_mic_files:
        raise FileNotFoundError(f"No RIGHT MIC WAV file found in {data_dir}")

    ldv_path = str(ldv_files[0])
    mic_path = str(left_mic_files[0]) if args.target == 'left' else str(right_mic_files[0])
    target_name = "MicL" if args.target == 'left' else "MicR"

    # Create output directory with speaker and target info
    output_dir = Path(args.output_dir) / args.speaker / args.target

    # Run evaluation
    results = run_stage1_evaluation(
        ldv_path=ldv_path,
        mic_path=mic_path,
        config=config,
        output_dir=str(output_dir),
        target_name=target_name
    )

    return results


if __name__ == '__main__':
    main()
