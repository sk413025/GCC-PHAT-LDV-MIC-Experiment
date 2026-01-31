#!/usr/bin/env python3
"""
Stage 3: Cross-Mic TDoA Evaluation

Goal: Verify that OMP-aligned LDV can replace a mic for TDoA estimation.

Signal Pairings:
1. (MicL, MicR) - Baseline (true mic pair)
2. (Raw_LDV, MicR) - LDV at original position
3. (LDV_as_MicL, MicR) - LDV aligned to MicL position

Pass Conditions:
- |τ_OMP - τ_baseline| < |τ_Raw - τ_baseline|
- PSR_OMP > PSR_Raw
- τ_error < 0.5 ms

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
# Utility Functions
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

    return tau, psr, gcc, lags


# ==============================================================================
# OMP Alignment (copied from stage2)
# ==============================================================================
def apply_omp_alignment(
    Zxx_ldv: np.ndarray,
    Zxx_mic: np.ndarray,
    config: dict,
    start_t: int
) -> tuple:
    """Apply OMP alignment using direct reconstruction."""
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
    Y_norm, Y_scale = normalize_per_freq_maxabs(Y_chunk)

    Dict_full = build_lagged_dictionary(Zxx_ldv, max_lag, tw, start_t)
    Dict_selected = Dict_full[freq_mask, :, :]
    Dict_norm, Dict_scale = normalize_per_freq_maxabs(Dict_selected)

    n_freq_selected = len(freq_indices)
    omp_lags_all = []
    dominant_lags = np.zeros(n_freq)

    Zxx_omp = Zxx_ldv.copy()

    for f_idx in range(n_freq_selected):
        Dict_f = Dict_norm[f_idx]
        Y_f = Y_norm[f_idx]

        selected_lags, coeffs, reconstructed_norm = omp_single_freq(Dict_f, Y_f, max_k)
        omp_lags_all.append(selected_lags)

        lag_idx = selected_lags[0]
        lag_value = lag_idx - max_lag
        dominant_lags[freq_indices[f_idx]] = lag_value

        D_orig = Dict_selected[f_idx].T
        A = D_orig[:, selected_lags]
        Y_orig = Zxx_mic[freq_indices[f_idx], start_t:start_t + tw]
        coeffs_orig, _, _, _ = np.linalg.lstsq(A, Y_orig, rcond=None)
        reconstructed_orig = A @ coeffs_orig

        Zxx_omp[freq_indices[f_idx], start_t:start_t + tw] = reconstructed_orig

    return Zxx_omp, omp_lags_all, dominant_lags


# ==============================================================================
# Geometry
# ==============================================================================
def compute_expected_tdoa(speaker_id: str, config: dict) -> float:
    """Compute expected TDoA between MicL and MicR based on geometry."""
    speaker_key = speaker_id.split('-')[0]
    if speaker_key not in GEOMETRY['speakers']:
        speaker_key = '20'

    speaker_pos = GEOMETRY['speakers'][speaker_key]
    mic_l_pos = GEOMETRY['mic_left']
    mic_r_pos = GEOMETRY['mic_right']

    d_left = np.sqrt((speaker_pos[0] - mic_l_pos[0])**2 + (speaker_pos[1] - mic_l_pos[1])**2)
    d_right = np.sqrt((speaker_pos[0] - mic_r_pos[0])**2 + (speaker_pos[1] - mic_r_pos[1])**2)

    c = config['speed_of_sound']
    tdoa = (d_left - d_right) / c  # positive if sound reaches right first

    return tdoa


# ==============================================================================
# Main Stage 3 Evaluation
# ==============================================================================
def run_stage3_evaluation(
    ldv_path: str,
    mic_left_path: str,
    mic_right_path: str,
    config: dict,
    output_dir: str
) -> dict:
    """Run Stage 3 cross-mic TDoA evaluation."""

    logger.info("Loading audio files...")
    sr_ldv, ldv_signal = load_wav(ldv_path)
    sr_left, mic_left_signal = load_wav(mic_left_path)
    sr_right, mic_right_signal = load_wav(mic_right_path)

    assert sr_ldv == sr_left == sr_right == config['fs']

    logger.info(f"Sample rate: {sr_ldv} Hz")
    logger.info(f"Duration: {len(ldv_signal)/sr_ldv:.2f} s")

    # Compute expected TDoA
    speaker_id = Path(ldv_path).parent.name
    expected_tdoa = compute_expected_tdoa(speaker_id, config)
    logger.info(f"Expected TDoA (MicL-MicR): {expected_tdoa*1000:.3f} ms")

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

    logger.info(f"STFT shapes: LDV={Zxx_ldv.shape}, Left={Zxx_left.shape}, Right={Zxx_right.shape}")

    # Select time chunk for alignment
    n_time = min(Zxx_ldv.shape[1], Zxx_left.shape[1], Zxx_right.shape[1])
    tw = config['tw']
    max_lag = config['max_lag']
    start_t = n_time // 2 - tw // 2
    start_t = max(start_t, max_lag + 1)

    logger.info(f"Alignment window: start_t={start_t}, tw={tw}")

    # Apply OMP alignment: LDV → MicL
    logger.info("Applying OMP alignment (LDV → MicL)...")
    Zxx_ldv_as_left, _, _ = apply_omp_alignment(Zxx_ldv, Zxx_left, config, start_t)

    # Convert all signals to time domain
    logger.info("Converting to time domain (ISTFT)...")

    _, ldv_raw_td = istft(Zxx_ldv, fs=config['fs'],
                          nperseg=config['n_fft'],
                          noverlap=config['n_fft'] - config['hop_length'],
                          window='hann')

    _, ldv_as_left_td = istft(Zxx_ldv_as_left, fs=config['fs'],
                               nperseg=config['n_fft'],
                               noverlap=config['n_fft'] - config['hop_length'],
                               window='hann')

    # Extract 1-second segment for comparison
    hop = config['hop_length']
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

    logger.info(f"Segment length: {len(mic_left_seg)} samples ({len(mic_left_seg)/config['fs']*1000:.1f} ms)")

    # Compute GCC-PHAT for all pairings
    logger.info("Computing GCC-PHAT TDoA...")
    max_lag_samples = int(config['gcc_max_lag_ms'] * config['fs'] / 1000)

    # 1. Baseline: MicL vs MicR
    tau_baseline, psr_baseline, _, _ = gcc_phat(mic_left_seg, mic_right_seg, config['fs'], max_lag_samples)

    # 2. Raw LDV vs MicR
    tau_raw, psr_raw, _, _ = gcc_phat(ldv_raw_seg, mic_right_seg, config['fs'], max_lag_samples)

    # 3. LDV_as_MicL vs MicR
    tau_omp, psr_omp, _, _ = gcc_phat(ldv_as_left_seg, mic_right_seg, config['fs'], max_lag_samples)

    # Compute errors
    error_raw = np.abs(tau_raw - tau_baseline)
    error_omp = np.abs(tau_omp - tau_baseline)

    # Check pass conditions
    error_improved = error_omp < error_raw
    psr_improved = psr_omp > psr_raw
    error_small = error_omp < 0.5e-3  # < 0.5 ms

    passed = error_improved and error_small

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'geometry': {
            'speaker_id': speaker_id,
            'expected_tdoa_ms': float(expected_tdoa * 1000)
        },
        'files': {
            'ldv': ldv_path,
            'mic_left': mic_left_path,
            'mic_right': mic_right_path
        },
        'metrics': {
            'baseline': {
                'pairing': '(MicL, MicR)',
                'tau_ms': float(tau_baseline * 1000),
                'psr_db': float(psr_baseline)
            },
            'raw_ldv': {
                'pairing': '(Raw_LDV, MicR)',
                'tau_ms': float(tau_raw * 1000),
                'psr_db': float(psr_raw),
                'error_ms': float(error_raw * 1000)
            },
            'omp_ldv': {
                'pairing': '(LDV_as_MicL, MicR)',
                'tau_ms': float(tau_omp * 1000),
                'psr_db': float(psr_omp),
                'error_ms': float(error_omp * 1000)
            }
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
    logger.info("Stage 3: Cross-Mic TDoA Evaluation Results")
    logger.info("=" * 70)
    logger.info("GCC-PHAT TDoA:")
    logger.info(f"  (MicL, MicR):        τ = {tau_baseline*1000:+.3f} ms, PSR = {psr_baseline:.1f} dB [BASELINE]")
    logger.info(f"  (Raw_LDV, MicR):     τ = {tau_raw*1000:+.3f} ms, PSR = {psr_raw:.1f} dB, error = {error_raw*1000:.3f} ms")
    logger.info(f"  (LDV_as_MicL, MicR): τ = {tau_omp*1000:+.3f} ms, PSR = {psr_omp:.1f} dB, error = {error_omp*1000:.3f} ms")
    logger.info("-" * 70)
    logger.info(f"Expected TDoA: {expected_tdoa*1000:.3f} ms")
    logger.info("-" * 70)
    logger.info(f"Error improved (Raw→OMP):  {'✓ PASS' if error_improved else '✗ FAIL'} ({error_raw*1000:.3f} → {error_omp*1000:.3f} ms)")
    logger.info(f"PSR improved:              {'✓ PASS' if psr_improved else '✗ FAIL'} ({psr_raw:.1f} → {psr_omp:.1f} dB)")
    logger.info(f"Error < 0.5 ms:            {'✓ PASS' if error_small else '✗ FAIL'} ({error_omp*1000:.3f} ms)")
    logger.info("-" * 70)
    logger.info(f"STAGE 3:                   {'✓ PASSED' if passed else '✗ FAILED'}")
    logger.info("=" * 70)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'summary.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Stage 3: Cross-Mic TDoA Evaluation')
    parser.add_argument('--data_root', type=str,
                        default='dataset/GCC-PHAT-LDV-MIC-Experiment')
    parser.add_argument('--speaker', type=str, default='20-0.1V')
    parser.add_argument('--output_dir', type=str,
                        default='results/stage3_tdoa_evaluation')

    parser.add_argument('--n_fft', type=int, default=None)
    parser.add_argument('--max_lag', type=int, default=None)
    parser.add_argument('--max_k', type=int, default=None)
    parser.add_argument('--tw', type=int, default=None)

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.n_fft is not None:
        config['n_fft'] = args.n_fft
    if args.max_lag is not None:
        config['max_lag'] = args.max_lag
    if args.max_k is not None:
        config['max_k'] = args.max_k
    if args.tw is not None:
        config['tw'] = args.tw

    data_dir = Path(args.data_root) / args.speaker

    ldv_files = list(data_dir.glob('*LDV*.wav'))
    left_mic_files = list(data_dir.glob('*LEFT*.wav'))
    right_mic_files = list(data_dir.glob('*RIGHT*.wav'))

    if not ldv_files:
        raise FileNotFoundError(f"No LDV WAV file found in {data_dir}")
    if not left_mic_files:
        raise FileNotFoundError(f"No LEFT MIC WAV file found in {data_dir}")
    if not right_mic_files:
        raise FileNotFoundError(f"No RIGHT MIC WAV file found in {data_dir}")

    ldv_path = str(ldv_files[0])
    mic_left_path = str(left_mic_files[0])
    mic_right_path = str(right_mic_files[0])

    output_dir = Path(args.output_dir) / args.speaker

    results = run_stage3_evaluation(
        ldv_path=ldv_path,
        mic_left_path=mic_left_path,
        mic_right_path=mic_right_path,
        config=config,
        output_dir=str(output_dir)
    )

    return results


if __name__ == '__main__':
    main()
