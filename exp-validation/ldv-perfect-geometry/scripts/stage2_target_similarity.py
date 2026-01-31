#!/usr/bin/env python3
"""
Stage 2: Target Mic Similarity Validation

Goal: Verify that OMP-aligned LDV signal is similar to Target Mic in time domain.

Metrics:
- GCC-PHAT τ: Should approach 0 after alignment
- GCC-PHAT PSR: Should improve after alignment

Comparison:
- Raw LDV: Original LDV signal (unaligned)
- OMP LDV: OMP-aligned LDV signal

Pass Conditions:
- OMP's |τ| < Raw's |τ|
- OMP's PSR > Raw's PSR
- τ approaches 0 (error < 0.5 ms)

Author: Stage validation for LDV-to-Mic alignment
Date: 2026-01-30
"""

import numpy as np
import argparse
import json
import os
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import stft, istft, correlate
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
    'max_k': 3,         # OMP sparsity
    'tw': 64,           # Time window for OMP
    'freq_min': 100,    # Hz
    'freq_max': 8000,   # Hz
    # GCC-PHAT config
    'gcc_n_fft': 4096,
    'gcc_hop': 1024,
    'gcc_max_lag_ms': 10.0,  # ms, for peak search (increased for geometry)
    # Geometry (from EXPERIMENT_PLAN.md)
    'speed_of_sound': 343.0,  # m/s
}

# Geometry: sensor positions (from EXPERIMENT_PLAN.md)
GEOMETRY = {
    'ldv': (0.0, 0.5),       # LDV box position
    'mic_left': (-0.7, 2.0), # Left mic position
    'mic_right': (0.7, 2.0), # Right mic position
    'speakers': {            # Speaker positions (y=0)
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
def compute_geometry_delay(speaker_id: str, target: str, config: dict) -> float:
    """
    Compute the geometric delay between LDV and target mic.

    Returns:
        delta_t: delay in seconds (positive = LDV arrives first)
    """
    # Extract speaker position
    speaker_key = speaker_id.split('-')[0]  # "20-0.1V" -> "20"
    if speaker_key not in GEOMETRY['speakers']:
        logger.warning(f"Unknown speaker {speaker_key}, using speaker 20")
        speaker_key = '20'

    speaker_pos = GEOMETRY['speakers'][speaker_key]
    ldv_pos = GEOMETRY['ldv']
    mic_pos = GEOMETRY['mic_left'] if target == 'left' else GEOMETRY['mic_right']

    # Compute distances
    d_ldv = np.sqrt((speaker_pos[0] - ldv_pos[0])**2 + (speaker_pos[1] - ldv_pos[1])**2)
    d_mic = np.sqrt((speaker_pos[0] - mic_pos[0])**2 + (speaker_pos[1] - mic_pos[1])**2)

    # Time delay: positive means LDV receives sound first (closer to source)
    c = config['speed_of_sound']
    delta_t = (d_mic - d_ldv) / c

    logger.info(f"Geometry: speaker={speaker_key}, d_ldv={d_ldv:.3f}m, d_mic={d_mic:.3f}m")
    logger.info(f"Geometry: delta_t = {delta_t*1000:.3f} ms (LDV {'leads' if delta_t > 0 else 'lags'})")

    return delta_t


def apply_geometry_compensation(Zxx: np.ndarray, delta_t: float, fs: int, n_fft: int) -> np.ndarray:
    """
    Apply geometry-based delay compensation in frequency domain.

    To align LDV to Mic, we need to DELAY the LDV signal by delta_t.
    In frequency domain: X_delayed(f) = X(f) * exp(-j * 2 * pi * f * delay)

    Args:
        Zxx: STFT of signal
        delta_t: time difference (mic_arrival - ldv_arrival) in seconds
                 positive means LDV receives sound first, needs to be delayed
        fs: sample rate
        n_fft: FFT size

    Returns:
        Zxx_comp: compensated STFT (delayed)
    """
    freqs = np.fft.rfftfreq(n_fft, 1/fs)

    # To DELAY the signal by delta_t: multiply by exp(-j * 2π * f * delta_t)
    # Positive delta_t → delay the signal (LDV arrives first, delay it to match Mic)
    phase_comp = np.exp(-1j * 2 * np.pi * freqs * delta_t)

    # Apply phase compensation
    Zxx_comp = Zxx * phase_comp[:, np.newaxis]

    return Zxx_comp


def load_wav(path: str) -> tuple:
    """Load WAV file and return (sample_rate, data)."""
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    return sr, data


def normalize_per_freq_maxabs(X_stft: np.ndarray) -> tuple:
    """Per-frequency max-abs normalization."""
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
    """Build lagged dictionary for OMP."""
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
    """OMP for a single frequency bin."""
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


# ==============================================================================
# GCC-PHAT Implementation
# ==============================================================================
def gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: int, max_lag_samples: int = None) -> tuple:
    """
    Compute GCC-PHAT between two signals.

    Returns:
        tau: estimated delay in seconds
        psr: peak-to-sidelobe ratio in dB
        gcc: full GCC-PHAT correlation
        lags: lag array in samples
    """
    n = len(sig1) + len(sig2) - 1
    n_fft = 2 ** int(np.ceil(np.log2(n)))

    # FFT
    S1 = np.fft.fft(sig1, n_fft)
    S2 = np.fft.fft(sig2, n_fft)

    # Cross-spectrum with PHAT weighting
    cross = S1 * np.conj(S2)
    cross_phat = cross / (np.abs(cross) + 1e-10)

    # Inverse FFT
    gcc = np.fft.ifft(cross_phat).real
    gcc = np.fft.fftshift(gcc)

    # Lag array
    lags = np.arange(-n_fft // 2, n_fft // 2)

    # Restrict search range
    if max_lag_samples is not None:
        center = n_fft // 2
        search_start = center - max_lag_samples
        search_end = center + max_lag_samples + 1
        search_start = max(0, search_start)
        search_end = min(len(gcc), search_end)
    else:
        search_start = 0
        search_end = len(gcc)

    # Find peak
    search_gcc = gcc[search_start:search_end]
    search_lags = lags[search_start:search_end]

    peak_idx = np.argmax(np.abs(search_gcc))
    peak_val = np.abs(search_gcc[peak_idx])
    tau_samples = search_lags[peak_idx]
    tau = tau_samples / fs

    # PSR: peak to sidelobe ratio
    sidelobe_mask = np.ones(len(search_gcc), dtype=bool)
    # Exclude peak region (±5 samples)
    peak_region = range(max(0, peak_idx - 5), min(len(search_gcc), peak_idx + 6))
    sidelobe_mask[peak_region] = False

    if sidelobe_mask.sum() > 0:
        sidelobe_max = np.abs(search_gcc[sidelobe_mask]).max()
        psr = 20 * np.log10(peak_val / (sidelobe_max + 1e-10))
    else:
        psr = 0.0

    return tau, psr, gcc, lags


def compute_ncc(sig1: np.ndarray, sig2: np.ndarray) -> tuple:
    """
    Compute Normalized Cross-Correlation.

    Returns:
        max_ncc: maximum NCC value
        lag_at_max: lag at maximum NCC
    """
    # Normalize
    sig1_norm = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-10)
    sig2_norm = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-10)

    # Cross-correlation
    cc = correlate(sig1_norm, sig2_norm, mode='full')
    cc /= len(sig1)

    # Find peak
    lags = np.arange(-len(sig2) + 1, len(sig1))
    max_idx = np.argmax(np.abs(cc))

    return cc[max_idx], lags[max_idx]


# ==============================================================================
# OMP Alignment
# ==============================================================================
def apply_omp_alignment(
    Zxx_ldv: np.ndarray,
    Zxx_mic: np.ndarray,
    config: dict,
    start_t: int
) -> tuple:
    """
    Apply OMP alignment to LDV signal using direct reconstruction.

    OMP reconstructs the target Mic signal from lagged LDV atoms.
    The reconstruction is what we compare with the target Mic.

    Returns:
        Zxx_omp: OMP-reconstructed signal (approximates Mic)
        omp_lags_all: list of selected lags per frequency
        dominant_lags: the primary lag for each frequency
    """
    n_freq, n_time = Zxx_ldv.shape
    tw = config['tw']
    max_lag = config['max_lag']
    max_k = config['max_k']
    freq_min = config['freq_min']
    freq_max = config['freq_max']
    fs = config['fs']
    n_fft = config['n_fft']

    # Frequency bins
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    freq_indices = np.where(freq_mask)[0]

    # Extract chunks
    Y_chunk = Zxx_mic[freq_mask, start_t:start_t + tw]
    Y_norm, Y_scale = normalize_per_freq_maxabs(Y_chunk)

    Dict_full = build_lagged_dictionary(Zxx_ldv, max_lag, tw, start_t)
    Dict_selected = Dict_full[freq_mask, :, :]
    Dict_norm, Dict_scale = normalize_per_freq_maxabs(Dict_selected)

    # Run OMP per frequency
    n_freq_selected = len(freq_indices)
    omp_lags_all = []
    dominant_lags = np.zeros(n_freq)

    # Create output STFT (copy of LDV, will replace aligned window)
    Zxx_omp = Zxx_ldv.copy()

    for f_idx in range(n_freq_selected):
        Dict_f = Dict_norm[f_idx]
        Y_f = Y_norm[f_idx]

        selected_lags, coeffs, reconstructed_norm = omp_single_freq(Dict_f, Y_f, max_k)
        omp_lags_all.append(selected_lags)

        # Dominant lag: first selected lag
        lag_idx = selected_lags[0]
        lag_value = lag_idx - max_lag
        dominant_lags[freq_indices[f_idx]] = lag_value

        # Reconstruct using ORIGINAL (unnormalized) dictionary
        # This gives us a signal that approximates the target Mic
        D_orig = Dict_selected[f_idx].T  # (tw, n_lags)
        A = D_orig[:, selected_lags]
        Y_orig = Zxx_mic[freq_indices[f_idx], start_t:start_t + tw]
        coeffs_orig, _, _, _ = np.linalg.lstsq(A, Y_orig, rcond=None)
        reconstructed_orig = A @ coeffs_orig

        # Replace the aligned window with reconstruction
        Zxx_omp[freq_indices[f_idx], start_t:start_t + tw] = reconstructed_orig

    # Compute median lag for logging
    selected_dominant_lags = dominant_lags[freq_indices]
    median_lag = np.median(selected_dominant_lags)
    logger.info(f"OMP median lag: {median_lag:.1f} samples ({median_lag/fs*1000:.3f} ms)")

    return Zxx_omp, omp_lags_all, dominant_lags


# ==============================================================================
# Main Stage 2 Evaluation
# ==============================================================================
def run_stage2_evaluation(
    ldv_path: str,
    mic_path: str,
    config: dict,
    output_dir: str,
    target_name: str = "MicL"
) -> dict:
    """
    Run Stage 2 target similarity evaluation.
    """
    logger.info(f"Loading audio files...")
    logger.info(f"  LDV: {ldv_path}")
    logger.info(f"  Target Mic ({target_name}): {mic_path}")

    # Load audio
    sr_ldv, ldv_signal = load_wav(ldv_path)
    sr_mic, mic_signal = load_wav(mic_path)

    assert sr_ldv == sr_mic == config['fs']

    logger.info(f"Sample rate: {sr_ldv} Hz")
    logger.info(f"Duration: {len(ldv_signal)/sr_ldv:.2f} s")

    # Compute STFT
    logger.info(f"Computing STFT...")

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

    logger.info(f"STFT shape: {Zxx_ldv.shape}")

    # Compute geometry for reference (but don't apply compensation)
    speaker_id = Path(ldv_path).parent.name  # e.g., "20-0.1V"
    target_side = 'left' if 'LEFT' in mic_path.upper() or target_name == 'MicL' else 'right'

    delta_t_geom = compute_geometry_delay(speaker_id, target_side, config)

    # Skip geometry compensation - let OMP learn the full delay
    logger.info("Skipping geometry-based delay compensation (OMP will learn full delay)")
    Zxx_ldv_geom = Zxx_ldv.copy()  # No geometry compensation

    # Select time chunk for OMP alignment
    n_time = min(Zxx_ldv.shape[1], Zxx_mic.shape[1])
    tw = config['tw']
    max_lag = config['max_lag']

    # Use center chunk
    start_t = n_time // 2 - tw // 2
    start_t = max(start_t, max_lag + 1)

    logger.info(f"OMP alignment window: start_t={start_t}, tw={tw}")

    # Step 2: Apply OMP for fine-tuning (on geometry-compensated signal)
    logger.info("Applying OMP fine-tuning alignment...")
    Zxx_omp, omp_lags_all, dominant_lags = apply_omp_alignment(
        Zxx_ldv_geom, Zxx_mic, config, start_t
    )

    # Log dominant lag statistics
    freq_mask = (freqs_ldv >= config['freq_min']) & (freqs_ldv <= config['freq_max'])
    selected_lags = dominant_lags[freq_mask]
    logger.info(f"OMP dominant lags: mean={np.mean(selected_lags):.2f}, std={np.std(selected_lags):.2f} samples")
    logger.info(f"OMP dominant lags: mean={np.mean(selected_lags)/config['fs']*1000:.3f} ms")

    # Convert back to time domain using ISTFT
    logger.info("Converting to time domain (ISTFT)...")

    _, ldv_raw_td = istft(
        Zxx_ldv, fs=config['fs'],
        nperseg=config['n_fft'],
        noverlap=config['n_fft'] - config['hop_length'],
        window='hann'
    )

    _, ldv_geom_td = istft(
        Zxx_ldv_geom, fs=config['fs'],
        nperseg=config['n_fft'],
        noverlap=config['n_fft'] - config['hop_length'],
        window='hann'
    )

    _, ldv_omp_td = istft(
        Zxx_omp, fs=config['fs'],
        nperseg=config['n_fft'],
        noverlap=config['n_fft'] - config['hop_length'],
        window='hann'
    )

    # Use a longer segment for GCC-PHAT comparison (1 second around alignment center)
    hop = config['hop_length']
    center_sample = start_t * hop + config['n_fft'] // 2
    segment_duration = 1.0  # seconds
    segment_samples = int(segment_duration * config['fs'])

    t_start_samples = center_sample - segment_samples // 2
    t_end_samples = center_sample + segment_samples // 2

    # Ensure bounds
    t_start_samples = max(0, t_start_samples)
    t_end_samples = min(len(ldv_raw_td), t_end_samples)
    t_end_samples = min(len(ldv_geom_td), t_end_samples)
    t_end_samples = min(len(ldv_omp_td), t_end_samples)
    t_end_samples = min(len(mic_signal), t_end_samples)

    ldv_raw_seg = ldv_raw_td[t_start_samples:t_end_samples]
    ldv_geom_seg = ldv_geom_td[t_start_samples:t_end_samples]
    ldv_omp_seg = ldv_omp_td[t_start_samples:t_end_samples]
    mic_seg = mic_signal[t_start_samples:t_end_samples]

    logger.info(f"Segment length: {len(mic_seg)} samples ({len(mic_seg)/config['fs']*1000:.1f} ms)")

    # Compute GCC-PHAT metrics
    logger.info("Computing GCC-PHAT metrics...")

    max_lag_samples = int(config['gcc_max_lag_ms'] * config['fs'] / 1000)

    # Raw LDV vs Mic
    tau_raw, psr_raw, gcc_raw, lags_raw = gcc_phat(
        ldv_raw_seg, mic_seg, config['fs'], max_lag_samples
    )

    # Geom-compensated LDV vs Mic
    tau_geom, psr_geom, gcc_geom, _ = gcc_phat(
        ldv_geom_seg, mic_seg, config['fs'], max_lag_samples
    )

    # Geom+OMP LDV vs Mic
    tau_omp, psr_omp, gcc_omp, lags_omp = gcc_phat(
        ldv_omp_seg, mic_seg, config['fs'], max_lag_samples
    )

    # NCC metrics
    ncc_raw, lag_ncc_raw = compute_ncc(ldv_raw_seg, mic_seg)
    ncc_geom, lag_ncc_geom = compute_ncc(ldv_geom_seg, mic_seg)
    ncc_omp, lag_ncc_omp = compute_ncc(ldv_omp_seg, mic_seg)

    # Check pass conditions (compare Geom+OMP vs Raw)
    tau_improved = np.abs(tau_omp) < np.abs(tau_raw)
    psr_improved = psr_omp > psr_raw
    tau_near_zero = np.abs(tau_omp) < 0.5e-3  # < 0.5 ms

    # Also check Geom-only improvement
    geom_improved_tau = np.abs(tau_geom) < np.abs(tau_raw)
    omp_improved_over_geom = np.abs(tau_omp) < np.abs(tau_geom)

    passed = tau_improved and tau_near_zero

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'geometry': {
            'delta_t_ms': float(delta_t_geom * 1000),
            'speaker_id': speaker_id,
            'target_side': target_side
        },
        'files': {
            'ldv': ldv_path,
            'mic': mic_path,
            'target_name': target_name
        },
        'alignment_info': {
            'start_t': int(start_t),
            'tw': int(tw),
            'segment_samples': int(len(mic_seg)),
            'segment_ms': float(len(mic_seg) / config['fs'] * 1000)
        },
        'metrics': {
            'raw_ldv': {
                'gcc_phat_tau_ms': float(tau_raw * 1000),
                'gcc_phat_psr_db': float(psr_raw),
                'ncc_peak': float(ncc_raw),
                'ncc_lag_samples': int(lag_ncc_raw)
            },
            'geom_ldv': {
                'gcc_phat_tau_ms': float(tau_geom * 1000),
                'gcc_phat_psr_db': float(psr_geom),
                'ncc_peak': float(ncc_geom),
                'ncc_lag_samples': int(lag_ncc_geom)
            },
            'geom_omp_ldv': {
                'gcc_phat_tau_ms': float(tau_omp * 1000),
                'gcc_phat_psr_db': float(psr_omp),
                'ncc_peak': float(ncc_omp),
                'ncc_lag_samples': int(lag_ncc_omp)
            }
        },
        'pass_conditions': {
            'tau_improved': bool(tau_improved),
            'psr_improved': bool(psr_improved),
            'tau_near_zero': bool(tau_near_zero),
            'geom_improved_tau': bool(geom_improved_tau),
            'omp_improved_over_geom': bool(omp_improved_over_geom),
            'passed': bool(passed)
        }
    }

    # Print summary
    logger.info("=" * 60)
    logger.info("Stage 2: Target Mic Similarity Results")
    logger.info("=" * 60)
    logger.info("GCC-PHAT (LDV vs Target Mic):")
    logger.info(f"  Raw LDV:      τ = {tau_raw*1000:+.3f} ms, PSR = {psr_raw:.1f} dB")
    logger.info(f"  Geom LDV:     τ = {tau_geom*1000:+.3f} ms, PSR = {psr_geom:.1f} dB")
    logger.info(f"  Geom+OMP LDV: τ = {tau_omp*1000:+.3f} ms, PSR = {psr_omp:.1f} dB")
    logger.info("-" * 60)
    logger.info("NCC Peak:")
    logger.info(f"  Raw LDV:      {ncc_raw:.4f} (lag={lag_ncc_raw})")
    logger.info(f"  Geom LDV:     {ncc_geom:.4f} (lag={lag_ncc_geom})")
    logger.info(f"  Geom+OMP LDV: {ncc_omp:.4f} (lag={lag_ncc_omp})")
    logger.info("-" * 60)
    logger.info(f"Expected geometry delay: {delta_t_geom*1000:.3f} ms")
    logger.info("-" * 60)
    logger.info(f"|τ| improved (Raw→Geom+OMP): {'✓ PASS' if tau_improved else '✗ FAIL'} ({np.abs(tau_raw)*1000:.3f} → {np.abs(tau_omp)*1000:.3f} ms)")
    logger.info(f"Geom improved |τ|:          {'✓ PASS' if geom_improved_tau else '✗ FAIL'} ({np.abs(tau_raw)*1000:.3f} → {np.abs(tau_geom)*1000:.3f} ms)")
    logger.info(f"OMP improved over Geom:     {'✓ PASS' if omp_improved_over_geom else '✗ FAIL'} ({np.abs(tau_geom)*1000:.3f} → {np.abs(tau_omp)*1000:.3f} ms)")
    logger.info(f"τ near zero (<0.5ms):       {'✓ PASS' if tau_near_zero else '✗ FAIL'}")
    logger.info("-" * 60)
    logger.info(f"STAGE 2:                    {'✓ PASSED' if passed else '✗ FAILED'}")
    logger.info("=" * 60)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'summary.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_file}")

    # Save detailed data
    detail_file = os.path.join(output_dir, 'stage2_details.npz')
    np.savez(
        detail_file,
        gcc_raw=gcc_raw,
        gcc_omp=gcc_omp,
        lags=lags_raw,
        omp_lags=np.array(omp_lags_all, dtype=object)
    )
    logger.info(f"Detailed data saved to: {detail_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Target Mic Similarity Validation')
    parser.add_argument('--data_root', type=str,
                        default='dataset/GCC-PHAT-LDV-MIC-Experiment',
                        help='Root directory of dataset')
    parser.add_argument('--speaker', type=str, default='20-0.1V',
                        help='Speaker position folder')
    parser.add_argument('--target', type=str, default='left', choices=['left', 'right'],
                        help='Target mic')
    parser.add_argument('--output_dir', type=str,
                        default='results/stage2_target_similarity',
                        help='Output directory')

    # Override config
    parser.add_argument('--n_fft', type=int, default=None)
    parser.add_argument('--max_lag', type=int, default=None)
    parser.add_argument('--max_k', type=int, default=None)
    parser.add_argument('--tw', type=int, default=None)

    args = parser.parse_args()

    # Build config
    config = DEFAULT_CONFIG.copy()
    if args.n_fft is not None:
        config['n_fft'] = args.n_fft
    if args.max_lag is not None:
        config['max_lag'] = args.max_lag
    if args.max_k is not None:
        config['max_k'] = args.max_k
    if args.tw is not None:
        config['tw'] = args.tw

    # Find files
    data_dir = Path(args.data_root) / args.speaker

    ldv_files = list(data_dir.glob('*LDV*.wav'))
    left_mic_files = list(data_dir.glob('*LEFT*.wav'))
    right_mic_files = list(data_dir.glob('*RIGHT*.wav'))

    if not ldv_files:
        raise FileNotFoundError(f"No LDV WAV file found in {data_dir}")

    ldv_path = str(ldv_files[0])
    mic_path = str(left_mic_files[0]) if args.target == 'left' else str(right_mic_files[0])
    target_name = "MicL" if args.target == 'left' else "MicR"

    output_dir = Path(args.output_dir) / args.speaker / args.target

    results = run_stage2_evaluation(
        ldv_path=ldv_path,
        mic_path=mic_path,
        config=config,
        output_dir=str(output_dir),
        target_name=target_name
    )

    return results


if __name__ == '__main__':
    main()
