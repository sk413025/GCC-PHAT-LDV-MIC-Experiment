#!/usr/bin/env python3
"""
Stage 4: DoA Multi-Method Validation with Geometric Ground Truth

Goal: Validate DoA estimation accuracy using multiple methods and signal pairings.

Signal Pairings (4):
- (MicL, MicR): Baseline reference
- (Raw_LDV, MicR): Raw LDV signal
- (Random_LDV, MicR): Random-aligned LDV
- (OMP_LDV, MicR): OMP-aligned LDV

DoA Methods (4):
- GCC-PHAT: Phase Transform weighted cross-correlation
- CC: Standard cross-correlation
- NCC: Normalized cross-correlation
- MUSIC: Multiple Signal Classification (subspace method)

Metrics:
- τ: TDoA estimate (ms)
- θ: DoA angle (degrees)
- θ_error: |θ_est - θ_true| (degrees)
- PSR: Peak-to-Sidelobe Ratio (dB)

Author: Stage 4 DoA validation
Date: 2026-01-31
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
# Configuration
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
    # GCC config
    'gcc_max_lag_ms': 10.0,  # ms, for peak search
    # Geometry
    'speed_of_sound': 343.0,  # m/s
    'mic_spacing': 1.4,       # m
    # MUSIC config
    'music_n_sources': 1,
    'music_n_snapshots': 256,
}

# Geometry: sensor positions
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
# Ground Truth Calculation
# ==============================================================================
def compute_ground_truth(speaker_id: str, config: dict) -> dict:
    """
    Compute geometric ground truth TDoA and DoA for a speaker.

    Returns:
        dict with tau_true (s), theta_true (degrees)
    """
    speaker_key = speaker_id.split('-')[0]
    if speaker_key not in GEOMETRY['speakers']:
        logger.warning(f"Unknown speaker {speaker_key}, using speaker 20")
        speaker_key = '20'

    speaker_pos = GEOMETRY['speakers'][speaker_key]
    mic_left = GEOMETRY['mic_left']
    mic_right = GEOMETRY['mic_right']
    c = config['speed_of_sound']
    d = config['mic_spacing']

    # Distances
    d_left = np.sqrt((speaker_pos[0] - mic_left[0])**2 +
                     (speaker_pos[1] - mic_left[1])**2)
    d_right = np.sqrt((speaker_pos[0] - mic_right[0])**2 +
                      (speaker_pos[1] - mic_right[1])**2)

    # TDoA convention for GCC-PHAT(MicL, MicR):
    # - Positive τ: MicL signal leads (sound arrives at MicL first)
    # - Negative τ: MicR signal leads (sound arrives at MicR first)
    # If d_left < d_right: sound arrives at MicL first → τ > 0
    # Formula: τ = (d_right - d_left) / c
    tau_true = (d_right - d_left) / c

    # DoA angle: sin(θ) = τ * c / d
    sin_theta = tau_true * c / d
    sin_theta = np.clip(sin_theta, -1, 1)  # Handle numerical errors
    theta_true = np.degrees(np.arcsin(sin_theta))

    return {
        'tau_true_ms': tau_true * 1000,
        'theta_true_deg': theta_true,
        'd_left': d_left,
        'd_right': d_right,
        'speaker_pos': speaker_pos
    }


# ==============================================================================
# Utility Functions
# ==============================================================================
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
# DoA Estimation Methods
# ==============================================================================
def estimate_tdoa_gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: int,
                           max_lag_samples: int = None) -> dict:
    """GCC-PHAT TDoA estimation."""
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

    # PSR calculation
    sidelobe_mask = np.ones(len(search_gcc), dtype=bool)
    peak_region = range(max(0, peak_idx - 5), min(len(search_gcc), peak_idx + 6))
    sidelobe_mask[list(peak_region)] = False

    if sidelobe_mask.sum() > 0:
        sidelobe_max = np.abs(search_gcc[sidelobe_mask]).max()
        psr = 20 * np.log10(peak_val / (sidelobe_max + 1e-10))
    else:
        psr = 0.0

    return {'tau_ms': tau * 1000, 'psr_db': psr}


def estimate_tdoa_cc(sig1: np.ndarray, sig2: np.ndarray, fs: int,
                     max_lag_samples: int = None) -> dict:
    """Standard cross-correlation TDoA estimation."""
    cc = correlate(sig1, sig2, mode='full')
    lags = np.arange(-len(sig2) + 1, len(sig1))

    if max_lag_samples is not None:
        center = len(sig2) - 1
        search_start = max(0, center - max_lag_samples)
        search_end = min(len(cc), center + max_lag_samples + 1)
    else:
        search_start = 0
        search_end = len(cc)

    search_cc = cc[search_start:search_end]
    search_lags = lags[search_start:search_end]

    peak_idx = np.argmax(np.abs(search_cc))
    peak_val = np.abs(search_cc[peak_idx])
    tau_samples = search_lags[peak_idx]
    tau = tau_samples / fs

    # PSR
    sidelobe_mask = np.ones(len(search_cc), dtype=bool)
    peak_region = range(max(0, peak_idx - 5), min(len(search_cc), peak_idx + 6))
    sidelobe_mask[list(peak_region)] = False

    if sidelobe_mask.sum() > 0:
        sidelobe_max = np.abs(search_cc[sidelobe_mask]).max()
        psr = 20 * np.log10(peak_val / (sidelobe_max + 1e-10))
    else:
        psr = 0.0

    return {'tau_ms': tau * 1000, 'psr_db': psr}


def estimate_tdoa_ncc(sig1: np.ndarray, sig2: np.ndarray, fs: int,
                      max_lag_samples: int = None) -> dict:
    """Normalized cross-correlation TDoA estimation."""
    # Normalize
    sig1_norm = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-10)
    sig2_norm = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-10)

    cc = correlate(sig1_norm, sig2_norm, mode='full')
    cc /= len(sig1)

    lags = np.arange(-len(sig2) + 1, len(sig1))

    if max_lag_samples is not None:
        center = len(sig2) - 1
        search_start = max(0, center - max_lag_samples)
        search_end = min(len(cc), center + max_lag_samples + 1)
    else:
        search_start = 0
        search_end = len(cc)

    search_cc = cc[search_start:search_end]
    search_lags = lags[search_start:search_end]

    peak_idx = np.argmax(np.abs(search_cc))
    peak_val = np.abs(search_cc[peak_idx])
    tau_samples = search_lags[peak_idx]
    tau = tau_samples / fs

    # PSR
    sidelobe_mask = np.ones(len(search_cc), dtype=bool)
    peak_region = range(max(0, peak_idx - 5), min(len(search_cc), peak_idx + 6))
    sidelobe_mask[list(peak_region)] = False

    if sidelobe_mask.sum() > 0:
        sidelobe_max = np.abs(search_cc[sidelobe_mask]).max()
        psr = 20 * np.log10(peak_val / (sidelobe_max + 1e-10))
    else:
        psr = 0.0

    return {'tau_ms': tau * 1000, 'psr_db': psr}


def estimate_tdoa_music(sig1: np.ndarray, sig2: np.ndarray, fs: int, config: dict,
                        max_lag_samples: int = None) -> dict:
    """
    MUSIC-based TDoA estimation.

    Simplified MUSIC for 2-element array using covariance matrix.
    """
    n_sources = config.get('music_n_sources', 1)
    n_snapshots = config.get('music_n_snapshots', 256)

    # Create observation matrix (2 channels x time)
    min_len = min(len(sig1), len(sig2))
    n_frames = min_len // n_snapshots

    if n_frames < 2:
        # Fall back to GCC-PHAT if not enough data
        return estimate_tdoa_gcc_phat(sig1, sig2, fs, max_lag_samples)

    # Build data matrix
    X = np.zeros((2, n_frames * n_snapshots), dtype=complex)

    # Use STFT for frequency-domain processing
    n_fft = 512
    hop = 128

    _, _, S1 = stft(sig1[:n_frames * n_snapshots], fs=fs, nperseg=n_fft, noverlap=n_fft-hop)
    _, _, S2 = stft(sig2[:n_frames * n_snapshots], fs=fs, nperseg=n_fft, noverlap=n_fft-hop)

    # Frequency bins for narrowband MUSIC
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    freq_mask = (freqs >= 500) & (freqs <= 2000)  # Focus on speech frequencies

    # TDoA search grid
    if max_lag_samples is None:
        max_lag_samples = int(10e-3 * fs)  # 10ms

    tau_search = np.linspace(-max_lag_samples/fs, max_lag_samples/fs, 1000)

    # Accumulate MUSIC spectrum across frequencies
    music_spectrum = np.zeros(len(tau_search))

    for freq_idx in np.where(freq_mask)[0]:
        f = freqs[freq_idx]

        # Build covariance matrix for this frequency
        X_f = np.vstack([S1[freq_idx, :], S2[freq_idx, :]])
        R = X_f @ X_f.conj().T / X_f.shape[1]

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R)

        # Noise subspace (smallest eigenvalue)
        noise_idx = np.argsort(eigenvalues)[:2 - n_sources]
        En = eigenvectors[:, noise_idx]

        # MUSIC spectrum
        for tau_idx, tau in enumerate(tau_search):
            # Steering vector for 2-element array
            a = np.array([1, np.exp(-1j * 2 * np.pi * f * tau)])

            # MUSIC pseudo-spectrum
            a = a.reshape(-1, 1)
            denom = a.conj().T @ En @ En.conj().T @ a
            music_spectrum[tau_idx] += 1.0 / (np.abs(denom[0, 0]) + 1e-10)

    # Find peak
    peak_idx = np.argmax(music_spectrum)
    tau_est = tau_search[peak_idx]

    # PSR
    peak_val = music_spectrum[peak_idx]
    sidelobe_mask = np.ones(len(music_spectrum), dtype=bool)
    peak_region = range(max(0, peak_idx - 20), min(len(music_spectrum), peak_idx + 21))
    sidelobe_mask[list(peak_region)] = False

    if sidelobe_mask.sum() > 0:
        sidelobe_max = music_spectrum[sidelobe_mask].max()
        psr = 20 * np.log10(peak_val / (sidelobe_max + 1e-10))
    else:
        psr = 0.0

    return {'tau_ms': tau_est * 1000, 'psr_db': psr}


def tau_to_doa(tau_ms: float, config: dict) -> float:
    """Convert TDoA (ms) to DoA angle (degrees)."""
    c = config['speed_of_sound']
    d = config['mic_spacing']

    sin_theta = (tau_ms / 1000) * c / d
    sin_theta = np.clip(sin_theta, -1, 1)
    theta = np.degrees(np.arcsin(sin_theta))

    return theta


# ==============================================================================
# OMP Alignment (from Stage 2)
# ==============================================================================
def apply_omp_alignment(Zxx_ldv: np.ndarray, Zxx_mic: np.ndarray,
                        config: dict, start_t: int) -> np.ndarray:
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

    Zxx_omp = Zxx_ldv.copy()

    for f_idx in range(len(freq_indices)):
        Dict_f = Dict_norm[f_idx]
        Y_f = Y_norm[f_idx]

        selected_lags, coeffs, _ = omp_single_freq(Dict_f, Y_f, max_k)

        D_orig = Dict_selected[f_idx].T
        A = D_orig[:, selected_lags]
        Y_orig = Zxx_mic[freq_indices[f_idx], start_t:start_t + tw]
        coeffs_orig, _, _, _ = np.linalg.lstsq(A, Y_orig, rcond=None)
        reconstructed_orig = A @ coeffs_orig

        Zxx_omp[freq_indices[f_idx], start_t:start_t + tw] = reconstructed_orig

    return Zxx_omp


def apply_random_alignment(Zxx_ldv: np.ndarray, Zxx_mic: np.ndarray,
                           config: dict, start_t: int) -> np.ndarray:
    """Apply random lag alignment (baseline)."""
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

    n_lags = 2 * max_lag + 1

    Dict_full = build_lagged_dictionary(Zxx_ldv, max_lag, tw, start_t)
    Dict_selected = Dict_full[freq_mask, :, :]

    Zxx_random = Zxx_ldv.copy()

    for f_idx in range(len(freq_indices)):
        # Random lag selection
        selected_lags = np.random.choice(n_lags, max_k, replace=False).tolist()

        D_orig = Dict_selected[f_idx].T
        A = D_orig[:, selected_lags]
        Y_orig = Zxx_mic[freq_indices[f_idx], start_t:start_t + tw]
        coeffs_orig, _, _, _ = np.linalg.lstsq(A, Y_orig, rcond=None)
        reconstructed_orig = A @ coeffs_orig

        Zxx_random[freq_indices[f_idx], start_t:start_t + tw] = reconstructed_orig

    return Zxx_random


# ==============================================================================
# Main Stage 4 Evaluation
# ==============================================================================
def run_stage4_evaluation(
    ldv_path: str,
    mic_left_path: str,
    mic_right_path: str,
    config: dict,
    output_dir: str,
    n_segments: int = 5
) -> dict:
    """
    Run Stage 4 DoA multi-method validation.
    """
    logger.info("=" * 70)
    logger.info("Stage 4: DoA Multi-Method Validation")
    logger.info("=" * 70)

    # Get speaker ID
    speaker_id = Path(ldv_path).parent.name
    logger.info(f"Speaker: {speaker_id}")

    # Compute ground truth
    ground_truth = compute_ground_truth(speaker_id, config)
    tau_true = ground_truth['tau_true_ms']
    theta_true = ground_truth['theta_true_deg']

    logger.info(f"Ground Truth: τ = {tau_true:.3f} ms, θ = {theta_true:.2f}°")
    logger.info(f"Distances: d_left = {ground_truth['d_left']:.3f} m, d_right = {ground_truth['d_right']:.3f} m")

    # Load audio
    logger.info(f"\nLoading audio files...")
    sr_ldv, ldv_signal = load_wav(ldv_path)
    sr_mic_l, mic_left_signal = load_wav(mic_left_path)
    sr_mic_r, mic_right_signal = load_wav(mic_right_path)

    assert sr_ldv == sr_mic_l == sr_mic_r == config['fs']

    logger.info(f"Sample rate: {sr_ldv} Hz")
    logger.info(f"Duration: {len(ldv_signal)/sr_ldv:.2f} s")

    # Compute STFT
    logger.info(f"Computing STFT...")

    _, _, Zxx_ldv = stft(
        ldv_signal, fs=config['fs'],
        nperseg=config['n_fft'],
        noverlap=config['n_fft'] - config['hop_length'],
        window='hann'
    )

    _, _, Zxx_mic_left = stft(
        mic_left_signal, fs=config['fs'],
        nperseg=config['n_fft'],
        noverlap=config['n_fft'] - config['hop_length'],
        window='hann'
    )

    _, _, Zxx_mic_right = stft(
        mic_right_signal, fs=config['fs'],
        nperseg=config['n_fft'],
        noverlap=config['n_fft'] - config['hop_length'],
        window='hann'
    )

    logger.info(f"STFT shape: {Zxx_ldv.shape}")

    # Prepare segments
    n_time = min(Zxx_ldv.shape[1], Zxx_mic_left.shape[1], Zxx_mic_right.shape[1])
    duration_s = len(ldv_signal) / config['fs']
    segment_spacing_frames = int(50 * config['fs'] / config['hop_length'])  # 50s spacing

    segment_starts = []
    for i in range(n_segments):
        start_t = 51 * config['fs'] // config['hop_length'] + i * segment_spacing_frames
        if start_t + config['tw'] < n_time:
            segment_starts.append(start_t)

    if len(segment_starts) < n_segments:
        logger.warning(f"Only {len(segment_starts)} segments available")
        n_segments = len(segment_starts)

    logger.info(f"\nEvaluating {n_segments} segments...")

    # DoA methods
    methods = ['GCC-PHAT', 'CC', 'NCC', 'MUSIC']

    # Signal pairings
    pairings = ['MicL-MicR', 'Raw_LDV', 'Random_LDV', 'OMP_LDV']

    # Results storage
    all_results = {method: {pairing: [] for pairing in pairings} for method in methods}

    max_lag_samples = int(config['gcc_max_lag_ms'] * config['fs'] / 1000)

    for seg_idx, start_t in enumerate(segment_starts):
        logger.info(f"  Segment {seg_idx+1}/{n_segments}: start_t={start_t}")

        # Apply OMP alignment
        Zxx_omp = apply_omp_alignment(Zxx_ldv, Zxx_mic_left, config, start_t)

        # Apply Random alignment
        Zxx_random = apply_random_alignment(Zxx_ldv, Zxx_mic_left, config, start_t)

        # Convert to time domain
        _, ldv_raw_td = istft(Zxx_ldv, fs=config['fs'],
                             nperseg=config['n_fft'],
                             noverlap=config['n_fft'] - config['hop_length'],
                             window='hann')

        _, ldv_omp_td = istft(Zxx_omp, fs=config['fs'],
                             nperseg=config['n_fft'],
                             noverlap=config['n_fft'] - config['hop_length'],
                             window='hann')

        _, ldv_random_td = istft(Zxx_random, fs=config['fs'],
                                nperseg=config['n_fft'],
                                noverlap=config['n_fft'] - config['hop_length'],
                                window='hann')

        # Extract segment for evaluation
        hop = config['hop_length']
        center_sample = start_t * hop + config['n_fft'] // 2
        segment_samples = int(1.0 * config['fs'])  # 1 second

        t_start = max(0, center_sample - segment_samples // 2)
        t_end = min(len(ldv_raw_td), center_sample + segment_samples // 2)
        t_end = min(len(mic_left_signal), t_end)
        t_end = min(len(mic_right_signal), t_end)

        ldv_raw_seg = ldv_raw_td[t_start:t_end]
        ldv_omp_seg = ldv_omp_td[t_start:t_end]
        ldv_random_seg = ldv_random_td[t_start:t_end]
        mic_left_seg = mic_left_signal[t_start:t_end]
        mic_right_seg = mic_right_signal[t_start:t_end]

        # Evaluate each method and pairing
        for method in methods:
            if method == 'GCC-PHAT':
                est_func = estimate_tdoa_gcc_phat
            elif method == 'CC':
                est_func = estimate_tdoa_cc
            elif method == 'NCC':
                est_func = estimate_tdoa_ncc
            elif method == 'MUSIC':
                est_func = lambda s1, s2, fs, mls: estimate_tdoa_music(s1, s2, fs, config, mls)

            # Baseline: MicL-MicR
            result = est_func(mic_left_seg, mic_right_seg, config['fs'], max_lag_samples)
            result['theta_deg'] = tau_to_doa(result['tau_ms'], config)
            result['theta_error'] = abs(result['theta_deg'] - theta_true)
            all_results[method]['MicL-MicR'].append(result)

            # Raw LDV
            result = est_func(ldv_raw_seg, mic_right_seg, config['fs'], max_lag_samples)
            result['theta_deg'] = tau_to_doa(result['tau_ms'], config)
            result['theta_error'] = abs(result['theta_deg'] - theta_true)
            all_results[method]['Raw_LDV'].append(result)

            # Random LDV
            result = est_func(ldv_random_seg, mic_right_seg, config['fs'], max_lag_samples)
            result['theta_deg'] = tau_to_doa(result['tau_ms'], config)
            result['theta_error'] = abs(result['theta_deg'] - theta_true)
            all_results[method]['Random_LDV'].append(result)

            # OMP LDV
            result = est_func(ldv_omp_seg, mic_right_seg, config['fs'], max_lag_samples)
            result['theta_deg'] = tau_to_doa(result['tau_ms'], config)
            result['theta_error'] = abs(result['theta_deg'] - theta_true)
            all_results[method]['OMP_LDV'].append(result)

    # Compute statistics
    logger.info("\n" + "=" * 70)
    logger.info(f"Stage 4 Results: Speaker {speaker_id}")
    logger.info(f"Ground Truth: τ = {tau_true:.3f} ms, θ = {theta_true:.2f}°")
    logger.info("=" * 70)

    summary = {
        'speaker_id': speaker_id,
        'ground_truth': ground_truth,
        'n_segments': n_segments,
        'config': config,
        'results': {}
    }

    for method in methods:
        logger.info(f"\n{method}:")
        logger.info("-" * 60)
        logger.info(f"{'Pairing':<15} {'τ (ms)':<12} {'θ (°)':<10} {'θ_err':<10} {'PSR (dB)':<10}")
        logger.info("-" * 60)

        summary['results'][method] = {}

        for pairing in pairings:
            results_list = all_results[method][pairing]

            tau_median = np.median([r['tau_ms'] for r in results_list])
            tau_std = np.std([r['tau_ms'] for r in results_list])
            theta_median = np.median([r['theta_deg'] for r in results_list])
            theta_err_median = np.median([r['theta_error'] for r in results_list])
            psr_median = np.median([r['psr_db'] for r in results_list])

            logger.info(f"{pairing:<15} {tau_median:+.3f} ± {tau_std:.3f}  {theta_median:+.1f}     {theta_err_median:.1f}      {psr_median:.1f}")

            summary['results'][method][pairing] = {
                'tau_median_ms': float(tau_median),
                'tau_std_ms': float(tau_std),
                'theta_median_deg': float(theta_median),
                'theta_error_median_deg': float(theta_err_median),
                'psr_median_db': float(psr_median),
                'per_segment': [
                    {k: float(v) for k, v in r.items()} for r in results_list
                ]
            }

    # Pass conditions
    logger.info("\n" + "=" * 70)
    logger.info("Pass Conditions:")
    logger.info("=" * 70)

    pass_conditions = {}

    for method in methods:
        omp_err = summary['results'][method]['OMP_LDV']['theta_error_median_deg']
        raw_err = summary['results'][method]['Raw_LDV']['theta_error_median_deg']
        random_err = summary['results'][method]['Random_LDV']['theta_error_median_deg']
        omp_psr = summary['results'][method]['OMP_LDV']['psr_median_db']
        raw_psr = summary['results'][method]['Raw_LDV']['psr_median_db']

        omp_better_than_raw = omp_err < raw_err
        omp_better_than_random = omp_err < random_err
        omp_error_small = omp_err < 5.0  # < 5 degrees
        omp_psr_improved = omp_psr > raw_psr

        pass_conditions[method] = {
            'omp_better_than_raw': bool(omp_better_than_raw),
            'omp_better_than_random': bool(omp_better_than_random),
            'omp_error_small': bool(omp_error_small),
            'omp_psr_improved': bool(omp_psr_improved),
            'passed': bool(omp_better_than_raw and omp_error_small)
        }

        status = "✓ PASS" if pass_conditions[method]['passed'] else "✗ FAIL"
        logger.info(f"{method}: OMP θ_err={omp_err:.1f}° (Raw={raw_err:.1f}°) {status}")

    summary['pass_conditions'] = pass_conditions

    # Overall pass
    overall_passed = all(pc['passed'] for pc in pass_conditions.values())
    logger.info("-" * 60)
    logger.info(f"OVERALL: {'✓ PASSED' if overall_passed else '✗ FAILED'} ({sum(pc['passed'] for pc in pass_conditions.values())}/{len(methods)} methods)")
    logger.info("=" * 70)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'summary.json')
    summary['timestamp'] = datetime.now().isoformat()

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved to: {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Stage 4: DoA Multi-Method Validation')
    parser.add_argument('--data_root', type=str,
                        default='dataset/GCC-PHAT-LDV-MIC-Experiment',
                        help='Root directory of dataset')
    parser.add_argument('--speaker', type=str, default='20-0.1V',
                        help='Speaker position folder')
    parser.add_argument('--output_dir', type=str,
                        default='results/stage4_doa_validation',
                        help='Output directory')
    parser.add_argument('--n_segments', type=int, default=5,
                        help='Number of segments to evaluate')

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()

    # Find files
    data_dir = Path(args.data_root) / args.speaker

    ldv_files = list(data_dir.glob('*LDV*.wav'))
    left_mic_files = list(data_dir.glob('*LEFT*.wav'))
    right_mic_files = list(data_dir.glob('*RIGHT*.wav'))

    if not ldv_files or not left_mic_files or not right_mic_files:
        raise FileNotFoundError(f"Missing audio files in {data_dir}")

    output_dir = Path(args.output_dir) / args.speaker

    results = run_stage4_evaluation(
        ldv_path=str(ldv_files[0]),
        mic_left_path=str(left_mic_files[0]),
        mic_right_path=str(right_mic_files[0]),
        config=config,
        output_dir=str(output_dir),
        n_segments=args.n_segments
    )

    return results


if __name__ == '__main__':
    main()
