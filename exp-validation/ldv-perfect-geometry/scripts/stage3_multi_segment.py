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
from scipy.signal import stft, istft, butter, filtfilt
from scipy.fft import fft, ifft
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
    'segment_offset_sec': 0.0,    # skip initial silence if needed
    # GCC-PHAT config (match full_analysis.py by default)
    'gcc_bandpass_low': 500,
    'gcc_bandpass_high': 2000,
    'gcc_segment_sec': 1.0,
    # To avoid huge full-audio STFTs, analyze a limited time slice per segment
    'analysis_slice_sec': 5.0,
    # Baseline reference (to match GCC-PHAT_LDV_MIC_完整實驗報告.md / full_analysis.py)
    'baseline_method': 'report',  # segment | report | windowed | geometry
    'baseline_start_sec': 100.0,
    'baseline_end_sec': 600.0,
    'baseline_window_sec': 5.0,  # only used when baseline_method=windowed
    'baseline_hop_sec': 5.0,     # only used when baseline_method=windowed
    'baseline_psr_min_db': None,  # only used when baseline_method=windowed
    'baseline_psr_exclude_samples': 50,
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


def extract_centered_slice(signals: list[np.ndarray], center_sample: int, slice_samples: int) -> tuple[int, int]:
    """Return (slice_start, slice_end) centered at `center_sample` within all signals' bounds."""
    if slice_samples <= 0:
        raise ValueError(f"slice_samples must be > 0, got {slice_samples}")
    min_len = min(len(s) for s in signals)
    center_sample = int(np.clip(center_sample, 0, max(0, min_len - 1)))

    slice_start = max(0, center_sample - slice_samples // 2)
    slice_end = slice_start + slice_samples
    if slice_end > min_len:
        slice_end = min_len
        slice_start = max(0, slice_end - slice_samples)
    return slice_start, slice_end


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


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Butterworth bandpass filter with zero-phase filtering."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)


def gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: int, max_lag_samples: int = None,
             bandpass: tuple = None) -> tuple:
    if bandpass is not None:
        sig1 = bandpass_filter(sig1, bandpass[0], bandpass[1], fs)
        sig2 = bandpass_filter(sig2, bandpass[0], bandpass[1], fs)
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


def gcc_phat_full_analysis(
    sig1: np.ndarray,
    sig2: np.ndarray,
    fs: int,
    *,
    max_tau: float | None = None,
    bandpass: tuple[float, float] | None = None,
    psr_exclude_samples: int = 50,
) -> tuple[float, float]:
    """
    GCC-PHAT implementation matching full_analysis.py (parabolic interpolation + PSR exclude ±50 samples).

    Returns:
        tau_sec, psr_db
    """
    if bandpass is not None:
        sig1 = bandpass_filter(sig1, bandpass[0], bandpass[1], fs)
        sig2 = bandpass_filter(sig2, bandpass[0], bandpass[1], fs)

    n = len(sig1) + len(sig2)
    SIG1 = fft(sig1, n)
    SIG2 = fft(sig2, n)
    R = SIG1 * np.conj(SIG2)
    R = R / (np.abs(R) + 1e-10)
    cc = np.real(ifft(R))

    if max_tau is not None:
        max_shift = int(max_tau * fs)
    else:
        max_shift = n // 2

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    abs_cc = np.abs(cc)

    peak_idx = int(np.argmax(abs_cc))
    peak_val = float(abs_cc[peak_idx])

    refined_idx = float(peak_idx)
    if 1 <= peak_idx <= len(abs_cc) - 2:
        y0 = float(abs_cc[peak_idx - 1])
        y1 = float(abs_cc[peak_idx])
        y2 = float(abs_cc[peak_idx + 1])
        denom = 2 * (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            refined_idx += (y0 - y2) / denom

    shift = refined_idx - max_shift
    tau = shift / fs

    sidelobe_mask = np.ones(len(abs_cc), dtype=bool)
    lo = max(0, peak_idx - psr_exclude_samples)
    hi = min(len(abs_cc), peak_idx + psr_exclude_samples + 1)
    sidelobe_mask[lo:hi] = False

    sidelobes = abs_cc[sidelobe_mask]
    if sidelobes.size > 0:
        sidelobe_max = float(np.max(sidelobes))
        psr_db = 20 * np.log10(peak_val / (sidelobe_max + 1e-10))
    else:
        psr_db = 0.0

    return float(tau), float(psr_db)


def estimate_baseline_report(
    mic_left: np.ndarray,
    mic_right: np.ndarray,
    config: dict,
) -> dict:
    """Compute baseline τ using the same long-segment method as full_analysis.py (e.g., 100–600s)."""
    fs = int(config['fs'])
    start_sec = float(config['baseline_start_sec'])
    end_sec = float(config['baseline_end_sec'])

    min_len = min(len(mic_left), len(mic_right))
    start = int(max(0, start_sec * fs))
    end = int(min(min_len, end_sec * fs))

    if end <= start:
        raise ValueError(f"Invalid baseline range: {start_sec}–{end_sec}s (start must be < end)")

    mic_left_seg = mic_left[start:end].astype(np.float64, copy=False)
    mic_right_seg = mic_right[start:end].astype(np.float64, copy=False)

    bp = (float(config['gcc_bandpass_low']), float(config['gcc_bandpass_high']))
    max_tau = float(config['gcc_max_lag_ms']) / 1000.0
    psr_exclude = int(config.get('baseline_psr_exclude_samples', 50))

    tau, psr_db = gcc_phat_full_analysis(
        mic_left_seg,
        mic_right_seg,
        fs,
        max_tau=max_tau,
        bandpass=bp,
        psr_exclude_samples=psr_exclude,
    )

    return {
        'start_sec': float(start / fs),
        'end_sec': float(end / fs),
        'duration_sec': float((end - start) / fs),
        'tau_ms': float(tau * 1000),
        'psr_db': float(psr_db),
    }


def estimate_baseline_windowed(
    mic_left: np.ndarray,
    mic_right: np.ndarray,
    config: dict,
) -> dict:
    """Estimate baseline τ over a long range using multiple windows and take median (optionally PSR-filtered)."""
    fs = int(config['fs'])
    start_sec = float(config['baseline_start_sec'])
    end_sec = float(config['baseline_end_sec'])
    window_sec = float(config['baseline_window_sec'])
    hop_sec = float(config['baseline_hop_sec'])
    psr_min_db = config.get('baseline_psr_min_db', None)

    min_len = min(len(mic_left), len(mic_right))
    start = int(max(0, start_sec * fs))
    end = int(min(min_len, end_sec * fs))
    window = int(max(1, window_sec * fs))
    hop = int(max(1, hop_sec * fs))

    if end <= start:
        raise ValueError(f"Invalid baseline range: {start_sec}–{end_sec}s (start must be < end)")
    if end - start < window:
        raise ValueError(
            f"Baseline range too short: {(end-start)/fs:.2f}s < window_sec={window_sec:.2f}s"
        )

    bp = (float(config['gcc_bandpass_low']), float(config['gcc_bandpass_high']))
    max_tau = float(config['gcc_max_lag_ms']) / 1000.0
    psr_exclude = int(config.get('baseline_psr_exclude_samples', 50))

    taus = []
    psrs = []
    for win_start in range(start, end - window + 1, hop):
        win_end = win_start + window
        tau, psr_db = gcc_phat_full_analysis(
            mic_left[win_start:win_end].astype(np.float64, copy=False),
            mic_right[win_start:win_end].astype(np.float64, copy=False),
            fs,
            max_tau=max_tau,
            bandpass=bp,
            psr_exclude_samples=psr_exclude,
        )
        taus.append(tau)
        psrs.append(psr_db)

    taus = np.asarray(taus, dtype=np.float64)
    psrs = np.asarray(psrs, dtype=np.float64)

    if psr_min_db is not None:
        psr_min_db = float(psr_min_db)
        mask = psrs >= psr_min_db
        if np.any(mask):
            taus_used = taus[mask]
            psrs_used = psrs[mask]
        else:
            taus_used = taus
            psrs_used = psrs
    else:
        taus_used = taus
        psrs_used = psrs

    return {
        'start_sec': float(start / fs),
        'end_sec': float(end / fs),
        'window_sec': float(window / fs),
        'hop_sec': float(hop / fs),
        'n_windows': int(len(taus)),
        'n_used': int(len(taus_used)),
        'psr_min_db': None if psr_min_db is None else float(psr_min_db),
        'tau_median_ms': float(np.median(taus_used) * 1000),
        'tau_std_ms': float(np.std(taus_used) * 1000),
        'psr_median_db': float(np.median(psrs_used)),
        'psr_std_db': float(np.std(psrs_used)),
    }


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
    return (d_left - d_right) / c


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

    duration = min(len(ldv_signal), len(mic_left_signal), len(mic_right_signal)) / sr_ldv
    logger.info(f"Sample rate: {sr_ldv} Hz, Duration: {duration:.2f} s")

    speaker_id = Path(ldv_path).parent.name
    expected_tdoa = compute_expected_tdoa(speaker_id, config)
    logger.info(f"Expected TDoA (MicL-MicR): {expected_tdoa*1000:+.3f} ms")

    baseline_method = str(config.get('baseline_method', 'segment')).lower()
    if baseline_method not in {'segment', 'report', 'windowed', 'geometry'}:
        raise ValueError(
            f"Invalid baseline_method={baseline_method!r} (expected: segment|report|windowed|geometry)"
        )

    baseline_report = None
    baseline_windowed = None
    if baseline_method == 'report':
        logger.info(
            f"Computing baseline τ using report method "
            f"({config['baseline_start_sec']:.1f}–{config['baseline_end_sec']:.1f}s, "
            f"{config['gcc_bandpass_low']:.0f}–{config['gcc_bandpass_high']:.0f} Hz)..."
        )
        baseline_report = estimate_baseline_report(mic_left_signal, mic_right_signal, config)
        logger.info(
            f"Baseline(report): τ={baseline_report['tau_ms']:+.4f} ms, PSR={baseline_report['psr_db']:.2f} dB"
        )
    elif baseline_method == 'windowed':
        logger.info(
            f"Computing baseline τ using windowed median "
            f"({config['baseline_start_sec']:.1f}–{config['baseline_end_sec']:.1f}s, "
            f"window={config['baseline_window_sec']:.2f}s, hop={config['baseline_hop_sec']:.2f}s)..."
        )
        baseline_windowed = estimate_baseline_windowed(mic_left_signal, mic_right_signal, config)
        logger.info(
            f"Baseline(windowed): τ_med={baseline_windowed['tau_median_ms']:+.4f} ms "
            f"(std={baseline_windowed['tau_std_ms']:.4f} ms), "
            f"PSR_med={baseline_windowed['psr_median_db']:.2f} dB "
            f"(n_used={baseline_windowed['n_used']}/{baseline_windowed['n_windows']})"
        )

    bp = (config['gcc_bandpass_low'], config['gcc_bandpass_high'])
    max_tau = float(config['gcc_max_lag_ms']) / 1000.0
    psr_exclude = int(config.get('baseline_psr_exclude_samples', 50))

    # Determine segment centers in seconds (avoid huge full-audio STFTs)
    n_segments = int(config['n_segments'])
    segment_spacing = float(config['segment_spacing_sec'])
    segment_offset = float(config.get('segment_offset_sec', 0.0))

    slice_samples = int(config['analysis_slice_sec'] * config['fs'])
    half_slice_sec = (slice_samples / config['fs']) / 2

    min_center = half_slice_sec
    max_center = max(min_center, duration - half_slice_sec)
    start_center = max(min_center, segment_offset)

    if n_segments <= 1:
        segment_centers_sec = [min(max_center, max(min_center, duration / 2))]
    else:
        segment_centers_sec = [start_center + i * segment_spacing for i in range(n_segments)]
        segment_centers_sec = [t for t in segment_centers_sec if t <= max_center]

        if len(segment_centers_sec) < n_segments:
            logger.warning(
                f"Requested {n_segments} segments but only {len(segment_centers_sec)} fit in "
                f"[{min_center:.2f}s, {max_center:.2f}s]."
            )

    logger.info(f"Evaluating {len(segment_centers_sec)} segments...")

    # Results storage
    results_baseline = []
    results_raw = []
    results_omp = []

    for seg_idx, center_sec in enumerate(segment_centers_sec):
        logger.info(f"  Segment {seg_idx+1}/{len(segment_centers_sec)}: center_t={center_sec:.2f}s")

        center_sample = int(center_sec * config['fs'])
        slice_start, slice_end = extract_centered_slice(
            [ldv_signal, mic_left_signal, mic_right_signal],
            center_sample=center_sample,
            slice_samples=slice_samples
        )

        ldv_slice = ldv_signal[slice_start:slice_end]
        mic_left_slice = mic_left_signal[slice_start:slice_end]
        mic_right_slice = mic_right_signal[slice_start:slice_end]

        # Compute STFT on the slice
        _, _, Zxx_ldv = stft(ldv_slice, fs=config['fs'],
                             nperseg=config['n_fft'],
                             noverlap=config['n_fft'] - config['hop_length'],
                             window='hann')

        _, _, Zxx_left = stft(mic_left_slice, fs=config['fs'],
                              nperseg=config['n_fft'],
                              noverlap=config['n_fft'] - config['hop_length'],
                              window='hann')

        n_time = min(Zxx_ldv.shape[1], Zxx_left.shape[1])
        tw = config['tw']
        max_lag = config['max_lag']

        start_t = n_time // 2 - tw // 2
        start_t = max(start_t, max_lag + 1)
        start_t = min(start_t, n_time - tw - max_lag - 1)
        if start_t < max_lag + 1:
            raise ValueError(
                "analysis_slice_sec too short for OMP window; "
                f"need > {(tw + 2*max_lag) * config['hop_length'] / config['fs']:.3f}s of STFT support"
            )

        # Apply OMP alignment for this segment
        Zxx_ldv_as_left = apply_omp_alignment_segment(Zxx_ldv, Zxx_left, config, start_t)

        # Convert slice to time domain
        _, ldv_raw_td = istft(Zxx_ldv, fs=config['fs'],
                              nperseg=config['n_fft'],
                              noverlap=config['n_fft'] - config['hop_length'],
                              window='hann')

        _, ldv_as_left_td = istft(Zxx_ldv_as_left, fs=config['fs'],
                                  nperseg=config['n_fft'],
                                  noverlap=config['n_fft'] - config['hop_length'],
                                  window='hann')

        # Extract segment for GCC-PHAT evaluation
        hop = config['hop_length']
        gcc_center_sample = start_t * hop + config['n_fft'] // 2
        segment_samples = int(config['gcc_segment_sec'] * config['fs'])

        t_start = max(0, gcc_center_sample - segment_samples // 2)
        t_end = min(len(ldv_raw_td), t_start + segment_samples)
        t_end = min(len(mic_left_slice), t_end)
        t_end = min(len(mic_right_slice), t_end)

        ldv_raw_seg = ldv_raw_td[t_start:t_end]
        ldv_as_left_seg = ldv_as_left_td[t_start:t_end]
        mic_left_seg = mic_left_slice[t_start:t_end]
        mic_right_seg = mic_right_slice[t_start:t_end]

        tau_baseline, psr_baseline = gcc_phat_full_analysis(
            mic_left_seg.astype(np.float64, copy=False),
            mic_right_seg.astype(np.float64, copy=False),
            config['fs'],
            max_tau=max_tau,
            bandpass=bp,
            psr_exclude_samples=psr_exclude,
        )
        tau_raw, psr_raw = gcc_phat_full_analysis(
            ldv_raw_seg.astype(np.float64, copy=False),
            mic_right_seg.astype(np.float64, copy=False),
            config['fs'],
            max_tau=max_tau,
            bandpass=bp,
            psr_exclude_samples=psr_exclude,
        )
        tau_omp, psr_omp = gcc_phat_full_analysis(
            ldv_as_left_seg.astype(np.float64, copy=False),
            mic_right_seg.astype(np.float64, copy=False),
            config['fs'],
            max_tau=max_tau,
            bandpass=bp,
            psr_exclude_samples=psr_exclude,
        )

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

    # Select baseline reference τ (for Stage 3 criteria)
    if baseline_method == 'segment':
        tau_ref = None  # per-segment baseline
        psr_ref = None
    elif baseline_method == 'report':
        tau_ref = float(baseline_report['tau_ms']) / 1000.0
        psr_ref = float(baseline_report['psr_db'])
    elif baseline_method == 'windowed':
        tau_ref = float(baseline_windowed['tau_median_ms']) / 1000.0
        psr_ref = float(baseline_windowed['psr_median_db'])
    else:  # geometry
        tau_ref = float(expected_tdoa)
        psr_ref = None

    # Errors vs baseline reference (segment baseline is per-segment)
    if baseline_method == 'segment':
        error_baseline_arr = np.zeros_like(tau_baseline_arr)
        error_raw_arr = np.abs(tau_raw_arr - tau_baseline_arr)
        error_omp_arr = np.abs(tau_omp_arr - tau_baseline_arr)
    else:
        error_baseline_arr = np.abs(tau_baseline_arr - tau_ref)
        error_raw_arr = np.abs(tau_raw_arr - tau_ref)
        error_omp_arr = np.abs(tau_omp_arr - tau_ref)

    # Errors vs geometric expected τ (for debugging)
    error_baseline_vs_theory_arr = np.abs(tau_baseline_arr - expected_tdoa)
    error_raw_vs_theory_arr = np.abs(tau_raw_arr - expected_tdoa)
    error_omp_vs_theory_arr = np.abs(tau_omp_arr - expected_tdoa)

    # Statistics
    stats = {
        'expected_tdoa_ms': float(expected_tdoa * 1000),
        'baseline_reference': {
            'method': baseline_method,
            'tau_ms': float(np.median(tau_baseline_arr) * 1000) if baseline_method == 'segment' else float(tau_ref * 1000),
            'psr_db': None if psr_ref is None else float(psr_ref),
        },
        'baseline_segment': {
            'tau_median_ms': float(np.median(tau_baseline_arr) * 1000),
            'tau_std_ms': float(np.std(tau_baseline_arr) * 1000),
            'psr_median_db': float(np.median(psr_baseline_arr)),
            'psr_std_db': float(np.std(psr_baseline_arr)),
            'error_vs_baseline_median_ms': float(np.median(error_baseline_arr) * 1000),
            'error_vs_theory_median_ms': float(np.median(error_baseline_vs_theory_arr) * 1000),
        },
        'raw_ldv': {
            'tau_median_ms': float(np.median(tau_raw_arr) * 1000),
            'tau_std_ms': float(np.std(tau_raw_arr) * 1000),
            'psr_median_db': float(np.median(psr_raw_arr)),
            'error_vs_baseline_median_ms': float(np.median(error_raw_arr) * 1000),
            'error_vs_baseline_std_ms': float(np.std(error_raw_arr) * 1000),
            'error_vs_theory_median_ms': float(np.median(error_raw_vs_theory_arr) * 1000),
            'error_vs_theory_std_ms': float(np.std(error_raw_vs_theory_arr) * 1000),
        },
        'omp_ldv': {
            'tau_median_ms': float(np.median(tau_omp_arr) * 1000),
            'tau_std_ms': float(np.std(tau_omp_arr) * 1000),
            'psr_median_db': float(np.median(psr_omp_arr)),
            'error_vs_baseline_median_ms': float(np.median(error_omp_arr) * 1000),
            'error_vs_baseline_std_ms': float(np.std(error_omp_arr) * 1000),
            'error_vs_theory_median_ms': float(np.median(error_omp_vs_theory_arr) * 1000),
            'error_vs_theory_std_ms': float(np.std(error_omp_vs_theory_arr) * 1000),
        }
    }

    # Pass conditions (compare against selected baseline reference)
    error_improved = np.median(error_omp_arr) < np.median(error_raw_arr)
    psr_improved = np.median(psr_omp_arr) > np.median(psr_raw_arr)
    error_small = np.median(error_omp_arr) < 0.5e-3

    passed = error_improved and psr_improved and error_small

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'speaker_id': speaker_id,
        'n_segments': len(segment_centers_sec),
        'segment_centers_sec': [float(t) for t in segment_centers_sec],
        'baseline_report': baseline_report,
        'baseline_windowed': baseline_windowed,
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
    logger.info(f"Stage 3 Multi-Segment Results ({len(segment_centers_sec)} segments)")
    logger.info("=" * 70)
    logger.info(f"Baseline reference ({baseline_method}): τ = {stats['baseline_reference']['tau_ms']:+.4f} ms")
    logger.info(f"Geometric expected TDoA:              τ = {stats['expected_tdoa_ms']:+.4f} ms")
    logger.info(f"GCC-PHAT TDoA ({bp[0]:.0f}-{bp[1]:.0f} Hz bandpass, segment={config['gcc_segment_sec']:.2f}s):")
    logger.info(f"  (MicL, MicR) segment:    τ = {stats['baseline_segment']['tau_median_ms']:+.4f} ± {stats['baseline_segment']['tau_std_ms']:.4f} ms, err vs baseline = {stats['baseline_segment']['error_vs_baseline_median_ms']:.4f} ms")
    logger.info(f"  (Raw_LDV, MicR):         τ = {stats['raw_ldv']['tau_median_ms']:+.4f} ± {stats['raw_ldv']['tau_std_ms']:.4f} ms, err vs baseline = {stats['raw_ldv']['error_vs_baseline_median_ms']:.4f} ms")
    logger.info(f"  (LDV_as_MicL, MicR):     τ = {stats['omp_ldv']['tau_median_ms']:+.4f} ± {stats['omp_ldv']['tau_std_ms']:.4f} ms, err vs baseline = {stats['omp_ldv']['error_vs_baseline_median_ms']:.4f} ms")
    logger.info("-" * 70)
    logger.info(f"OMP closer to baseline than raw: {'✓ PASS' if error_improved else '✗ FAIL'} ({stats['omp_ldv']['error_vs_baseline_median_ms']:.4f} vs {stats['raw_ldv']['error_vs_baseline_median_ms']:.4f} ms)")
    logger.info(f"PSR improved:                   {'✓ PASS' if psr_improved else '✗ FAIL'}")
    logger.info(f"OMP error vs baseline < 0.5 ms: {'✓ PASS' if error_small else '✗ FAIL'} ({stats['omp_ldv']['error_vs_baseline_median_ms']:.4f} ms)")
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
    parser.add_argument('--segment_offset', type=float, default=0.0,
                        help='Offset (seconds) for the first segment center (e.g., 100 to match full_analysis.py).')
    parser.add_argument('--freq_min', type=float, default=None)
    parser.add_argument('--freq_max', type=float, default=None)
    parser.add_argument('--gcc_bandpass_low', type=float, default=None)
    parser.add_argument('--gcc_bandpass_high', type=float, default=None)
    parser.add_argument('--gcc_segment_sec', type=float, default=None)
    parser.add_argument('--analysis_slice_sec', type=float, default=None)

    # Baseline reference (Stage 3)
    parser.add_argument('--baseline_method', type=str, default=None,
                        choices=['segment', 'report', 'windowed', 'geometry'],
                        help='Which τ to treat as τ_baseline for Stage 3 pass conditions.')
    parser.add_argument('--baseline_start_sec', type=float, default=None,
                        help='Start time (sec) for baseline_method=report/windowed.')
    parser.add_argument('--baseline_end_sec', type=float, default=None,
                        help='End time (sec) for baseline_method=report/windowed.')
    parser.add_argument('--baseline_window_sec', type=float, default=None,
                        help='Window length (sec) for baseline_method=windowed.')
    parser.add_argument('--baseline_hop_sec', type=float, default=None,
                        help='Hop length (sec) for baseline_method=windowed.')
    parser.add_argument('--baseline_psr_min_db', type=float, default=None,
                        help='Optional PSR threshold for baseline_method=windowed.')
    parser.add_argument('--baseline_psr_exclude_samples', type=int, default=None,
                        help='PSR sidelobe exclusion half-width in samples (default: 50 to match full_analysis.py).')

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config['n_segments'] = args.n_segments
    config['segment_spacing_sec'] = args.segment_spacing
    config['segment_offset_sec'] = args.segment_offset
    if args.freq_min is not None:
        config['freq_min'] = args.freq_min
    if args.freq_max is not None:
        config['freq_max'] = args.freq_max
    if args.gcc_bandpass_low is not None:
        config['gcc_bandpass_low'] = args.gcc_bandpass_low
    if args.gcc_bandpass_high is not None:
        config['gcc_bandpass_high'] = args.gcc_bandpass_high
    if args.gcc_segment_sec is not None:
        config['gcc_segment_sec'] = args.gcc_segment_sec
    if args.analysis_slice_sec is not None:
        config['analysis_slice_sec'] = args.analysis_slice_sec
    if args.baseline_method is not None:
        config['baseline_method'] = args.baseline_method
    if args.baseline_start_sec is not None:
        config['baseline_start_sec'] = args.baseline_start_sec
    if args.baseline_end_sec is not None:
        config['baseline_end_sec'] = args.baseline_end_sec
    if args.baseline_window_sec is not None:
        config['baseline_window_sec'] = args.baseline_window_sec
    if args.baseline_hop_sec is not None:
        config['baseline_hop_sec'] = args.baseline_hop_sec
    if args.baseline_psr_min_db is not None:
        config['baseline_psr_min_db'] = args.baseline_psr_min_db
    if args.baseline_psr_exclude_samples is not None:
        config['baseline_psr_exclude_samples'] = args.baseline_psr_exclude_samples

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
