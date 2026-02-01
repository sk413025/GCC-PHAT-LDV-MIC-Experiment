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
from scipy.signal import stft, istft, correlate, butter, filtfilt
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
    'max_lag': 50,      # ±50 samples @ 48kHz = ±1.04 ms
    'max_k': 3,         # OMP sparsity
    'tw': 64,           # Time window for OMP
    'freq_min': 100,    # Hz
    'freq_max': 8000,   # Hz
    # GCC config
    'gcc_max_lag_ms': 10.0,  # ms, for peak search
    'gcc_bandpass_low': 500.0,
    'gcc_bandpass_high': 2000.0,
    'psr_exclude_samples': 50,
    # Optional: guided peak search around a reference τ (useful when global peak locks to a wrong, stable sidelobe)
    # If set (ms), GCC-PHAT will search the peak within ±radius around the current truth reference τ.
    'gcc_guided_peak_radius_ms': None,
    # Segmenting config (avoid full-file STFTs)
    'analysis_slice_sec': 5.0,
    'eval_window_sec': 1.0,
    'segment_spacing_sec': 50.0,
    'segment_offset_sec': 100.0,
    # Optional prealignment (useful for chirp where sub-hop delays matter)
    # - 'none': no prealignment
    # - 'gcc_phat': estimate LDV->MicL delay on eval window via GCC-PHAT, then apply fractional delay to LDV slice
    'ldv_prealign': 'none',
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

    # TDoA convention (match Stage 3 / GCC-PHAT_LDV_MIC_完整實驗報告.md):
    # - τ = (d_left - d_right) / c
    # - Positive τ means the RIGHT mic is closer (sound reaches right earlier).
    tau_true = (d_left - d_right) / c

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


def extract_centered_slice(
    signals: list[np.ndarray],
    *,
    center_sample: int,
    slice_samples: int,
) -> tuple[int, int]:
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


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 5) -> np.ndarray:
    """Butterworth bandpass filter with zero-phase filtering."""
    nyq = 0.5 * fs
    low = float(lowcut) / nyq
    high = float(highcut) / nyq
    if not (0.0 < low < high < 1.0):
        raise ValueError(f"Invalid bandpass range: lowcut={lowcut}, highcut={highcut}, fs={fs}")
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def apply_fractional_delay_fd(signal: np.ndarray, fs: int, delay_sec: float) -> np.ndarray:
    """
    Apply a (possibly fractional) time delay using an FFT phase ramp.

    Notes:
    - Uses zero-padding (2x length) to reduce circular wrap-around artifacts.
    - Returns a float64 array (even if input is float32).
    """
    delay_sec = float(delay_sec)
    if abs(delay_sec) < 1e-12:
        return signal.astype(np.float64, copy=True)

    x = signal.astype(np.float64, copy=False)
    n = len(x)
    if n == 0:
        return x.copy()

    n_fft = 1 << int(np.ceil(np.log2(max(2, n * 2))))
    X = np.fft.rfft(x, n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(fs))
    X *= np.exp(-1j * 2 * np.pi * freqs * delay_sec)
    y = np.fft.irfft(X, n_fft)[:n]
    return y.astype(np.float64, copy=False)


def scan_segment_centers_by_mic_mic(
    ldv_signal: np.ndarray | None,
    mic_left_signal: np.ndarray,
    mic_right_signal: np.ndarray,
    *,
    fs: int,
    eval_window_sec: float,
    max_lag_ms: float,
    bandpass: tuple[float, float] | None,
    psr_exclude_samples: int,
    tau_true_ms: float,
    guided_tau_ms: float | None = None,
    guided_radius_ms: float | None = None,
    scan_start_sec: float,
    scan_end_sec: float,
    scan_hop_sec: float,
    scan_psr_min_db: float | None,
    scan_ldv_micl_psr_min_db: float | None,
    scan_tau_err_max_ms: float | None,
    scan_sort_by: str = 'tau_err',
    n_segments: int,
    min_separation_sec: float | None,
    allow_fallback: bool = False,
) -> tuple[list[float], dict]:
    """Scan mic-mic windows and pick segment centers with good PSR and τ close to theory."""
    eval_window_samples = int(eval_window_sec * fs)
    if eval_window_samples <= 0:
        raise ValueError(f"eval_window_sec must be > 0, got {eval_window_sec}")

    duration_s = min(len(mic_left_signal), len(mic_right_signal)) / fs
    half_win_sec = (eval_window_samples / fs) / 2
    min_center = half_win_sec
    max_center = max(min_center, duration_s - half_win_sec)

    start_center = max(min_center, scan_start_sec)
    end_center = min(max_center, scan_end_sec if scan_end_sec is not None else max_center)

    if scan_hop_sec <= 0:
        raise ValueError(f"scan_hop_sec must be > 0, got {scan_hop_sec}")

    max_tau = float(max_lag_ms) / 1000.0
    candidates = []

    center = start_center
    n_scanned = 0
    while center <= end_center + 1e-9:
        center_sample = int(center * fs)
        t_start = center_sample - eval_window_samples // 2
        t_end = t_start + eval_window_samples
        if t_start < 0 or t_end > len(mic_left_signal) or t_end > len(mic_right_signal):
            center += scan_hop_sec
            continue

        mic_l_seg = mic_left_signal[t_start:t_end]
        mic_r_seg = mic_right_signal[t_start:t_end]
        tau_sec, psr_db = gcc_phat_full_analysis(
            mic_l_seg.astype(np.float64, copy=False),
            mic_r_seg.astype(np.float64, copy=False),
            fs,
            max_tau=max_tau,
            bandpass=bandpass,
            psr_exclude_samples=psr_exclude_samples,
            guided_tau=None if guided_tau_ms is None else float(guided_tau_ms) / 1000.0,
            guided_radius=None if guided_radius_ms is None else float(guided_radius_ms) / 1000.0,
        )
        tau_ms = tau_sec * 1000.0
        tau_err_ms = float(abs(tau_ms - tau_true_ms))

        n_scanned += 1
        cand = {
            'center_sec': float(center),
            'tau_ms': float(tau_ms),
            'psr_db': float(psr_db),
            'tau_err_ms': tau_err_ms,
        }

        if ldv_signal is not None:
            if t_end <= len(ldv_signal):
                ldv_seg = ldv_signal[t_start:t_end]
                tau_ldv_micl_sec, psr_ldv_micl_db = gcc_phat_full_analysis(
                    ldv_seg.astype(np.float64, copy=False),
                    mic_l_seg.astype(np.float64, copy=False),
                    fs,
                    max_tau=max_tau,
                    bandpass=bandpass,
                    psr_exclude_samples=psr_exclude_samples,
                )
                cand['ldv_micl_tau_ms'] = float(tau_ldv_micl_sec * 1000.0)
                cand['ldv_micl_psr_db'] = float(psr_ldv_micl_db)
            else:
                cand['ldv_micl_tau_ms'] = None
                cand['ldv_micl_psr_db'] = None

        candidates.append(cand)

        center += scan_hop_sec

    # Filter candidates
    filtered = candidates
    if scan_psr_min_db is not None:
        filtered = [c for c in filtered if c['psr_db'] >= float(scan_psr_min_db)]
    if scan_ldv_micl_psr_min_db is not None:
        filtered = [
            c for c in filtered
            if c.get('ldv_micl_psr_db') is not None and c['ldv_micl_psr_db'] >= float(scan_ldv_micl_psr_min_db)
        ]
    if scan_tau_err_max_ms is not None:
        filtered = [c for c in filtered if c['tau_err_ms'] <= float(scan_tau_err_max_ms)]

    scan_sort_by = str(scan_sort_by or 'tau_err').lower()
    if scan_sort_by not in {'tau_err', 'psr'}:
        raise ValueError(f"Invalid scan_sort_by={scan_sort_by!r} (expected: tau_err|psr)")

    # Sort by scan preference
    if scan_sort_by == 'psr':
        # Prefer strong, stable peaks first; break ties by tau_err
        filtered.sort(key=lambda c: (-c['psr_db'], c['tau_err_ms']))
    else:
        # Prefer tau closeness first; break ties by PSR
        filtered.sort(key=lambda c: (c['tau_err_ms'], -c['psr_db']))

    # Greedy pick with separation
    min_sep = float(min_separation_sec) if min_separation_sec is not None else 0.0
    selected = []
    for c in filtered:
        if len(selected) >= int(n_segments):
            break
        if min_sep > 0 and any(abs(c['center_sec'] - s) < min_sep for s in selected):
            continue
        selected.append(c['center_sec'])

    used_fallback = False
    # If insufficient, optionally fall back to best PSR windows (still obey separation)
    if allow_fallback and len(selected) < int(n_segments):
        used_fallback = True
        fallback = sorted(candidates, key=lambda c: (-c['psr_db'], c['tau_err_ms']))
        for c in fallback:
            if len(selected) >= int(n_segments):
                break
            if min_sep > 0 and any(abs(c['center_sec'] - s) < min_sep for s in selected):
                continue
            if c['center_sec'] not in selected:
                selected.append(c['center_sec'])

    selected.sort()
    scan_summary = {
        'duration_sec': float(duration_s),
        'eval_window_sec': float(eval_window_sec),
        'scan_start_sec': float(scan_start_sec),
        'scan_end_sec': float(end_center),
        'scan_hop_sec': float(scan_hop_sec),
        'guided_tau_ms': None if guided_tau_ms is None else float(guided_tau_ms),
        'guided_radius_ms': None if guided_radius_ms is None else float(guided_radius_ms),
        'scan_psr_min_db': None if scan_psr_min_db is None else float(scan_psr_min_db),
        'scan_ldv_micl_psr_min_db': None if scan_ldv_micl_psr_min_db is None else float(scan_ldv_micl_psr_min_db),
        'scan_tau_err_max_ms': None if scan_tau_err_max_ms is None else float(scan_tau_err_max_ms),
        'scan_sort_by': scan_sort_by,
        'min_separation_sec': float(min_sep),
        'n_scanned': int(n_scanned),
        'n_candidates': int(len(candidates)),
        'n_filtered': int(len(filtered)),
        'allow_fallback': bool(allow_fallback),
        'used_fallback': bool(used_fallback),
        'selected_centers_sec': selected,
        # Keep a small preview for debugging
        'best_by_tau_err': filtered[:10],
    }

    return selected, scan_summary


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
def gcc_phat_full_analysis(
    sig1: np.ndarray,
    sig2: np.ndarray,
    fs: int,
    *,
    max_tau: float | None = None,
    bandpass: tuple[float, float] | None = None,
    psr_exclude_samples: int = 50,
    guided_tau: float | None = None,
    guided_radius: float | None = None,
) -> tuple[float, float]:
    """
    GCC-PHAT implementation matching full_analysis.py (parabolic interpolation + PSR exclude).

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

    peak_idx = None
    if guided_tau is not None and guided_radius is not None and float(guided_radius) > 0:
        guided_center_idx = int(round(float(guided_tau) * fs)) + max_shift
        guided_radius_samples = int(round(float(guided_radius) * fs))
        lo = max(0, guided_center_idx - guided_radius_samples)
        hi = min(len(abs_cc) - 1, guided_center_idx + guided_radius_samples)
        if lo <= hi:
            peak_idx = int(np.argmax(abs_cc[lo:hi + 1])) + lo

    if peak_idx is None:
        peak_idx = int(np.argmax(abs_cc))

    # Parabolic interpolation
    if 0 < peak_idx < len(abs_cc) - 1:
        y0 = abs_cc[peak_idx - 1]
        y1 = abs_cc[peak_idx]
        y2 = abs_cc[peak_idx + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            shift = 0.5 * (y0 - y2) / denom
        else:
            shift = 0.0
    else:
        shift = 0.0

    tau = ((peak_idx - max_shift) + shift) / fs

    # PSR exclude region around peak
    mask = np.ones_like(abs_cc, dtype=bool)
    lo = max(0, peak_idx - psr_exclude_samples)
    hi = min(len(abs_cc), peak_idx + psr_exclude_samples + 1)
    mask[lo:hi] = False

    sidelobe_max = abs_cc[mask].max() if np.any(mask) else 0.0
    peak_val = abs_cc[peak_idx]
    psr = 20 * np.log10(peak_val / (sidelobe_max + 1e-10))

    return float(tau), float(psr)


def estimate_tdoa_gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: int,
                           max_lag_samples: int = None,
                           bandpass: tuple[float, float] | None = None,
                           psr_exclude_samples: int = 50,
                           guided_tau_ms: float | None = None,
                           guided_radius_ms: float | None = None) -> dict:
    """GCC-PHAT TDoA estimation (full_analysis.py-style)."""
    if max_lag_samples is not None:
        max_tau = float(max_lag_samples) / float(fs)
    else:
        max_tau = None
    tau_sec, psr_db = gcc_phat_full_analysis(
        sig1.astype(np.float64, copy=False),
        sig2.astype(np.float64, copy=False),
        fs,
        max_tau=max_tau,
        bandpass=bandpass,
        psr_exclude_samples=int(psr_exclude_samples),
        guided_tau=None if guided_tau_ms is None else float(guided_tau_ms) / 1000.0,
        guided_radius=None if guided_radius_ms is None else float(guided_radius_ms) / 1000.0,
    )
    return {'tau_ms': tau_sec * 1000.0, 'psr_db': psr_db}


def estimate_tdoa_cc(sig1: np.ndarray, sig2: np.ndarray, fs: int,
                     max_lag_samples: int = None,
                     bandpass: tuple[float, float] | None = None,
                     psr_exclude_samples: int = 50) -> dict:
    """Standard cross-correlation TDoA estimation."""
    if bandpass is not None:
        sig1 = bandpass_filter(sig1, bandpass[0], bandpass[1], fs)
        sig2 = bandpass_filter(sig2, bandpass[0], bandpass[1], fs)

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
    lo = max(0, peak_idx - int(psr_exclude_samples))
    hi = min(len(search_cc), peak_idx + int(psr_exclude_samples) + 1)
    sidelobe_mask[lo:hi] = False

    if sidelobe_mask.sum() > 0:
        sidelobe_max = np.abs(search_cc[sidelobe_mask]).max()
        psr = 20 * np.log10(peak_val / (sidelobe_max + 1e-10))
    else:
        psr = 0.0

    return {'tau_ms': tau * 1000, 'psr_db': psr}


def estimate_tdoa_ncc(sig1: np.ndarray, sig2: np.ndarray, fs: int,
                      max_lag_samples: int = None,
                      bandpass: tuple[float, float] | None = None,
                      psr_exclude_samples: int = 50) -> dict:
    """Normalized cross-correlation TDoA estimation."""
    if bandpass is not None:
        sig1 = bandpass_filter(sig1, bandpass[0], bandpass[1], fs)
        sig2 = bandpass_filter(sig2, bandpass[0], bandpass[1], fs)

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
    lo = max(0, peak_idx - int(psr_exclude_samples))
    hi = min(len(search_cc), peak_idx + int(psr_exclude_samples) + 1)
    sidelobe_mask[lo:hi] = False

    if sidelobe_mask.sum() > 0:
        sidelobe_max = np.abs(search_cc[sidelobe_mask]).max()
        psr = 20 * np.log10(peak_val / (sidelobe_max + 1e-10))
    else:
        psr = 0.0

    return {'tau_ms': tau * 1000, 'psr_db': psr}


def estimate_tdoa_music(sig1: np.ndarray, sig2: np.ndarray, fs: int, config: dict,
                        max_lag_samples: int = None,
                        bandpass: tuple[float, float] | None = None) -> dict:
    """
    MUSIC-based TDoA estimation.

    Simplified MUSIC for 2-element array using covariance matrix.
    """
    if bandpass is not None:
        sig1 = bandpass_filter(sig1, bandpass[0], bandpass[1], fs)
        sig2 = bandpass_filter(sig2, bandpass[0], bandpass[1], fs)

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
    n_segments: int = 5,
    *,
    segment_mode: str = 'fixed',
    scan_start_sec: float | None = None,
    scan_end_sec: float | None = None,
    scan_hop_sec: float = 1.0,
    scan_psr_min_db: float | None = None,
    scan_ldv_micl_psr_min_db: float | None = None,
    scan_tau_err_max_ms: float | None = 0.2,
    scan_sort_by: str = 'tau_err',
    scan_min_separation_sec: float | None = None,
    scan_allow_fallback: bool = False,
    speaker_key_override: str | None = None,
    truth_tau_ms: float | None = None,
    truth_theta_deg: float | None = None,
    truth_label: str | None = None,
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

    # Compute geometry ground truth (optionally override speaker key for non-standard folder names)
    ground_truth = compute_ground_truth(speaker_key_override or speaker_id, config)
    tau_geom_ms = float(ground_truth['tau_true_ms'])
    theta_geom_deg = float(ground_truth['theta_true_deg'])

    # Truth reference: default to geometry truth, but allow overriding (e.g., use chirp mic τ as truth-ref).
    truth_mode = 'geometry'
    if truth_tau_ms is not None:
        truth_mode = 'override'
        tau_ref_ms = float(truth_tau_ms)
        if truth_theta_deg is not None:
            theta_ref_deg = float(truth_theta_deg)
        else:
            theta_ref_deg = float(tau_to_doa(tau_ref_ms, config))
    else:
        tau_ref_ms = tau_geom_ms
        theta_ref_deg = theta_geom_deg

    logger.info(f"Geometry Truth:  τ = {tau_geom_ms:.3f} ms, θ = {theta_geom_deg:.2f}°")
    logger.info(f"Reference({truth_mode}): τ = {tau_ref_ms:.3f} ms, θ = {theta_ref_deg:.2f}°")
    logger.info(f"Distances: d_left = {ground_truth['d_left']:.3f} m, d_right = {ground_truth['d_right']:.3f} m")

    # Load audio
    logger.info(f"\nLoading audio files...")
    sr_ldv, ldv_signal = load_wav(ldv_path)
    sr_mic_l, mic_left_signal = load_wav(mic_left_path)
    sr_mic_r, mic_right_signal = load_wav(mic_right_path)

    assert sr_ldv == sr_mic_l == sr_mic_r == config['fs']

    logger.info(f"Sample rate: {sr_ldv} Hz")
    duration_s = min(len(ldv_signal), len(mic_left_signal), len(mic_right_signal)) / sr_ldv
    logger.info(f"Duration: {duration_s:.2f} s")

    slice_samples = int(float(config.get('analysis_slice_sec', 5.0)) * config['fs'])
    half_slice_sec = (slice_samples / config['fs']) / 2
    segment_spacing_sec = float(config.get('segment_spacing_sec', 50.0))
    segment_offset_sec = float(config.get('segment_offset_sec', 100.0))

    min_center = half_slice_sec
    max_center = max(min_center, duration_s - half_slice_sec)
    start_center = max(min_center, segment_offset_sec)

    bp_low = float(config.get('gcc_bandpass_low', 0.0))
    bp_high = float(config.get('gcc_bandpass_high', 0.0))
    bp = (bp_low, bp_high) if (bp_low > 0 and bp_high > 0 and bp_high > bp_low) else None
    psr_exclude_samples = int(config.get('psr_exclude_samples', 50))

    if scan_start_sec is None:
        scan_start_sec = float(segment_offset_sec)
    if scan_end_sec is None:
        scan_end_sec = float(min(max_center, 600.0)) if duration_s > 600.0 else float(max_center)

    segment_mode = (segment_mode or 'fixed').lower()
    if segment_mode not in {'fixed', 'scan'}:
        raise ValueError(f"Invalid segment_mode={segment_mode!r} (expected: fixed|scan)")

    if segment_mode == 'scan':
        if scan_min_separation_sec is None:
            scan_min_separation_sec = float(config.get('eval_window_sec', 1.0))

        scan_use_ldv = str(config.get('ldv_prealign', 'none')).lower() == 'gcc_phat'
        segment_centers_sec, scan_summary = scan_segment_centers_by_mic_mic(
            ldv_signal if scan_use_ldv else None,
            mic_left_signal,
            mic_right_signal,
            fs=config['fs'],
            eval_window_sec=float(config.get('eval_window_sec', 1.0)),
            max_lag_ms=float(config['gcc_max_lag_ms']),
            bandpass=bp,
            psr_exclude_samples=psr_exclude_samples,
            tau_true_ms=float(tau_ref_ms),
            guided_tau_ms=float(tau_ref_ms) if (config.get('gcc_guided_peak_radius_ms') is not None and float(config.get('gcc_guided_peak_radius_ms')) > 0) else None,
            guided_radius_ms=None if config.get('gcc_guided_peak_radius_ms') is None else float(config.get('gcc_guided_peak_radius_ms')),
            scan_start_sec=float(scan_start_sec),
            scan_end_sec=float(scan_end_sec),
            scan_hop_sec=float(scan_hop_sec),
            scan_psr_min_db=scan_psr_min_db,
            scan_ldv_micl_psr_min_db=scan_ldv_micl_psr_min_db,
            scan_tau_err_max_ms=scan_tau_err_max_ms,
            scan_sort_by=scan_sort_by,
            n_segments=int(n_segments),
            min_separation_sec=scan_min_separation_sec,
            allow_fallback=bool(scan_allow_fallback),
        )
        if not segment_centers_sec:
            raise ValueError(
                "segment_mode=scan selected 0 segments. "
                "Try relaxing --scan_tau_err_max_ms / --scan_psr_min_db or set --scan_allow_fallback."
            )
    else:
        scan_summary = None
        if n_segments <= 1:
            segment_centers_sec = [min(max_center, max(min_center, duration_s / 2))]
        else:
            segment_centers_sec = [start_center + i * segment_spacing_sec for i in range(n_segments)]
            segment_centers_sec = [t for t in segment_centers_sec if t <= max_center]
            if len(segment_centers_sec) < n_segments:
                logger.warning(
                    f"Requested {n_segments} segments but only {len(segment_centers_sec)} fit in "
                    f"[{min_center:.2f}s, {max_center:.2f}s]."
                )

    n_segments_used = len(segment_centers_sec)
    logger.info(f"\nEvaluating {n_segments_used} segments...")

    # DoA methods
    methods = ['GCC-PHAT', 'CC', 'NCC', 'MUSIC']

    # Signal pairings
    pairings = ['MicL-MicR', 'Raw_LDV', 'Random_LDV', 'OMP_LDV']

    # Results storage
    all_results = {method: {pairing: [] for pairing in pairings} for method in methods}

    max_lag_samples = int(config['gcc_max_lag_ms'] * config['fs'] / 1000)
    eval_window_samples = int(float(config.get('eval_window_sec', 1.0)) * config['fs'])
    ldv_prealign = str(config.get('ldv_prealign', 'none')).lower()
    if ldv_prealign not in {'none', 'gcc_phat'}:
        raise ValueError(f"Invalid ldv_prealign={ldv_prealign!r} (expected: none|gcc_phat)")

    gcc_guided_radius_ms = config.get('gcc_guided_peak_radius_ms', None)
    use_guided_gcc = gcc_guided_radius_ms is not None and float(gcc_guided_radius_ms) > 0

    for seg_idx, center_sec in enumerate(segment_centers_sec):
        logger.info(f"  Segment {seg_idx+1}/{n_segments_used}: center_t={center_sec:.2f}s")

        center_sample_global = int(center_sec * config['fs'])
        slice_start, slice_end = extract_centered_slice(
            [ldv_signal, mic_left_signal, mic_right_signal],
            center_sample=center_sample_global,
            slice_samples=slice_samples,
        )

        ldv_slice = ldv_signal[slice_start:slice_end]
        mic_left_slice = mic_left_signal[slice_start:slice_end]
        mic_right_slice = mic_right_signal[slice_start:slice_end]
        desired_center_in_slice = int(center_sample_global - slice_start)

        # Decide evaluation window on the slice (for prealignment and final evaluation)
        eval_center_sample = int(np.clip(desired_center_in_slice, 0, max(0, len(ldv_slice) - 1)))
        t_start = eval_center_sample - eval_window_samples // 2
        t_end = t_start + eval_window_samples
        max_len_pre = min(len(ldv_slice), len(mic_left_slice), len(mic_right_slice))
        if t_start < 0:
            t_start = 0
            t_end = min(max_len_pre, eval_window_samples)
        if t_end > max_len_pre:
            t_end = max_len_pre
            t_start = max(0, t_end - eval_window_samples)

        # Optional: fractional-delay prealignment of LDV to MicL (useful for chirp)
        ldv_for_alignment_slice = ldv_slice
        prealign_info = None
        if ldv_prealign == 'gcc_phat':
            ldv_seg_for_tau = ldv_slice[t_start:t_end]
            mic_l_seg_for_tau = mic_left_slice[t_start:t_end]
            tau_ldv_to_micl_sec, psr_ldv_to_micl_db = gcc_phat_full_analysis(
                ldv_seg_for_tau.astype(np.float64, copy=False),
                mic_l_seg_for_tau.astype(np.float64, copy=False),
                config['fs'],
                max_tau=float(config['gcc_max_lag_ms']) / 1000.0,
                bandpass=bp,
                psr_exclude_samples=psr_exclude_samples,
            )
            # gcc_phat_full_analysis returns τ ≈ (t_ldv - t_micl). To align LDV to MicL, delay by -τ.
            delay_sec = -float(tau_ldv_to_micl_sec)
            ldv_for_alignment_slice = apply_fractional_delay_fd(ldv_slice, config['fs'], delay_sec)
            prealign_info = {
                'mode': 'gcc_phat',
                'tau_ldv_to_micl_ms': float(tau_ldv_to_micl_sec * 1000.0),
                'psr_ldv_to_micl_db': float(psr_ldv_to_micl_db),
                'applied_delay_ms': float(delay_sec * 1000.0),
            }

        # Compute STFT on the slice (avoid huge full-file STFTs)
        _, _, Zxx_ldv = stft(
            ldv_for_alignment_slice,
            fs=config['fs'],
            nperseg=config['n_fft'],
            noverlap=config['n_fft'] - config['hop_length'],
            window='hann',
        )

        _, _, Zxx_mic_left = stft(
            mic_left_slice,
            fs=config['fs'],
            nperseg=config['n_fft'],
            noverlap=config['n_fft'] - config['hop_length'],
            window='hann',
        )

        # Select time chunk for alignment
        n_time = min(Zxx_ldv.shape[1], Zxx_mic_left.shape[1])
        tw = int(config['tw'])
        max_lag = int(config['max_lag'])
        desired_frame = int(round((desired_center_in_slice - int(config['n_fft']) // 2) / int(config['hop_length'])))
        start_t = desired_frame - tw // 2

        start_t = max(start_t, max_lag + 1)
        start_t = min(start_t, n_time - tw - max_lag - 1)
        if start_t < max_lag + 1:
            raise ValueError(
                "analysis_slice_sec too short for OMP window; "
                f"need > {(tw + 2*max_lag) * config['hop_length'] / config['fs']:.3f}s of STFT support"
            )

        # Apply OMP alignment
        Zxx_omp = apply_omp_alignment(Zxx_ldv, Zxx_mic_left, config, start_t)

        # Apply Random alignment
        Zxx_random = apply_random_alignment(Zxx_ldv, Zxx_mic_left, config, start_t)

        # Convert aligned LDV to time domain
        _, ldv_omp_td = istft(
            Zxx_omp,
            fs=config['fs'],
            nperseg=config['n_fft'],
            noverlap=config['n_fft'] - config['hop_length'],
            window='hann',
        )

        _, ldv_random_td = istft(
            Zxx_random,
            fs=config['fs'],
            nperseg=config['n_fft'],
            noverlap=config['n_fft'] - config['hop_length'],
            window='hann',
        )

        # Extract evaluation window centered at the requested segment center (stable across scan/fixed).
        max_len = min(
            len(ldv_slice),
            len(mic_left_slice),
            len(mic_right_slice),
            len(ldv_omp_td),
            len(ldv_random_td),
        )
        if t_start < 0:
            t_start = 0
        if t_end > max_len:
            t_end = max_len
            t_start = max(0, t_end - eval_window_samples)

        ldv_raw_seg = ldv_slice[t_start:t_end]
        ldv_omp_seg = ldv_omp_td[t_start:t_end]
        ldv_random_seg = ldv_random_td[t_start:t_end]
        mic_left_seg = mic_left_slice[t_start:t_end]
        mic_right_seg = mic_right_slice[t_start:t_end]

        # Evaluate each method and pairing
        for method in methods:
            if method == 'GCC-PHAT':
                est_func = lambda s1, s2, fs, mls: estimate_tdoa_gcc_phat(
                    s1,
                    s2,
                    fs,
                    mls,
                    bandpass=bp,
                    psr_exclude_samples=psr_exclude_samples,
                    guided_tau_ms=float(tau_ref_ms) if use_guided_gcc else None,
                    guided_radius_ms=float(gcc_guided_radius_ms) if use_guided_gcc else None,
                )
            elif method == 'CC':
                est_func = lambda s1, s2, fs, mls: estimate_tdoa_cc(
                    s1, s2, fs, mls, bandpass=bp, psr_exclude_samples=psr_exclude_samples
                )
            elif method == 'NCC':
                est_func = lambda s1, s2, fs, mls: estimate_tdoa_ncc(
                    s1, s2, fs, mls, bandpass=bp, psr_exclude_samples=psr_exclude_samples
                )
            elif method == 'MUSIC':
                est_func = lambda s1, s2, fs, mls: estimate_tdoa_music(
                    s1, s2, fs, config, mls, bandpass=bp
                )

            # Baseline: MicL-MicR
            result = est_func(mic_left_seg, mic_right_seg, config['fs'], max_lag_samples)
            result['theta_deg'] = tau_to_doa(result['tau_ms'], config)
            result['theta_error'] = abs(result['theta_deg'] - theta_ref_deg)
            all_results[method]['MicL-MicR'].append(result)

            # Raw LDV
            result = est_func(ldv_raw_seg, mic_right_seg, config['fs'], max_lag_samples)
            result['theta_deg'] = tau_to_doa(result['tau_ms'], config)
            result['theta_error'] = abs(result['theta_deg'] - theta_ref_deg)
            all_results[method]['Raw_LDV'].append(result)

            # Random LDV
            result = est_func(ldv_random_seg, mic_right_seg, config['fs'], max_lag_samples)
            result['theta_deg'] = tau_to_doa(result['tau_ms'], config)
            result['theta_error'] = abs(result['theta_deg'] - theta_ref_deg)
            all_results[method]['Random_LDV'].append(result)

            # OMP LDV
            result = est_func(ldv_omp_seg, mic_right_seg, config['fs'], max_lag_samples)
            result['theta_deg'] = tau_to_doa(result['tau_ms'], config)
            result['theta_error'] = abs(result['theta_deg'] - theta_ref_deg)
            all_results[method]['OMP_LDV'].append(result)

    # Compute statistics
    logger.info("\n" + "=" * 70)
    logger.info(f"Stage 4 Results: Speaker {speaker_id}")
    logger.info(f"Geometry Truth:  τ = {tau_geom_ms:.3f} ms, θ = {theta_geom_deg:.2f}°")
    logger.info(f"Reference({truth_mode}): τ = {tau_ref_ms:.3f} ms, θ = {theta_ref_deg:.2f}°")
    logger.info("=" * 70)

    summary = {
        'speaker_id': speaker_id,
        'ground_truth': ground_truth,
        'truth_reference': {
            'mode': truth_mode,
            'label': None if truth_label is None else str(truth_label),
            'tau_ref_ms': float(tau_ref_ms),
            'theta_ref_deg': float(theta_ref_deg),
        },
        'n_segments': n_segments_used,
        'config': config,
        'segment_mode': segment_mode,
        'scan_summary': scan_summary,
        'segment_centers_sec': segment_centers_sec,
        'ldv_prealign': ldv_prealign,
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
    parser.add_argument('--speaker_key', type=str, default=None,
                        help='Override geometry speaker key (e.g., 22 for -0.8m chirp folders)')
    parser.add_argument('--output_dir', type=str,
                        default='results/stage4_doa_validation',
                        help='Output directory')
    parser.add_argument('--n_segments', type=int, default=5,
                        help='Number of segments to evaluate')
    parser.add_argument('--segment_mode', type=str, default='fixed',
                        choices=['fixed', 'scan'],
                        help='How to choose segment centers: fixed grid or scan mic-mic windows')
    parser.add_argument('--segment_spacing_sec', type=float, default=None,
                        help='Seconds between segment centers (default: 50s)')
    parser.add_argument('--segment_offset_sec', type=float, default=None,
                        help='Start time (seconds) for first segment center (default: 100s)')
    parser.add_argument('--analysis_slice_sec', type=float, default=None,
                        help='Slice length (seconds) around each segment for STFT/ISTFT')
    parser.add_argument('--eval_window_sec', type=float, default=None,
                        help='Evaluation window length (seconds) for TDoA/DoA per segment')
    parser.add_argument('--ldv_prealign', type=str, default=None,
                        choices=['none', 'gcc_phat'],
                        help='Optional fractional-delay prealignment of LDV to MicL using GCC-PHAT on eval window (useful for chirp)')
    parser.add_argument('--gcc_bandpass_low', type=float, default=None,
                        help='Bandpass low cutoff (Hz) for TDoA methods (<=0 disables bandpass)')
    parser.add_argument('--gcc_bandpass_high', type=float, default=None,
                        help='Bandpass high cutoff (Hz) for TDoA methods (<=0 disables bandpass)')
    parser.add_argument('--gcc_guided_peak_radius_ms', type=float, default=None,
                        help='Optional guided peak radius (ms) for GCC-PHAT around truth τ (requires truth reference).')
    parser.add_argument('--psr_exclude_samples', type=int, default=None,
                        help='PSR sidelobe exclusion half-width in samples')
    parser.add_argument('--scan_start_sec', type=float, default=None,
                        help='Scan start time (seconds) for segment_mode=scan (default: segment_offset_sec)')
    parser.add_argument('--scan_end_sec', type=float, default=None,
                        help='Scan end time (seconds) for segment_mode=scan (default: min(600s, end))')
    parser.add_argument('--scan_hop_sec', type=float, default=1.0,
                        help='Scan hop (seconds) for segment_mode=scan')
    parser.add_argument('--scan_psr_min_db', type=float, default=None,
                        help='Optional PSR threshold for scan candidates')
    parser.add_argument('--scan_ldv_micl_psr_min_db', type=float, default=None,
                        help='Optional LDV-MicL PSR threshold for scan candidates (only used when --ldv_prealign gcc_phat)')
    parser.add_argument('--scan_tau_err_max_ms', type=float, default=0.2,
                        help='Max |tau_est - tau_true| (ms) for scan candidates (default: 0.2ms)')
    parser.add_argument('--scan_sort_by', type=str, default='tau_err',
                        choices=['tau_err', 'psr'],
                        help='How to rank scan candidates: tau_err (default) or psr')
    parser.add_argument('--scan_min_separation_sec', type=float, default=None,
                        help='Min separation between selected scan centers (default: eval_window_sec)')
    parser.add_argument('--scan_allow_fallback', action='store_true',
                        help='Allow filling missing scan segments by PSR-only fallback windows')
    parser.add_argument('--truth_tau_ms', type=float, default=None,
                        help='Override truth reference τ (ms). If set, scan/passing use this τ instead of geometry truth.')
    parser.add_argument('--truth_theta_deg', type=float, default=None,
                        help='Optional override truth reference θ (deg). If omitted, computed from truth_tau_ms.')
    parser.add_argument('--truth_label', type=str, default=None,
                        help='Optional label to record in summary.json for the truth reference source.')

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.segment_spacing_sec is not None:
        config['segment_spacing_sec'] = float(args.segment_spacing_sec)
    if args.segment_offset_sec is not None:
        config['segment_offset_sec'] = float(args.segment_offset_sec)
    if args.analysis_slice_sec is not None:
        config['analysis_slice_sec'] = float(args.analysis_slice_sec)
    if args.eval_window_sec is not None:
        config['eval_window_sec'] = float(args.eval_window_sec)
    if args.ldv_prealign is not None:
        config['ldv_prealign'] = str(args.ldv_prealign)
    if args.gcc_bandpass_low is not None:
        config['gcc_bandpass_low'] = float(args.gcc_bandpass_low)
    if args.gcc_bandpass_high is not None:
        config['gcc_bandpass_high'] = float(args.gcc_bandpass_high)
    if args.gcc_guided_peak_radius_ms is not None:
        config['gcc_guided_peak_radius_ms'] = float(args.gcc_guided_peak_radius_ms)
    if args.psr_exclude_samples is not None:
        config['psr_exclude_samples'] = int(args.psr_exclude_samples)

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
        n_segments=args.n_segments,
        segment_mode=args.segment_mode,
        scan_start_sec=args.scan_start_sec,
        scan_end_sec=args.scan_end_sec,
        scan_hop_sec=args.scan_hop_sec,
        scan_psr_min_db=args.scan_psr_min_db,
        scan_ldv_micl_psr_min_db=args.scan_ldv_micl_psr_min_db,
        scan_tau_err_max_ms=args.scan_tau_err_max_ms,
        scan_sort_by=args.scan_sort_by,
        scan_min_separation_sec=args.scan_min_separation_sec,
        scan_allow_fallback=args.scan_allow_fallback,
        speaker_key_override=args.speaker_key,
        truth_tau_ms=args.truth_tau_ms,
        truth_theta_deg=args.truth_theta_deg,
        truth_label=args.truth_label,
    )

    return results


if __name__ == '__main__':
    main()
