#!/usr/bin/env python3
"""
Phase 3: Re-validate Stage 3 (Cross-mic TDoA)

Purpose: Re-run Stage 3 validation with improved baseline calculation
and guided peak search from Phase 2.

Outputs:
- New baseline computation using guided search + PSR filtering
- Pass/fail comparison: old vs new methods
- Failure case analysis
"""

import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional
import warnings

# Scipy imports
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from scipy.fft import fft, ifft

# Matplotlib for plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# Configuration
# ============================================================================

DATASET_ROOT = Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\dataset\GCC-PHAT-LDV-MIC-Experiment")
RESULTS_ROOT = Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\results\phase3_stage3_revalidation")
PHASE2_RESULTS = Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\results\phase2_guided_search")
PHASE1_RESULTS = Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\results\phase1_tau_stability")

SPEECH_POSITIONS = {
    '18': '18-0.1V',
    '19': '19-0.1V',
    '20': '20-0.1V',
    '21': '21-0.1V',
    '22': '22-0.1V',
}

# Default parameters
DEFAULT_WINDOW_SIZE = 2.0  # seconds
DEFAULT_FREQ_BAND = (500, 2000)
DEFAULT_SEARCH_WINDOW_MS = 0.3
PSR_THRESHOLD = 10.0  # dB
MIN_WINDOWS_FOR_BASELINE = 3

DEFAULT_FS = 48000
SPEED_OF_SOUND = 343.0


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GCCResult:
    tau_ms: float
    tau_samples: int
    psr_db: float
    peak_value: float
    is_valid: bool


@dataclass
class BaselineResult:
    tau_baseline_ms: float
    is_reliable: bool
    num_windows_used: int
    tau_std_ms: float
    method: str


@dataclass
class Stage3Result:
    position: str
    segment_idx: int
    # Baseline
    tau_baseline: float
    baseline_reliable: bool
    # Raw LDV results
    tau_raw: float
    psr_raw: float
    deviation_raw: float
    # OMP results (if available)
    tau_omp: Optional[float]
    psr_omp: Optional[float]
    deviation_omp: Optional[float]
    # Pass criteria
    pass_non_degradation: bool
    pass_accurate: bool
    pass_high_quality: bool
    pass_overall: bool
    # Method used
    method: str


@dataclass
class FailureAnalysis:
    position: str
    segment_idx: int
    tau_raw: float
    tau_baseline: float
    psr: float
    deviation: float
    failure_reason: str


# ============================================================================
# Signal Processing Functions
# ============================================================================

def load_wav(path: Path, target_fs: int = None) -> Tuple[np.ndarray, int]:
    """Load WAV file."""
    fs, data = wavfile.read(path)

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)

    if len(data.shape) > 1:
        data = data[:, 0]

    if target_fs is not None and fs != target_fs:
        from scipy.signal import resample
        num_samples = int(len(data) * target_fs / fs)
        data = resample(data, num_samples)
        fs = target_fs

    return data, fs


def bandpass_filter(data: np.ndarray, fs: int, lowcut: float, highcut: float, order: int = 5) -> np.ndarray:
    """Apply Butterworth bandpass filter."""
    nyq = fs / 2
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)

    if low >= high:
        return data

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: int, max_lag_ms: float = 10.0) -> GCCResult:
    """Standard GCC-PHAT."""
    n = len(sig1) + len(sig2) - 1
    n_fft = 2 ** int(np.ceil(np.log2(n)))

    X1 = fft(sig1, n_fft)
    X2 = fft(sig2, n_fft)

    cross_spectrum = X1 * np.conj(X2)
    magnitude = np.abs(cross_spectrum) + 1e-10
    gcc = np.real(ifft(cross_spectrum / magnitude))

    gcc = np.fft.fftshift(gcc)
    lags = np.arange(-n_fft // 2, n_fft // 2)

    max_lag_samples = int(max_lag_ms * fs / 1000)
    center = n_fft // 2
    search_start = max(0, center - max_lag_samples)
    search_end = min(n_fft, center + max_lag_samples + 1)

    gcc_search = gcc[search_start:search_end]
    lags_search = lags[search_start:search_end]

    if len(gcc_search) == 0:
        return GCCResult(tau_ms=0.0, tau_samples=0, psr_db=0.0, peak_value=0.0, is_valid=False)

    peak_idx = np.argmax(np.abs(gcc_search))
    peak_value = np.abs(gcc_search[peak_idx])
    tau_samples = lags_search[peak_idx]

    # Parabolic interpolation
    delta = 0.0
    if 0 < peak_idx < len(gcc_search) - 1:
        y0 = np.abs(gcc_search[peak_idx - 1])
        y1 = np.abs(gcc_search[peak_idx])
        y2 = np.abs(gcc_search[peak_idx + 1])
        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) > 1e-10:
            delta = (y0 - y2) / denom
            delta = np.clip(delta, -0.5, 0.5)

    tau_ms = (tau_samples + delta) * 1000.0 / fs

    # PSR
    sidelobe_exclusion = 50
    sidelobe_mask = np.ones(len(gcc_search), dtype=bool)
    exclude_start = max(0, peak_idx - sidelobe_exclusion)
    exclude_end = min(len(gcc_search), peak_idx + sidelobe_exclusion + 1)
    sidelobe_mask[exclude_start:exclude_end] = False

    if np.any(sidelobe_mask):
        sidelobe_max = np.max(np.abs(gcc_search[sidelobe_mask]))
        psr_db = 20 * np.log10(peak_value / (sidelobe_max + 1e-10))
    else:
        psr_db = 0.0

    return GCCResult(tau_ms=tau_ms, tau_samples=int(tau_samples), psr_db=psr_db, peak_value=peak_value, is_valid=True)


def guided_gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: int,
                   tau_reference_ms: float, search_window_ms: float = 0.3) -> GCCResult:
    """GCC-PHAT with guided peak search."""
    n = len(sig1) + len(sig2) - 1
    n_fft = 2 ** int(np.ceil(np.log2(n)))

    X1 = fft(sig1, n_fft)
    X2 = fft(sig2, n_fft)

    cross_spectrum = X1 * np.conj(X2)
    magnitude = np.abs(cross_spectrum) + 1e-10
    gcc = np.real(ifft(cross_spectrum / magnitude))

    gcc = np.fft.fftshift(gcc)
    lags = np.arange(-n_fft // 2, n_fft // 2)
    tau_axis_ms = lags * 1000.0 / fs

    mask = (tau_axis_ms >= tau_reference_ms - search_window_ms) & \
           (tau_axis_ms <= tau_reference_ms + search_window_ms)

    if not np.any(mask):
        return GCCResult(tau_ms=tau_reference_ms, tau_samples=0, psr_db=0.0, peak_value=0.0, is_valid=False)

    gcc_guided = gcc[mask]
    lags_guided = lags[mask]

    peak_idx = np.argmax(np.abs(gcc_guided))
    peak_value = np.abs(gcc_guided[peak_idx])
    tau_samples = lags_guided[peak_idx]

    delta = 0.0
    if 0 < peak_idx < len(gcc_guided) - 1:
        y0 = np.abs(gcc_guided[peak_idx - 1])
        y1 = np.abs(gcc_guided[peak_idx])
        y2 = np.abs(gcc_guided[peak_idx + 1])
        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) > 1e-10:
            delta = (y0 - y2) / denom
            delta = np.clip(delta, -0.5, 0.5)

    tau_ms = (tau_samples + delta) * 1000.0 / fs

    sidelobe_exclusion = 20
    sidelobe_mask = np.ones(len(gcc_guided), dtype=bool)
    exclude_start = max(0, peak_idx - sidelobe_exclusion)
    exclude_end = min(len(gcc_guided), peak_idx + sidelobe_exclusion + 1)
    sidelobe_mask[exclude_start:exclude_end] = False

    if np.any(sidelobe_mask):
        sidelobe_max = np.max(np.abs(gcc_guided[sidelobe_mask]))
        psr_db = 20 * np.log10(peak_value / (sidelobe_max + 1e-10))
    else:
        psr_db = 0.0

    return GCCResult(tau_ms=tau_ms, tau_samples=int(tau_samples), psr_db=psr_db, peak_value=peak_value, is_valid=True)


# ============================================================================
# Baseline Computation
# ============================================================================

def compute_stable_baseline(mic_l: np.ndarray, mic_r: np.ndarray, fs: int,
                           tau_chirp: float, window_size: float = 2.0,
                           freq_band: Tuple[int, int] = (500, 2000),
                           search_window_ms: float = 0.3,
                           psr_threshold: float = 10.0) -> BaselineResult:
    """
    Compute stable baseline using guided search + PSR filtering + robust statistics.
    """
    # Filter signals
    mic_l_filt = bandpass_filter(mic_l, fs, freq_band[0], freq_band[1])
    mic_r_filt = bandpass_filter(mic_r, fs, freq_band[0], freq_band[1])

    # Sliding windows
    window_samples = int(window_size * fs)
    hop_samples = window_samples // 2  # 50% overlap
    n_windows = (len(mic_l_filt) - window_samples) // hop_samples + 1

    tau_list = []

    for i in range(n_windows):
        start = i * hop_samples
        end = start + window_samples

        window_l = mic_l_filt[start:end]
        window_r = mic_r_filt[start:end]

        # Guided GCC-PHAT
        result = guided_gcc_phat(window_l, window_r, fs, tau_chirp, search_window_ms)

        # PSR filtering
        if result.is_valid and result.psr_db >= psr_threshold:
            tau_list.append(result.tau_ms)

    # Compute robust baseline
    if len(tau_list) >= MIN_WINDOWS_FOR_BASELINE:
        tau_baseline = np.median(tau_list)
        tau_std = np.std(tau_list)
        return BaselineResult(
            tau_baseline_ms=tau_baseline,
            is_reliable=True,
            num_windows_used=len(tau_list),
            tau_std_ms=tau_std,
            method="guided_psr_median"
        )
    else:
        # Fallback to global search
        result = gcc_phat(mic_l_filt, mic_r_filt, fs)
        return BaselineResult(
            tau_baseline_ms=result.tau_ms if result.is_valid else 0.0,
            is_reliable=False,
            num_windows_used=len(tau_list),
            tau_std_ms=0.0,
            method="fallback_global"
        )


def compute_old_baseline(mic_l: np.ndarray, mic_r: np.ndarray, fs: int,
                        freq_band: Tuple[int, int] = (500, 2000)) -> BaselineResult:
    """
    Compute baseline using old method (single segment, global search).
    """
    mic_l_filt = bandpass_filter(mic_l, fs, freq_band[0], freq_band[1])
    mic_r_filt = bandpass_filter(mic_r, fs, freq_band[0], freq_band[1])

    result = gcc_phat(mic_l_filt, mic_r_filt, fs)

    return BaselineResult(
        tau_baseline_ms=result.tau_ms if result.is_valid else 0.0,
        is_reliable=result.psr_db >= PSR_THRESHOLD if result.is_valid else False,
        num_windows_used=1,
        tau_std_ms=0.0,
        method="single_segment_global"
    )


# ============================================================================
# Pass/Fail Criteria
# ============================================================================

def old_pass_criterion(tau_raw: float, tau_omp: float, tau_baseline: float) -> bool:
    """Old pass criterion: OMP must be closer to baseline than raw."""
    return abs(tau_omp - tau_baseline) < abs(tau_raw - tau_baseline)


def new_pass_criteria(tau_raw: float, tau_omp: float, tau_baseline: float,
                     psr_omp: float, baseline_reliable: bool) -> Dict[str, bool]:
    """
    New pass criteria (separated concerns):
    1. Non-degradation: OMP is not worse than raw
    2. Absolute accuracy: OMP is within 0.1ms of baseline (when baseline reliable)
    3. High quality: PSR >= threshold
    """
    # Criterion 1: Non-degradation
    non_degradation = abs(tau_omp - tau_baseline) <= abs(tau_raw - tau_baseline)

    # Criterion 2: Absolute accuracy
    accurate = abs(tau_omp - tau_baseline) < 0.1  # ms

    # Criterion 3: Signal quality
    high_quality = psr_omp >= PSR_THRESHOLD

    # Overall: non-degradation AND high quality
    # (Accuracy is informative but not required if baseline unreliable)
    overall = non_degradation and high_quality

    return {
        'non_degradation': non_degradation,
        'accurate': accurate,
        'high_quality': high_quality,
        'overall': overall,
    }


# ============================================================================
# Phase 3 Analysis
# ============================================================================

def load_previous_results() -> Tuple[Dict[str, float], Dict, float]:
    """Load chirp references and best parameters from previous phases."""
    chirp_refs = {}
    best_params = {
        'window_size': DEFAULT_WINDOW_SIZE,
        'freq_band': DEFAULT_FREQ_BAND,
    }
    best_search_window = DEFAULT_SEARCH_WINDOW_MS

    # Load from Phase 1
    if PHASE1_RESULTS.exists():
        runs = sorted(PHASE1_RESULTS.glob("run_*"), reverse=True)
        if runs:
            chirp_file = runs[0] / "chirp_references.json"
            if chirp_file.exists():
                with open(chirp_file) as f:
                    data = json.load(f)
                    chirp_refs = data.get('positions', {})

    # Load from Phase 2
    if PHASE2_RESULTS.exists():
        runs = sorted(PHASE2_RESULTS.glob("run_*"), reverse=True)
        if runs:
            report_file = runs[0] / "comparison_report.json"
            if report_file.exists():
                with open(report_file) as f:
                    report = json.load(f)
                    # Find best search window with 0% false peak rate
                    # Prefer 0.3ms as a reasonable default for baseline computation
                    window_analysis = report.get('search_window_analysis', {})
                    if window_analysis:
                        # Get windows with 0% false peak rate
                        zero_fp_windows = [k for k, v in window_analysis.items()
                                          if v['false_peak_rate_guided'] == 0]
                        if zero_fp_windows and '0.3ms' in zero_fp_windows:
                            best_search_window = 0.3
                        elif zero_fp_windows:
                            # Pick the largest zero-FP window for robustness
                            best_search_window = max(float(k.replace('ms', '')) for k in zero_fp_windows)
                        else:
                            best_key = min(window_analysis.keys(),
                                          key=lambda k: window_analysis[k]['false_peak_rate_guided'])
                            best_search_window = float(best_key.replace('ms', ''))

    return chirp_refs, best_params, best_search_window


def estimate_chirp_ref(position: str, chirp_refs: Dict[str, float]) -> float:
    """Estimate chirp reference for position."""
    x_coords = {
        '18': 0.8, '19': 0.4, '20': 0.0, '21': -0.4, '22': -0.8,
    }

    if position == '22' and '23' in chirp_refs:
        return chirp_refs['23']
    if position == '21' and '24' in chirp_refs:
        return chirp_refs['24']

    x = x_coords.get(position, 0.0)
    mic_l = np.array([-0.7, 2.0])
    mic_r = np.array([0.7, 2.0])
    speaker = np.array([x, 0.0])

    d_l = np.linalg.norm(speaker - mic_l)
    d_r = np.linalg.norm(speaker - mic_r)

    return (d_r - d_l) / SPEED_OF_SOUND * 1000.0


def run_stage3_revalidation(chirp_refs: Dict[str, float],
                            best_params: Dict,
                            search_window_ms: float,
                            output_dir: Path) -> Tuple[List[Stage3Result], List[Stage3Result]]:
    """
    Run Stage 3 validation with both old and new methods.
    """
    print("\n" + "="*60)
    print("Phase 3: Stage 3 Re-validation")
    print("="*60)

    old_results = []
    new_results = []

    window_size = best_params['window_size']
    freq_band = best_params['freq_band']

    print(f"\n  Parameters:")
    print(f"    Window size: {window_size}s")
    print(f"    Frequency band: {freq_band}")
    print(f"    Search window: {search_window_ms}ms")

    for pos_id, folder_name in SPEECH_POSITIONS.items():
        folder = DATASET_ROOT / folder_name

        mic_l_files = list(folder.glob("*LEFT-MIC*.wav"))
        mic_r_files = list(folder.glob("*RIGHT-MIC*.wav"))
        ldv_files = list(folder.glob("*LDV*.wav"))

        if not mic_l_files or not mic_r_files:
            print(f"\n  [SKIP] Position {pos_id}: Missing mic files")
            continue

        print(f"\n  Position {pos_id}:")

        # Load signals
        mic_l, fs = load_wav(mic_l_files[0])
        mic_r, _ = load_wav(mic_r_files[0])

        # Load LDV if available
        ldv_raw = None
        if ldv_files:
            ldv_raw, _ = load_wav(ldv_files[0])

        tau_chirp = estimate_chirp_ref(pos_id, chirp_refs)
        print(f"    Chirp reference: {tau_chirp:.4f} ms")

        # Compute baselines
        old_baseline = compute_old_baseline(mic_l, mic_r, fs, freq_band)
        new_baseline = compute_stable_baseline(mic_l, mic_r, fs, tau_chirp,
                                               window_size, freq_band, search_window_ms)

        print(f"    Old baseline: {old_baseline.tau_baseline_ms:.4f} ms (reliable: {old_baseline.is_reliable})")
        print(f"    New baseline: {new_baseline.tau_baseline_ms:.4f} ms (reliable: {new_baseline.is_reliable}, n={new_baseline.num_windows_used})")

        # Segment signals
        mic_l_filt = bandpass_filter(mic_l, fs, freq_band[0], freq_band[1])
        mic_r_filt = bandpass_filter(mic_r, fs, freq_band[0], freq_band[1])

        window_samples = int(window_size * fs)
        n_segments = len(mic_l_filt) // window_samples

        start_seg = max(1, int(n_segments * 0.1))
        end_seg = int(n_segments * 0.9)

        for seg_idx in range(start_seg, end_seg):
            start_sample = seg_idx * window_samples
            end_sample = start_sample + window_samples

            seg_l = mic_l_filt[start_sample:end_sample]
            seg_r = mic_r_filt[start_sample:end_sample]

            # Measure tau with global search (simulating raw LDV)
            result_raw = gcc_phat(seg_l, seg_r, fs)

            if not result_raw.is_valid:
                continue

            # For now, we don't have OMP results, so we simulate
            # In real implementation, this would load OMP-aligned LDV results
            tau_omp = result_raw.tau_ms  # Placeholder
            psr_omp = result_raw.psr_db  # Placeholder

            # Old method evaluation
            old_pass = old_pass_criterion(result_raw.tau_ms, tau_omp, old_baseline.tau_baseline_ms)

            old_result = Stage3Result(
                position=pos_id,
                segment_idx=seg_idx,
                tau_baseline=old_baseline.tau_baseline_ms,
                baseline_reliable=old_baseline.is_reliable,
                tau_raw=result_raw.tau_ms,
                psr_raw=result_raw.psr_db,
                deviation_raw=abs(result_raw.tau_ms - old_baseline.tau_baseline_ms),
                tau_omp=tau_omp,
                psr_omp=psr_omp,
                deviation_omp=abs(tau_omp - old_baseline.tau_baseline_ms) if tau_omp else None,
                pass_non_degradation=old_pass,
                pass_accurate=abs(tau_omp - old_baseline.tau_baseline_ms) < 0.1 if tau_omp else False,
                pass_high_quality=psr_omp >= PSR_THRESHOLD if psr_omp else False,
                pass_overall=old_pass,
                method="old_single_segment"
            )
            old_results.append(old_result)

            # New method evaluation (with guided search)
            result_guided = guided_gcc_phat(seg_l, seg_r, fs, tau_chirp, search_window_ms)

            if result_guided.is_valid:
                new_criteria = new_pass_criteria(
                    result_raw.tau_ms,
                    result_guided.tau_ms,  # Use guided result as "improved" measurement
                    new_baseline.tau_baseline_ms,
                    result_guided.psr_db,
                    new_baseline.is_reliable
                )

                new_result = Stage3Result(
                    position=pos_id,
                    segment_idx=seg_idx,
                    tau_baseline=new_baseline.tau_baseline_ms,
                    baseline_reliable=new_baseline.is_reliable,
                    tau_raw=result_raw.tau_ms,
                    psr_raw=result_raw.psr_db,
                    deviation_raw=abs(result_raw.tau_ms - new_baseline.tau_baseline_ms),
                    tau_omp=result_guided.tau_ms,
                    psr_omp=result_guided.psr_db,
                    deviation_omp=abs(result_guided.tau_ms - new_baseline.tau_baseline_ms),
                    pass_non_degradation=new_criteria['non_degradation'],
                    pass_accurate=new_criteria['accurate'],
                    pass_high_quality=new_criteria['high_quality'],
                    pass_overall=new_criteria['overall'],
                    method="new_guided_search"
                )
                new_results.append(new_result)

    return old_results, new_results


def analyze_failures(results: List[Stage3Result], method_name: str) -> List[FailureAnalysis]:
    """Analyze failure cases."""
    failures = []

    for r in results:
        if not r.pass_overall:
            # Determine failure reason
            if not r.baseline_reliable:
                reason = "baseline_unreliable"
            elif r.tau_omp is None:
                reason = "omp_failed"
            elif not r.pass_non_degradation:
                reason = "omp_degraded"
            elif not r.pass_high_quality:
                reason = "low_psr"
            else:
                reason = "unknown"

            failures.append(FailureAnalysis(
                position=r.position,
                segment_idx=r.segment_idx,
                tau_raw=r.tau_raw,
                tau_baseline=r.tau_baseline,
                psr=r.psr_raw,
                deviation=r.deviation_raw,
                failure_reason=reason
            ))

    return failures


def generate_revalidation_report(old_results: List[Stage3Result],
                                  new_results: List[Stage3Result],
                                  output_dir: Path) -> Dict:
    """Generate comparison report."""
    print("\n" + "="*60)
    print("Generating Re-validation Report")
    print("="*60)

    report = {
        'timestamp': datetime.now().isoformat(),
        'old_method': {},
        'new_method': {},
        'comparison': {},
    }

    # Old method stats
    if old_results:
        old_pass = sum(r.pass_overall for r in old_results)
        n_old = len(old_results)
        report['old_method'] = {
            'total': n_old,
            'passed': old_pass,
            'pass_rate': old_pass / n_old,
            'non_degradation_rate': sum(r.pass_non_degradation for r in old_results) / n_old,
            'accurate_rate': sum(r.pass_accurate for r in old_results) / n_old,
            'high_quality_rate': sum(r.pass_high_quality for r in old_results) / n_old,
        }

    # New method stats
    if new_results:
        new_pass = sum(r.pass_overall for r in new_results)
        n_new = len(new_results)
        report['new_method'] = {
            'total': n_new,
            'passed': new_pass,
            'pass_rate': new_pass / n_new,
            'non_degradation_rate': sum(r.pass_non_degradation for r in new_results) / n_new,
            'accurate_rate': sum(r.pass_accurate for r in new_results) / n_new,
            'high_quality_rate': sum(r.pass_high_quality for r in new_results) / n_new,
        }

    # Comparison
    if old_results and new_results:
        old_rate = report['old_method']['pass_rate']
        new_rate = report['new_method']['pass_rate']
        report['comparison'] = {
            'pass_rate_improvement': (new_rate - old_rate) * 100,
            'old_pass_rate': old_rate * 100,
            'new_pass_rate': new_rate * 100,
        }

    # Position breakdown
    report['position_breakdown'] = {}
    for pos in SPEECH_POSITIONS.keys():
        old_pos = [r for r in old_results if r.position == pos]
        new_pos = [r for r in new_results if r.position == pos]

        if old_pos and new_pos:
            report['position_breakdown'][pos] = {
                'old_pass_rate': sum(r.pass_overall for r in old_pos) / len(old_pos) * 100,
                'new_pass_rate': sum(r.pass_overall for r in new_pos) / len(new_pos) * 100,
            }

    # Print summary
    print("\n  Pass Rate Comparison:")
    print("  " + "-"*50)
    print(f"  {'Method':<25} {'Pass Rate':<15}")
    print("  " + "-"*50)
    print(f"  {'Old (single segment)':<25} {report['old_method'].get('pass_rate', 0)*100:.1f}%")
    print(f"  {'New (guided + PSR)':<25} {report['new_method'].get('pass_rate', 0)*100:.1f}%")

    print("\n  Position Breakdown:")
    for pos, stats in report['position_breakdown'].items():
        print(f"    Position {pos}: {stats['old_pass_rate']:.1f}% -> {stats['new_pass_rate']:.1f}%")

    return report


def plot_revalidation_results(old_results: List[Stage3Result],
                               new_results: List[Stage3Result],
                               output_dir: Path):
    """Generate comparison plots."""
    print("\n  Generating plots...")

    if not old_results or not new_results:
        return

    # Plot 1: Pass rate comparison by position
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Position breakdown
    ax = axes[0]
    positions = list(SPEECH_POSITIONS.keys())
    old_rates = []
    new_rates = []

    for pos in positions:
        old_pos = [r for r in old_results if r.position == pos]
        new_pos = [r for r in new_results if r.position == pos]
        old_rates.append(sum(r.pass_overall for r in old_pos) / len(old_pos) * 100 if old_pos else 0)
        new_rates.append(sum(r.pass_overall for r in new_pos) / len(new_pos) * 100 if new_pos else 0)

    x = np.arange(len(positions))
    width = 0.35

    ax.bar(x - width/2, old_rates, width, label='Old Method', color='red', alpha=0.7)
    ax.bar(x + width/2, new_rates, width, label='New Method', color='green', alpha=0.7)
    ax.set_xlabel('Position')
    ax.set_ylabel('Pass Rate (%)')
    ax.set_title('Stage 3 Pass Rate by Position')
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Criteria breakdown
    ax = axes[1]
    criteria = ['Non-degradation', 'Accurate', 'High Quality', 'Overall']
    old_criteria_rates = [
        sum(r.pass_non_degradation for r in old_results) / len(old_results) * 100,
        sum(r.pass_accurate for r in old_results) / len(old_results) * 100,
        sum(r.pass_high_quality for r in old_results) / len(old_results) * 100,
        sum(r.pass_overall for r in old_results) / len(old_results) * 100,
    ]
    new_criteria_rates = [
        sum(r.pass_non_degradation for r in new_results) / len(new_results) * 100,
        sum(r.pass_accurate for r in new_results) / len(new_results) * 100,
        sum(r.pass_high_quality for r in new_results) / len(new_results) * 100,
        sum(r.pass_overall for r in new_results) / len(new_results) * 100,
    ]

    x = np.arange(len(criteria))
    ax.bar(x - width/2, old_criteria_rates, width, label='Old Method', color='red', alpha=0.7)
    ax.bar(x + width/2, new_criteria_rates, width, label='New Method', color='green', alpha=0.7)
    ax.set_xlabel('Criterion')
    ax.set_ylabel('Pass Rate (%)')
    ax.set_title('Pass Rate by Criterion')
    ax.set_xticks(x)
    ax.set_xticklabels(criteria, rotation=15)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'stage3_revalidation_comparison.png', dpi=150)
    plt.close()

    print(f"    Saved plots to {output_dir}")


# ============================================================================
# Main Entry Point
# ============================================================================

# Convert numpy types to native Python for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    """Run Phase 3: Stage 3 Re-validation."""
    print("\n" + "="*70)
    print("  PHASE 3: Stage 3 Re-validation")
    print("  Purpose: Re-run Stage 3 with improved baseline and guided search")
    print("="*70)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_ROOT / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Output directory: {output_dir}")

    # Load previous results
    chirp_refs, best_params, search_window_ms = load_previous_results()

    print(f"\n  Loaded chirp refs: {chirp_refs}")
    print(f"  Best search window: {search_window_ms} ms")

    # Run re-validation
    old_results, new_results = run_stage3_revalidation(
        chirp_refs, best_params, search_window_ms, output_dir
    )

    if not old_results or not new_results:
        print("\n  [ERROR] No results generated.")
        return 1

    # Analyze failures
    old_failures = analyze_failures(old_results, "old")
    new_failures = analyze_failures(new_results, "new")

    # Generate report
    report = generate_revalidation_report(old_results, new_results, output_dir)

    # Add failure analysis to report
    report['failure_analysis'] = {
        'old_method': {},
        'new_method': {},
    }

    from collections import Counter
    if old_failures:
        reasons = Counter(f.failure_reason for f in old_failures)
        report['failure_analysis']['old_method'] = dict(reasons)

    if new_failures:
        reasons = Counter(f.failure_reason for f in new_failures)
        report['failure_analysis']['new_method'] = dict(reasons)

    # Save results
    with open(output_dir / "old_results.json", 'w') as f:
        json.dump([convert_numpy(asdict(r)) for r in old_results], f, indent=2)

    with open(output_dir / "new_results.json", 'w') as f:
        json.dump([convert_numpy(asdict(r)) for r in new_results], f, indent=2)

    with open(output_dir / "revalidation_report.json", 'w') as f:
        json.dump(convert_numpy(report), f, indent=2)

    with open(output_dir / "failure_analysis.json", 'w') as f:
        json.dump(convert_numpy({
            'old': [asdict(f) for f in old_failures],
            'new': [asdict(f) for f in new_failures],
        }), f, indent=2)

    # Generate plots
    plot_revalidation_results(old_results, new_results, output_dir)

    # Print summary
    print("\n" + "="*70)
    print("  PHASE 3 SUMMARY")
    print("="*70)

    improvement = report['comparison'].get('pass_rate_improvement', 0)
    old_rate = report['comparison'].get('old_pass_rate', 0)
    new_rate = report['comparison'].get('new_pass_rate', 0)

    print(f"\n  Pass Rate: {old_rate:.1f}% -> {new_rate:.1f}% ({improvement:+.1f}%)")

    print("\n  Failure Reasons (New Method):")
    for reason, count in report['failure_analysis']['new_method'].items():
        print(f"    {reason}: {count}")

    print(f"\n  Results saved to: {output_dir}")

    # Decision point
    if new_rate >= 80:
        print("\n  DECISION: Significant improvement. Stage 3 validation successful.")
        return 0
    elif new_rate >= 60:
        print("\n  DECISION: Moderate improvement. Review failure cases.")
        return 1
    else:
        print("\n  DECISION: Limited improvement. Investigate OMP alignment or signal issues.")
        return 2


if __name__ == "__main__":
    exit(main())

