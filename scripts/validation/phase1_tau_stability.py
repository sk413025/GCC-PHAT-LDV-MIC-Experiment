#!/usr/bin/env python3
"""
Phase 1: Speech tau Stability Diagnosis

Purpose: Find conditions under which speech mic-mic tau measurement is stable
compared to chirp reference.

Outputs:
- Chirp reference tau for each speaker position
- tau stability analysis across window sizes, frequency bands
- tau distribution plots
- Collapse pattern analysis (tau â†’ 0 cases)
"""

import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import warnings

# Scipy imports
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, correlate
from scipy.fft import fft, ifft, fftfreq

# Matplotlib for plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# Configuration
# ============================================================================

DATASET_ROOT = Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\dataset\GCC-PHAT-LDV-MIC-Experiment")
RESULTS_ROOT = Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\results\phase1_tau_stability")

# Speaker positions with speech data
SPEECH_POSITIONS = {
    '18': '18-0.1V',
    '19': '19-0.1V',
    '20': '20-0.1V',
    '21': '21-0.1V',
    '22': '22-0.1V',
}

# Chirp positions (for reference)
CHIRP_POSITIONS = {
    '23': '23-chirp(-0.8m)',
    '24': '24-chirp(-0.4m)',
}

# Test parameters
WINDOW_SIZES_SEC = [0.5, 1.0, 2.0, 5.0]
FREQ_BANDS = {
    '500-2000Hz': (500, 2000),
    '200-4000Hz': (200, 4000),
    '100-8000Hz': (100, 8000),
}
PSR_THRESHOLD = 10.0  # dB
TAU_STABLE_THRESHOLD_MS = 0.3  # within 0.3ms of chirp reference

DEFAULT_FS = 48000
SPEED_OF_SOUND = 343.0  # m/s


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
    reason: str = ""


@dataclass
class StabilityResult:
    position: str
    window_size: float
    freq_band: str
    segment_idx: int
    segment_start_sec: float
    tau_ms: float
    psr_db: float
    tau_chirp_ref: float
    deviation_ms: float
    is_stable: bool
    is_valid: bool


# ============================================================================
# Signal Processing Functions
# ============================================================================

def load_wav(path: Path, target_fs: int = None) -> Tuple[np.ndarray, int]:
    """Load WAV file and optionally resample."""
    fs, data = wavfile.read(path)

    # Convert to float
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)

    # Handle stereo
    if len(data.shape) > 1:
        data = data[:, 0]

    # Resample if needed
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
        warnings.warn(f"Invalid bandpass: {lowcut}-{highcut} Hz at fs={fs}")
        return data

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: int, max_lag_ms: float = 10.0) -> GCCResult:
    """
    Compute GCC-PHAT with parabolic interpolation for sub-sample precision.

    Returns:
        GCCResult with tau_ms, psr_db, etc.
    """
    n = len(sig1) + len(sig2) - 1
    n_fft = 2 ** int(np.ceil(np.log2(n)))

    # FFT
    X1 = fft(sig1, n_fft)
    X2 = fft(sig2, n_fft)

    # Cross-spectrum with phase transform
    cross_spectrum = X1 * np.conj(X2)
    magnitude = np.abs(cross_spectrum) + 1e-10
    gcc = np.real(ifft(cross_spectrum / magnitude))

    # Shift to center
    gcc = np.fft.fftshift(gcc)
    lags = np.arange(-n_fft // 2, n_fft // 2)

    # Limit search to max_lag
    max_lag_samples = int(max_lag_ms * fs / 1000)
    center = n_fft // 2
    search_start = max(0, center - max_lag_samples)
    search_end = min(n_fft, center + max_lag_samples + 1)

    gcc_search = gcc[search_start:search_end]
    lags_search = lags[search_start:search_end]

    if len(gcc_search) == 0:
        return GCCResult(
            tau_ms=0.0, tau_samples=0, psr_db=0.0,
            peak_value=0.0, is_valid=False, reason="Empty search window"
        )

    # Find peak
    peak_idx = np.argmax(np.abs(gcc_search))
    peak_value = np.abs(gcc_search[peak_idx])
    tau_samples = lags_search[peak_idx]

    # Parabolic interpolation for sub-sample precision
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

    # Compute PSR
    sidelobe_exclusion = 50  # samples
    sidelobe_mask = np.ones(len(gcc_search), dtype=bool)
    exclude_start = max(0, peak_idx - sidelobe_exclusion)
    exclude_end = min(len(gcc_search), peak_idx + sidelobe_exclusion + 1)
    sidelobe_mask[exclude_start:exclude_end] = False

    if np.any(sidelobe_mask):
        sidelobe_max = np.max(np.abs(gcc_search[sidelobe_mask]))
        psr_db = 20 * np.log10(peak_value / (sidelobe_max + 1e-10))
    else:
        psr_db = 0.0

    return GCCResult(
        tau_ms=tau_ms,
        tau_samples=int(tau_samples),
        psr_db=psr_db,
        peak_value=peak_value,
        is_valid=True
    )


def guided_gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: int,
                   tau_reference_ms: float, search_window_ms: float = 0.5) -> GCCResult:
    """
    GCC-PHAT with guided peak search around a reference tau.

    Args:
        sig1, sig2: Input signals
        fs: Sample rate
        tau_reference_ms: Reference tau from chirp
        search_window_ms: Search within Â±search_window_ms of reference

    Returns:
        GCCResult from guided search
    """
    n = len(sig1) + len(sig2) - 1
    n_fft = 2 ** int(np.ceil(np.log2(n)))

    # FFT
    X1 = fft(sig1, n_fft)
    X2 = fft(sig2, n_fft)

    # Cross-spectrum with phase transform
    cross_spectrum = X1 * np.conj(X2)
    magnitude = np.abs(cross_spectrum) + 1e-10
    gcc = np.real(ifft(cross_spectrum / magnitude))

    # Shift to center
    gcc = np.fft.fftshift(gcc)
    lags = np.arange(-n_fft // 2, n_fft // 2)
    tau_axis_ms = lags * 1000.0 / fs

    # Guided search window around reference
    mask = (tau_axis_ms >= tau_reference_ms - search_window_ms) & \
           (tau_axis_ms <= tau_reference_ms + search_window_ms)

    if not np.any(mask):
        return GCCResult(
            tau_ms=tau_reference_ms, tau_samples=0, psr_db=0.0,
            peak_value=0.0, is_valid=False, reason="Empty guided search window"
        )

    gcc_guided = gcc[mask]
    lags_guided = lags[mask]
    tau_axis_guided = tau_axis_ms[mask]

    # Find peak in guided window
    peak_idx = np.argmax(np.abs(gcc_guided))
    peak_value = np.abs(gcc_guided[peak_idx])
    tau_samples = lags_guided[peak_idx]

    # Parabolic interpolation
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

    # PSR within guided window
    sidelobe_exclusion = 20  # smaller for guided search
    sidelobe_mask = np.ones(len(gcc_guided), dtype=bool)
    exclude_start = max(0, peak_idx - sidelobe_exclusion)
    exclude_end = min(len(gcc_guided), peak_idx + sidelobe_exclusion + 1)
    sidelobe_mask[exclude_start:exclude_end] = False

    if np.any(sidelobe_mask):
        sidelobe_max = np.max(np.abs(gcc_guided[sidelobe_mask]))
        psr_db = 20 * np.log10(peak_value / (sidelobe_max + 1e-10))
    else:
        psr_db = 0.0

    return GCCResult(
        tau_ms=tau_ms,
        tau_samples=int(tau_samples),
        psr_db=psr_db,
        peak_value=peak_value,
        is_valid=True
    )


# ============================================================================
# Phase 1.1: Build Chirp Reference
# ============================================================================

def build_chirp_reference(output_dir: Path) -> Dict[str, float]:
    """
    Measure mic-mic tau using chirp signals for each position.
    These serve as ground truth references.
    """
    print("\n" + "="*60)
    print("Phase 1.1: Building Chirp Reference Values")
    print("="*60)

    chirp_refs = {}

    for pos_id, folder_name in CHIRP_POSITIONS.items():
        folder = DATASET_ROOT / folder_name

        # Find mic files
        mic_l_files = list(folder.glob("*LEFT-MIC*.wav"))
        mic_r_files = list(folder.glob("*RIGHT-MIC*.wav"))

        if not mic_l_files or not mic_r_files:
            print(f"  [SKIP] Position {pos_id}: Missing mic files")
            continue

        mic_l_path = mic_l_files[0]
        mic_r_path = mic_r_files[0]

        print(f"\n  Position {pos_id} ({folder_name}):")
        print(f"    MicL: {mic_l_path.name}")
        print(f"    MicR: {mic_r_path.name}")

        # Load signals
        mic_l, fs = load_wav(mic_l_path)
        mic_r, _ = load_wav(mic_r_path)

        # Use full bandwidth for chirp
        result = gcc_phat(mic_l, mic_r, fs, max_lag_ms=5.0)

        print(f"    tau = {result.tau_ms:.4f} ms, PSR = {result.psr_db:.1f} dB")

        chirp_refs[pos_id] = result.tau_ms

    # Save reference values
    ref_file = output_dir / "chirp_references.json"
    with open(ref_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'positions': chirp_refs,
            'description': 'Mic-mic tau (ms) measured from chirp signals'
        }, f, indent=2)

    print(f"\n  Saved chirp references to: {ref_file}")

    return chirp_refs


def estimate_speech_position_chirp_ref(position: str, chirp_refs: Dict[str, float]) -> Optional[float]:
    """
    Estimate chirp reference for speech position by interpolation.

    Speech positions 18-22 correspond to x = 0.8, 0.4, 0, -0.4, -0.8 m
    Chirp positions 23, 24 correspond to x = -0.8, -0.4 m
    """
    # Map speech position to x coordinate
    x_coords = {
        '18': 0.8,
        '19': 0.4,
        '20': 0.0,
        '21': -0.4,
        '22': -0.8,
    }

    if position not in x_coords:
        return None

    x = x_coords[position]

    # If we have chirp reference for position 23 (-0.8m), use it for 22
    if position == '22' and '23' in chirp_refs:
        return chirp_refs['23']

    # If we have chirp reference for position 24 (-0.4m), use it for 21
    if position == '21' and '24' in chirp_refs:
        return chirp_refs['24']

    # For other positions, estimate from geometry
    # tau = (d_R - d_L) / c, where d is distance to each mic
    mic_l = np.array([-0.7, 2.0])
    mic_r = np.array([0.7, 2.0])
    speaker = np.array([x, 0.0])

    d_l = np.linalg.norm(speaker - mic_l)
    d_r = np.linalg.norm(speaker - mic_r)

    tau_geo_ms = (d_r - d_l) / SPEED_OF_SOUND * 1000.0

    return tau_geo_ms


# ============================================================================
# Phase 1.2: Speech tau Stability Analysis
# ============================================================================

def analyze_speech_stability(chirp_refs: Dict[str, float], output_dir: Path) -> List[StabilityResult]:
    """
    Analyze tau stability for speech signals across different parameters.
    """
    print("\n" + "="*60)
    print("Phase 1.2: Speech tau Stability Analysis")
    print("="*60)

    all_results = []

    for pos_id, folder_name in SPEECH_POSITIONS.items():
        folder = DATASET_ROOT / folder_name

        # Find mic files
        mic_l_files = list(folder.glob("*LEFT-MIC*.wav"))
        mic_r_files = list(folder.glob("*RIGHT-MIC*.wav"))

        if not mic_l_files or not mic_r_files:
            print(f"  [SKIP] Position {pos_id}: Missing mic files")
            continue

        mic_l_path = mic_l_files[0]
        mic_r_path = mic_r_files[0]

        print(f"\n  Position {pos_id} ({folder_name}):")

        # Load signals
        mic_l, fs = load_wav(mic_l_path)
        mic_r, _ = load_wav(mic_r_path)

        # Get chirp reference for this position
        tau_chirp = estimate_speech_position_chirp_ref(pos_id, chirp_refs)
        if tau_chirp is None:
            print(f"    [WARN] No chirp reference for position {pos_id}")
            tau_chirp = 0.0

        print(f"    Chirp reference tau = {tau_chirp:.4f} ms")
        print(f"    Signal length: {len(mic_l)/fs:.1f} sec")

        # Test across parameters
        for window_size in WINDOW_SIZES_SEC:
            for band_name, (low, high) in FREQ_BANDS.items():
                # Filter signals
                mic_l_filt = bandpass_filter(mic_l, fs, low, high)
                mic_r_filt = bandpass_filter(mic_r, fs, low, high)

                # Segment signal
                window_samples = int(window_size * fs)
                n_segments = len(mic_l_filt) // window_samples

                # Skip first and last 10% to avoid transients
                start_seg = max(1, int(n_segments * 0.1))
                end_seg = int(n_segments * 0.9)

                for seg_idx in range(start_seg, end_seg):
                    start_sample = seg_idx * window_samples
                    end_sample = start_sample + window_samples

                    seg_l = mic_l_filt[start_sample:end_sample]
                    seg_r = mic_r_filt[start_sample:end_sample]

                    # Measure tau
                    result = gcc_phat(seg_l, seg_r, fs, max_lag_ms=5.0)

                    # Compute deviation from chirp reference
                    deviation = abs(result.tau_ms - tau_chirp)
                    is_stable = deviation < TAU_STABLE_THRESHOLD_MS

                    stability_result = StabilityResult(
                        position=pos_id,
                        window_size=window_size,
                        freq_band=band_name,
                        segment_idx=seg_idx,
                        segment_start_sec=start_sample / fs,
                        tau_ms=result.tau_ms,
                        psr_db=result.psr_db,
                        tau_chirp_ref=tau_chirp,
                        deviation_ms=deviation,
                        is_stable=is_stable,
                        is_valid=result.is_valid
                    )
                    all_results.append(stability_result)

        # Progress indicator
        print(f"    Analyzed {len([r for r in all_results if r.position == pos_id])} segments")

    return all_results


def generate_stability_report(results: List[StabilityResult], output_dir: Path) -> Dict:
    """
    Generate statistical report on tau stability.
    """
    print("\n" + "="*60)
    print("Generating Stability Report")
    print("="*60)

    # Group by (window_size, freq_band)
    from collections import defaultdict
    grouped = defaultdict(list)

    for r in results:
        if r.is_valid:
            key = (r.window_size, r.freq_band)
            grouped[key].append(r)

    report = {
        'timestamp': datetime.now().isoformat(),
        'total_segments': len(results),
        'valid_segments': len([r for r in results if r.is_valid]),
        'parameter_analysis': {},
        'position_analysis': {},
        'collapse_analysis': {},
    }

    # Parameter analysis
    print("\n  Parameter Analysis:")
    print("  " + "-"*70)
    print(f"  {'Window':<10} {'Band':<15} {'N':<6} {'Mean Dev':<10} {'Std Dev':<10} {'Stable %':<10} {'Mean PSR':<10}")
    print("  " + "-"*70)

    for (window, band), group in sorted(grouped.items()):
        deviations = [r.deviation_ms for r in group]
        stabilities = [r.is_stable for r in group]
        psrs = [r.psr_db for r in group]

        mean_dev = np.mean(deviations)
        std_dev = np.std(deviations)
        stable_rate = np.mean(stabilities) * 100
        mean_psr = np.mean(psrs)

        print(f"  {window:<10.1f} {band:<15} {len(group):<6} {mean_dev:<10.4f} {std_dev:<10.4f} {stable_rate:<10.1f} {mean_psr:<10.1f}")

        report['parameter_analysis'][f"{window}s_{band}"] = {
            'window_size': window,
            'freq_band': band,
            'num_segments': len(group),
            'deviation_mean_ms': mean_dev,
            'deviation_std_ms': std_dev,
            'stability_rate': stable_rate / 100,
            'psr_mean_db': mean_psr,
        }

    # Position analysis
    print("\n  Position Analysis:")
    pos_grouped = defaultdict(list)
    for r in results:
        if r.is_valid:
            pos_grouped[r.position].append(r)

    for pos_id, group in sorted(pos_grouped.items()):
        deviations = [r.deviation_ms for r in group]
        stabilities = [r.is_stable for r in group]

        report['position_analysis'][pos_id] = {
            'num_segments': len(group),
            'deviation_mean_ms': np.mean(deviations),
            'deviation_std_ms': np.std(deviations),
            'stability_rate': np.mean(stabilities),
            'chirp_reference_ms': group[0].tau_chirp_ref if group else None,
        }

        print(f"    Position {pos_id}: Stable rate = {np.mean(stabilities)*100:.1f}%, Mean deviation = {np.mean(deviations):.4f} ms")

    # Collapse analysis (tau â†’ 0)
    collapse_threshold_ms = 0.1
    collapse_cases = [r for r in results if r.is_valid and abs(r.tau_ms) < collapse_threshold_ms]

    report['collapse_analysis'] = {
        'threshold_ms': collapse_threshold_ms,
        'total_collapse_cases': len(collapse_cases),
        'collapse_rate': len(collapse_cases) / len([r for r in results if r.is_valid]) if results else 0,
        'by_window_size': {},
        'by_freq_band': {},
    }

    # Collapse by window size
    for window in WINDOW_SIZES_SEC:
        window_results = [r for r in results if r.is_valid and r.window_size == window]
        window_collapse = [r for r in window_results if abs(r.tau_ms) < collapse_threshold_ms]
        rate = len(window_collapse) / len(window_results) if window_results else 0
        report['collapse_analysis']['by_window_size'][f"{window}s"] = rate

    # Collapse by freq band
    for band_name in FREQ_BANDS.keys():
        band_results = [r for r in results if r.is_valid and r.freq_band == band_name]
        band_collapse = [r for r in band_results if abs(r.tau_ms) < collapse_threshold_ms]
        rate = len(band_collapse) / len(band_results) if band_results else 0
        report['collapse_analysis']['by_freq_band'][band_name] = rate

    print(f"\n  Collapse Analysis (|tau| < {collapse_threshold_ms} ms):")
    print(f"    Total collapse cases: {len(collapse_cases)} ({report['collapse_analysis']['collapse_rate']*100:.1f}%)")

    return report


def plot_tau_distribution(results: List[StabilityResult], output_dir: Path):
    """
    Generate tau distribution plots.
    """
    print("\n  Generating distribution plots...")

    # Filter valid results
    valid_results = [r for r in results if r.is_valid]

    if not valid_results:
        print("    No valid results to plot")
        return

    # Plot 1: tau distribution by window size
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, window in enumerate(WINDOW_SIZES_SEC):
        ax = axes[idx]
        window_data = [r.tau_ms for r in valid_results if r.window_size == window]

        if window_data:
            ax.hist(window_data, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', label='tau=0')

            # Add chirp reference lines
            chirp_refs = set(r.tau_chirp_ref for r in valid_results if r.window_size == window)
            for ref in chirp_refs:
                ax.axvline(ref, color='green', linestyle=':', alpha=0.7)

        ax.set_xlabel('tau (ms)')
        ax.set_ylabel('Count')
        ax.set_title(f'Window Size: {window}s')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'tau_distribution_by_window.png', dpi=150)
    plt.close()

    # Plot 2: tau distribution by frequency band
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, band_name in enumerate(FREQ_BANDS.keys()):
        ax = axes[idx]
        band_data = [r.tau_ms for r in valid_results if r.freq_band == band_name]

        if band_data:
            ax.hist(band_data, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', label='tau=0')

        ax.set_xlabel('tau (ms)')
        ax.set_ylabel('Count')
        ax.set_title(f'Frequency Band: {band_name}')

    plt.tight_layout()
    plt.savefig(output_dir / 'tau_distribution_by_band.png', dpi=150)
    plt.close()

    # Plot 3: Deviation vs PSR scatter
    fig, ax = plt.subplots(figsize=(10, 6))

    deviations = [r.deviation_ms for r in valid_results]
    psrs = [r.psr_db for r in valid_results]

    scatter = ax.scatter(psrs, deviations, alpha=0.3, s=10)
    ax.axhline(TAU_STABLE_THRESHOLD_MS, color='red', linestyle='--', label=f'Stable threshold ({TAU_STABLE_THRESHOLD_MS} ms)')
    ax.axvline(PSR_THRESHOLD, color='green', linestyle='--', label=f'PSR threshold ({PSR_THRESHOLD} dB)')

    ax.set_xlabel('PSR (dB)')
    ax.set_ylabel('Deviation from chirp reference (ms)')
    ax.set_title('tau Deviation vs PSR')
    ax.legend()
    ax.set_xlim(-5, 30)
    ax.set_ylim(0, 3)

    plt.tight_layout()
    plt.savefig(output_dir / 'deviation_vs_psr.png', dpi=150)
    plt.close()

    # Plot 4: Stability heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    stability_matrix = np.zeros((len(WINDOW_SIZES_SEC), len(FREQ_BANDS)))

    for i, window in enumerate(WINDOW_SIZES_SEC):
        for j, band_name in enumerate(FREQ_BANDS.keys()):
            group = [r for r in valid_results if r.window_size == window and r.freq_band == band_name]
            if group:
                stability_matrix[i, j] = np.mean([r.is_stable for r in group]) * 100

    im = ax.imshow(stability_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(FREQ_BANDS)))
    ax.set_xticklabels(FREQ_BANDS.keys(), rotation=45, ha='right')
    ax.set_yticks(range(len(WINDOW_SIZES_SEC)))
    ax.set_yticklabels([f'{w}s' for w in WINDOW_SIZES_SEC])
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Window Size')
    ax.set_title('Stability Rate (%)')

    # Add text annotations
    for i in range(len(WINDOW_SIZES_SEC)):
        for j in range(len(FREQ_BANDS)):
            text = ax.text(j, i, f'{stability_matrix[i, j]:.0f}%',
                          ha='center', va='center', color='black', fontweight='bold')

    plt.colorbar(im, ax=ax, label='Stability Rate (%)')
    plt.tight_layout()
    plt.savefig(output_dir / 'stability_heatmap.png', dpi=150)
    plt.close()

    print(f"    Saved plots to {output_dir}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run Phase 1: tau Stability Diagnosis."""
    print("\n" + "="*70)
    print("  PHASE 1: Speech tau Stability Diagnosis")
    print("  Purpose: Find conditions where speech mic-mic tau is stable")
    print("="*70)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_ROOT / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Output directory: {output_dir}")

    # Phase 1.1: Build chirp reference
    chirp_refs = build_chirp_reference(output_dir)

    if not chirp_refs:
        print("\n  [WARN] No chirp data found. Using geometric estimation as reference.")
        print("  (This is less accurate but allows analysis to proceed)")

        # Create geometric references for all speech positions
        chirp_refs = {}
        for pos_id in SPEECH_POSITIONS.keys():
            geo_ref = estimate_speech_position_chirp_ref(pos_id, {})
            if geo_ref is not None:
                chirp_refs[pos_id] = geo_ref
                print(f"    Position {pos_id}: geometric tau = {geo_ref:.4f} ms")

        # Save geometric references
        ref_file = output_dir / "chirp_references.json"
        with open(ref_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'positions': chirp_refs,
                'description': 'Geometric tau estimation (no chirp data available)',
                'method': 'geometric'
            }, f, indent=2)

    # Phase 1.2: Analyze speech stability
    results = analyze_speech_stability(chirp_refs, output_dir)

    if not results:
        print("\n  [ERROR] No stability results generated.")
        return

    # Generate report
    report = generate_stability_report(results, output_dir)

    # Save detailed results (convert numpy types to native Python)
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

    results_file = output_dir / "detailed_results.json"
    with open(results_file, 'w') as f:
        json.dump([convert_numpy(asdict(r)) for r in results], f, indent=2)

    # Save summary report
    report_file = output_dir / "stability_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    # Generate plots
    plot_tau_distribution(results, output_dir)

    # Print summary
    print("\n" + "="*70)
    print("  PHASE 1 SUMMARY")
    print("="*70)

    # Find best parameter combination
    best_params = max(
        report['parameter_analysis'].items(),
        key=lambda x: x[1]['stability_rate']
    )

    print(f"\n  Best parameter combination:")
    print(f"    {best_params[0]}: {best_params[1]['stability_rate']*100:.1f}% stable")

    # Answer key questions
    print("\n  Key Questions:")
    print("\n  Q1: Which window sizes yield tau close to chirp?")
    for window in WINDOW_SIZES_SEC:
        params = [v for k, v in report['parameter_analysis'].items() if f"{window}s" in k]
        if params:
            avg_rate = np.mean([p['stability_rate'] for p in params])
            print(f"      {window}s: {avg_rate*100:.1f}% average stability")

    print("\n  Q2: Which frequency bands are stable?")
    for band in FREQ_BANDS.keys():
        params = [v for k, v in report['parameter_analysis'].items() if band in k]
        if params:
            avg_rate = np.mean([p['stability_rate'] for p in params])
            print(f"      {band}: {avg_rate*100:.1f}% average stability")

    print("\n  Q3: PSR vs stability relationship:")
    high_psr = [r for r in results if r.is_valid and r.psr_db >= PSR_THRESHOLD]
    low_psr = [r for r in results if r.is_valid and r.psr_db < PSR_THRESHOLD]
    if high_psr:
        print(f"      PSR >= {PSR_THRESHOLD} dB: {np.mean([r.is_stable for r in high_psr])*100:.1f}% stable")
    if low_psr:
        print(f"      PSR < {PSR_THRESHOLD} dB: {np.mean([r.is_stable for r in low_psr])*100:.1f}% stable")

    print("\n  Q4: tau collapse conditions:")
    for key, rate in report['collapse_analysis']['by_window_size'].items():
        print(f"      {key}: {rate*100:.1f}% collapse rate")

    print(f"\n  Results saved to: {output_dir}")
    print("\n" + "="*70)

    # Return decision point
    best_rate = best_params[1]['stability_rate']
    if best_rate >= 0.7:
        print("\n  DECISION: [PASS] Stable parameters found. Proceed to Phase 2.")
        return 0
    elif best_rate >= 0.4:
        print("\n  DECISION: âš ï¸ Marginal stability. Review parameters before Phase 2.")
        return 1
    else:
        print("\n  DECISION: âŒ No stable parameters. Consider alternative signals.")
        return 2


if __name__ == "__main__":
    exit(main())

