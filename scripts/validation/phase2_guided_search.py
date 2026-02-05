#!/usr/bin/env python3
"""
Phase 2: Guided Peak Search Mechanism

Purpose: Implement and validate guided GCC-PHAT peak search
using chirp-derived references.

Outputs:
- Global vs Guided comparison
- False peak rate analysis
- Optimal search window determination
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
RESULTS_ROOT = Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\results\phase2_guided_search")
PHASE1_RESULTS = Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\results\phase1_tau_stability")

SPEECH_POSITIONS = {
    '18': '18-0.1V',
    '19': '19-0.1V',
    '20': '20-0.1V',
    '21': '21-0.1V',
    '22': '22-0.1V',
}

# Best parameters from Phase 1 (default, will be loaded from Phase 1 if available)
DEFAULT_WINDOW_SIZE = 2.0  # seconds
DEFAULT_FREQ_BAND = (500, 2000)  # Hz

# Search window sizes to test
SEARCH_WINDOWS_MS = [0.1, 0.2, 0.3, 0.5, 1.0]

FALSE_PEAK_THRESHOLD_MS = 0.5  # tau deviation > 0.5ms = false peak
PSR_THRESHOLD = 10.0  # dB

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
    method: str = "global"
    search_window_ms: float = 0.0


@dataclass
class ComparisonResult:
    position: str
    segment_idx: int
    segment_start_sec: float
    tau_chirp_ref: float
    # Global search results
    tau_global: float
    psr_global: float
    deviation_global: float
    is_false_peak_global: bool
    # Guided search results
    tau_guided: float
    psr_guided: float
    deviation_guided: float
    is_false_peak_guided: bool
    search_window_ms: float
    # Comparison
    guided_better: bool


# ============================================================================
# Signal Processing Functions (same as Phase 1)
# ============================================================================

def load_wav(path: Path, target_fs: int = None) -> Tuple[np.ndarray, int]:
    """Load WAV file and optionally resample."""
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


def gcc_phat_global(sig1: np.ndarray, sig2: np.ndarray, fs: int, max_lag_ms: float = 10.0) -> GCCResult:
    """Standard GCC-PHAT with global peak search."""
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
        return GCCResult(tau_ms=0.0, tau_samples=0, psr_db=0.0,
                        peak_value=0.0, is_valid=False, method="global")

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

    return GCCResult(
        tau_ms=tau_ms,
        tau_samples=int(tau_samples),
        psr_db=psr_db,
        peak_value=peak_value,
        is_valid=True,
        method="global"
    )


def gcc_phat_guided(sig1: np.ndarray, sig2: np.ndarray, fs: int,
                   tau_reference_ms: float, search_window_ms: float = 0.3) -> GCCResult:
    """GCC-PHAT with guided peak search around reference tau."""
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

    # Guided search window
    mask = (tau_axis_ms >= tau_reference_ms - search_window_ms) & \
           (tau_axis_ms <= tau_reference_ms + search_window_ms)

    if not np.any(mask):
        return GCCResult(tau_ms=tau_reference_ms, tau_samples=0, psr_db=0.0,
                        peak_value=0.0, is_valid=False, method="guided",
                        search_window_ms=search_window_ms)

    gcc_guided = gcc[mask]
    lags_guided = lags[mask]

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

    return GCCResult(
        tau_ms=tau_ms,
        tau_samples=int(tau_samples),
        psr_db=psr_db,
        peak_value=peak_value,
        is_valid=True,
        method="guided",
        search_window_ms=search_window_ms
    )


# ============================================================================
# Phase 2 Analysis Functions
# ============================================================================

def load_phase1_results() -> Tuple[Dict[str, float], Dict]:
    """Load chirp references and best parameters from Phase 1."""
    chirp_refs = {}
    best_params = {
        'window_size': DEFAULT_WINDOW_SIZE,
        'freq_band': DEFAULT_FREQ_BAND,
    }

    # Find most recent Phase 1 run
    if PHASE1_RESULTS.exists():
        runs = sorted(PHASE1_RESULTS.glob("run_*"), reverse=True)
        if runs:
            latest_run = runs[0]
            print(f"  Loading Phase 1 results from: {latest_run}")

            # Load chirp references
            chirp_file = latest_run / "chirp_references.json"
            if chirp_file.exists():
                with open(chirp_file) as f:
                    data = json.load(f)
                    chirp_refs = data.get('positions', {})

            # Load stability report to find best parameters
            report_file = latest_run / "stability_report.json"
            if report_file.exists():
                with open(report_file) as f:
                    report = json.load(f)
                    # Find best parameter combination
                    param_analysis = report.get('parameter_analysis', {})
                    if param_analysis:
                        best_key = max(param_analysis.keys(),
                                      key=lambda k: param_analysis[k]['stability_rate'])
                        best = param_analysis[best_key]
                        best_params['window_size'] = best['window_size']
                        # Parse freq band
                        band_str = best['freq_band']
                        if '-' in band_str:
                            parts = band_str.replace('Hz', '').split('-')
                            best_params['freq_band'] = (int(parts[0]), int(parts[1]))

    return chirp_refs, best_params


def estimate_chirp_ref_for_position(position: str, chirp_refs: Dict[str, float]) -> float:
    """Estimate chirp reference for speech position."""
    x_coords = {
        '18': 0.8,
        '19': 0.4,
        '20': 0.0,
        '21': -0.4,
        '22': -0.8,
    }

    if position == '22' and '23' in chirp_refs:
        return chirp_refs['23']
    if position == '21' and '24' in chirp_refs:
        return chirp_refs['24']

    # Geometric estimation
    x = x_coords.get(position, 0.0)
    mic_l = np.array([-0.7, 2.0])
    mic_r = np.array([0.7, 2.0])
    speaker = np.array([x, 0.0])

    d_l = np.linalg.norm(speaker - mic_l)
    d_r = np.linalg.norm(speaker - mic_r)

    return (d_r - d_l) / SPEED_OF_SOUND * 1000.0


def run_global_vs_guided_comparison(chirp_refs: Dict[str, float],
                                    best_params: Dict,
                                    output_dir: Path) -> List[ComparisonResult]:
    """
    Compare global vs guided peak search across all speech data.
    """
    print("\n" + "="*60)
    print("Phase 2.2: Global vs Guided Peak Search Comparison")
    print("="*60)

    all_results = []

    window_size = best_params['window_size']
    freq_band = best_params['freq_band']

    print(f"\n  Using parameters: window={window_size}s, band={freq_band[0]}-{freq_band[1]}Hz")

    for search_window_ms in SEARCH_WINDOWS_MS:
        print(f"\n  Testing search window: {search_window_ms} ms")

        for pos_id, folder_name in SPEECH_POSITIONS.items():
            folder = DATASET_ROOT / folder_name

            mic_l_files = list(folder.glob("*LEFT-MIC*.wav"))
            mic_r_files = list(folder.glob("*RIGHT-MIC*.wav"))

            if not mic_l_files or not mic_r_files:
                continue

            mic_l, fs = load_wav(mic_l_files[0])
            mic_r, _ = load_wav(mic_r_files[0])

            tau_chirp = estimate_chirp_ref_for_position(pos_id, chirp_refs)

            # Filter
            mic_l_filt = bandpass_filter(mic_l, fs, freq_band[0], freq_band[1])
            mic_r_filt = bandpass_filter(mic_r, fs, freq_band[0], freq_band[1])

            # Segment
            window_samples = int(window_size * fs)
            n_segments = len(mic_l_filt) // window_samples

            start_seg = max(1, int(n_segments * 0.1))
            end_seg = int(n_segments * 0.9)

            for seg_idx in range(start_seg, end_seg):
                start_sample = seg_idx * window_samples
                end_sample = start_sample + window_samples

                seg_l = mic_l_filt[start_sample:end_sample]
                seg_r = mic_r_filt[start_sample:end_sample]

                # Global search
                result_global = gcc_phat_global(seg_l, seg_r, fs)

                # Guided search
                result_guided = gcc_phat_guided(seg_l, seg_r, fs, tau_chirp, search_window_ms)

                if not result_global.is_valid or not result_guided.is_valid:
                    continue

                # Compute deviations
                dev_global = abs(result_global.tau_ms - tau_chirp)
                dev_guided = abs(result_guided.tau_ms - tau_chirp)

                comparison = ComparisonResult(
                    position=pos_id,
                    segment_idx=seg_idx,
                    segment_start_sec=start_sample / fs,
                    tau_chirp_ref=tau_chirp,
                    tau_global=result_global.tau_ms,
                    psr_global=result_global.psr_db,
                    deviation_global=dev_global,
                    is_false_peak_global=dev_global > FALSE_PEAK_THRESHOLD_MS,
                    tau_guided=result_guided.tau_ms,
                    psr_guided=result_guided.psr_db,
                    deviation_guided=dev_guided,
                    is_false_peak_guided=dev_guided > FALSE_PEAK_THRESHOLD_MS,
                    search_window_ms=search_window_ms,
                    guided_better=dev_guided < dev_global
                )
                all_results.append(comparison)

    return all_results


def generate_comparison_report(results: List[ComparisonResult], output_dir: Path) -> Dict:
    """Generate comparison report between global and guided search."""
    print("\n" + "="*60)
    print("Generating Comparison Report")
    print("="*60)

    report = {
        'timestamp': datetime.now().isoformat(),
        'total_comparisons': len(results),
        'search_window_analysis': {},
        'overall_comparison': {},
    }

    # Analysis by search window
    print("\n  Search Window Analysis:")
    print("  " + "-"*80)
    print(f"  {'Window (ms)':<12} {'N':<6} {'FP Global':<12} {'FP Guided':<12} {'Guided Better':<15} {'Dev Reduction':<15}")
    print("  " + "-"*80)

    for search_window in SEARCH_WINDOWS_MS:
        window_results = [r for r in results if r.search_window_ms == search_window]

        if not window_results:
            continue

        fp_global = sum(r.is_false_peak_global for r in window_results)
        fp_guided = sum(r.is_false_peak_guided for r in window_results)
        guided_better = sum(r.guided_better for r in window_results)
        n = len(window_results)

        # Deviation reduction
        dev_global_mean = np.mean([r.deviation_global for r in window_results])
        dev_guided_mean = np.mean([r.deviation_guided for r in window_results])
        dev_reduction = (dev_global_mean - dev_guided_mean) / dev_global_mean * 100 if dev_global_mean > 0 else 0

        print(f"  {search_window:<12.1f} {n:<6} {fp_global/n*100:<12.1f}% {fp_guided/n*100:<12.1f}% {guided_better/n*100:<15.1f}% {dev_reduction:<15.1f}%")

        report['search_window_analysis'][f"{search_window}ms"] = {
            'num_comparisons': n,
            'false_peak_rate_global': fp_global / n,
            'false_peak_rate_guided': fp_guided / n,
            'guided_better_rate': guided_better / n,
            'deviation_mean_global_ms': dev_global_mean,
            'deviation_mean_guided_ms': dev_guided_mean,
            'deviation_reduction_percent': dev_reduction,
        }

    # Overall comparison
    if results:
        fp_global_all = sum(r.is_false_peak_global for r in results) / len(results)
        fp_guided_best = min(
            report['search_window_analysis'].values(),
            key=lambda x: x['false_peak_rate_guided']
        )['false_peak_rate_guided']

        report['overall_comparison'] = {
            'false_peak_rate_global': fp_global_all,
            'best_false_peak_rate_guided': fp_guided_best,
            'improvement': (fp_global_all - fp_guided_best) / fp_global_all * 100 if fp_global_all > 0 else 0,
        }

    # Stability comparison
    global_std = np.std([r.tau_global for r in results]) if results else 0
    guided_std = np.std([r.tau_guided for r in results]) if results else 0

    report['stability_comparison'] = {
        'tau_std_global': global_std,
        'tau_std_guided': guided_std,
        'stability_improvement': (global_std - guided_std) / global_std * 100 if global_std > 0 else 0,
    }

    print(f"\n  Overall False Peak Rate:")
    print(f"    Global: {report['overall_comparison'].get('false_peak_rate_global', 0)*100:.1f}%")
    print(f"    Best Guided: {report['overall_comparison'].get('best_false_peak_rate_guided', 0)*100:.1f}%")

    print(f"\n  Stability (tau std):")
    print(f"    Global: {global_std:.4f} ms")
    print(f"    Guided: {guided_std:.4f} ms")

    return report


def plot_comparison_results(results: List[ComparisonResult], output_dir: Path):
    """Generate comparison plots."""
    print("\n  Generating comparison plots...")

    if not results:
        return

    # Plot 1: False peak rate by search window
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: False peak rate
    ax = axes[0]
    windows = []
    fp_global = []
    fp_guided = []

    for search_window in SEARCH_WINDOWS_MS:
        window_results = [r for r in results if r.search_window_ms == search_window]
        if window_results:
            windows.append(search_window)
            fp_global.append(sum(r.is_false_peak_global for r in window_results) / len(window_results) * 100)
            fp_guided.append(sum(r.is_false_peak_guided for r in window_results) / len(window_results) * 100)

    x = np.arange(len(windows))
    width = 0.35

    ax.bar(x - width/2, fp_global, width, label='Global', color='red', alpha=0.7)
    ax.bar(x + width/2, fp_guided, width, label='Guided', color='green', alpha=0.7)
    ax.set_xlabel('Search Window (ms)')
    ax.set_ylabel('False Peak Rate (%)')
    ax.set_title('False Peak Rate: Global vs Guided')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{w:.1f}' for w in windows])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Right: Deviation scatter
    ax = axes[1]
    dev_global = [r.deviation_global for r in results if r.search_window_ms == 0.3]
    dev_guided = [r.deviation_guided for r in results if r.search_window_ms == 0.3]

    ax.scatter(dev_global, dev_guided, alpha=0.3, s=10)
    max_dev = max(max(dev_global) if dev_global else 1, max(dev_guided) if dev_guided else 1)
    ax.plot([0, max_dev], [0, max_dev], 'k--', label='y=x')
    ax.set_xlabel('Deviation Global (ms)')
    ax.set_ylabel('Deviation Guided (ms)')
    ax.set_title('Deviation Comparison (search window = 0.3ms)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'global_vs_guided_comparison.png', dpi=150)
    plt.close()

    # Plot 2: tau distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Using best search window (0.3ms default)
    best_window = 0.3
    window_results = [r for r in results if r.search_window_ms == best_window]

    if window_results:
        # Global tau distribution
        ax = axes[0]
        tau_global = [r.tau_global for r in window_results]
        tau_refs = [r.tau_chirp_ref for r in window_results]
        ax.hist(tau_global, bins=50, alpha=0.7, label='Global', color='red')
        for ref in set(tau_refs):
            ax.axvline(ref, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('tau (ms)')
        ax.set_ylabel('Count')
        ax.set_title('Global Search tau Distribution')
        ax.legend()

        # Guided tau distribution
        ax = axes[1]
        tau_guided = [r.tau_guided for r in window_results]
        ax.hist(tau_guided, bins=50, alpha=0.7, label='Guided', color='green')
        for ref in set(tau_refs):
            ax.axvline(ref, color='blue', linestyle='--', alpha=0.5)
        ax.set_xlabel('tau (ms)')
        ax.set_ylabel('Count')
        ax.set_title('Guided Search tau Distribution')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'tau_distribution_comparison.png', dpi=150)
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
    """Run Phase 2: Guided Peak Search Mechanism."""
    print("\n" + "="*70)
    print("  PHASE 2: Guided Peak Search Mechanism")
    print("  Purpose: Implement and validate guided GCC-PHAT peak search")
    print("="*70)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_ROOT / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Output directory: {output_dir}")

    # Load Phase 1 results
    chirp_refs, best_params = load_phase1_results()

    if not chirp_refs:
        print("\n  [WARN] No chirp references found from Phase 1.")
        print("  Using geometric estimation for references.")

    print(f"\n  Chirp references: {chirp_refs}")
    print(f"  Best parameters: {best_params}")

    # Run comparison
    results = run_global_vs_guided_comparison(chirp_refs, best_params, output_dir)

    if not results:
        print("\n  [ERROR] No comparison results generated.")
        return 1

    # Generate report
    report = generate_comparison_report(results, output_dir)

    # Save detailed results
    results_file = output_dir / "detailed_results.json"
    with open(results_file, 'w') as f:
        json.dump([convert_numpy(asdict(r)) for r in results], f, indent=2)

    # Save report
    report_file = output_dir / "comparison_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    # Generate plots
    plot_comparison_results(results, output_dir)

    # Print summary
    print("\n" + "="*70)
    print("  PHASE 2 SUMMARY")
    print("="*70)

    # Find best search window
    best_window = min(
        report['search_window_analysis'].items(),
        key=lambda x: x[1]['false_peak_rate_guided']
    )

    print(f"\n  Best search window: {best_window[0]}")
    print(f"    False peak rate: {best_window[1]['false_peak_rate_guided']*100:.1f}%")
    print(f"    Deviation reduction: {best_window[1]['deviation_reduction_percent']:.1f}%")

    # Decision point
    improvement = report['overall_comparison'].get('improvement', 0)

    print(f"\n  Overall improvement: {improvement:.1f}%")

    if improvement > 30:
        print("\n  DECISION: Guided search significantly reduces false peaks.")
        print("  Proceed to Phase 3.")
        return 0
    elif improvement > 10:
        print("\n  DECISION: Guided search shows moderate improvement.")
        print("  Review parameters, then proceed to Phase 3.")
        return 1
    else:
        print("\n  DECISION: Guided search shows limited improvement.")
        print("  Problem may not be peak search. Re-examine Phase 1.")
        return 2


if __name__ == "__main__":
    exit(main())

