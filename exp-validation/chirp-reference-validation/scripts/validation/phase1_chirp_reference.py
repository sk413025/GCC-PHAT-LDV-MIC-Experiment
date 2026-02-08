#!/usr/bin/env python3
"""
Phase 1: Speech tau Stability with Chirp References

Purpose: Measure speech mic-mic tau stability using chirp-derived references
from dataset/chirp and dataset/chirp_2 calibration summaries.

Outputs (under --out_dir):
- chirp_cross_validation.json
- chirp_references.json
- detailed_results.json
- stability_report.json
- plots (tau distributions, stability heatmap, deviation vs PSR)
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from scipy.fft import fft, ifft

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Local imports (chirp_reference)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from chirp_reference import get_chirp_references, SPEECH_TO_CHIRP  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPEECH_POSITIONS = {
    '18': '18-0.1V',
    '19': '19-0.1V',
    '20': '20-0.1V',
    '21': '21-0.1V',
    '22': '22-0.1V',
}

WINDOW_SIZES_SEC = [0.5, 1.0, 2.0, 5.0]
FREQ_BANDS = {
    '500-2000Hz': (500, 2000),
    '200-4000Hz': (200, 4000),
    '100-8000Hz': (100, 8000),
}

PSR_THRESHOLD = 10.0
TAU_STABLE_THRESHOLD_MS = 0.3

DEFAULT_FS = 48000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_project_root_with_dataset(start: Path) -> Path:
    """Walk upward from start to find a directory containing dataset/."""
    start = start.resolve()
    for parent in [start] + list(start.parents):
        if (parent / 'dataset').exists():
            return parent
    raise FileNotFoundError(f"Could not find dataset/ above {start}")


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def load_wav(path: Path, target_fs: int = None) -> Tuple[np.ndarray, int]:
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
    nyq = fs / 2
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)

    if low >= high:
        warnings.warn(f"Invalid bandpass: {lowcut}-{highcut} Hz at fs={fs}")
        return data

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: int, max_lag_ms: float = 10.0) -> GCCResult:
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
        return GCCResult(
            tau_ms=0.0, tau_samples=0, psr_db=0.0,
            peak_value=0.0, is_valid=False, reason="Empty search window"
        )

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
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_speech_stability(chirp_refs: Dict[str, float], data_root: Path) -> List[StabilityResult]:
    print("\n" + "=" * 60)
    print("Phase 1.2: Speech tau Stability Analysis")
    print("=" * 60)

    all_results: List[StabilityResult] = []

    for pos_id, folder_name in SPEECH_POSITIONS.items():
        folder = data_root / folder_name

        mic_l_files = list(folder.glob("*LEFT-MIC*.wav"))
        mic_r_files = list(folder.glob("*RIGHT-MIC*.wav"))

        if not mic_l_files or not mic_r_files:
            raise FileNotFoundError(f"Missing mic files for position {pos_id} in {folder}")

        mic_l_path = mic_l_files[0]
        mic_r_path = mic_r_files[0]

        print(f"\n  Position {pos_id} ({folder_name}):")

        mic_l, fs = load_wav(mic_l_path)
        mic_r, _ = load_wav(mic_r_path)

        tau_chirp = chirp_refs.get(pos_id)
        if tau_chirp is None:
            raise ValueError(f"No chirp reference for position {pos_id}")

        print(f"    Chirp reference tau = {tau_chirp:.4f} ms")
        print(f"    Signal length: {len(mic_l) / fs:.1f} sec")

        for window_size in WINDOW_SIZES_SEC:
            for band_name, (low, high) in FREQ_BANDS.items():
                mic_l_filt = bandpass_filter(mic_l, fs, low, high)
                mic_r_filt = bandpass_filter(mic_r, fs, low, high)

                window_samples = int(window_size * fs)
                n_segments = len(mic_l_filt) // window_samples

                start_seg = max(1, int(n_segments * 0.1))
                end_seg = int(n_segments * 0.9)

                for seg_idx in range(start_seg, end_seg):
                    start_sample = seg_idx * window_samples
                    end_sample = start_sample + window_samples

                    seg_l = mic_l_filt[start_sample:end_sample]
                    seg_r = mic_r_filt[start_sample:end_sample]

                    result = gcc_phat(seg_l, seg_r, fs, max_lag_ms=5.0)

                    deviation = abs(result.tau_ms - tau_chirp)
                    is_stable = deviation < TAU_STABLE_THRESHOLD_MS

                    all_results.append(
                        StabilityResult(
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
                            is_valid=result.is_valid,
                        )
                    )

        print(f"    Analyzed {len([r for r in all_results if r.position == pos_id])} segments")

    return all_results


def generate_stability_report(results: List[StabilityResult], chirp_refs: Dict[str, float]) -> Dict:
    print("\n" + "=" * 60)
    print("Generating Stability Report")
    print("=" * 60)

    from collections import defaultdict
    grouped = defaultdict(list)

    for r in results:
        if r.is_valid:
            grouped[(r.window_size, r.freq_band)].append(r)

    report = {
        'timestamp': datetime.now().isoformat(),
        'total_segments': len(results),
        'valid_segments': len([r for r in results if r.is_valid]),
        'parameter_analysis': {},
        'position_analysis': {},
        'collapse_analysis': {},
        'chirp_references_ms': chirp_refs,
    }

    print("\n  Parameter Analysis:")
    print("  " + "-" * 70)
    print(f"  {'Window':<10} {'Band':<15} {'N':<6} {'Mean Dev':<10} {'Std Dev':<10} {'Stable %':<10} {'Mean PSR':<10}")
    print("  " + "-" * 70)

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

        print(f"    Position {pos_id}: Stable rate = {np.mean(stabilities) * 100:.1f}%, Mean deviation = {np.mean(deviations):.4f} ms")

    collapse_threshold_ms = 0.1
    collapse_cases = [r for r in results if r.is_valid and abs(r.tau_ms) < collapse_threshold_ms]

    report['collapse_analysis'] = {
        'threshold_ms': collapse_threshold_ms,
        'total_collapse_cases': len(collapse_cases),
        'collapse_rate': len(collapse_cases) / len([r for r in results if r.is_valid]) if results else 0,
        'by_window_size': {},
        'by_freq_band': {},
    }

    for window in WINDOW_SIZES_SEC:
        window_results = [r for r in results if r.is_valid and r.window_size == window]
        window_collapse = [r for r in window_results if abs(r.tau_ms) < collapse_threshold_ms]
        rate = len(window_collapse) / len(window_results) if window_results else 0
        report['collapse_analysis']['by_window_size'][f"{window}s"] = rate

    for band_name in FREQ_BANDS.keys():
        band_results = [r for r in results if r.is_valid and r.freq_band == band_name]
        band_collapse = [r for r in band_results if abs(r.tau_ms) < collapse_threshold_ms]
        rate = len(band_collapse) / len(band_results) if band_results else 0
        report['collapse_analysis']['by_freq_band'][band_name] = rate

    print(f"\n  Collapse Analysis (|tau| < {collapse_threshold_ms} ms):")
    print(f"    Total collapse cases: {len(collapse_cases)} ({report['collapse_analysis']['collapse_rate'] * 100:.1f}%)")

    return report


def plot_tau_distribution(results: List[StabilityResult], output_dir: Path):
    print("\n  Generating distribution plots...")

    valid_results = [r for r in results if r.is_valid]

    if not valid_results:
        print("    No valid results to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, window in enumerate(WINDOW_SIZES_SEC):
        ax = axes[idx]
        window_data = [r.tau_ms for r in valid_results if r.window_size == window]

        if window_data:
            ax.hist(window_data, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', label='tau=0')
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

    fig, ax = plt.subplots(figsize=(10, 6))

    deviations = [r.deviation_ms for r in valid_results]
    psrs = [r.psr_db for r in valid_results]

    ax.scatter(psrs, deviations, alpha=0.3, s=10)
    ax.axhline(TAU_STABLE_THRESHOLD_MS, color='red', linestyle='--',
               label=f'Stable threshold ({TAU_STABLE_THRESHOLD_MS} ms)')
    ax.axvline(PSR_THRESHOLD, color='green', linestyle='--',
               label=f'PSR threshold ({PSR_THRESHOLD} dB)')

    ax.set_xlabel('PSR (dB)')
    ax.set_ylabel('Deviation from chirp reference (ms)')
    ax.set_title('tau Deviation vs PSR')
    ax.legend()
    ax.set_xlim(-5, 30)
    ax.set_ylim(0, 3)

    plt.tight_layout()
    plt.savefig(output_dir / 'deviation_vs_psr.png', dpi=150)
    plt.close()

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

    for i in range(len(WINDOW_SIZES_SEC)):
        for j in range(len(FREQ_BANDS)):
            ax.text(j, i, f'{stability_matrix[i, j]:.0f}%',
                    ha='center', va='center', color='black', fontweight='bold')

    plt.colorbar(im, ax=ax, label='Stability Rate (%)')
    plt.tight_layout()
    plt.savefig(output_dir / 'stability_heatmap.png', dpi=150)
    plt.close()

    print(f"    Saved plots to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Phase 1: Speech tau stability with chirp references')
    parser.add_argument('--dataset_root', type=Path, default=None,
                        help='Project root containing dataset/ (auto-detected if omitted)')
    parser.add_argument('--out_dir', type=Path, default=None,
                        help='Output directory (default: results/chirp_ref_phase1/run_<ts>)')
    parser.add_argument('--no_geometric_fallback', action='store_true',
                        help='Disable geometric fallback for positions without reliable chirp data')
    return parser.parse_args()


def main() -> int:
    print("\n" + "=" * 70)
    print("  PHASE 1: Speech tau Stability with Chirp References")
    print("  Purpose: Find conditions where speech mic-mic tau is stable")
    print("=" * 70)

    args = parse_args()

    script_root = Path(__file__).resolve().parents[2]
    project_root = find_project_root_with_dataset(script_root)

    dataset_root = args.dataset_root.resolve() if args.dataset_root else project_root
    data_root = dataset_root / 'dataset' / 'GCC-PHAT-LDV-MIC-Experiment'
    if not data_root.exists():
        raise FileNotFoundError(f"Speech dataset not found: {data_root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        output_dir = args.out_dir
    else:
        output_dir = script_root / 'results' / 'chirp_ref_phase1' / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Output directory: {output_dir}")

    allow_geometric_fallback = not args.no_geometric_fallback
    chirp_refs, chirp_report = get_chirp_references(
        dataset_root,
        allow_geometric_fallback=allow_geometric_fallback,
    )

    # Save chirp cross-validation and references
    cross_file = output_dir / 'chirp_cross_validation.json'
    with open(cross_file, 'w', encoding='utf-8') as f:
        json.dump(chirp_report, f, indent=2)

    refs_file = output_dir / 'chirp_references.json'
    with open(refs_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'references': chirp_refs,
            'position_mapping': SPEECH_TO_CHIRP,
            'allow_geometric_fallback': bool(allow_geometric_fallback),
            'dataset_root': str(dataset_root),
        }, f, indent=2)

    # Phase 1.2: Analyze stability
    results = analyze_speech_stability(chirp_refs, data_root)
    if not results:
        print("\n  [ERROR] No stability results generated.")
        return 1

    report = generate_stability_report(results, chirp_refs)

    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_file = output_dir / 'detailed_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump([convert_numpy(asdict(r)) for r in results], f, indent=2)

    report_file = output_dir / 'stability_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    plot_tau_distribution(results, output_dir)

    print("\n" + "=" * 70)
    print("  PHASE 1 SUMMARY")
    print("=" * 70)

    best_params = max(
        report['parameter_analysis'].items(),
        key=lambda x: x[1]['stability_rate']
    )

    print(f"\n  Best parameter combination:")
    print(f"    {best_params[0]}: {best_params[1]['stability_rate'] * 100:.1f}% stable")

    print("\n  Key Questions:")
    print("\n  Q1: Which window sizes yield tau close to chirp?")
    for window in WINDOW_SIZES_SEC:
        params = [v for k, v in report['parameter_analysis'].items() if f"{window}s" in k]
        if params:
            avg_rate = np.mean([p['stability_rate'] for p in params])
            print(f"      {window}s: {avg_rate * 100:.1f}% average stability")

    print("\n  Q2: Which frequency bands are stable?")
    for band in FREQ_BANDS.keys():
        params = [v for k, v in report['parameter_analysis'].items() if band in k]
        if params:
            avg_rate = np.mean([p['stability_rate'] for p in params])
            print(f"      {band}: {avg_rate * 100:.1f}% average stability")

    print("\n  Q3: PSR vs stability relationship:")
    high_psr = [r for r in results if r.is_valid and r.psr_db >= PSR_THRESHOLD]
    low_psr = [r for r in results if r.is_valid and r.psr_db < PSR_THRESHOLD]
    if high_psr:
        print(f"      PSR >= {PSR_THRESHOLD} dB: {np.mean([r.is_stable for r in high_psr]) * 100:.1f}% stable")
    if low_psr:
        print(f"      PSR < {PSR_THRESHOLD} dB: {np.mean([r.is_stable for r in low_psr]) * 100:.1f}% stable")

    print("\n  Q4: tau collapse conditions:")
    for key, rate in report['collapse_analysis']['by_window_size'].items():
        print(f"      {key}: {rate * 100:.1f}% collapse rate")

    print(f"\n  Results saved to: {output_dir}")
    print("\n" + "=" * 70)

    best_rate = best_params[1]['stability_rate']
    if best_rate >= 0.7:
        print("\n  DECISION: [PASS] Stable parameters found. Proceed to Phase 2.")
        return 0
    if best_rate >= 0.4:
        print("\n  DECISION: [WARN] Marginal stability. Review parameters before Phase 2.")
        return 1

    print("\n  DECISION: [FAIL] No stable parameters. Consider alternative signals.")
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
