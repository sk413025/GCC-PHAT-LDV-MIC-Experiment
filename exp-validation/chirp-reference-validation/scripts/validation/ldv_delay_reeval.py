#!/usr/bin/env python3
"""
LDV Delay Re-evaluation (Chirp References)

Purpose: Compare LDV-MIC TDoA residuals using chirp-calibrated sensor delays
versus the older speech-derived LDV delay estimate.

Outputs (under --out_dir):
- ldv_delay_reeval_report.json
- residuals_by_pair.png
"""

import argparse
import json
import sys
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
# Local imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from chirp_reference import get_chirp_references  # noqa: E402


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

MIC_LEFT = (-0.7, 2.0)
MIC_RIGHT = (0.7, 2.0)
LDV_POS = (0.0, 0.5)
SPEED_OF_SOUND = 343.0

SPEECH_POSITIONS = {
    '18': ('18-0.1V', 0.8),
    '19': ('19-0.1V', 0.4),
    '20': ('20-0.1V', 0.0),
    '21': ('21-0.1V', -0.4),
    '22': ('22-0.1V', -0.8),
}

DEFAULT_WINDOW_SIZE = 2.0
DEFAULT_FREQ_BAND = (500, 2000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_project_root_with_dataset(start: Path) -> Path:
    start = start.resolve()
    for parent in [start] + list(start.parents):
        if (parent / 'dataset').exists():
            return parent
    raise FileNotFoundError(f"Could not find dataset/ above {start}")


def find_latest_run(phase_dir: Path) -> Path:
    if not phase_dir.exists():
        return None
    runs = sorted(phase_dir.glob('run_*'), reverse=True)
    return runs[0] if runs else None


def load_phase1_best_params(phase1_dir: Path) -> Dict:
    best = {
        'window_size': DEFAULT_WINDOW_SIZE,
        'freq_band': DEFAULT_FREQ_BAND,
    }

    if not phase1_dir:
        return best

    report_file = phase1_dir / 'stability_report.json'
    if not report_file.exists():
        return best

    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)

    param_analysis = report.get('parameter_analysis', {})
    if param_analysis:
        best_key = max(param_analysis.keys(), key=lambda k: param_analysis[k]['stability_rate'])
        param = param_analysis[best_key]
        best['window_size'] = param['window_size']
        band_str = param['freq_band']
        if '-' in band_str:
            parts = band_str.replace('Hz', '').split('-')
            best['freq_band'] = (int(parts[0]), int(parts[1]))

    return best


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    fs, data = wavfile.read(path)

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)

    if len(data.shape) > 1:
        data = data[:, 0]

    return data, fs


def bandpass_filter(data: np.ndarray, fs: int, lowcut: float, highcut: float, order: int = 5) -> np.ndarray:
    nyq = fs / 2
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    if low >= high:
        return data
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: int, max_lag_ms: float = 10.0) -> Tuple[float, float]:
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
        return 0.0, 0.0

    peak_idx = np.argmax(np.abs(gcc_search))
    peak_value = np.abs(gcc_search[peak_idx])
    tau_samples = lags_search[peak_idx]

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

    return tau_ms, psr_db


def calc_theory_ldv_mic_tau_ms(speaker_x: float) -> Dict[str, float]:
    spk = (speaker_x, 0.0)
    d_left = np.hypot(spk[0] - MIC_LEFT[0], spk[1] - MIC_LEFT[1])
    d_right = np.hypot(spk[0] - MIC_RIGHT[0], spk[1] - MIC_RIGHT[1])
    d_ldv = np.hypot(spk[0] - LDV_POS[0], spk[1] - LDV_POS[1])

    return {
        'LDV_LEFT': (d_ldv - d_left) / SPEED_OF_SOUND * 1000.0,
        'LDV_RIGHT': (d_ldv - d_right) / SPEED_OF_SOUND * 1000.0,
    }


def stats(arr: List[float]) -> Dict:
    if not arr:
        return {'n': 0}
    vals = np.asarray(arr, dtype=np.float64)
    return {
        'n': int(vals.size),
        'mean_ms': float(np.mean(vals)),
        'median_ms': float(np.median(vals)),
        'std_ms': float(np.std(vals)),
        'abs_mean_ms': float(np.mean(np.abs(vals))),
        'abs_p95_ms': float(np.percentile(np.abs(vals), 95)),
        'abs_max_ms': float(np.max(np.abs(vals))),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='LDV delay re-evaluation with chirp references')
    parser.add_argument('--dataset_root', type=Path, default=None,
                        help='Project root containing dataset/ (auto-detected if omitted)')
    parser.add_argument('--out_dir', type=Path, default=None,
                        help='Output directory (default: results/ldv_delay_reeval/run_<ts>)')
    parser.add_argument('--phase1_dir', type=Path, default=None,
                        help='Phase 1 run directory (defaults to latest under results/chirp_ref_phase1)')
    parser.add_argument('--old_delay_ms', type=float, action='append', default=None,
                        help='Old speech-derived LDV delay (ms). Repeatable. Defaults: 3.8, 4.3, 4.8')
    parser.add_argument('--no_geometric_fallback', action='store_true',
                        help='Disable geometric fallback for missing chirp references')
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    script_root = Path(__file__).resolve().parents[2]
    project_root = find_project_root_with_dataset(script_root)

    dataset_root = args.dataset_root.resolve() if args.dataset_root else project_root
    data_root = dataset_root / 'dataset' / 'GCC-PHAT-LDV-MIC-Experiment'
    if not data_root.exists():
        raise FileNotFoundError(f"Speech dataset not found: {data_root}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.out_dir:
        output_dir = args.out_dir
    else:
        output_dir = script_root / 'results' / 'ldv_delay_reeval' / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_dir = args.phase1_dir
    if phase1_dir is None:
        phase1_dir = find_latest_run(script_root / 'results' / 'chirp_ref_phase1')

    best_params = load_phase1_best_params(phase1_dir)

    allow_geometric_fallback = not args.no_geometric_fallback
    chirp_refs, chirp_report = get_chirp_references(dataset_root, allow_geometric_fallback=allow_geometric_fallback)
    sensor_delays = chirp_report.get('sensor_delays', {})

    chirp_delays = sensor_delays.get('chirp', {})
    chirp2_delays = sensor_delays.get('chirp_2', {})

    def require_delay(d: Dict, key: str) -> float:
        if key not in d or not isinstance(d[key], (int, float)):
            raise ValueError(f"Missing numeric delay for {key} in {d}")
        return float(d[key])

    delay_sets = {
        'chirp': {
            'LDV': require_delay(chirp_delays, 'LDV'),
            'LEFT-MIC': 0.0,
            'RIGHT-MIC': require_delay(chirp_delays, 'RIGHT-MIC'),
        },
        'chirp_2': {
            'LDV': require_delay(chirp2_delays, 'LDV'),
            'LEFT-MIC': 0.0,
            'RIGHT-MIC': require_delay(chirp2_delays, 'RIGHT-MIC'),
        },
    }

    old_list = args.old_delay_ms if args.old_delay_ms else [3.8, 4.3, 4.8]
    for v in old_list:
        delay_sets[f'old_{v:.1f}ms'] = {
            'LDV': float(v),
            'LEFT-MIC': 0.0,
            'RIGHT-MIC': 0.0,
        }

    window_size = best_params['window_size']
    freq_band = best_params['freq_band']

    print("\n" + "=" * 70)
    print("  LDV Delay Re-evaluation")
    print("=" * 70)
    print(f"  Output directory: {output_dir}")
    print(f"  Window size: {window_size}s, Band: {freq_band}")

    residuals = {pair: {name: [] for name in delay_sets} for pair in ['LDV_LEFT', 'LDV_RIGHT']}
    residuals_by_pos = {pos_id: {pair: {name: [] for name in delay_sets} for pair in ['LDV_LEFT', 'LDV_RIGHT']} for pos_id in SPEECH_POSITIONS}

    for pos_id, (folder_name, x_pos) in SPEECH_POSITIONS.items():
        folder = data_root / folder_name

        ldv_files = list(folder.glob('*LDV*.wav'))
        mic_l_files = list(folder.glob('*LEFT-MIC*.wav'))
        mic_r_files = list(folder.glob('*RIGHT-MIC*.wav'))

        if not ldv_files or not mic_l_files or not mic_r_files:
            raise FileNotFoundError(f"Missing LDV/MIC files for position {pos_id} in {folder}")

        ldv, fs = load_wav(ldv_files[0])
        mic_l, _ = load_wav(mic_l_files[0])
        mic_r, _ = load_wav(mic_r_files[0])

        ldv = bandpass_filter(ldv, fs, freq_band[0], freq_band[1])
        mic_l = bandpass_filter(mic_l, fs, freq_band[0], freq_band[1])
        mic_r = bandpass_filter(mic_r, fs, freq_band[0], freq_band[1])

        window_samples = int(window_size * fs)
        n_segments = len(ldv) // window_samples

        start_seg = max(1, int(n_segments * 0.1))
        end_seg = int(n_segments * 0.9)

        theory = calc_theory_ldv_mic_tau_ms(x_pos)

        for seg_idx in range(start_seg, end_seg):
            start_sample = seg_idx * window_samples
            end_sample = start_sample + window_samples

            seg_ldv = ldv[start_sample:end_sample]
            seg_l = mic_l[start_sample:end_sample]
            seg_r = mic_r[start_sample:end_sample]

            tau_ll, _ = gcc_phat(seg_ldv, seg_l, fs)
            tau_lr, _ = gcc_phat(seg_ldv, seg_r, fs)

            for name, delays in delay_sets.items():
                tau_ll_corr = tau_ll - (delays['LDV'] - delays['LEFT-MIC'])
                tau_lr_corr = tau_lr - (delays['LDV'] - delays['RIGHT-MIC'])

                res_ll = tau_ll_corr - theory['LDV_LEFT']
                res_lr = tau_lr_corr - theory['LDV_RIGHT']

                residuals['LDV_LEFT'][name].append(res_ll)
                residuals['LDV_RIGHT'][name].append(res_lr)

                residuals_by_pos[pos_id]['LDV_LEFT'][name].append(res_ll)
                residuals_by_pos[pos_id]['LDV_RIGHT'][name].append(res_lr)

    report = {
        'timestamp': datetime.now().isoformat(),
        'window_size_sec': window_size,
        'freq_band_hz': list(freq_band),
        'delay_sets_ms': delay_sets,
        'overall_stats': {},
        'per_position_stats': {},
    }

    for pair in residuals:
        report['overall_stats'][pair] = {}
        for name, vals in residuals[pair].items():
            report['overall_stats'][pair][name] = stats(vals)

    for pos_id in residuals_by_pos:
        report['per_position_stats'][pos_id] = {}
        for pair in residuals_by_pos[pos_id]:
            report['per_position_stats'][pos_id][pair] = {}
            for name, vals in residuals_by_pos[pos_id][pair].items():
                report['per_position_stats'][pos_id][pair][name] = stats(vals)

    report_file = output_dir / 'ldv_delay_reeval_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    # Plot residual distributions per pair
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, pair in enumerate(['LDV_LEFT', 'LDV_RIGHT']):
        ax = axes[idx]
        for name, vals in residuals[pair].items():
            if not vals:
                continue
            ax.hist(vals, bins=60, alpha=0.4, label=name)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f'Residuals (Corrected - Theory) for {pair}')
        ax.set_xlabel('Residual (ms)')
        ax.set_ylabel('Count')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'residuals_by_pair.png', dpi=150)
    plt.close()

    print("\n  Overall Residual Summary (abs mean / abs p95):")
    for pair in report['overall_stats']:
        print(f"  {pair}:")
        for name, st in report['overall_stats'][pair].items():
            if st.get('n', 0) == 0:
                continue
            print(f"    {name:<12} abs_mean={st['abs_mean_ms']:.4f} ms, abs_p95={st['abs_p95_ms']:.4f} ms")

    print(f"\n  Report saved to: {report_file}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
