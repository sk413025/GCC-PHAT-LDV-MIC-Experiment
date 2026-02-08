#!/usr/bin/env python3
"""
Negative Position Diagnosis (-0.4m, -0.8m)

Purpose: Diagnose why negative positions fail chirp event quality gates by
inspecting event-level metrics, relaxing gating thresholds, and analyzing
full GCC-PHAT correlation curves.

Outputs (under --out_dir):
- negative_position_diagnosis.json
- plots/*.png (GCC-PHAT correlation curves)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import wavfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_project_root_with_dataset(start: Path) -> Path:
    start = start.resolve()
    for parent in [start] + list(start.parents):
        if (parent / 'dataset').exists():
            return parent
    raise FileNotFoundError(f"Could not find dataset/ above {start}")


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    fs, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    else:
        data = data.astype(np.float64)
    if len(data.shape) > 1:
        data = data[:, 0]
    return data, fs


def compute_gcc_cc(sig_a: np.ndarray, sig_b: np.ndarray, fs: int,
                   max_lag_ms: float, bandpass: Tuple[float, float] = None):
    x = sig_a
    y = sig_b

    if bandpass is not None:
        x = vcc.bandpass_filter(x, bandpass[0], bandpass[1], fs)
        y = vcc.bandpass_filter(y, bandpass[0], bandpass[1], fs)

    x = x - np.mean(x)
    y = y - np.mean(y)

    n = len(x) + len(y)
    n_fft = 1 << int(np.ceil(np.log2(max(2, n))))
    X = np.fft.rfft(x, n_fft)
    Y = np.fft.rfft(y, n_fft)
    R = X * np.conj(Y)
    R /= (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R, n_fft)

    max_shift = int(round((max_lag_ms / 1000.0) * fs))
    max_shift = max(1, min(max_shift, n_fft // 2 - 1))

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    lags = np.arange(-max_shift, max_shift + 1, dtype=int)
    lags_ms = lags * 1000.0 / fs

    cc_abs = np.abs(cc)
    if np.max(cc_abs) > 0:
        cc_abs = cc_abs / np.max(cc_abs)

    return lags_ms, cc_abs


def top_peaks(lags_ms: np.ndarray, cc_abs: np.ndarray, k: int = 5, min_sep: int = 10):
    indices = np.argsort(cc_abs)[::-1]
    peaks = []
    used = []
    for idx in indices:
        if any(abs(idx - u) < min_sep for u in used):
            continue
        peaks.append({'lag_ms': float(lags_ms[idx]), 'value': float(cc_abs[idx])})
        used.append(idx)
        if len(peaks) >= k:
            break
    return peaks


def count_passes(event_rows: List[Dict], micmic_err_max_ms: float,
                 micmic_psr_min_db: float, consistency_max_ms: float) -> int:
    passes = 0
    for e in event_rows:
        mic = e['pairs'].get('LEFT-MIC_RIGHT-MIC')
        if not mic:
            continue
        mic_err = mic.get('tau_err_ms')
        mic_psr = mic.get('psr_db')
        cons = e.get('consistency_ms')

        psr_ok = True if micmic_psr_min_db is None else (mic_psr is not None and mic_psr >= micmic_psr_min_db)
        pass_evt = (
            mic_err is not None and mic_err <= micmic_err_max_ms and
            psr_ok and
            cons is not None and abs(cons) <= consistency_max_ms
        )
        if pass_evt:
            passes += 1
    return passes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Diagnose negative position chirp failures')
    parser.add_argument('--dataset_root', type=Path, default=None,
                        help='Project root containing dataset/ (auto-detected if omitted)')
    parser.add_argument('--out_dir', type=Path, default=None,
                        help='Output directory (default: results/negative_pos_diagnosis/run_<ts>)')
    parser.add_argument('--positions', nargs='*', default=['-0.4', '-0.8'],
                        help='Positions to diagnose (default: -0.4 -0.8)')
    parser.add_argument('--max_events_plot', type=int, default=3,
                        help='Max events to plot per dataset/position')
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    script_root = Path(__file__).resolve().parents[2]
    project_root = find_project_root_with_dataset(script_root)

    dataset_root = args.dataset_root.resolve() if args.dataset_root else project_root
    chirp_root = dataset_root / 'dataset' / 'chirp'
    chirp2_root = dataset_root / 'dataset' / 'chirp_2'

    if not chirp_root.exists() or not chirp2_root.exists():
        raise FileNotFoundError('chirp or chirp_2 dataset not found')

    # Import validate_chirp_calibration from chirp dataset
    sys.path.insert(0, str(chirp_root))
    global vcc
    import validate_chirp_calibration as vcc  # type: ignore

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.out_dir:
        output_dir = args.out_dir
    else:
        output_dir = script_root / 'results' / 'negative_pos_diagnosis' / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'timestamp': datetime.now().isoformat(),
        'positions': args.positions,
        'datasets': {},
    }

    default_cfg = dict(vcc.DEFAULT_CONFIG)

    for dataset_name, root in [('chirp', chirp_root), ('chirp_2', chirp2_root)]:
        positions_info = vcc.discover_chirp_files(str(root))
        report['datasets'][dataset_name] = {}

        for pos_label in args.positions:
            pos_info = next((p for p in positions_info if p['pos_label'] == pos_label), None)
            if pos_info is None:
                raise FileNotFoundError(f"Position {pos_label} not found in {root}")

            res, _obs = vcc.validate_position(pos_info, default_cfg)
            summary = res.get('summary', {})
            event_rows = res.get('events', [])

            # Gate relax sweeps (one parameter at a time)
            sweeps = {
                'micmic_err_max_ms': [],
                'consistency_max_ms': [],
                'micmic_psr_min_db': [],
            }

            for err in [0.30, 0.50, 0.80, 1.20]:
                passes = count_passes(event_rows, err, default_cfg['micmic_psr_min_db'], default_cfg['consistency_max_ms'])
                sweeps['micmic_err_max_ms'].append({'value': err, 'events_used': passes})

            for cons in [0.80, 1.20, 1.60, 2.40]:
                passes = count_passes(event_rows, default_cfg['micmic_err_max_ms'], default_cfg['micmic_psr_min_db'], cons)
                sweeps['consistency_max_ms'].append({'value': cons, 'events_used': passes})

            for psr in [None, 5.0, 10.0, 15.0]:
                passes = count_passes(event_rows, default_cfg['micmic_err_max_ms'], psr, default_cfg['consistency_max_ms'])
                sweeps['micmic_psr_min_db'].append({'value': psr, 'events_used': passes})

            # Geometry metrics
            spk = vcc.SPEAKER_POSITIONS[pos_label]
            mic_l = vcc.GEOMETRY['mic_left']
            mic_r = vcc.GEOMETRY['mic_right']
            d_l = float(np.hypot(spk[0] - mic_l[0], spk[1] - mic_l[1]))
            d_r = float(np.hypot(spk[0] - mic_r[0], spk[1] - mic_r[1]))
            geometry = {
                'speaker_x': float(spk[0]),
                'dist_left_m': d_l,
                'dist_right_m': d_r,
                'delta_dist_m': d_l - d_r,
            }

            # Full GCC-PHAT curves for a few events
            signals = {}
            for sensor, path in pos_info['paths'].items():
                sig, fs = load_wav(Path(path))
                if fs != int(default_cfg['fs']):
                    raise ValueError(f"Expected fs={default_cfg['fs']} for {path}, got {fs}")
                signals[sensor] = sig

            events = vcc.detect_chirp_events(
                signals['LEFT-MIC'],
                int(default_cfg['fs']),
                smooth_ms=float(default_cfg['env_smooth_ms']),
                peak_quantile=float(default_cfg['peak_quantile']),
                thr_scale=float(default_cfg['peak_thr_scale']),
                min_distance_sec=float(default_cfg['peak_min_distance_sec']),
                max_events=int(default_cfg['max_events']),
            )
            env = vcc.smooth_abs_envelope(signals['LEFT-MIC'], int(default_cfg['fs']), float(default_cfg['env_smooth_ms']))
            pre_samples = int(round(float(default_cfg['event_pre_sec']) * default_cfg['fs']))
            post_samples = int(round(float(default_cfg['event_post_sec']) * default_cfg['fs']))

            event_plots = []
            for idx, peak_center in enumerate(events[: args.max_events_plot]):
                onset = vcc.estimate_onset_from_peak(
                    env,
                    int(default_cfg['fs']),
                    peak_center,
                    back_sec=float(default_cfg['onset_search_back_sec']),
                    frac=float(default_cfg['onset_frac']),
                )

                win_l = vcc.extract_asymmetric_window(signals['LEFT-MIC'], onset,
                                                      pre_samples=pre_samples, post_samples=post_samples)
                win_r = vcc.extract_asymmetric_window(signals['RIGHT-MIC'], onset,
                                                      pre_samples=pre_samples, post_samples=post_samples)

                lags_ms, cc_abs = compute_gcc_cc(
                    win_l, win_r, int(default_cfg['fs']),
                    max_lag_ms=float(default_cfg['gcc_max_lag_ms']),
                    bandpass=default_cfg.get('bandpass'),
                )

                peaks = top_peaks(lags_ms, cc_abs, k=5, min_sep=10)

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(lags_ms, cc_abs, color='black', linewidth=1)
                ax.axvline(0, color='red', linestyle='--', linewidth=1)
                for p in peaks:
                    ax.axvline(p['lag_ms'], color='blue', linestyle=':', linewidth=1)
                ax.set_title(f"{dataset_name} {pos_label} event {idx}: GCC-PHAT (MicL vs MicR)")
                ax.set_xlabel('Lag (ms)')
                ax.set_ylabel('Normalized |cc|')
                ax.grid(alpha=0.3)

                plot_name = f"{dataset_name}_{pos_label}_event{idx}_gcc.png"
                fig.savefig(plots_dir / plot_name, dpi=150)
                plt.close(fig)

                # Downsample for JSON
                max_points = 2000
                if len(lags_ms) > max_points:
                    idxs = np.linspace(0, len(lags_ms) - 1, max_points).astype(int)
                    lags_ms_ds = lags_ms[idxs]
                    cc_abs_ds = cc_abs[idxs]
                else:
                    lags_ms_ds = lags_ms
                    cc_abs_ds = cc_abs

                event_plots.append({
                    'event_idx': int(idx),
                    'peak_center_sample': int(peak_center),
                    'onset_sample': int(onset),
                    'plot': plot_name,
                    'top_peaks': peaks,
                    'lags_ms': lags_ms_ds.tolist(),
                    'cc_abs': cc_abs_ds.tolist(),
                })

            report['datasets'][dataset_name][pos_label] = {
                'summary': {
                    'events_detected': summary.get('events_detected', 0),
                    'events_used': summary.get('events_used', 0),
                    'micmic_tau_median_ms': summary.get('pairs', {}).get('LEFT-MIC_RIGHT-MIC', {}).get('tau_median_ms'),
                    'micmic_psr_median_db': summary.get('pairs', {}).get('LEFT-MIC_RIGHT-MIC', {}).get('psr_median_db'),
                    'micmic_err_median_ms': summary.get('pairs', {}).get('LEFT-MIC_RIGHT-MIC', {}).get('err_median_ms'),
                },
                'geometry': geometry,
                'gate_relax_sweeps': sweeps,
                'gcc_event_plots': event_plots,
            }

    out_file = output_dir / 'negative_position_diagnosis.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {out_file}")
    print(f"Plots saved to: {plots_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
