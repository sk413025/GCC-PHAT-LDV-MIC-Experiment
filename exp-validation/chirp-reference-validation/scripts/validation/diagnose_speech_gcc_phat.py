#!/usr/bin/env python3
"""
Comprehensive GCC-PHAT Diagnosis for Speech
============================================

Three-priority diagnostic to understand why Speech GCC-PHAT fails:
Priority 1: Visualize GCC-PHAT failure symptoms
Priority 2: Check 4ms delay discrepancy root cause
Priority 3: Validate OMP-DTmin quality on passing cases

Run: python diagnose_speech_gcc_phat.py --data_root /path/to/dataset --out_dir ./diagnosis_results
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, resample
from scipy.fft import fft, ifft

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from chirp_reference import get_chirp_references
except ImportError:
    print("Warning: Could not import chirp_reference module")


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

DEFAULT_FS = 48000
MAX_LAG_MS = 10.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GCCDiagnosis:
    """Result of GCC-PHAT analysis"""
    position: str
    segment_idx: int
    segment_start_sec: float
    tau_ms: float
    psr_db: float
    peak_value: float
    noise_floor: float
    signal_to_noise: float
    num_peaks_above_threshold: int
    chirp_reference_tau: Optional[float]
    deviation_from_reference: Optional[float]
    collapse_detected: bool
    multi_peak_detected: bool
    diagnosis: str


# ---------------------------------------------------------------------------
# Signal Processing
# ---------------------------------------------------------------------------

def load_wav(path: Path, target_fs: int = None) -> Tuple[np.ndarray, int]:
    """Load WAV file with optional resampling"""
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
        num_samples = int(len(data) * target_fs / fs)
        data = resample(data, num_samples)
        fs = target_fs

    return data, fs


def bandpass_filter(data: np.ndarray, fs: int, lowcut: float, highcut: float, order: int = 5) -> np.ndarray:
    """Apply bandpass filter"""
    nyq = fs / 2
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)

    if low >= high:
        warnings.warn(f"Invalid bandpass: {lowcut}-{highcut} Hz at fs={fs}")
        return data

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def gcc_phat_detailed(sig1: np.ndarray, sig2: np.ndarray, fs: int, max_lag_ms: float = 10.0) -> Dict:
    """
    GCC-PHAT with detailed diagnosis information

    Returns:
        dict with keys:
        - tau_ms: estimated delay in ms
        - psr_db: peak-to-sidelobe ratio
        - peak_value: peak of cross-correlation
        - noise_floor: maximum sidelobe level
        - gcc_curve: the full GCC-PHAT curve
        - lags: lag values in samples
        - lags_ms: lag values in ms
    """
    n = len(sig1) + len(sig2) - 1
    n_fft = 2 ** int(np.ceil(np.log2(n)))

    X1 = fft(sig1, n_fft)
    X2 = fft(sig2, n_fft)

    cross_spectrum = X1 * np.conj(X2)
    magnitude = np.abs(cross_spectrum) + 1e-10
    gcc = np.real(ifft(cross_spectrum / magnitude))

    gcc = np.fft.fftshift(gcc)
    lags = np.arange(-n_fft // 2, n_fft // 2)

    # Search window
    max_lag_samples = int(max_lag_ms * fs / 1000)
    center = n_fft // 2
    search_start = max(0, center - max_lag_samples)
    search_end = min(n_fft, center + max_lag_samples + 1)

    gcc_search = gcc[search_start:search_end]
    lags_search = lags[search_start:search_end]

    if len(gcc_search) == 0:
        return {
            'tau_ms': 0.0,
            'tau_samples': 0,
            'psr_db': 0.0,
            'peak_value': 0.0,
            'noise_floor': 0.0,
            'gcc_curve': gcc_search,
            'lags': lags_search,
            'lags_ms': lags_search * 1000.0 / fs,
            'valid': False,
            'reason': "Empty search window"
        }

    # Find peak
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

    # Sidelobe analysis
    sidelobe_exclusion = 50
    sidelobe_mask = np.ones(len(gcc_search), dtype=bool)
    exclude_start = max(0, peak_idx - sidelobe_exclusion)
    exclude_end = min(len(gcc_search), peak_idx + sidelobe_exclusion + 1)
    sidelobe_mask[exclude_start:exclude_end] = False

    noise_floor = 0.0
    psr_db = 0.0
    if np.any(sidelobe_mask):
        noise_floor = np.max(np.abs(gcc_search[sidelobe_mask]))
        psr_db = 20 * np.log10(peak_value / (noise_floor + 1e-10))

    return {
        'tau_ms': tau_ms,
        'tau_samples': int(tau_samples),
        'psr_db': psr_db,
        'peak_value': peak_value,
        'noise_floor': noise_floor,
        'gcc_curve': gcc_search,
        'lags': lags_search,
        'lags_ms': lags_search * 1000.0 / fs,
        'valid': True,
        'peak_idx': peak_idx,
        'sidelobe_mask': sidelobe_mask,
    }


# ---------------------------------------------------------------------------
# Priority 1: Visualize GCC-PHAT Failure Symptoms
# ---------------------------------------------------------------------------

def priority1_visualize_failures(data_root: Path, out_dir: Path, chirp_refs: Dict[str, float]) -> Dict:
    """
    Priority 1: Diagnose what exactly is wrong with Speech GCC-PHAT

    Returns:
        dict with diagnosis results and failure modes
    """
    print("\n" + "=" * 70)
    print("PRIORITY 1: Visualize GCC-PHAT Failure Symptoms (1 hour)")
    print("=" * 70)

    results = {
        'timestamp': datetime.now().isoformat(),
        'priority': 1,
        'failure_modes': {},
        'statistics': {}
    }

    failed_diagnoses = []

    for pos_id, folder_name in SPEECH_POSITIONS.items():
        folder = data_root / folder_name

        if not folder.exists():
            print(f"  [WARN] Position {pos_id}: folder not found: {folder}")
            continue

        mic_l_files = list(folder.glob("*LEFT-MIC*.wav"))
        mic_r_files = list(folder.glob("*RIGHT-MIC*.wav"))

        if not mic_l_files or not mic_r_files:
            print(f"  [WARN] Position {pos_id}: missing mic files")
            continue

        mic_l, fs = load_wav(mic_l_files[0])
        mic_r, _ = load_wav(mic_r_files[0])
        tau_chirp = chirp_refs.get(pos_id, None)

        print(f"\n  Position {pos_id} (tau_chirp = {tau_chirp:.3f} ms if available):")
        print(f"    Data length: {len(mic_l) / fs:.2f} sec ({len(mic_l)} samples)")

        # Sample some segments
        segment_length = int(fs * 2.0)  # 2 second segments
        num_segments = min(5, len(mic_l) // segment_length)
        segment_indices = np.linspace(0, len(mic_l) - segment_length, num_segments, dtype=int)

        position_diagnoses = []

        for seg_idx, start_sample in enumerate(segment_indices):
            end_sample = start_sample + segment_length
            seg_l = mic_l[start_sample:end_sample]
            seg_r = mic_r[start_sample:end_sample]

            result = gcc_phat_detailed(seg_l, seg_r, fs, MAX_LAG_MS)

            if not result['valid']:
                continue

            # Diagnose failure mode
            collapse_detected = result['peak_value'] < 0.1
            multi_peak_detected = np.sum(result['sidelobe_mask']) > 0 and \
                                  np.max(np.abs(result['gcc_curve'][result['sidelobe_mask']])) > result['peak_value'] * 0.5

            diagnosis_obj = GCCDiagnosis(
                position=pos_id,
                segment_idx=seg_idx,
                segment_start_sec=start_sample / fs,
                tau_ms=result['tau_ms'],
                psr_db=result['psr_db'],
                peak_value=result['peak_value'],
                noise_floor=result['noise_floor'],
                signal_to_noise=result['peak_value'] / (result['noise_floor'] + 1e-10),
                num_peaks_above_threshold=np.sum(np.abs(result['gcc_curve']) > result['peak_value'] * 0.5),
                chirp_reference_tau=tau_chirp,
                deviation_from_reference=abs(result['tau_ms'] - tau_chirp) if tau_chirp else None,
                collapse_detected=collapse_detected,
                multi_peak_detected=multi_peak_detected,
                diagnosis="COLLAPSE" if collapse_detected else ("MULTI-PEAK" if multi_peak_detected else "NORMAL")
            )

            position_diagnoses.append((result, diagnosis_obj))
            failed_diagnoses.append(diagnosis_obj)

        # Visualize sample failures for this position
        if position_diagnoses:
            fig = plt.figure(figsize=(14, 10))
            gs = GridSpec(3, 2, figure=fig)

            for plot_idx, (result, diag) in enumerate(position_diagnoses[:3]):
                row = plot_idx // 2
                col = plot_idx % 2
                ax = fig.add_subplot(gs[row, col])

                ax.plot(result['lags_ms'], np.abs(result['gcc_curve']), 'b-', linewidth=1, label='GCC-PHAT')
                ax.axvline(diag.tau_ms, color='g', linestyle='--', linewidth=2, label=f'Detected tau={diag.tau_ms:.2f}ms')

                if diag.chirp_reference_tau:
                    ax.axvline(diag.chirp_reference_tau, color='r', linestyle=':', linewidth=2, label=f'Chirp ref tau={diag.chirp_reference_tau:.2f}ms')

                # Mark noise floor
                ax.axhline(diag.noise_floor, color='orange', linestyle='--', alpha=0.5, label=f'Noise floor')

                ax.set_xlabel('Lag (ms)')
                ax.set_ylabel('Cross-correlation magnitude')
                ax.set_title(f'Seg{diag.segment_idx} ({diag.segment_start_sec:.1f}s): {diag.diagnosis} (PSR={diag.psr_db:.1f}dB)')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            # Summary subplot
            ax_summary = fig.add_subplot(gs[2, :])
            diagnoses_summary = [f"{d.diagnosis}" for _, d in position_diagnoses[:3]]
            psr_summary = [d.psr_db for _, d in position_diagnoses[:3]]
            colors = ['red' if d.collapse_detected else 'orange' if d.multi_peak_detected else 'green' for _, d in position_diagnoses[:3]]

            ax_summary.bar(range(len(diagnoses_summary)), psr_summary, color=colors, alpha=0.7)
            ax_summary.set_ylabel('PSR (dB)')
            ax_summary.set_xlabel('Segment')
            ax_summary.set_title('PSR Distribution (Red=Collapse, Orange=Multi-peak, Green=Normal)')
            ax_summary.grid(True, alpha=0.3, axis='y')

            plt.suptitle(f'GCC-PHAT Failure Diagnosis - Position {pos_id}', fontsize=12, fontweight='bold')
            plt.tight_layout()

            plot_path = out_dir / f"p1_gcc_phat_diagnosis_pos_{pos_id}.png"
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"    [OK] Saved diagnosis plot: {plot_path.name}")

    # Aggregate statistics
    if failed_diagnoses:
        collapses = [d for d in failed_diagnoses if d.collapse_detected]
        multi_peaks = [d for d in failed_diagnoses if d.multi_peak_detected]

        results['failure_modes'] = {
            'total_segments': len(failed_diagnoses),
            'collapse_rate': len(collapses) / len(failed_diagnoses) if failed_diagnoses else 0,
            'multi_peak_rate': len(multi_peaks) / len(failed_diagnoses) if failed_diagnoses else 0,
            'avg_psr_db': np.mean([d.psr_db for d in failed_diagnoses]),
            'avg_peak_value': np.mean([d.peak_value for d in failed_diagnoses]),
            'avg_signal_to_noise': np.mean([d.signal_to_noise for d in failed_diagnoses]),
        }

        results['sample_failures'] = [asdict(d) for d in failed_diagnoses[:10]]

    print(f"\n  Summary:")
    print(f"    Total segments analyzed: {len(failed_diagnoses)}")
    if failed_diagnoses:
        print(f"    Collapse rate: {results['failure_modes']['collapse_rate']*100:.1f}%")
        print(f"    Multi-peak rate: {results['failure_modes']['multi_peak_rate']*100:.1f}%")
        print(f"    Average PSR: {results['failure_modes']['avg_psr_db']:.2f} dB")
        print(f"    Average Signal/Noise: {results['failure_modes']['avg_signal_to_noise']:.2f}")

    return results


# ---------------------------------------------------------------------------
# Priority 2: Check 4ms Delay Discrepancy
# ---------------------------------------------------------------------------

def priority2_check_delay_discrepancy(data_root: Path, out_dir: Path, chirp_refs: Dict[str, float]) -> Dict:
    """
    Priority 2: Diagnose the 4ms delay difference between Chirp and Speech

    Check:
    1. Sampling rate
    2. Channel alignment
    3. Time sync
    4. Frequency band effects
    """
    print("\n" + "=" * 70)
    print("PRIORITY 2: Check 4ms Delay Discrepancy Root Cause (30 min)")
    print("=" * 70)

    results = {
        'timestamp': datetime.now().isoformat(),
        'priority': 2,
        'checks': {},
    }

    # Check 1: Sampling rate consistency
    print("\n  Check 1: Sampling rate consistency")
    for pos_id, folder_name in SPEECH_POSITIONS.items():
        folder = data_root / folder_name

        if not folder.exists():
            continue

        mic_l_files = list(folder.glob("*LEFT-MIC*.wav"))
        mic_r_files = list(folder.glob("*RIGHT-MIC*.wav"))

        if not mic_l_files or not mic_r_files:
            continue

        fs_l, _ = wavfile.read(mic_l_files[0])[:2]
        fs_r, _ = wavfile.read(mic_r_files[0])[:2]

        fs_l = fs_l if isinstance(fs_l, int) else fs_l[0]
        fs_r = fs_r if isinstance(fs_r, int) else fs_r[0]

        results['checks']['sampling_rate'] = {
            'position': pos_id,
            'fs_left': fs_l,
            'fs_right': fs_r,
            'match': fs_l == fs_r,
            'both_48k': fs_l == 48000 and fs_r == 48000,
        }

        print(f"    Position {pos_id}: LEFT={fs_l} Hz, RIGHT={fs_r} Hz, Match={fs_l == fs_r}")

    # Check 2: Channel length alignment
    print("\n  Check 2: Channel length alignment")
    for pos_id, folder_name in SPEECH_POSITIONS.items():
        folder = data_root / folder_name

        if not folder.exists():
            continue

        mic_l_files = list(folder.glob("*LEFT-MIC*.wav"))
        mic_r_files = list(folder.glob("*RIGHT-MIC*.wav"))

        if not mic_l_files or not mic_r_files:
            continue

        _, data_l = wavfile.read(mic_l_files[0])
        _, data_r = wavfile.read(mic_r_files[0])

        len_l = len(data_l)
        len_r = len(data_r)

        results['checks']['channel_alignment'] = {
            'position': pos_id,
            'length_left_samples': len_l,
            'length_right_samples': len_r,
            'length_match': len_l == len_r,
            'max_difference': abs(len_l - len_r),
        }

        print(f"    Position {pos_id}: LEFT={len_l} samples, RIGHT={len_r} samples, Diff={abs(len_l - len_r)}")

    # Check 3: Frequency band effect on delay
    print("\n  Check 3: Frequency band effect on delay estimation")

    freq_bands = {
        'full': (20, 22000),
        'chirp_like': (50, 10000),  # Typical chirp range
        'narrow_speech': (200, 4000),
        'wide_speech': (100, 8000),
    }

    band_results = {}

    for pos_id, folder_name in SPEECH_POSITIONS.items():
        folder = data_root / folder_name

        if not folder.exists():
            continue

        mic_l_files = list(folder.glob("*LEFT-MIC*.wav"))
        mic_r_files = list(folder.glob("*RIGHT-MIC*.wav"))

        if not mic_l_files or not mic_r_files:
            continue

        mic_l, fs = load_wav(mic_l_files[0])
        mic_r, _ = load_wav(mic_r_files[0])

        # Take first 2 seconds
        seg_len = int(fs * 2.0)
        mic_l = mic_l[:seg_len]
        mic_r = mic_r[:seg_len]

        tau_chirp = chirp_refs.get(pos_id, None)

        band_results[pos_id] = {}

        print(f"\n    Position {pos_id} (tau_chirp = {tau_chirp:.3f} ms):")

        for band_name, (low_freq, high_freq) in freq_bands.items():
            mic_l_filt = bandpass_filter(mic_l, fs, low_freq, high_freq)
            mic_r_filt = bandpass_filter(mic_r, fs, low_freq, high_freq)

            result = gcc_phat_detailed(mic_l_filt, mic_r_filt, fs, MAX_LAG_MS)

            if result['valid']:
                deviation = abs(result['tau_ms'] - tau_chirp) if tau_chirp else None
                band_results[pos_id][band_name] = {
                    'tau_ms': result['tau_ms'],
                    'psr_db': result['psr_db'],
                    'deviation_from_chirp': deviation,
                }

                print(f"      {band_name:15}: tau={result['tau_ms']:6.2f}ms (PSR={result['psr_db']:6.1f}dB), dev={deviation:.2f}ms" if deviation else f"      {band_name:15}: tau={result['tau_ms']:6.2f}ms (PSR={result['psr_db']:6.1f}dB)")

    results['checks']['frequency_band_effect'] = band_results

    return results


# ---------------------------------------------------------------------------
# Priority 3: Validate OMP-DTmin Quality
# ---------------------------------------------------------------------------

def priority3_validate_omp_dtmin(data_root: Path, out_dir: Path, chirp_refs: Dict[str, float]) -> Dict:
    """
    Priority 3: Check if OMP-DTmin estimates are reliable

    On passing cases, check:
    1. Consistency of per-frequency delays
    2. Deviation from chirp reference
    """
    print("\n" + "=" * 70)
    print("PRIORITY 3: Validate OMP-DTmin Quality (2 hours)")
    print("=" * 70)

    results = {
        'timestamp': datetime.now().isoformat(),
        'priority': 3,
        'note': 'This requires OMP-DTmin implementation and passing case data',
        'placeholder': 'Manual inspection of OMP-DTmin results needed'
    }

    print("\n  [INFO] This priority requires:")
    print("    1. OMP-DTmin implementation code")
    print("    2. Identification of 'passing' cases from Phase 3")
    print("    3. Per-frequency delay extraction from OMP-DTmin")
    print("\n  Recommendation: After priorities 1-2, we can implement this")
    print("  if we have access to OMP-DTmin results and passing case metadata.")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive diagnosis of Speech GCC-PHAT failures'
    )
    parser.add_argument(
        '--data_root',
        type=Path,
        default=Path(__file__).resolve().parents[4] / 'dataset',
        help='Path to dataset directory (default: ../../dataset)'
    )
    parser.add_argument(
        '--out_dir',
        type=Path,
        default=Path(__file__).resolve().parent / 'diagnosis_results',
        help='Output directory for diagnosis results'
    )

    args = parser.parse_args()

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("SPEECH GCC-PHAT COMPREHENSIVE DIAGNOSIS")
    print("=" * 70)
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.out_dir}")

    # Load chirp references
    print("\nLoading chirp references...")
    try:
        refs, report = get_chirp_references(args.data_root.parent.parent)  # Go up to find dataset
        chirp_refs = refs
        print(f"[OK] Loaded {len(chirp_refs)} chirp references")
    except Exception as e:
        print(f"[WARN] Could not load chirp references: {e}")
        print("  Using geometric fallback values instead")
        # Use geometric fallback
        from chirp_reference import geometric_tau_ms, SPEECH_TO_CHIRP, POSITION_X
        chirp_refs = {
            pos_id: geometric_tau_ms(POSITION_X[SPEECH_TO_CHIRP[pos_id]])
            for pos_id in SPEECH_POSITIONS.keys()
        }
        print(f"[OK] Using geometric tau values for {len(chirp_refs)} positions")

    # Execute priorities
    all_results = {
        'execution_timestamp': datetime.now().isoformat(),
        'data_root': str(args.data_root),
        'output_dir': str(args.out_dir),
        'chirp_references': chirp_refs,
        'priorities': {}
    }

    # Priority 1
    try:
        p1_results = priority1_visualize_failures(args.data_root, args.out_dir, chirp_refs)
        all_results['priorities']['priority_1'] = p1_results
    except Exception as e:
        print(f"[ERROR] Priority 1 failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['priorities']['priority_1'] = {'error': str(e)}

    # Priority 2
    try:
        p2_results = priority2_check_delay_discrepancy(args.data_root, args.out_dir, chirp_refs)
        all_results['priorities']['priority_2'] = p2_results
    except Exception as e:
        print(f"[ERROR] Priority 2 failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['priorities']['priority_2'] = {'error': str(e)}

    # Priority 3
    try:
        p3_results = priority3_validate_omp_dtmin(args.data_root, args.out_dir, chirp_refs)
        all_results['priorities']['priority_3'] = p3_results
    except Exception as e:
        print(f"[ERROR] Priority 3 failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['priorities']['priority_3'] = {'error': str(e)}

    # Save results
    results_json = args.out_dir / 'diagnosis_results.json'
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[OK] Diagnosis complete. Results saved to: {results_json}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
