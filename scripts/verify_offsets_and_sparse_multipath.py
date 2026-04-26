#!/usr/bin/env python
"""
Verify an "offsets + sparse multipath" measurement-chain model using existing WAVs only.

Goal
----
Produce reproducible diagnostics that test whether observed GCC-PHAT peaks are dominated by:
1) stable constant offsets / stable non-LOS paths (speaker-position invariant), and
2) a small number of dominant paths (sparse multipath).

This script does NOT try to recover LDV↔MIC geometry TDOA (τ₂/τ₃). It provides evidence
for/against an offset-dominated measurement chain and proposes a simple, testable
"silence-derived offset" subtraction.

CLI (plan-locked)
-----------------
python -u scripts/verify_offsets_and_sparse_multipath.py \\
  --data_root /home/sbplab/jiawei/data \\
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \\
  --out_dir results/offsets_multipath_verify_<YYYYMMDD_HHMMSS> \\
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)


# Locked constants
FS_EXPECTED = 48_000
WINDOW_SEC = 5.0
CENTER_START_SEC = 100.0
CENTER_END_SEC = 600.0
CENTER_STEP_SEC = 1.0

BAND_HZ = (500.0, 2000.0)
N_BANDS = 64  # used only for reporting edges

MAX_LAG_MS = 10.0
GUIDED_RADIUS_MS = 0.3
PSR_EXCLUDE_SAMPLES = 50

SPEECH_RMS_PERCENTILE = 50.0
SILENCE_PERCENT = 1.0  # bottom 1% by RMS

TOPK_PEAKS = 5
TOPK_EXCLUDE_SAMPLES = 50

TAU_BIN_MS = 0.1
RIDGE_EPS = 1e-12

GEOMETRY = {
    "mic_left": (-0.7, 2.0),
    "mic_right": (0.7, 2.0),
    "speakers": {
        "18": (0.8, 0.0),
        "19": (0.4, 0.0),
        "20": (0.0, 0.0),
        "21": (-0.4, 0.0),
        "22": (-0.8, 0.0),
    },
}


def configure_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_fingerprint(files: list[Path], *, root: Path) -> str:
    entries: list[str] = []
    for fp in sorted(files, key=lambda p: p.as_posix().lower()):
        entries.append(f"{sha256_file(fp)} {fp.relative_to(root).as_posix()}")
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def git_head_and_dirty() -> tuple[str | None, bool | None]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() != ""
        return head, dirty
    except Exception:
        return None, None


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_code_state(out_dir: Path, script_path: Path) -> None:
    head, dirty = git_head_and_dirty()
    payload = {
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "git_head": head,
        "dirty": dirty,
        "timestamp": datetime.now().isoformat(),
    }
    write_json(out_dir / "code_state.json", payload)


def load_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(str(path))
    if data.ndim != 1:
        raise ValueError(f"Expected mono WAV: {path} (shape={data.shape})")
    if data.dtype == np.int16:
        data = (data.astype(np.float32) / 32768.0).astype(np.float32, copy=False)
    elif data.dtype == np.int32:
        data = (data.astype(np.float32) / 2147483648.0).astype(np.float32, copy=False)
    elif data.dtype == np.float64:
        data = data.astype(np.float32)
    else:
        data = data.astype(np.float32)
    return int(sr), data.astype(np.float32, copy=False)


def list_triplet_files(data_root: Path, speaker: str) -> tuple[Path, Path, Path]:
    sp_dir = data_root / speaker
    ldv_files = sorted(sp_dir.glob("*LDV*.wav"))
    micl_files = sorted(sp_dir.glob("*LEFT*.wav"))
    micr_files = sorted(sp_dir.glob("*RIGHT*.wav"))
    if not ldv_files or not micl_files or not micr_files:
        raise FileNotFoundError(
            f"Missing WAV triplet in {sp_dir} (LDV={len(ldv_files)}, LEFT={len(micl_files)}, RIGHT={len(micr_files)})"
        )
    return ldv_files[0], micl_files[0], micr_files[0]


def extract_centered_window(sig: np.ndarray, *, fs: int, center_sec: float, window_sec: float) -> np.ndarray:
    win_samples = int(round(float(window_sec) * float(fs)))
    center_samp = int(round(float(center_sec) * float(fs)))
    start = int(center_samp - win_samples // 2)
    end = int(start + win_samples)
    if start < 0 or end > len(sig):
        raise ValueError(
            f"Window out of bounds: center_sec={center_sec}, start={start}, end={end}, len={len(sig)}"
        )
    return sig[start:end]


def rms(x: np.ndarray) -> float:
    x64 = x.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(x64 * x64)))


def band_edges_linear(band_hz: tuple[float, float], n_bands: int) -> np.ndarray:
    lo, hi = float(band_hz[0]), float(band_hz[1])
    return np.linspace(lo, hi, int(n_bands) + 1, dtype=np.float64)


def fft_freqs(fs: int, n_fft: int) -> np.ndarray:
    return np.fft.rfftfreq(int(n_fft), d=1.0 / float(fs)).astype(np.float64, copy=False)


def ccwin_from_spectrum(R_w: np.ndarray, *, n_fft: int, max_shift: int) -> np.ndarray:
    cc = np.fft.irfft(R_w, int(n_fft))
    cc = np.real(cc)
    return np.concatenate((cc[-max_shift:], cc[: max_shift + 1])).astype(np.float64, copy=False)


def estimate_guided_tau_psr(
    cc_win: np.ndarray,
    *,
    fs: int,
    max_shift: int,
    guided_tau_sec: float,
    guided_radius_sec: float,
    psr_exclude_samples: int,
) -> tuple[float, float]:
    abs_cc = np.abs(cc_win)
    guided_center = int(round(float(guided_tau_sec) * float(fs))) + int(max_shift)
    guided_radius = int(round(float(guided_radius_sec) * float(fs)))
    lo = max(0, guided_center - guided_radius)
    hi = min(len(abs_cc) - 1, guided_center + guided_radius)
    if lo > hi:
        raise ValueError("Invalid guided window")

    peak_idx = int(np.argmax(abs_cc[lo : hi + 1])) + lo

    shift = 0.0
    if 0 < peak_idx < len(abs_cc) - 1:
        y0 = abs_cc[peak_idx - 1]
        y1 = abs_cc[peak_idx]
        y2 = abs_cc[peak_idx + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(float(denom)) > 1e-12:
            shift = float(0.5 * (y0 - y2) / denom)

    tau_sec = ((peak_idx - max_shift) + shift) / float(fs)

    mask = np.ones_like(abs_cc, dtype=bool)
    exc = int(psr_exclude_samples)
    lo_e = max(0, peak_idx - exc)
    hi_e = min(len(abs_cc), peak_idx + exc + 1)
    mask[lo_e:hi_e] = False
    sidelobe_max = float(abs_cc[mask].max()) if np.any(mask) else 0.0
    peak_val = float(abs_cc[peak_idx])
    psr_db = 20.0 * float(np.log10(peak_val / (sidelobe_max + 1e-10)))
    return float(tau_sec), float(psr_db)


def topk_peaks_from_ccwin(cc_win: np.ndarray, *, fs: int, max_shift: int, k: int) -> list[dict[str, float]]:
    abs_cc = np.abs(cc_win).copy()
    peaks: list[dict[str, float]] = []
    for _ in range(int(k)):
        idx = int(np.argmax(abs_cc))
        val = float(abs_cc[idx])
        if not np.isfinite(val) or val <= 0.0:
            break
        tau_ms = float((idx - max_shift) * 1000.0 / float(fs))
        peaks.append({"tau_ms": tau_ms, "peak_abs": val})
        lo = max(0, idx - int(TOPK_EXCLUDE_SAMPLES))
        hi = min(len(abs_cc), idx + int(TOPK_EXCLUDE_SAMPLES) + 1)
        abs_cc[lo:hi] = 0.0
    return peaks


def compute_geometry_truth(speaker_id: str, *, c: float = 343.0, d: float = 1.4) -> dict[str, float]:
    key = speaker_id.split("-")[0]
    if key not in GEOMETRY["speakers"]:
        raise ValueError(f"Unknown speaker key: {key}")
    x_s, y_s = GEOMETRY["speakers"][key]
    x_l, y_l = GEOMETRY["mic_left"]
    x_r, y_r = GEOMETRY["mic_right"]
    d_l = float(np.hypot(x_s - x_l, y_s - y_l))
    d_r = float(np.hypot(x_s - x_r, y_s - y_r))
    tau_true = (d_l - d_r) / float(c)
    sin_theta = float(np.clip(tau_true * float(c) / float(d), -1.0, 1.0))
    theta_true = float(np.degrees(np.arcsin(sin_theta)))
    return {"tau_true_ms": float(tau_true * 1000.0), "theta_true_deg": theta_true}


def load_truth_reference(summary_path: Path) -> dict[str, float]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    ref = payload.get("truth_reference", None)
    if not isinstance(ref, dict):
        raise ValueError(f"Missing truth_reference in {summary_path}")
    if "tau_ref_ms" not in ref or "theta_ref_deg" not in ref:
        raise ValueError(f"truth_reference missing tau_ref_ms/theta_ref_deg in {summary_path}")
    return {"tau_ref_ms": float(ref["tau_ref_ms"]), "theta_ref_deg": float(ref["theta_ref_deg"]), "label": str(ref.get("label", ""))}


def mode_bin_center(values_ms: list[float], *, bin_ms: float) -> float:
    if not values_ms:
        return float("nan")
    b = float(bin_ms)
    bins = [int(np.floor(v / b)) for v in values_ms]
    # mode bin
    counts: dict[int, int] = {}
    for x in bins:
        counts[x] = counts.get(x, 0) + 1
    m = max(counts.items(), key=lambda kv: kv[1])[0]
    return float((m + 0.5) * b)


def entropy_from_counts(counts: dict[int, int]) -> float:
    n = float(sum(counts.values()))
    if n <= 0.0:
        return float("nan")
    p = np.asarray([c / n for c in counts.values()], dtype=np.float64)
    p = p[p > 0.0]
    return float(-(p * np.log(p)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--truth_ref_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--speakers", nargs="+", required=True, type=str)
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    truth_ref_root = Path(args.truth_ref_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    speakers = [str(s) for s in args.speakers]

    configure_logging(out_dir)
    write_code_state(out_dir, Path(__file__).resolve())

    edges_hz = band_edges_linear(BAND_HZ, N_BANDS)

    centers_grid = np.arange(CENTER_START_SEC, CENTER_END_SEC + 1e-9, CENTER_STEP_SEC, dtype=np.float64)
    win_samples = int(round(WINDOW_SEC * FS_EXPECTED))
    n_fft = int(win_samples * 2)
    f_fft = fft_freqs(FS_EXPECTED, n_fft)
    analysis_mask_fft = ((f_fft >= BAND_HZ[0]) & (f_fft <= BAND_HZ[1])).astype(np.float64)
    max_shift = int(round(float(MAX_LAG_MS) * float(FS_EXPECTED) / 1000.0))
    guided_radius_sec = float(GUIDED_RADIUS_MS) / 1000.0

    per_speaker_dir = out_dir / "per_speaker"
    per_speaker_dir.mkdir(parents=True, exist_ok=True)

    all_files: list[Path] = []

    pair_names = ["micl_micr", "ldv_micl", "ldv_micr"]
    offsets_by_pair: dict[str, list[float]] = {p: [] for p in pair_names}
    speech_global_peak_by_pair: dict[str, list[float]] = {p: [] for p in pair_names}
    silence_top1_bins_by_pair: dict[str, list[int]] = {p: [] for p in pair_names}

    # For MIC-MIC calibration effect vs chirp reference
    tau_err_before: list[float] = []
    tau_err_after: list[float] = []

    for sp in speakers:
        ldv_path, micl_path, micr_path = list_triplet_files(data_root, sp)
        all_files.extend([ldv_path, micl_path, micr_path])

        truth_ref = load_truth_reference(truth_ref_root / sp / "summary.json")
        geom = compute_geometry_truth(sp)

        sr_ldv, ldv = load_wav_mono(ldv_path)
        sr_l, micl = load_wav_mono(micl_path)
        sr_r, micr = load_wav_mono(micr_path)
        if not (sr_ldv == sr_l == sr_r == FS_EXPECTED):
            raise ValueError(f"fs mismatch for {sp}: ldv={sr_ldv}, micl={sr_l}, micr={sr_r}, expected={FS_EXPECTED}")
        duration_sec = min(len(ldv), len(micl), len(micr)) / float(FS_EXPECTED)
        logger.info("Speaker %s duration: %.2f s", sp, duration_sec)

        # Build candidate windows (same rule as teacher scripts)
        candidates: list[dict[str, Any]] = []
        for csec in centers_grid.tolist():
            try:
                seg_micl = extract_centered_window(micl, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)
                seg_micr = extract_centered_window(micr, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)
                seg_ldv = extract_centered_window(ldv, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)
            except ValueError:
                continue
            candidates.append(
                {
                    "center_sec": float(csec),
                    "micl": seg_micl.astype(np.float64, copy=False),
                    "micr": seg_micr.astype(np.float64, copy=False),
                    "ldv": seg_ldv.astype(np.float64, copy=False),
                    "rms_micl": rms(seg_micl),
                }
            )
        if not candidates:
            raise RuntimeError(f"No candidate windows for speaker {sp}")

        rms_vals = np.asarray([c["rms_micl"] for c in candidates], dtype=np.float64)
        speech_thresh = float(np.percentile(rms_vals, SPEECH_RMS_PERCENTILE))
        speech_windows = [c for c in candidates if float(c["rms_micl"]) >= speech_thresh]
        if not speech_windows:
            raise RuntimeError(f"No speech windows after RMS filter for speaker {sp}")

        n_silence = max(3, int(round(len(candidates) * (SILENCE_PERCENT / 100.0))))
        silence_windows = sorted(candidates, key=lambda d: float(d["rms_micl"]))[:n_silence]

        guided_tau_sec = float(truth_ref["tau_ref_ms"]) / 1000.0

        def ccwin_pair(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            X = np.fft.rfft(x, n_fft)
            Y = np.fft.rfft(y, n_fft)
            R = X * np.conj(Y)
            R_phat = R / (np.abs(R) + RIDGE_EPS)
            return ccwin_from_spectrum((R_phat * analysis_mask_fft).astype(np.complex128, copy=False), n_fft=n_fft, max_shift=max_shift)

        # 1) silence-derived offsets (global peak)
        silence_top1: dict[str, list[float]] = {p: [] for p in pair_names}
        silence_bin_counts: dict[str, dict[int, int]] = {p: {} for p in pair_names}
        for w in silence_windows:
            seg_ldv = w["ldv"]
            seg_l = w["micl"]
            seg_r = w["micr"]

            cc_lr = ccwin_pair(seg_l, seg_r)
            cc_ldvL = ccwin_pair(seg_ldv, seg_l)
            cc_ldvR = ccwin_pair(seg_ldv, seg_r)

            for pname, cc in [("micl_micr", cc_lr), ("ldv_micl", cc_ldvL), ("ldv_micr", cc_ldvR)]:
                peaks = topk_peaks_from_ccwin(cc, fs=FS_EXPECTED, max_shift=max_shift, k=1)
                if not peaks:
                    continue
                tau_ms = float(peaks[0]["tau_ms"])
                silence_top1[pname].append(tau_ms)
                b = int(np.floor(tau_ms / float(TAU_BIN_MS)))
                silence_bin_counts[pname][b] = silence_bin_counts[pname].get(b, 0) + 1

        tau_offset_ms: dict[str, float] = {p: mode_bin_center(silence_top1[p], bin_ms=TAU_BIN_MS) for p in pair_names}
        for p in pair_names:
            offsets_by_pair[p].append(float(tau_offset_ms[p]))
            if silence_bin_counts[p]:
                # store top-1 mass proxy
                top1 = max(silence_bin_counts[p].values())
                silence_top1_bins_by_pair[p].append(int(top1))

        # 2) write per-window peaks
        spk_out = per_speaker_dir / sp
        spk_out.mkdir(parents=True, exist_ok=True)
        windows_path = spk_out / "windows.jsonl"
        with windows_path.open("w", encoding="utf-8") as f_jsonl:
            # silence windows
            for w in silence_windows:
                seg_ldv = w["ldv"]
                seg_l = w["micl"]
                seg_r = w["micr"]
                cc_lr = ccwin_pair(seg_l, seg_r)
                cc_ldvL = ccwin_pair(seg_ldv, seg_l)
                cc_ldvR = ccwin_pair(seg_ldv, seg_r)
                rec = {
                    "speaker_id": sp,
                    "center_sec": float(w["center_sec"]),
                    "kind": "silence",
                    "peaks": {
                        "micl_micr": topk_peaks_from_ccwin(cc_lr, fs=FS_EXPECTED, max_shift=max_shift, k=TOPK_PEAKS),
                        "ldv_micl": topk_peaks_from_ccwin(cc_ldvL, fs=FS_EXPECTED, max_shift=max_shift, k=TOPK_PEAKS),
                        "ldv_micr": topk_peaks_from_ccwin(cc_ldvR, fs=FS_EXPECTED, max_shift=max_shift, k=TOPK_PEAKS),
                    },
                }
                f_jsonl.write(json.dumps(rec) + "\n")

            # speech windows
            for w in speech_windows:
                seg_ldv = w["ldv"]
                seg_l = w["micl"]
                seg_r = w["micr"]
                cc_lr = ccwin_pair(seg_l, seg_r)
                cc_ldvL = ccwin_pair(seg_ldv, seg_l)
                cc_ldvR = ccwin_pair(seg_ldv, seg_r)

                peaks_lr = topk_peaks_from_ccwin(cc_lr, fs=FS_EXPECTED, max_shift=max_shift, k=TOPK_PEAKS)
                peaks_ldvL = topk_peaks_from_ccwin(cc_ldvL, fs=FS_EXPECTED, max_shift=max_shift, k=TOPK_PEAKS)
                peaks_ldvR = topk_peaks_from_ccwin(cc_ldvR, fs=FS_EXPECTED, max_shift=max_shift, k=TOPK_PEAKS)

                # guided MIC-MIC tau around chirp reference
                tau_guided_sec, psr_guided = estimate_guided_tau_psr(
                    cc_lr,
                    fs=FS_EXPECTED,
                    max_shift=max_shift,
                    guided_tau_sec=guided_tau_sec,
                    guided_radius_sec=guided_radius_sec,
                    psr_exclude_samples=int(PSR_EXCLUDE_SAMPLES),
                )
                tau_guided_ms = float(tau_guided_sec * 1000.0)
                err_before = abs(tau_guided_ms - float(truth_ref["tau_ref_ms"]))
                err_after = abs((tau_guided_ms - float(tau_offset_ms["micl_micr"])) - float(truth_ref["tau_ref_ms"]))
                tau_err_before.append(float(err_before))
                tau_err_after.append(float(err_after))

                # global peak stability stats (top1)
                if peaks_lr:
                    speech_global_peak_by_pair["micl_micr"].append(float(peaks_lr[0]["tau_ms"]))
                if peaks_ldvL:
                    speech_global_peak_by_pair["ldv_micl"].append(float(peaks_ldvL[0]["tau_ms"]))
                if peaks_ldvR:
                    speech_global_peak_by_pair["ldv_micr"].append(float(peaks_ldvR[0]["tau_ms"]))

                rec = {
                    "speaker_id": sp,
                    "center_sec": float(w["center_sec"]),
                    "kind": "speech",
                    "truth_reference": truth_ref,
                    "geometry_truth": geom,
                    "tau_offset_silence_ms": tau_offset_ms,
                    "micl_micr_guided": {"tau_ms": tau_guided_ms, "psr_db": float(psr_guided)},
                    "micl_micr_guided_err_ref_ms": {"before": float(err_before), "after": float(err_after)},
                    "peaks": {"micl_micr": peaks_lr, "ldv_micl": peaks_ldvL, "ldv_micr": peaks_ldvR},
                }
                f_jsonl.write(json.dumps(rec) + "\n")

        # Per-speaker summary
        write_json(
            spk_out / "summary.json",
            {
                "generated": datetime.now().isoformat(),
                "speaker_id": sp,
                "truth_reference": truth_ref,
                "geometry_truth": geom,
                "counts": {"n_candidate_windows": len(candidates), "n_speech_windows": len(speech_windows), "n_silence_windows": len(silence_windows)},
                "rms_thresholds": {"speech_rms_micl_p50": speech_thresh},
                "tau_offset_silence_ms": tau_offset_ms,
                "silence_top1_tau_ms": {p: silence_top1[p] for p in pair_names},
            },
        )

    # Write manifest
    manifest = {
        "generated": datetime.now().isoformat(),
        "data_root": str(data_root),
        "truth_ref_root": str(truth_ref_root),
        "speakers": speakers,
        "wav_files": [{"path": str(p), "sha256": sha256_file(p)} for p in sorted(set(all_files))],
        "dataset_fingerprint_sha256": dataset_fingerprint(sorted(set(all_files)), root=data_root),
    }
    write_json(out_dir / "manifest.json", manifest)

    run_config = {
        "generated": datetime.now().isoformat(),
        "data_root": str(data_root),
        "truth_ref_root": str(truth_ref_root),
        "speakers": speakers,
        "fs_expected": int(FS_EXPECTED),
        "window_sec": float(WINDOW_SEC),
        "center_grid": [float(CENTER_START_SEC), float(CENTER_END_SEC), float(CENTER_STEP_SEC)],
        "band_hz": [float(BAND_HZ[0]), float(BAND_HZ[1])],
        "silence_percent": float(SILENCE_PERCENT),
        "speech_rms_percentile": float(SPEECH_RMS_PERCENTILE),
        "max_lag_ms": float(MAX_LAG_MS),
        "topk_peaks": int(TOPK_PEAKS),
        "tau_bin_ms": float(TAU_BIN_MS),
        "guided_radius_ms": float(GUIDED_RADIUS_MS),
    }
    write_json(out_dir / "run_config.json", run_config)

    # Aggregate checks
    offsets_report: dict[str, Any] = {"generated": datetime.now().isoformat(), "run_dir": str(out_dir), "pairs": {}}
    for p in pair_names:
        xs = np.asarray(offsets_by_pair[p], dtype=np.float64)
        offsets_report["pairs"][p] = {
            "tau_offset_silence_ms": offsets_by_pair[p],
            "across_speaker_mean_ms": float(np.mean(xs)) if xs.size else float("nan"),
            "across_speaker_std_ms": float(np.std(xs)) if xs.size else float("nan"),
        }

    # Calibration effect vs chirp-ref (MIC-MIC guided)
    eb = np.asarray(tau_err_before, dtype=np.float64)
    ea = np.asarray(tau_err_after, dtype=np.float64)
    med_before = float(np.median(eb)) if eb.size else float("nan")
    med_after = float(np.median(ea)) if ea.size else float("nan")
    cal_improve_frac = float((med_before - med_after) / med_before) if np.isfinite(med_before) and med_before > 0 else float("nan")

    # Acceptance criteria (plan-locked; may fail and must be recorded)
    accept = {"computed": {}, "overall_pass": False}
    # 1) low std of silence-derived offsets across speakers
    std_ok = True
    for p in ["ldv_micl", "ldv_micr"]:
        std_ms = float(offsets_report["pairs"][p]["across_speaker_std_ms"])
        ok = bool(np.isfinite(std_ms) and std_ms <= 0.3)
        accept["computed"][f"offset_std_le_0p3ms_{p}"] = ok
        std_ok = std_ok and ok

    # 2) mic-mic offset subtraction reduces median |tau - tau_ref| by >=20%
    mic_ok = bool(np.isfinite(cal_improve_frac) and cal_improve_frac >= 0.20)
    accept["computed"]["micl_micr_median_abs_tauerr_ref_improve_ge_0p20"] = mic_ok

    # 3) sparse multipath proxy: top-1 bin count mass >=0.6 (approx via max bin count / n_silence)
    sparse_ok = False
    for p in ["ldv_micl", "ldv_micr"]:
        # silence_top1_bins_by_pair stores max bin count per speaker; normalize by n_silence (approx 5)
        xs = np.asarray(silence_top1_bins_by_pair[p], dtype=np.float64)
        if xs.size:
            frac = float(np.max(xs) / 5.0)
            if frac >= 0.6:
                sparse_ok = True
    accept["computed"]["sparse_top1_bin_mass_ge_0p6_any_ldv_pair"] = bool(sparse_ok)

    accept["overall_pass"] = bool(std_ok and mic_ok and sparse_ok)

    write_json(
        out_dir / "pair_offset_estimates.json",
        {"generated": datetime.now().isoformat(), "offsets_report": offsets_report, "mic_mic_tauerr_ref": {"median_before_ms": med_before, "median_after_ms": med_after, "improve_frac": cal_improve_frac}, "acceptance": accept},
    )

    # Markdown report
    lines: list[str] = []
    lines.append("# Offsets + Sparse Multipath Verification Report\n\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n\n")
    lines.append(f"Run dir: `{out_dir.as_posix()}`\n\n")
    lines.append("## Offsets (silence-derived, global GCC peak)\n\n")
    lines.append("| pair | across-speaker mean (ms) | across-speaker std (ms) |\n")
    lines.append("| --- | ---: | ---: |\n")
    for p in pair_names:
        mean_ms = offsets_report["pairs"][p]["across_speaker_mean_ms"]
        std_ms = offsets_report["pairs"][p]["across_speaker_std_ms"]
        lines.append(f"| {p} | {mean_ms:.3f} | {std_ms:.3f} |\n")
    lines.append("\n## MIC-MIC guided tau error vs chirp reference (before/after offset subtraction)\n\n")
    lines.append(f"- median |tau_guided - tau_ref| before: {med_before:.4f} ms\n")
    lines.append(f"- median |(tau_guided - tau_offset_silence) - tau_ref| after: {med_after:.4f} ms\n")
    lines.append(f"- improvement frac: {cal_improve_frac:.3f} (>= 0.20 required by acceptance)\n\n")
    lines.append("## Acceptance (Claim 3)\n\n")
    lines.append(f"- OVERALL: {'PASS' if accept['overall_pass'] else 'FAIL'}\n")
    for k, v in sorted(accept["computed"].items()):
        lines.append(f"- {k}: {v}\n")
    lines.append("\n## Notes\n\n")
    lines.append("- If the MIC-MIC 'after' error worsens, this suggests the silence-derived mode does not represent a pure channel offset compatible with chirp reference; record as a negative result rather than tuning thresholds.\n")
    lines.append("- Per-window peak lists are stored under `per_speaker/<speaker>/windows.jsonl`.\n")

    (out_dir / "offsets_report.md").write_text("".join(lines), encoding="utf-8")
    logger.info("Wrote offsets_report.md")
    logger.info("Acceptance OVERALL: %s", "PASS" if accept["overall_pass"] else "FAIL")


if __name__ == "__main__":
    main()

