#!/usr/bin/env python
"""
Evaluate baseline vs student performance on truth-free "extreme" speech subsets.

Purpose
-------
We cannot (in this verification) record new wind/occlusion/saturation takes.
Instead, we define "extreme windows" using truth-free signal diagnostics computed
from the raw MIC waveforms and test whether LDV-assisted band policy remains
robust where mic-only tends to degrade.

Inputs
------
- Existing student run directory containing:
    <student_run>/test_windows.jsonl
  Each record contains baseline + student theta errors vs chirp-reference truth.

- Raw WAVs under --data_root:
    <data_root>/<speaker>/*LEFT*.wav
    <data_root>/<speaker>/*RIGHT*.wav

CLI (plan-locked)
-----------------
python -u scripts/eval_speech_extreme_subsets.py \\
  --data_root /home/sbplab/jiawei/data \\
  --student_run results/band_dtmin_student_<ts> \\
  --out_dir results/extreme_subset_eval_<YYYYMMDD_HHMMSS>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import csd, welch

logger = logging.getLogger(__name__)


# Locked constants
FS_EXPECTED = 48_000
WINDOW_SEC = 5.0

ANALYSIS_BAND_HZ = (500.0, 2000.0)
HF_BAND_HZ = (2000.0, 8000.0)

WELCH_NPERSEG = 8192
WELCH_NOVERLAP = 4096

EXTREME_PERCENT = 10.0  # bottom/top 10%
CLIP_THRESH = 0.99
CLIP_FRAC_THRESH = 1e-3

EPS = 1e-12


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


def list_mic_files(data_root: Path, speaker: str) -> tuple[Path, Path]:
    sp_dir = data_root / speaker
    micl_files = sorted(sp_dir.glob("*LEFT*.wav"))
    micr_files = sorted(sp_dir.glob("*RIGHT*.wav"))
    if not micl_files or not micr_files:
        raise FileNotFoundError(f"Missing MIC WAVs in {sp_dir} (LEFT={len(micl_files)}, RIGHT={len(micr_files)})")
    return micl_files[0], micr_files[0]


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


def coherence_gamma2(x: np.ndarray, y: np.ndarray, *, fs: int) -> tuple[np.ndarray, np.ndarray]:
    f, sxx = welch(x, fs=fs, nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP)
    _, syy = welch(y, fs=fs, nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP)
    _, sxy = csd(x, y, fs=fs, nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP)
    gamma2 = (np.abs(sxy) ** 2) / (np.maximum(sxx * syy, EPS))
    gamma2 = np.clip(gamma2, 0.0, 1.0)
    return f.astype(np.float64, copy=False), gamma2.astype(np.float64, copy=False)


def band_integral_psd(x: np.ndarray, *, fs: int, band_hz: tuple[float, float]) -> float:
    f, pxx = welch(x, fs=fs, nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP)
    f0, f1 = float(band_hz[0]), float(band_hz[1])
    m = (f >= f0) & (f <= f1)
    if not np.any(m):
        return 0.0
    return float(np.sum(pxx[m].astype(np.float64, copy=False)))


def quantile(vals: np.ndarray, q: float) -> float:
    return float(np.quantile(vals.astype(np.float64, copy=False), q))


def summarize_errors(errs: list[float]) -> dict[str, float]:
    x = np.asarray(errs, dtype=np.float64)
    if x.size == 0:
        return {"count": 0, "median": float("nan"), "p90": float("nan"), "p95": float("nan"), "fail_rate_gt5deg": float("nan")}
    return {
        "count": int(x.size),
        "median": quantile(x, 0.5),
        "p90": quantile(x, 0.9),
        "p95": quantile(x, 0.95),
        "fail_rate_gt5deg": float(np.mean(x > 5.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--student_run", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    student_run = Path(args.student_run).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    configure_logging(out_dir)
    write_code_state(out_dir, Path(__file__).resolve())

    tw_path = student_run / "test_windows.jsonl"
    if not tw_path.exists():
        raise FileNotFoundError(f"Missing test_windows.jsonl: {tw_path}")

    rows: list[dict[str, Any]] = []
    with tw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"Empty test_windows.jsonl: {tw_path}")

    speakers = sorted({str(r["speaker_id"]) for r in rows})
    logger.info("Loaded %d test windows across %d speakers: %s", len(rows), len(speakers), " ".join(speakers))

    # Load MIC WAVs once per speaker
    mic_cache: dict[str, tuple[int, np.ndarray, np.ndarray, Path, Path]] = {}
    wav_files_used: list[Path] = []
    for sp in speakers:
        micl_path, micr_path = list_mic_files(data_root, sp)
        sr_l, micl = load_wav_mono(micl_path)
        sr_r, micr = load_wav_mono(micr_path)
        if sr_l != FS_EXPECTED or sr_r != FS_EXPECTED:
            raise ValueError(f"fs mismatch for {sp}: micl={sr_l}, micr={sr_r}, expected={FS_EXPECTED}")
        if micl.shape != micr.shape:
            raise ValueError(f"Length mismatch for {sp}: micl={micl.shape}, micr={micr.shape}")
        mic_cache[sp] = (FS_EXPECTED, micl, micr, micl_path, micr_path)
        wav_files_used.extend([micl_path, micr_path])

    manifest = {
        "generated": datetime.now().isoformat(),
        "data_root": str(data_root),
        "student_run": str(student_run),
        "wav_files": [{"path": str(p), "sha256": sha256_file(p)} for p in sorted(set(wav_files_used))],
        "dataset_fingerprint_sha256": dataset_fingerprint(sorted(set(wav_files_used)), root=data_root),
    }
    write_json(out_dir / "manifest.json", manifest)

    run_config = {
        "generated": datetime.now().isoformat(),
        "data_root": str(data_root),
        "student_run": str(student_run),
        "fs_expected": FS_EXPECTED,
        "window_sec": WINDOW_SEC,
        "analysis_band_hz": list(ANALYSIS_BAND_HZ),
        "hf_band_hz": list(HF_BAND_HZ),
        "welch": {"nperseg": WELCH_NPERSEG, "noverlap": WELCH_NOVERLAP},
        "extreme_percent": EXTREME_PERCENT,
        "clip": {"clip_thresh": CLIP_THRESH, "clip_frac_thresh": CLIP_FRAC_THRESH},
        "subsets": ["LOW_COH", "HIGH_HF_IMB", "CLIPPED", "LOW_RMS"],
    }
    write_json(out_dir / "run_config.json", run_config)

    # Compute diagnostics per window
    diag_rows: list[dict[str, Any]] = []
    coh_vals: list[float] = []
    hfimb_vals: list[float] = []
    clip_vals: list[float] = []
    rmsdb_vals: list[float] = []

    for r in rows:
        sp = str(r["speaker_id"])
        center_sec = float(r["center_sec"])
        fs, micl_all, micr_all, _, _ = mic_cache[sp]
        seg_l = extract_centered_window(micl_all, fs=fs, center_sec=center_sec, window_sec=WINDOW_SEC)
        seg_r = extract_centered_window(micr_all, fs=fs, center_sec=center_sec, window_sec=WINDOW_SEC)

        # 1) median coherence in 500–2000
        f, g2 = coherence_gamma2(seg_l, seg_r, fs=fs)
        m = (f >= ANALYSIS_BAND_HZ[0]) & (f <= ANALYSIS_BAND_HZ[1])
        coh_med = float(np.median(g2[m])) if np.any(m) else float("nan")

        # 2) HF imbalance 2–8 kHz (PSD integral)
        e_l = band_integral_psd(seg_l, fs=fs, band_hz=HF_BAND_HZ)
        e_r = band_integral_psd(seg_r, fs=fs, band_hz=HF_BAND_HZ)
        hf_imb_db = float(10.0 * np.log10((e_l + EPS) / (e_r + EPS)))

        # 3) clipping fraction
        clip_l = float(np.mean(np.abs(seg_l) >= CLIP_THRESH))
        clip_r = float(np.mean(np.abs(seg_r) >= CLIP_THRESH))
        clip_max = max(clip_l, clip_r)

        # 4) RMS(MicL) in dB
        rms_l = rms(seg_l)
        rms_db = float(20.0 * np.log10(rms_l + EPS))

        diag = {
            "speaker_id": sp,
            "center_sec": center_sec,
            "mic_coh_median_500_2000": coh_med,
            "hf_imbalance_db": hf_imb_db,
            "clip_frac_max": clip_max,
            "speech_rms_db": rms_db,
        }
        diag_rows.append(diag)
        coh_vals.append(coh_med)
        hfimb_vals.append(abs(hf_imb_db))
        clip_vals.append(clip_max)
        rmsdb_vals.append(rms_db)

    # Thresholds computed on test windows only (truth-free)
    coh_arr = np.asarray(coh_vals, dtype=np.float64)
    hfimb_arr = np.asarray(hfimb_vals, dtype=np.float64)
    clip_arr = np.asarray(clip_vals, dtype=np.float64)
    rmsdb_arr = np.asarray(rmsdb_vals, dtype=np.float64)

    thr_low_coh = float(np.quantile(coh_arr, EXTREME_PERCENT / 100.0))
    thr_high_hfimb = float(np.quantile(hfimb_arr, 1.0 - EXTREME_PERCENT / 100.0))
    thr_low_rms = float(np.quantile(rmsdb_arr, EXTREME_PERCENT / 100.0))

    thresholds = {
        "LOW_COH": {"mic_coh_median_500_2000_le": thr_low_coh, "percent": EXTREME_PERCENT},
        "HIGH_HF_IMB": {"abs_hf_imbalance_db_ge": thr_high_hfimb, "percent": EXTREME_PERCENT},
        "CLIPPED": {"clip_frac_max_ge": CLIP_FRAC_THRESH},
        "LOW_RMS": {"speech_rms_db_le": thr_low_rms, "percent": EXTREME_PERCENT},
    }

    # Join diagnostics with baseline/student errors
    diag_map = {(d["speaker_id"], d["center_sec"]): d for d in diag_rows}
    joined: list[dict[str, Any]] = []
    for r in rows:
        key = (str(r["speaker_id"]), float(r["center_sec"]))
        d = diag_map[key]
        joined.append(
            {
                **d,
                "baseline": r["baseline"],
                "student": r["student"],
                "truth_reference": r.get("truth_reference", {}),
            }
        )

    # Define subset membership
    def in_subset(j: dict[str, Any], name: str) -> bool:
        if name == "LOW_COH":
            return float(j["mic_coh_median_500_2000"]) <= thr_low_coh
        if name == "HIGH_HF_IMB":
            return abs(float(j["hf_imbalance_db"])) >= thr_high_hfimb
        if name == "CLIPPED":
            return float(j["clip_frac_max"]) >= CLIP_FRAC_THRESH
        if name == "LOW_RMS":
            return float(j["speech_rms_db"]) <= thr_low_rms
        raise ValueError(name)

    subset_names = ["LOW_COH", "HIGH_HF_IMB", "CLIPPED", "LOW_RMS"]
    metrics: dict[str, Any] = {"generated": datetime.now().isoformat(), "run_dir": str(out_dir), "thresholds": thresholds, "subsets": {}}

    # Pooled + per-speaker metrics
    for name in subset_names:
        subset = [j for j in joined if in_subset(j, name)]
        by_sp: dict[str, list[dict[str, Any]]] = {}
        for j in subset:
            by_sp.setdefault(str(j["speaker_id"]), []).append(j)

        pooled_base = [float(j["baseline"]["theta_error_ref_deg"]) for j in subset]
        pooled_student = [float(j["student"]["theta_error_ref_deg"]) for j in subset]

        pooled = {
            "baseline": summarize_errors(pooled_base),
            "student": summarize_errors(pooled_student),
        }

        per_speaker: dict[str, Any] = {}
        for sp, items in sorted(by_sp.items()):
            base = [float(j["baseline"]["theta_error_ref_deg"]) for j in items]
            stud = [float(j["student"]["theta_error_ref_deg"]) for j in items]
            per_speaker[sp] = {"baseline": summarize_errors(base), "student": summarize_errors(stud)}

        # Improvement fractions
        def frac_improve(old: float, new: float) -> float:
            if not np.isfinite(old) or old == 0.0:
                return float("nan")
            return float((old - new) / old)

        p95_old = float(pooled["baseline"]["p95"])
        p95_new = float(pooled["student"]["p95"])
        fail_old = float(pooled["baseline"]["fail_rate_gt5deg"])
        fail_new = float(pooled["student"]["fail_rate_gt5deg"])

        improvements = {
            "p95_improvement_frac": frac_improve(p95_old, p95_new),
            "fail_rate_improvement_frac": frac_improve(fail_old, fail_new),
        }

        metrics["subsets"][name] = {"pooled": pooled, "per_speaker": per_speaker, "improvements": improvements}

    write_json(out_dir / "subset_metrics.json", metrics)

    # Acceptance for Claim 1
    accept_count = 0
    near_fail_met = False
    subset_accept: dict[str, Any] = {}
    for name in subset_names:
        imp = metrics["subsets"][name]["improvements"]
        pooled = metrics["subsets"][name]["pooled"]
        p95_ok = float(imp["p95_improvement_frac"]) >= 0.15
        fail_ok = float(imp["fail_rate_improvement_frac"]) >= 0.20
        if p95_ok and fail_ok:
            accept_count += 1
        base_fail = float(pooled["baseline"]["fail_rate_gt5deg"])
        stud_fail = float(pooled["student"]["fail_rate_gt5deg"])
        if base_fail >= 0.40 and stud_fail <= 0.10:
            near_fail_met = True
        subset_accept[name] = {"p95_ok": bool(p95_ok), "fail_ok": bool(fail_ok), "count": int(pooled["baseline"]["count"])}

    overall_pass = bool((accept_count >= 2) and near_fail_met)

    acceptance = {
        "generated": datetime.now().isoformat(),
        "criteria": {
            "need_subsets_ge_2": 2,
            "p95_improvement_frac_ge": 0.15,
            "fail_rate_improvement_frac_ge": 0.20,
            "need_near_fail_subset": True,
            "near_fail_baseline_fail_rate_ge": 0.40,
            "near_fail_student_fail_rate_le": 0.10,
        },
        "computed": {"n_subsets_meeting_p95_and_fail": int(accept_count), "near_fail_subset_met": bool(near_fail_met)},
        "per_subset": subset_accept,
        "overall_pass": bool(overall_pass),
    }
    write_json(out_dir / "summary.json", {"generated": datetime.now().isoformat(), "run_dir": str(out_dir), "acceptance": acceptance})

    # Markdown report
    lines: list[str] = []
    lines.append("# Extreme Subset Evaluation Report\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append(f"Run dir: {out_dir.as_posix()}\n")
    lines.append("## Subset definitions (truth-free)\n")
    lines.append(f"- LOW_COH: bottom {EXTREME_PERCENT:.0f}% by median MSC(MicL,MicR) in 500–2000 Hz\n")
    lines.append(f"- HIGH_HF_IMB: top {EXTREME_PERCENT:.0f}% by |HF imbalance| in 2–8 kHz\n")
    lines.append(f"- CLIPPED: clip_frac_max >= {CLIP_FRAC_THRESH}\n")
    lines.append(f"- LOW_RMS: bottom {EXTREME_PERCENT:.0f}% by RMS(MicL) dB\n")
    lines.append("\n## Acceptance (Claim 1)\n")
    lines.append(f"- subsets meeting (p95>=15% AND fail>=20%): {accept_count}\n")
    lines.append(f"- near-fail subset met (baseline fail>=0.40 and student<=0.10): {near_fail_met}\n")
    lines.append(f"- OVERALL: {'PASS' if overall_pass else 'FAIL'}\n")
    lines.append("\n## Pooled metrics (vs chirp reference)\n")
    for name in subset_names:
        pooled = metrics['subsets'][name]['pooled']
        imp = metrics['subsets'][name]['improvements']
        lines.append(f"\n### {name} (n={pooled['baseline']['count']})\n")
        lines.append("| Method | median | p90 | p95 | fail_rate(>5°) |\n")
        lines.append("| --- | ---: | ---: | ---: | ---: |\n")
        for method in ["baseline", "student"]:
            s = pooled[method]
            lines.append(f"| {method} | {s['median']:.3f} | {s['p90']:.3f} | {s['p95']:.3f} | {s['fail_rate_gt5deg']:.3f} |\n")
        lines.append(f"- p95 improvement frac: {imp['p95_improvement_frac']:.3f}\n")
        lines.append(f"- fail-rate improvement frac: {imp['fail_rate_improvement_frac']:.3f}\n")

    (out_dir / "subset_report.md").write_text("".join(lines), encoding="utf-8")
    logger.info("Wrote subset_report.md and subset_metrics.json")
    logger.info("Acceptance OVERALL: %s", "PASS" if overall_pass else "FAIL")


if __name__ == "__main__":
    main()

