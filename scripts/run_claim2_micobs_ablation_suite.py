#!/usr/bin/env python
"""
Claim-2 mic-observation ablation suite under a fixed near-fail occlusion setting.

Purpose
-------
Answer: is MicL–MicR coherence (and/or mic PSD) already sufficient to learn the band policy,
and what marginal benefit does LDV add beyond a strong mic-only observation?

This script runs:
  - teacher (Band-OMP) once per obs_mode, with identical corruption settings,
  - DTmin training/eval once per obs_mode,
  - and writes a pooled + per-speaker comparison report.

All outputs are written under --out_dir (never repo root).
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

logger = logging.getLogger(__name__)


# Fixed near-fail config (locked)
SNR_DB = 0.0
CORRUPT_SEED = 1337
PRECLIP_GAIN = 100.0
CLIP_LIMIT = 0.99
COUPLING_MODE = "mic_only"

OCCLUSION = {
    "occlusion_enable": 1,
    "occlusion_target": "micr",
    "occlusion_kind": "lowpass",
    "occlusion_lowpass_hz": 800.0,
    "occlusion_tilt_k": 2.0,
    "occlusion_tilt_pivot_hz": 800.0,
}

OBS_MODES = ["ldv_mic", "mic_only_control", "mic_only_coh_only", "mic_only_psd_only"]


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def git_head_and_dirty() -> tuple[str | None, bool | None]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() != ""
        return head, dirty
    except Exception:
        return None, None


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


def run_cmd(cmd: list[str], *, cwd: Path) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def assert_np_equal(a: np.ndarray, b: np.ndarray, *, name: str) -> None:
    if not np.array_equal(a, b):
        diff = int(np.sum(a != b))
        raise RuntimeError(f"Teacher identity check failed for {name}: diff_count={diff}")


def summarize(vals: list[float]) -> dict[str, float]:
    x = np.asarray(vals, dtype=np.float64)
    if x.size == 0:
        return {
            "count": 0,
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "fail_rate_gt5deg": float("nan"),
        }
    return {
        "count": int(x.size),
        "median": float(np.quantile(x, 0.5)),
        "p90": float(np.quantile(x, 0.9)),
        "p95": float(np.quantile(x, 0.95)),
        "fail_rate_gt5deg": float(np.mean(x > 5.0)),
    }


def frac_improve(old: float, new: float) -> float:
    if not np.isfinite(old) or old == 0.0:
        return float("nan")
    return float((old - new) / old)


def load_test_windows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "test_windows.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing test_windows.jsonl: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty test_windows.jsonl: {path}")
    return rows


def metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = summarize([float(r["baseline"]["theta_error_ref_deg"]) for r in rows])
    student = summarize([float(r["student"]["theta_error_ref_deg"]) for r in rows])
    per_speaker: dict[str, Any] = {}
    for spk in sorted({str(r["speaker_id"]) for r in rows}):
        rs = [r for r in rows if str(r["speaker_id"]) == spk]
        per_speaker[spk] = {
            "baseline": summarize([float(r["baseline"]["theta_error_ref_deg"]) for r in rs]),
            "student": summarize([float(r["student"]["theta_error_ref_deg"]) for r in rs]),
        }
    return {"pooled": {"baseline": baseline, "student": student}, "per_speaker": per_speaker}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run mic-observation ablation suite (Claim 2)")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--truth_ref_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--speakers",
        type=str,
        nargs="+",
        default=["18-0.1V", "19-0.1V", "20-0.1V", "21-0.1V", "22-0.1V"],
    )
    ap.add_argument("--smoke", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir).expanduser().resolve()
    configure_logging(out_dir)
    write_code_state(out_dir, Path(__file__).resolve())

    data_root = Path(args.data_root).expanduser().resolve()
    truth_ref_root = Path(args.truth_ref_root).expanduser().resolve()
    speakers = list(args.speakers)

    teacher_center_overrides: list[str] = []
    student_split_overrides: list[str] = []
    if bool(int(args.smoke)):
        speakers = ["20-0.1V"]
        teacher_center_overrides = ["--center_start_sec", "100", "--center_end_sec", "110", "--center_step_sec", "1"]
        student_split_overrides = ["--train_max_center_sec", "105"]

    run_cfg = {
        "generated": datetime.now().isoformat(),
        "data_root": str(data_root),
        "truth_ref_root": str(truth_ref_root),
        "speakers": speakers,
        "smoke": bool(int(args.smoke)),
        "analysis": {
            "band_hz": [500.0, 2000.0],
            "n_bands": 64,
            "k_horizon": 6,
            "guided_radius_ms": 0.3,
            "max_lag_ms": 10.0,
        },
        "coupling_mode": COUPLING_MODE,
        "corruption": {
            "corrupt_enable": 1,
            "corrupt_snr_db": float(SNR_DB),
            "corrupt_seed": int(CORRUPT_SEED),
            "preclip_gain": float(PRECLIP_GAIN),
            "clip_limit": float(CLIP_LIMIT),
        },
        "occlusion": OCCLUSION,
        "obs_modes": OBS_MODES,
    }
    write_json(out_dir / "run_config.json", run_cfg)

    # 1) Teachers
    teacher_dirs: dict[str, Path] = {}
    for obs_mode in OBS_MODES:
        tdir = out_dir / "teacher" / obs_mode
        teacher_dirs[obs_mode] = tdir
        cmd = [
            sys.executable,
            "-u",
            "scripts/teacher_band_omp_micmic.py",
            "--data_root",
            str(data_root),
            "--truth_ref_root",
            str(truth_ref_root),
            "--out_dir",
            str(tdir),
            "--speakers",
            *speakers,
            "--obs_mode",
            obs_mode,
            "--coupling_mode",
            COUPLING_MODE,
            "--corrupt_enable",
            "1",
            "--corrupt_snr_db",
            str(float(SNR_DB)),
            "--corrupt_seed",
            str(int(CORRUPT_SEED)),
            "--preclip_gain",
            str(float(PRECLIP_GAIN)),
            "--clip_limit",
            str(float(CLIP_LIMIT)),
            "--occlusion_enable",
            str(int(OCCLUSION["occlusion_enable"])),
            "--occlusion_target",
            str(OCCLUSION["occlusion_target"]),
            "--occlusion_kind",
            str(OCCLUSION["occlusion_kind"]),
            "--occlusion_lowpass_hz",
            str(float(OCCLUSION["occlusion_lowpass_hz"])),
            "--occlusion_tilt_k",
            str(float(OCCLUSION["occlusion_tilt_k"])),
            "--occlusion_tilt_pivot_hz",
            str(float(OCCLUSION["occlusion_tilt_pivot_hz"])),
            *teacher_center_overrides,
        ]
        run_cmd(cmd, cwd=repo_root)

    # 2) Teacher identity checks
    ref_npz = teacher_dirs[OBS_MODES[0]] / "teacher_trajectories.npz"
    if not ref_npz.exists():
        raise FileNotFoundError(f"Missing reference teacher npz: {ref_npz}")
    ref = np.load(str(ref_npz), allow_pickle=False)
    ref_actions = ref["actions"]
    ref_ncl = ref["noise_center_sec_L"]
    ref_ncr = ref["noise_center_sec_R"]
    ref_forbid = ref["forbidden_mask"]

    teacher_identity: dict[str, Any] = {
        "reference": str(ref_npz),
        "checks": {},
    }
    for obs_mode in OBS_MODES[1:]:
        npz = teacher_dirs[obs_mode] / "teacher_trajectories.npz"
        d = np.load(str(npz), allow_pickle=False)
        assert_np_equal(ref_actions, d["actions"], name=f"actions ({obs_mode})")
        assert_np_equal(ref_ncl, d["noise_center_sec_L"], name=f"noise_center_sec_L ({obs_mode})")
        assert_np_equal(ref_ncr, d["noise_center_sec_R"], name=f"noise_center_sec_R ({obs_mode})")
        assert_np_equal(ref_forbid, d["forbidden_mask"], name=f"forbidden_mask ({obs_mode})")
        teacher_identity["checks"][obs_mode] = {"ok": True, "npz": str(npz)}
    write_json(out_dir / "teacher_identity.json", teacher_identity)

    # 3) Students
    student_dirs: dict[str, Path] = {}
    for obs_mode in OBS_MODES:
        sdir = out_dir / "student" / obs_mode
        student_dirs[obs_mode] = sdir
        traj = teacher_dirs[obs_mode] / "teacher_trajectories.npz"
        cmd = [
            sys.executable,
            "-u",
            "scripts/train_dtmin_from_band_trajectories.py",
            "--traj_path",
            str(traj),
            "--data_root",
            str(data_root),
            "--truth_ref_root",
            str(truth_ref_root),
            "--out_dir",
            str(sdir),
            "--use_traj_corruption",
            "1",
            *student_split_overrides,
        ]
        run_cmd(cmd, cwd=repo_root)

    # 4) Metrics + report
    metrics: dict[str, Any] = {"generated": datetime.now().isoformat(), "out_dir": str(out_dir), "variants": {}}
    for obs_mode in OBS_MODES:
        rows = load_test_windows(student_dirs[obs_mode])
        metrics["variants"][obs_mode] = {
            "student_dir": str(student_dirs[obs_mode]),
            "sha256_test_windows": sha256_file(student_dirs[obs_mode] / "test_windows.jsonl"),
            "metrics": metrics_from_rows(rows),
        }

    # Baseline near-fail precondition (use first variant)
    base_fail = float(metrics["variants"][OBS_MODES[0]]["metrics"]["pooled"]["baseline"]["fail_rate_gt5deg"])
    near_fail = bool(base_fail >= 0.40)

    def pooled_student(obs_mode: str) -> dict[str, float]:
        return metrics["variants"][obs_mode]["metrics"]["pooled"]["student"]

    def pooled_baseline(obs_mode: str) -> dict[str, float]:
        return metrics["variants"][obs_mode]["metrics"]["pooled"]["baseline"]

    deltas = {}
    for a, b in [
        ("ldv_mic", "mic_only_control"),
        ("mic_only_control", "mic_only_coh_only"),
        ("mic_only_control", "mic_only_psd_only"),
    ]:
        pa = pooled_student(a)
        pb = pooled_student(b)
        deltas[f"{a}_vs_{b}"] = {
            "p95_improvement_frac": frac_improve(float(pb["p95"]), float(pa["p95"])),
            "fail_rate_improvement_frac": frac_improve(float(pb["fail_rate_gt5deg"]), float(pa["fail_rate_gt5deg"])),
        }

    summary = {
        "generated": datetime.now().isoformat(),
        "near_fail_precondition": {"baseline_fail_rate_gt5deg": base_fail, "near_fail_ge_0p40": near_fail},
        "obs_modes": OBS_MODES,
        "deltas": deltas,
    }

    write_json(out_dir / "summary_table.json", {"summary": summary, "metrics": metrics})

    lines: list[str] = []
    lines.append("# Claim-2 Mic-Observation Ablation Report\n\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n\n")
    lines.append(f"Run dir: `{out_dir.as_posix()}`\n\n")
    lines.append("## Fixed near-fail setting\n\n")
    lines.append(f"- corruption: SNR={SNR_DB:.1f} dB in-band (500–2000), seed={CORRUPT_SEED}, preclip_gain={PRECLIP_GAIN}, clip_limit={CLIP_LIMIT}\n")
    lines.append(
        f"- occlusion: target={OCCLUSION['occlusion_target']}, kind={OCCLUSION['occlusion_kind']}, lowpass_hz={OCCLUSION['occlusion_lowpass_hz']}\n\n"
    )
    lines.append("## Teacher identity checks\n\n")
    lines.append("- actions/noise centers/forbidden mask identical across obs modes: PASS\n\n")
    lines.append("## Near-fail precondition (baseline MIC–MIC)\n\n")
    lines.append(f"- baseline fail_rate_ref(>5°): {base_fail:.3f} (>= 0.40 required) => {'PASS' if near_fail else 'FAIL'}\n\n")

    lines.append("## Pooled test metrics (vs chirp reference)\n\n")
    lines.append("| obs_mode | baseline p95 | baseline fail | student p95 | student fail |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: |\n")
    for obs_mode in OBS_MODES:
        b = pooled_baseline(obs_mode)
        s = pooled_student(obs_mode)
        lines.append(
            f"| {obs_mode} | {b['p95']:.3f} | {b['fail_rate_gt5deg']:.3f} | {s['p95']:.3f} | {s['fail_rate_gt5deg']:.3f} |\n"
        )

    lines.append("\n## Key deltas (student vs student)\n\n")
    lines.append("| comparison | p95 improvement frac | fail-rate improvement frac |\n")
    lines.append("| --- | ---: | ---: |\n")
    for k, v in deltas.items():
        lines.append(f"| {k} | {v['p95_improvement_frac']:.3f} | {v['fail_rate_improvement_frac']:.3f} |\n")

    lines.append("\n## Interpretation rules (pre-registered)\n\n")
    lines.append("- Coherence nearly sufficient if coh_only is within <=5% p95 and <=0.02 abs fail-rate of mic_only_control.\n")
    lines.append("- PSD materially helps beyond coherence if psd_only closes >=50% of the p95 gap between coh_only and control.\n")
    lines.append("- LDV adds marginal info beyond strong mic-only if ldv_mic improves vs mic_only_control by >=10% p95 and >=20% fail-rate (relative).\n")

    (out_dir / "report.md").write_text("".join(lines), encoding="utf-8")
    logger.info("Wrote report: %s", (out_dir / "report.md").as_posix())


if __name__ == "__main__":
    main()

