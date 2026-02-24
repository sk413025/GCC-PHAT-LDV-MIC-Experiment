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

# Report thresholds (must be physically reachable under the guided radius)
THETA_FAIL_DEG = 4.0
TAU_ERR_FAIL_MS = 0.25
PSR_GOOD_DB = 3.0

# Quadratic sub-sample refinement is clamped in both teacher and student
SHIFT_CLAMP_SAMPLES = 0.5
FS_HZ = 48_000
SHIFT_CLAMP_MS = 1000.0 * SHIFT_CLAMP_SAMPLES / float(FS_HZ)


def tau_to_theta_deg(tau_sec: float, *, c: float = 343.0, d: float = 1.4) -> float:
    sin_theta = float(np.clip(float(tau_sec) * float(c) / float(d), -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_theta)))


def max_theta_error_possible_deg(*, tau_ref_ms: float, theta_ref_deg: float, guided_radius_ms: float) -> float:
    tau_lo_sec = (float(tau_ref_ms) - float(guided_radius_ms) - float(SHIFT_CLAMP_MS)) / 1000.0
    tau_hi_sec = (float(tau_ref_ms) + float(guided_radius_ms) + float(SHIFT_CLAMP_MS)) / 1000.0
    theta_lo = tau_to_theta_deg(tau_lo_sec)
    theta_hi = tau_to_theta_deg(tau_hi_sec)
    ref = float(theta_ref_deg)
    return float(max(abs(theta_lo - ref), abs(theta_hi - ref)))


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


def teacher_artifacts_exist(teacher_dir: Path, speakers: list[str]) -> bool:
    npz = teacher_dir / "teacher_trajectories.npz"
    per_speaker = teacher_dir / "per_speaker"
    if not npz.exists() or not per_speaker.exists():
        return False
    for spk in speakers:
        if not (per_speaker / spk / "windows.jsonl").exists():
            return False
    return True


def student_artifacts_exist(student_dir: Path) -> bool:
    # Require summary.json as a completion marker (test_windows/model can exist mid-run).
    return (
        (student_dir / "summary.json").exists()
        and (student_dir / "test_windows.jsonl").exists()
        and (student_dir / "model_dtmin_band_policy_k6.npz").exists()
    )


def assert_np_equal(a: np.ndarray, b: np.ndarray, *, name: str) -> None:
    if not np.array_equal(a, b):
        diff = int(np.sum(a != b))
        raise RuntimeError(f"Teacher identity check failed for {name}: diff_count={diff}")


def summarize_theta_err_deg(vals: list[float]) -> dict[str, float]:
    x = np.asarray(vals, dtype=np.float64)
    if x.size == 0:
        return {
            "count": 0,
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "fail_rate_gt4deg": float("nan"),
        }
    return {
        "count": int(x.size),
        "median": float(np.quantile(x, 0.5)),
        "p90": float(np.quantile(x, 0.9)),
        "p95": float(np.quantile(x, 0.95)),
        "fail_rate_gt4deg": float(np.mean(x > float(THETA_FAIL_DEG))),
    }


def summarize_tau_err_ms(vals: list[float]) -> dict[str, float]:
    x = np.asarray(vals, dtype=np.float64)
    if x.size == 0:
        return {
            "count": 0,
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "fail_rate_gt0p25ms": float("nan"),
        }
    return {
        "count": int(x.size),
        "median": float(np.quantile(x, 0.5)),
        "p90": float(np.quantile(x, 0.9)),
        "p95": float(np.quantile(x, 0.95)),
        "fail_rate_gt0p25ms": float(np.mean(x > float(TAU_ERR_FAIL_MS))),
    }


def summarize_psr_db(vals: list[float]) -> dict[str, float]:
    x = np.asarray(vals, dtype=np.float64)
    if x.size == 0:
        return {
            "count": 0,
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "frac_gt3db": float("nan"),
            "frac_gt0db": float("nan"),
        }
    return {
        "count": int(x.size),
        "median": float(np.quantile(x, 0.5)),
        "p90": float(np.quantile(x, 0.9)),
        "p95": float(np.quantile(x, 0.95)),
        "frac_gt3db": float(np.mean(x > float(PSR_GOOD_DB))),
        "frac_gt0db": float(np.mean(x > 0.0)),
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
    baseline_theta = summarize_theta_err_deg([float(r["baseline"]["theta_error_ref_deg"]) for r in rows])
    student_theta = summarize_theta_err_deg([float(r["student"]["theta_error_ref_deg"]) for r in rows])

    baseline_tau_err = summarize_tau_err_ms(
        [abs(float(r["baseline"]["tau_ms"]) - float(r["truth_reference"]["tau_ref_ms"])) for r in rows]
    )
    student_tau_err = summarize_tau_err_ms(
        [abs(float(r["student"]["tau_ms"]) - float(r["truth_reference"]["tau_ref_ms"])) for r in rows]
    )

    baseline_psr = summarize_psr_db([float(r["baseline"]["psr_db"]) for r in rows])
    student_psr = summarize_psr_db([float(r["student"]["psr_db"]) for r in rows])
    per_speaker: dict[str, Any] = {}
    for spk in sorted({str(r["speaker_id"]) for r in rows}):
        rs = [r for r in rows if str(r["speaker_id"]) == spk]
        base_tau_err_spk = [abs(float(r["baseline"]["tau_ms"]) - float(r["truth_reference"]["tau_ref_ms"])) for r in rs]
        stud_tau_err_spk = [abs(float(r["student"]["tau_ms"]) - float(r["truth_reference"]["tau_ref_ms"])) for r in rs]
        per_speaker[spk] = {
            "baseline": {
                "theta_error_ref_deg": summarize_theta_err_deg([float(r["baseline"]["theta_error_ref_deg"]) for r in rs]),
                "tau_error_ref_ms": summarize_tau_err_ms(base_tau_err_spk),
                "psr_db": summarize_psr_db([float(r["baseline"]["psr_db"]) for r in rs]),
            },
            "student": {
                "theta_error_ref_deg": summarize_theta_err_deg([float(r["student"]["theta_error_ref_deg"]) for r in rs]),
                "tau_error_ref_ms": summarize_tau_err_ms(stud_tau_err_spk),
                "psr_db": summarize_psr_db([float(r["student"]["psr_db"]) for r in rs]),
            },
        }
    return {
        "pooled": {
            "baseline": {
                "theta_error_ref_deg": baseline_theta,
                "tau_error_ref_ms": baseline_tau_err,
                "psr_db": baseline_psr,
            },
            "student": {
                "theta_error_ref_deg": student_theta,
                "tau_error_ref_ms": student_tau_err,
                "psr_db": student_psr,
            },
        },
        "per_speaker": per_speaker,
    }


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
    ap.add_argument(
        "--coupling_hard_forbid_enable",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 0, do not hard-forbid any band based on silence coupling (keep coupling as a soft penalty only). Default: 1.",
    )
    ap.add_argument(
        "--dynamic_coh_gate_enable",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, apply a per-window mic coherence gate (teacher forbidden mask includes coherence floor). Default: 0.",
    )
    ap.add_argument(
        "--dynamic_coh_min",
        type=float,
        default=0.05,
        help="Dynamic mic coherence floor for per-window gating. Default: 0.05.",
    )
    ap.add_argument(
        "--tau_ref_gate_enable",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, enable a tau_ref-support gate in the teacher forbidden mask (per-band guided-peak ratio). Default: 0.",
    )
    ap.add_argument(
        "--tau_ref_gate_ratio_min",
        type=float,
        default=0.60,
        help="Minimum guided-peak ratio to keep a band when --tau_ref_gate_enable=1. Default: 0.60.",
    )
    ap.add_argument(
        "--cm_interf_enable",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable common-mode coherent interference injected identically into MicL/MicR (mic-only). Default: 0.",
    )
    ap.add_argument(
        "--cm_interf_snr_db",
        type=float,
        default=None,
        help="Target in-band SNR (dB) for common-mode interference (required if --cm_interf_enable=1).",
    )
    ap.add_argument(
        "--cm_interf_seed",
        type=int,
        default=1337,
        help="Seed for deterministic common-mode interference noise selection. Default: 1337.",
    )
    ap.add_argument(
        "--cd_interf_enable",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable coherent interference with a fixed relative delay between mics (mic-only). Default: 0.",
    )
    ap.add_argument(
        "--cd_interf_snr_db",
        type=float,
        default=None,
        help="Target in-band SNR (dB) for delayed coherent interference (required if --cd_interf_enable=1).",
    )
    ap.add_argument(
        "--cd_interf_delay_ms",
        type=float,
        default=0.0,
        help="Relative delay (ms) applied to the interference on --cd_interf_target (integer-sample, zero-padded).",
    )
    ap.add_argument(
        "--cd_interf_target",
        type=str,
        default="micr",
        choices=["micl", "micr"],
        help="Which mic receives the delayed interference copy. Default: micr.",
    )
    ap.add_argument(
        "--cd_interf_seed",
        type=int,
        default=1337,
        help="Seed for deterministic delayed-interference noise selection. Default: 1337.",
    )
    ap.add_argument("--include_stop_action", type=int, default=0, choices=[0, 1])
    ap.add_argument("--stateful_obs_enable", type=int, default=0, choices=[0, 1])
    ap.add_argument(
        "--stop_target_frac_add",
        type=float,
        default=0.0,
        help="Passed to train_dtmin_from_band_trajectories.py. If STOP is enabled, bias STOP slightly earlier by increasing the target STOP rate by this additive amount.",
    )
    ap.add_argument(
        "--inference_psr_stop_enable",
        type=int,
        default=0,
        choices=[0, 1],
        help="Passed to train_dtmin_from_band_trajectories.py. If 1, apply inference-time PSR-monotone stopping for student-selected bands.",
    )
    ap.add_argument(
        "--inference_psr_stop_min_gain_db",
        type=float,
        default=0.01,
        help="Passed to train_dtmin_from_band_trajectories.py. Minimum guided-PSR improvement (dB) required to accept adding the next band when PSR stop is enabled.",
    )
    ap.add_argument(
        "--inference_psr_stop_mode",
        type=str,
        default="monotone",
        choices=["monotone", "best_prefix"],
        help="Passed to train_dtmin_from_band_trajectories.py. PSR-stop mode when enabled. Default: monotone.",
    )
    ap.add_argument(
        "--ldv_scale_mode",
        type=str,
        default="fixed",
        choices=["fixed", "coh_gate", "tau_ref_ratio_gate"],
        help="Passed to teacher_band_omp_micmic.py. LDV term scaling mode for obs_mode that include LDV. Default: fixed.",
    )
    ap.add_argument(
        "--ldv_scale_fixed",
        type=float,
        default=1.0,
        help="Passed to teacher_band_omp_micmic.py. Fixed LDV term scale when --ldv_scale_mode=fixed. Default: 1.0.",
    )
    ap.add_argument(
        "--ldv_scale_coh_lo",
        type=float,
        default=0.05,
        help="Passed to teacher_band_omp_micmic.py. Lower coherence bound for --ldv_scale_mode=coh_gate. Default: 0.05.",
    )
    ap.add_argument(
        "--ldv_scale_coh_hi",
        type=float,
        default=0.20,
        help="Passed to teacher_band_omp_micmic.py. Upper coherence bound for --ldv_scale_mode=coh_gate. Default: 0.20.",
    )
    ap.add_argument(
        "--ldv_scale_ratio_lo",
        type=float,
        default=0.40,
        help="Passed to teacher_band_omp_micmic.py. Lower guided/global peak ratio for --ldv_scale_mode=tau_ref_ratio_gate. Default: 0.40.",
    )
    ap.add_argument(
        "--ldv_scale_ratio_hi",
        type=float,
        default=0.90,
        help="Passed to teacher_band_omp_micmic.py. Upper guided/global peak ratio for --ldv_scale_mode=tau_ref_ratio_gate. Default: 0.90.",
    )
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

    coupling_hard_forbid_enable = int(args.coupling_hard_forbid_enable)
    dynamic_coh_gate_enable = int(args.dynamic_coh_gate_enable)
    dynamic_coh_min = float(args.dynamic_coh_min)
    tau_ref_gate_enable = int(args.tau_ref_gate_enable)
    tau_ref_gate_ratio_min = float(args.tau_ref_gate_ratio_min)
    cm_interf_enable = int(args.cm_interf_enable)
    cm_interf_snr_db = args.cm_interf_snr_db
    cm_interf_seed = int(args.cm_interf_seed)
    cd_interf_enable = int(args.cd_interf_enable)
    cd_interf_snr_db = args.cd_interf_snr_db
    cd_interf_delay_ms = float(args.cd_interf_delay_ms)
    cd_interf_target = str(args.cd_interf_target)
    cd_interf_seed = int(args.cd_interf_seed)
    include_stop_action = bool(int(args.include_stop_action))
    stateful_obs_enable = bool(int(args.stateful_obs_enable))
    stop_target_frac_add = float(args.stop_target_frac_add)
    inference_psr_stop_enable = int(args.inference_psr_stop_enable)
    inference_psr_stop_min_gain_db = float(args.inference_psr_stop_min_gain_db)
    inference_psr_stop_mode = str(args.inference_psr_stop_mode)
    ldv_scale_mode = str(args.ldv_scale_mode)
    ldv_scale_fixed = float(args.ldv_scale_fixed)
    ldv_scale_coh_lo = float(args.ldv_scale_coh_lo)
    ldv_scale_coh_hi = float(args.ldv_scale_coh_hi)
    ldv_scale_ratio_lo = float(args.ldv_scale_ratio_lo)
    ldv_scale_ratio_hi = float(args.ldv_scale_ratio_hi)

    if bool(cm_interf_enable) and cm_interf_snr_db is None:
        raise ValueError("--cm_interf_snr_db is required when --cm_interf_enable=1")
    if bool(cd_interf_enable) and cd_interf_snr_db is None:
        raise ValueError("--cd_interf_snr_db is required when --cd_interf_enable=1")

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
        "trajectories": {
            "include_stop_action": bool(include_stop_action),
            "stateful_obs_enable": bool(stateful_obs_enable),
            "stop_action_id": 64,
            "n_actions": 65 if bool(include_stop_action) else 64,
            "stop_target_frac_add": float(stop_target_frac_add),
            "inference_psr_stop_enable": bool(int(inference_psr_stop_enable)),
            "inference_psr_stop_min_gain_db": float(inference_psr_stop_min_gain_db),
            "inference_psr_stop_mode": str(inference_psr_stop_mode),
        },
        "ldv_term_scaling": {
            "mode": str(ldv_scale_mode),
            "fixed": float(ldv_scale_fixed),
            "coh_lo": float(ldv_scale_coh_lo),
            "coh_hi": float(ldv_scale_coh_hi),
            "ratio_lo": float(ldv_scale_ratio_lo),
            "ratio_hi": float(ldv_scale_ratio_hi),
        },
        "coupling_mode": COUPLING_MODE,
        "coupling_hard_forbid_enable": bool(coupling_hard_forbid_enable),
        "dynamic_coh_gate_enable": bool(dynamic_coh_gate_enable),
        "dynamic_coh_min": float(dynamic_coh_min),
        "tau_ref_gate_enable": bool(tau_ref_gate_enable),
        "tau_ref_gate_ratio_min": float(tau_ref_gate_ratio_min),
        "common_mode_interference": {
            "enabled": bool(cm_interf_enable),
            "snr_db": None if cm_interf_snr_db is None else float(cm_interf_snr_db),
            "seed": int(cm_interf_seed),
            "band_hz": [500.0, 2000.0],
        },
        "delayed_coherent_interference": {
            "enabled": bool(cd_interf_enable),
            "snr_db": None if cd_interf_snr_db is None else float(cd_interf_snr_db),
            "delay_ms": float(cd_interf_delay_ms),
            "target_delayed": str(cd_interf_target),
            "seed": int(cd_interf_seed),
            "band_hz": [500.0, 2000.0],
        },
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
        if teacher_artifacts_exist(tdir, speakers):
            logger.info("Teacher artifacts exist for obs_mode=%s; skipping teacher run.", obs_mode)
        else:
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
                "--include_stop_action",
                str(int(include_stop_action)),
                "--stateful_obs_enable",
                str(int(stateful_obs_enable)),
                "--ldv_scale_mode",
                str(ldv_scale_mode),
                "--ldv_scale_fixed",
                str(float(ldv_scale_fixed)),
                "--ldv_scale_coh_lo",
                str(float(ldv_scale_coh_lo)),
                "--ldv_scale_coh_hi",
                str(float(ldv_scale_coh_hi)),
                "--ldv_scale_ratio_lo",
                str(float(ldv_scale_ratio_lo)),
                "--ldv_scale_ratio_hi",
                str(float(ldv_scale_ratio_hi)),
                "--coupling_mode",
                COUPLING_MODE,
                "--coupling_hard_forbid_enable",
                str(int(coupling_hard_forbid_enable)),
                "--dynamic_coh_gate_enable",
                str(int(dynamic_coh_gate_enable)),
                "--dynamic_coh_min",
                str(float(dynamic_coh_min)),
                "--tau_ref_gate_enable",
                str(int(tau_ref_gate_enable)),
                "--tau_ref_gate_ratio_min",
                str(float(tau_ref_gate_ratio_min)),
                "--cm_interf_enable",
                str(int(cm_interf_enable)),
                "--cm_interf_snr_db",
                str(float(cm_interf_snr_db)) if cm_interf_snr_db is not None else "0.0",
                "--cm_interf_seed",
                str(int(cm_interf_seed)),
                "--cd_interf_enable",
                str(int(cd_interf_enable)),
                "--cd_interf_snr_db",
                str(float(cd_interf_snr_db)) if cd_interf_snr_db is not None else "0.0",
                "--cd_interf_delay_ms",
                str(float(cd_interf_delay_ms)),
                "--cd_interf_target",
                str(cd_interf_target),
                "--cd_interf_seed",
                str(int(cd_interf_seed)),
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
    ref_cm = ref["cm_noise_center_sec"] if "cm_noise_center_sec" in ref else None
    ref_cd = ref["cd_noise_center_sec"] if "cd_noise_center_sec" in ref else None
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
        if ref_cm is not None:
            assert_np_equal(ref_cm, d["cm_noise_center_sec"], name=f"cm_noise_center_sec ({obs_mode})")
        if ref_cd is not None:
            assert_np_equal(ref_cd, d["cd_noise_center_sec"], name=f"cd_noise_center_sec ({obs_mode})")
        assert_np_equal(ref_forbid, d["forbidden_mask"], name=f"forbidden_mask ({obs_mode})")
        teacher_identity["checks"][obs_mode] = {"ok": True, "npz": str(npz)}
    write_json(out_dir / "teacher_identity.json", teacher_identity)

    # 3) Students
    student_dirs: dict[str, Path] = {}
    for obs_mode in OBS_MODES:
        sdir = out_dir / "student" / obs_mode
        student_dirs[obs_mode] = sdir
        traj = teacher_dirs[obs_mode] / "teacher_trajectories.npz"
        if student_artifacts_exist(sdir):
            logger.info("Student artifacts exist for obs_mode=%s; skipping student run.", obs_mode)
        else:
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
                "--stop_target_frac_add",
                str(stop_target_frac_add),
                "--inference_psr_stop_enable",
                str(int(inference_psr_stop_enable)),
                "--inference_psr_stop_min_gain_db",
                str(float(inference_psr_stop_min_gain_db)),
                "--inference_psr_stop_mode",
                str(inference_psr_stop_mode),
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

    # Guardrail: ensure the theta-failure threshold is physically reachable given the guided radius.
    # We use the first variant's test windows as reference for tau_ref/theta_ref per speaker.
    guided_radius_ms = float(run_cfg["analysis"]["guided_radius_ms"])
    ref_rows = load_test_windows(student_dirs[OBS_MODES[0]])
    max_err_by_speaker: dict[str, float] = {}
    for spk in sorted({str(r["speaker_id"]) for r in ref_rows}):
        rs = [r for r in ref_rows if str(r["speaker_id"]) == spk]
        tau_ref_ms = float(rs[0]["truth_reference"]["tau_ref_ms"])
        theta_ref_deg = float(rs[0]["truth_reference"]["theta_ref_deg"])
        max_err_by_speaker[spk] = max_theta_error_possible_deg(
            tau_ref_ms=tau_ref_ms,
            theta_ref_deg=theta_ref_deg,
            guided_radius_ms=guided_radius_ms,
        )
    min_max_err = float(min(max_err_by_speaker.values())) if max_err_by_speaker else float("nan")
    if not np.isfinite(min_max_err):
        raise ValueError("Non-finite max theta-error bound (check tau_ref inputs)")
    if float(THETA_FAIL_DEG) > min_max_err + 1e-6:
        raise ValueError(
            f"THETA_FAIL_DEG={THETA_FAIL_DEG:.3f} is unreachable under guided_radius_ms={guided_radius_ms:.3f} "
            f"(min speaker max_err={min_max_err:.3f}). Lower THETA_FAIL_DEG or increase guided_radius_ms."
        )

    # Baseline near-fail precondition (use first variant)
    pooled_base_theta = metrics["variants"][OBS_MODES[0]]["metrics"]["pooled"]["baseline"]["theta_error_ref_deg"]
    pooled_base_psr = metrics["variants"][OBS_MODES[0]]["metrics"]["pooled"]["baseline"]["psr_db"]
    base_fail_theta = float(pooled_base_theta["fail_rate_gt4deg"])
    base_frac_psr_good = float(pooled_base_psr["frac_gt3db"])
    near_fail = bool((base_fail_theta >= 0.40) and (base_frac_psr_good <= 0.10))

    def pooled_student_metric(obs_mode: str, metric: str) -> dict[str, float]:
        return metrics["variants"][obs_mode]["metrics"]["pooled"]["student"][metric]

    def pooled_baseline_metric(obs_mode: str, metric: str) -> dict[str, float]:
        return metrics["variants"][obs_mode]["metrics"]["pooled"]["baseline"][metric]

    deltas = {}
    for a, b in [
        ("ldv_mic", "mic_only_control"),
        ("mic_only_control", "mic_only_coh_only"),
        ("mic_only_control", "mic_only_psd_only"),
    ]:
        pa = pooled_student_metric(a, "theta_error_ref_deg")
        pb = pooled_student_metric(b, "theta_error_ref_deg")
        deltas[f"{a}_vs_{b}"] = {
            "p95_improvement_frac": frac_improve(float(pb["p95"]), float(pa["p95"])),
            "fail_rate_improvement_frac": frac_improve(float(pb["fail_rate_gt4deg"]), float(pa["fail_rate_gt4deg"])),
        }

    summary = {
        "generated": datetime.now().isoformat(),
        "thresholds": {
            "theta_fail_deg": float(THETA_FAIL_DEG),
            "tau_err_fail_ms": float(TAU_ERR_FAIL_MS),
            "psr_good_db": float(PSR_GOOD_DB),
            "shift_clamp_ms": float(SHIFT_CLAMP_MS),
        },
        "theta_error_max_possible_deg": {"per_speaker": max_err_by_speaker, "min": float(min_max_err)},
        "near_fail_precondition": {
            "baseline_fail_rate_theta_gt4deg": float(base_fail_theta),
            "baseline_frac_psr_gt3db": float(base_frac_psr_good),
            "criteria": "fail_rate_theta_gt4deg>=0.40 AND frac_psr_gt3db<=0.10",
            "near_fail": bool(near_fail),
        },
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
    lines.append(
        f"- common-mode interference: enabled={bool(cm_interf_enable)}, snr_db={float(cm_interf_snr_db) if cm_interf_snr_db is not None else None}, seed={int(cm_interf_seed)}\n\n"
    )
    lines.append("## Policy gates (teacher forbidden mask)\n\n")
    lines.append(f"- coupling_hard_forbid_enable: {bool(coupling_hard_forbid_enable)}\n")
    lines.append(f"- dynamic_coh_gate_enable: {bool(dynamic_coh_gate_enable)} (coh_min={dynamic_coh_min:.3f})\n")
    lines.append(f"- tau_ref_gate_enable: {bool(tau_ref_gate_enable)} (ratio_min={tau_ref_gate_ratio_min:.2f})\n\n")
    lines.append("## Teacher identity checks\n\n")
    lines.append("- actions/noise centers/forbidden mask identical across obs modes: PASS\n\n")
    lines.append("## Metric reachability (guided window)\n\n")
    lines.append(f"- guided_radius_ms: {guided_radius_ms:.3f}\n")
    lines.append(f"- quadratic_shift_clamp_ms: {SHIFT_CLAMP_MS:.6f}\n")
    lines.append(f"- theta_fail_deg: {THETA_FAIL_DEG:.2f}\n\n")
    lines.append("| speaker | max_theta_error_possible_deg |\n")
    lines.append("| --- | ---: |\n")
    for spk in sorted(max_err_by_speaker.keys()):
        lines.append(f"| {spk} | {max_err_by_speaker[spk]:.3f} |\n")
    lines.append("\n")

    lines.append("## Near-fail precondition (baseline MIC–MIC)\n\n")
    lines.append("- Criteria: fail_rate_theta_gt4deg>=0.40 AND frac_psr_gt3db<=0.10 (pooled test windows)\n")
    lines.append(f"- baseline fail_rate_theta_gt4deg: {base_fail_theta:.3f}\n")
    lines.append(f"- baseline frac_psr_gt3db: {base_frac_psr_good:.3f}\n")
    lines.append(f"- near-fail => {'PASS' if near_fail else 'FAIL'}\n\n")

    lines.append("## Pooled test metrics (vs chirp reference)\n\n")
    lines.append("### Theta error\n\n")
    lines.append("| obs_mode | baseline p95 | baseline fail(>4°) | student p95 | student fail(>4°) |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: |\n")
    for obs_mode in OBS_MODES:
        b = pooled_baseline_metric(obs_mode, "theta_error_ref_deg")
        s = pooled_student_metric(obs_mode, "theta_error_ref_deg")
        lines.append(
            f"| {obs_mode} | {b['p95']:.3f} | {b['fail_rate_gt4deg']:.3f} | {s['p95']:.3f} | {s['fail_rate_gt4deg']:.3f} |\n"
        )

    lines.append("\n### Tau error (vs tau_ref)\n\n")
    lines.append("| obs_mode | baseline p95 (ms) | baseline fail(>0.25ms) | student p95 (ms) | student fail(>0.25ms) |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: |\n")
    for obs_mode in OBS_MODES:
        b = pooled_baseline_metric(obs_mode, "tau_error_ref_ms")
        s = pooled_student_metric(obs_mode, "tau_error_ref_ms")
        lines.append(
            f"| {obs_mode} | {b['p95']:.3f} | {b['fail_rate_gt0p25ms']:.3f} | {s['p95']:.3f} | {s['fail_rate_gt0p25ms']:.3f} |\n"
        )

    lines.append("\n### PSR (guided peak)\n\n")
    lines.append("| obs_mode | baseline median (dB) | baseline frac(>3dB) | student median (dB) | student frac(>3dB) |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: |\n")
    for obs_mode in OBS_MODES:
        b = pooled_baseline_metric(obs_mode, "psr_db")
        s = pooled_student_metric(obs_mode, "psr_db")
        lines.append(
            f"| {obs_mode} | {b['median']:.3f} | {b['frac_gt3db']:.3f} | {s['median']:.3f} | {s['frac_gt3db']:.3f} |\n"
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

    # Negative transfer check (per speaker): ldv_mic vs mic_only_control
    nt_threshold = {"p95_ratio": 1.05, "fail_abs": 0.02}
    flagged: list[str] = []
    per = metrics["variants"]["ldv_mic"]["metrics"]["per_speaker"]
    per_mic = metrics["variants"]["mic_only_control"]["metrics"]["per_speaker"]
    nt_rows: list[tuple[str, float, float, float, float]] = []
    for spk in sorted(per.keys()):
        p95_ldv = float(per[spk]["student"]["theta_error_ref_deg"]["p95"])
        fail_ldv = float(per[spk]["student"]["theta_error_ref_deg"]["fail_rate_gt4deg"])
        p95_mic = float(per_mic[spk]["student"]["theta_error_ref_deg"]["p95"])
        fail_mic = float(per_mic[spk]["student"]["theta_error_ref_deg"]["fail_rate_gt4deg"])
        nt_rows.append((spk, p95_ldv, p95_mic, fail_ldv, fail_mic))
        if (p95_ldv > float(nt_threshold["p95_ratio"]) * p95_mic) or (fail_ldv > fail_mic + float(nt_threshold["fail_abs"])):
            flagged.append(spk)

    lines.append("\n## Negative transfer check (per speaker)\n\n")
    lines.append("- Comparison: `ldv_mic` student vs `mic_only_control` student (test windows only)\n")
    lines.append(f"- Flag if: p95_ldv > {nt_threshold['p95_ratio']:.2f} * p95_mic OR fail_ldv > fail_mic + {nt_threshold['fail_abs']:.2f}\n\n")
    lines.append("| speaker | p95_ldv | p95_mic | Δp95 | fail_ldv(>4°) | fail_mic(>4°) | Δfail | flagged |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
    for spk, p95_ldv, p95_mic, fail_ldv, fail_mic in nt_rows:
        dp95 = p95_ldv - p95_mic
        dfail = fail_ldv - fail_mic
        is_flag = "YES" if spk in flagged else ""
        lines.append(f"| {spk} | {p95_ldv:.3f} | {p95_mic:.3f} | {dp95:+.3f} | {fail_ldv:.3f} | {fail_mic:.3f} | {dfail:+.3f} | {is_flag} |\n")
    lines.append("\n")
    lines.append(f"- Flagged speakers: `{flagged}`\n")

    (out_dir / "report.md").write_text("".join(lines), encoding="utf-8")
    logger.info("Wrote report: %s", (out_dir / "report.md").as_posix())


if __name__ == "__main__":
    main()
