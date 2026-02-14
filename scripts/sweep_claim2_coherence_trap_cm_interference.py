#!/usr/bin/env python
"""
Sweep a common-mode coherent interference level to create a "coherence trap":

- MicL–MicR coherence becomes high (misleading),
- but MIC–MIC guided GCC-PHAT becomes near-failing vs chirp-reference truth.

This script runs the Band-OMP teacher (single obs_mode) at multiple common-mode SNR levels,
parses the resulting windows.jsonl, and selects a candidate level for a full ablation run.

Outputs are written under --out_dir (never repo root).
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


# Locked grid + selection thresholds (pre-registered)
CM_SNR_GRID_DB = [20.0, 10.0, 5.0, 0.0, -5.0, -10.0, -15.0]
THETA_FAIL_DEG = 4.0
PSR_GOOD_DB = 3.0
NEAR_FAIL_MIN_FAIL_RATE = 0.40
NEAR_FAIL_MAX_FRAC_PSR_GOOD = 0.10

# "Coherence trap" requirement: coherence stays non-trivially high
COH_TRAP_MIN_MEDIAN = 0.20

# Time split consistent with DTmin eval
TEST_MIN_CENTER_SEC = 450.0


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


def git_head_and_dirty() -> tuple[str | None, bool | None]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() != ""
        return head, dirty
    except Exception:
        return None, None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def run_cmd(cmd: list[str], *, cwd: Path) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_rows(windows_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with windows_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty windows.jsonl: {windows_path}")
    return rows


def summarize_level(teacher_dir: Path, *, test_min_center_sec: float) -> dict[str, Any]:
    per_speaker_dir = teacher_dir / "per_speaker"
    if not per_speaker_dir.exists():
        raise FileNotFoundError(f"Missing per_speaker dir: {per_speaker_dir}")

    theta_err: list[float] = []
    psr: list[float] = []
    coh_medians: list[float] = []
    n_rows = 0

    for spk_dir in sorted(per_speaker_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        windows_path = spk_dir / "windows.jsonl"
        if not windows_path.exists():
            raise FileNotFoundError(f"Missing windows.jsonl: {windows_path}")
        rows = load_rows(windows_path)
        for r in rows:
            center = float(r["center_sec"])
            if center <= float(test_min_center_sec):
                continue
            b = r["baseline"]
            theta_err.append(float(b["theta_error_ref_deg"]))
            psr.append(float(b["psr_db"]))
            coh = r.get("mic_coh_speech_band_summary", {}).get("median", None)
            if coh is not None:
                coh_medians.append(float(coh))
            n_rows += 1

    if n_rows == 0:
        raise RuntimeError(f"No rows found for center_sec > {test_min_center_sec}.")

    theta_arr = np.asarray(theta_err, dtype=np.float64)
    psr_arr = np.asarray(psr, dtype=np.float64)
    coh_arr = np.asarray(coh_medians, dtype=np.float64) if coh_medians else np.asarray([], dtype=np.float64)

    out = {
        "n_test_windows": int(n_rows),
        "theta_error_ref_deg": {
            "p95": float(np.quantile(theta_arr, 0.95)),
            "fail_rate_gt4deg": float(np.mean(theta_arr > float(THETA_FAIL_DEG))),
            "median": float(np.quantile(theta_arr, 0.50)),
        },
        "psr_db": {
            "median": float(np.quantile(psr_arr, 0.50)),
            "frac_gt3db": float(np.mean(psr_arr > float(PSR_GOOD_DB))),
            "p95": float(np.quantile(psr_arr, 0.95)),
        },
        "mic_coh_band_median": {
            "median": float(np.quantile(coh_arr, 0.50)) if coh_arr.size else float("nan"),
            "p10": float(np.quantile(coh_arr, 0.10)) if coh_arr.size else float("nan"),
            "p90": float(np.quantile(coh_arr, 0.90)) if coh_arr.size else float("nan"),
        },
    }
    out["near_fail_pass"] = bool(
        (out["theta_error_ref_deg"]["fail_rate_gt4deg"] >= float(NEAR_FAIL_MIN_FAIL_RATE))
        and (out["psr_db"]["frac_gt3db"] <= float(NEAR_FAIL_MAX_FRAC_PSR_GOOD))
    )
    out["coherence_trap_pass"] = bool(np.isfinite(out["mic_coh_band_median"]["median"]) and (out["mic_coh_band_median"]["median"] >= float(COH_TRAP_MIN_MEDIAN)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep common-mode interference SNR to create a coherence trap (Claim2)")
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
    test_min_center_sec = float(TEST_MIN_CENTER_SEC)
    if bool(int(args.smoke)):
        speakers = ["20-0.1V"]
        teacher_center_overrides = ["--center_start_sec", "100", "--center_end_sec", "110", "--center_step_sec", "1"]
        # Smoke runs do not follow the DTmin time split; summarize over all smoke windows.
        test_min_center_sec = -1.0

    # Fixed base corruption/occlusion (same as suite defaults)
    base_flags = [
        "--coupling_mode",
        "mic_only",
        "--coupling_hard_forbid_enable",
        "0",
        "--dynamic_coh_gate_enable",
        "1",
        "--dynamic_coh_min",
        "0.05",
        "--tau_ref_gate_enable",
        "0",
        "--corrupt_enable",
        "1",
        "--corrupt_snr_db",
        "0.0",
        "--corrupt_seed",
        "1337",
        "--preclip_gain",
        "100.0",
        "--clip_limit",
        "0.99",
        "--occlusion_enable",
        "1",
        "--occlusion_target",
        "micr",
        "--occlusion_kind",
        "lowpass",
        "--occlusion_lowpass_hz",
        "800.0",
        "--occlusion_tilt_k",
        "2.0",
        "--occlusion_tilt_pivot_hz",
        "800.0",
        *teacher_center_overrides,
    ]

    run_cfg = {
        "generated": datetime.now().isoformat(),
        "data_root": str(data_root),
        "truth_ref_root": str(truth_ref_root),
        "speakers": speakers,
        "smoke": bool(int(args.smoke)),
        "cm_snr_grid_db": CM_SNR_GRID_DB,
        "selection": {
            "near_fail": {
                "theta_fail_deg": float(THETA_FAIL_DEG),
                "psr_good_db": float(PSR_GOOD_DB),
                "min_fail_rate": float(NEAR_FAIL_MIN_FAIL_RATE),
                "max_frac_psr_good": float(NEAR_FAIL_MAX_FRAC_PSR_GOOD),
                "test_min_center_sec": float(test_min_center_sec),
            },
            "coherence_trap": {"min_mic_coh_band_median": float(COH_TRAP_MIN_MEDIAN)},
        },
        "teacher_flags": base_flags,
    }
    write_json(out_dir / "run_config.json", run_cfg)

    sweep_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for cm_snr_db in CM_SNR_GRID_DB:
        level_dir = out_dir / f"cm_snr_{cm_snr_db:+.0f}db"
        teacher_dir = level_dir / "teacher_mic_only_control"
        teacher_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-u",
            "scripts/teacher_band_omp_micmic.py",
            "--data_root",
            str(data_root),
            "--truth_ref_root",
            str(truth_ref_root),
            "--out_dir",
            str(teacher_dir),
            "--speakers",
            *speakers,
            "--obs_mode",
            "mic_only_control",
            "--cm_interf_enable",
            "1",
            "--cm_interf_snr_db",
            str(float(cm_snr_db)),
            "--cm_interf_seed",
            "1337",
            *base_flags,
        ]
        run_cmd(cmd, cwd=repo_root)

        level_summary = summarize_level(teacher_dir, test_min_center_sec=float(test_min_center_sec))
        row = {"cm_snr_db": float(cm_snr_db), "teacher_dir": str(teacher_dir), **level_summary}
        sweep_rows.append(row)
        if bool(level_summary["near_fail_pass"]) and bool(level_summary["coherence_trap_pass"]):
            candidates.append(row)

        write_json(level_dir / "summary.json", row)

    # Selection rule: among passing candidates, pick the one with highest median coherence.
    chosen = None
    if candidates:
        chosen = sorted(candidates, key=lambda r: float(r["mic_coh_band_median"]["median"]), reverse=True)[0]

    sweep = {
        "generated": datetime.now().isoformat(),
        "out_dir": str(out_dir),
        "grid": sweep_rows,
        "chosen": chosen,
    }
    write_json(out_dir / "sweep_summary.json", sweep)

    # Human-readable report
    lines: list[str] = []
    lines.append("# Coherence-Trap Sweep: Common-Mode Interference (Teacher Baseline)\n\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n\n")
    lines.append(f"Run dir: `{out_dir.as_posix()}`\n\n")
    lines.append("## Selection criteria\n\n")
    lines.append(
        f"- Near-fail: fail_rate(theta_error_ref_deg > {THETA_FAIL_DEG:.1f}°) >= {NEAR_FAIL_MIN_FAIL_RATE:.2f} AND frac(PSR>{PSR_GOOD_DB:.1f}dB) <= {NEAR_FAIL_MAX_FRAC_PSR_GOOD:.2f}\n"
    )
    lines.append(f"- Coherence trap: median(mic_coh_band_median) >= {COH_TRAP_MIN_MEDIAN:.2f}\n")
    lines.append(f"- Windows summarized: center_sec > {test_min_center_sec:.0f}\n\n")
    lines.append("## Sweep results (test windows, pooled)\n\n")
    lines.append("| cm_snr_db | fail_rate(>4°) | frac_psr(>3dB) | psr_median | coh_median | near_fail | coh_trap |\n")
    lines.append("| ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
    for r in sweep_rows:
        lines.append(
            f"| {r['cm_snr_db']:+.0f} | {r['theta_error_ref_deg']['fail_rate_gt4deg']:.3f} | "
            f"{r['psr_db']['frac_gt3db']:.3f} | {r['psr_db']['median']:.2f} | "
            f"{r['mic_coh_band_median']['median']:.3f} | "
            f"{'PASS' if r['near_fail_pass'] else 'FAIL'} | "
            f"{'PASS' if r['coherence_trap_pass'] else 'FAIL'} |\n"
        )
    lines.append("\n")
    if chosen is None:
        lines.append("## Chosen level\n\n- None (no grid point met both near-fail and coherence-trap criteria).\n")
    else:
        lines.append("## Chosen level\n\n")
        lines.append(f"- cm_snr_db: {chosen['cm_snr_db']:+.0f}\n")
        lines.append(f"- teacher_dir: `{chosen['teacher_dir']}`\n")
        lines.append(f"- coh_median: {chosen['mic_coh_band_median']['median']:.3f}\n")

    (out_dir / "report.md").write_text("".join(lines), encoding="utf-8")
    logger.info("Wrote report: %s", (out_dir / "report.md").as_posix())

    if chosen is None and not bool(int(args.smoke)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
