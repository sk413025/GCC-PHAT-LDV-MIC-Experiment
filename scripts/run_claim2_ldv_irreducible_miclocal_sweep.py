#!/usr/bin/env python
"""
Claim-2 verification driver: mic-local corruption sweep (existing WAVs only).

This runs a fixed SNR grid with:
- two teachers (LDV+MIC obs vs MIC-only control obs), same actions
- two DTmin students (trained from each teacher trajectory)
- an ablation comparator (LDV+MIC student vs MIC-only student)

All outputs are written under --out_dir.
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


SNR_GRID_DB = [20.0, 10.0, 5.0, 0.0, -5.0, -10.0]
CORRUPT_SEED = 1337
PRECLIP_GAIN = 100.0
CLIP_LIMIT = 0.99
COUPLING_MODE = "mic_only"


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


def snr_dir_name(snr_db: float) -> str:
    if snr_db >= 0:
        return f"snr_p{int(snr_db)}" if float(snr_db).is_integer() else f"snr_p{snr_db}"
    return f"snr_m{int(abs(snr_db))}" if float(snr_db).is_integer() else f"snr_m{abs(snr_db)}"


def run_cmd(cmd: list[str], *, cwd: Path) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def assert_teacher_actions_identical(a_npz: Path, b_npz: Path) -> None:
    a = np.load(str(a_npz), allow_pickle=False)["actions"]
    b = np.load(str(b_npz), allow_pickle=False)["actions"]
    if not np.array_equal(a, b):
        diff = int(np.sum(a != b))
        raise RuntimeError(f"Teacher actions are not identical (diff_count={diff}): {a_npz} vs {b_npz}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Claim-2 mic-local corruption sweep (existing WAVs only)")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--truth_ref_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--speakers",
        type=str,
        nargs="+",
        default=["18-0.1V", "19-0.1V", "20-0.1V", "21-0.1V", "22-0.1V"],
    )
    ap.add_argument(
        "--smoke",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, run a minimal smoke subset (1 speaker, 1 SNR, short center range).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir).expanduser().resolve()
    configure_logging(out_dir)
    write_code_state(out_dir, Path(__file__).resolve())

    data_root = Path(args.data_root).expanduser().resolve()
    truth_ref_root = Path(args.truth_ref_root).expanduser().resolve()
    speakers = list(args.speakers)

    snr_grid = list(SNR_GRID_DB)
    teacher_center_overrides: list[str] = []
    student_split_overrides: list[str] = []
    if bool(int(args.smoke)):
        speakers = ["20-0.1V"]
        snr_grid = [0.0]
        teacher_center_overrides = ["--center_start_sec", "100", "--center_end_sec", "110", "--center_step_sec", "1"]
        student_split_overrides = ["--train_max_center_sec", "105"]

    run_config = {
        "generated": datetime.now().isoformat(),
        "data_root": str(data_root),
        "truth_ref_root": str(truth_ref_root),
        "speakers": speakers,
        "snr_grid_db": snr_grid,
        "corruption": {
            "seed": int(CORRUPT_SEED),
            "preclip_gain": float(PRECLIP_GAIN),
            "clip_limit": float(CLIP_LIMIT),
            "band_hz": [500.0, 2000.0],
        },
        "coupling_mode": str(COUPLING_MODE),
        "smoke": bool(int(args.smoke)),
    }
    write_json(out_dir / "run_config.json", run_config)

    summary_rows: list[dict[str, Any]] = []

    for snr_db in snr_grid:
        snr_dir = out_dir / snr_dir_name(float(snr_db))
        snr_dir.mkdir(parents=True, exist_ok=True)

        teacher_a = snr_dir / "teacher_ldv_mic"
        teacher_b = snr_dir / "teacher_mic_only"
        student_a = snr_dir / "student_ldv_mic"
        student_b = snr_dir / "student_mic_only"
        compare_dir = snr_dir / "compare"

        common_teacher = [
            sys.executable,
            "-u",
            str(repo_root / "scripts" / "teacher_band_omp_micmic.py"),
            "--data_root",
            str(data_root),
            "--truth_ref_root",
            str(truth_ref_root),
            "--speakers",
            *speakers,
            "--coupling_mode",
            str(COUPLING_MODE),
            "--corrupt_enable",
            "1",
            "--corrupt_snr_db",
            str(float(snr_db)),
            "--corrupt_seed",
            str(int(CORRUPT_SEED)),
            "--preclip_gain",
            str(float(PRECLIP_GAIN)),
            "--clip_limit",
            str(float(CLIP_LIMIT)),
            *teacher_center_overrides,
        ]

        run_cmd([*common_teacher, "--obs_mode", "ldv_mic", "--out_dir", str(teacher_a)], cwd=repo_root)
        run_cmd([*common_teacher, "--obs_mode", "mic_only_control", "--out_dir", str(teacher_b)], cwd=repo_root)

        a_npz = teacher_a / "teacher_trajectories.npz"
        b_npz = teacher_b / "teacher_trajectories.npz"
        assert_teacher_actions_identical(a_npz, b_npz)

        common_student = [
            sys.executable,
            "-u",
            str(repo_root / "scripts" / "train_dtmin_from_band_trajectories.py"),
            "--data_root",
            str(data_root),
            "--truth_ref_root",
            str(truth_ref_root),
            "--use_traj_corruption",
            "1",
            *student_split_overrides,
        ]
        run_cmd([*common_student, "--traj_path", str(a_npz), "--out_dir", str(student_a)], cwd=repo_root)
        run_cmd([*common_student, "--traj_path", str(b_npz), "--out_dir", str(student_b)], cwd=repo_root)

        run_cmd(
            [
                sys.executable,
                "-u",
                str(repo_root / "scripts" / "compare_ablation_runs.py"),
                "--run_a",
                str(student_a),
                "--run_b",
                str(student_b),
                "--teacher_a",
                str(a_npz),
                "--teacher_b",
                str(b_npz),
                "--out_dir",
                str(compare_dir),
            ],
            cwd=repo_root,
        )

        # Summarize this SNR
        sum_a = load_json(student_a / "summary.json")
        sum_b = load_json(student_b / "summary.json")
        comp = load_json(compare_dir / "summary.json")

        base_fail = float(sum_a["pooled"]["baseline"]["fail_rate_ref_gt5deg"])
        ldv_fail = float(sum_a["pooled"]["student"]["fail_rate_ref_gt5deg"])
        cmp_pass = bool(comp["acceptance"]["overall_pass"])

        near_fail = bool(base_fail >= 0.40)
        ldv_feasible = bool(ldv_fail <= 0.10)
        claim2_pass = bool(near_fail and ldv_feasible and cmp_pass)

        summary_rows.append(
            {
                "snr_db": float(snr_db),
                "paths": {
                    "teacher_ldv_mic": str(teacher_a),
                    "teacher_mic_only": str(teacher_b),
                    "student_ldv_mic": str(student_a),
                    "student_mic_only": str(student_b),
                    "compare": str(compare_dir),
                },
                "baseline_fail_rate_ref_gt5deg": base_fail,
                "ldv_mic_student_fail_rate_ref_gt5deg": ldv_fail,
                "compare_acceptance": comp["acceptance"],
                "claim2_checks": {"near_fail": near_fail, "ldv_feasible": ldv_feasible, "ldv_beats_mic_only": cmp_pass},
                "claim2_pass": claim2_pass,
            }
        )

    # Final decision (per plan): pass if any severity in {0,-5,-10} passes.
    target_set = {0.0, -5.0, -10.0}
    any_target_pass = any((r["snr_db"] in target_set) and bool(r["claim2_pass"]) for r in summary_rows)
    any_near_fail = any((r["snr_db"] in target_set) and bool(r["claim2_checks"]["near_fail"]) for r in summary_rows)

    final = {
        "generated": datetime.now().isoformat(),
        "run_dir": str(out_dir),
        "snr_rows": summary_rows,
        "final": {
            "claim2_supported_on_{0,-5,-10}": bool(any_target_pass),
            "near_fail_observed_on_{0,-5,-10}": bool(any_near_fail),
            "note": (
                "PASS: at least one severity in {0,-5,-10} satisfied near-fail + feasibility + ablation."
                if any_target_pass
                else (
                    "FAIL: baseline never reached near-fail on {0,-5,-10} (dataset not extreme enough under this proxy)."
                    if not any_near_fail
                    else "FAIL: near-fail occurred but LDV+MIC did not meet feasibility and/or did not beat MIC-only."
                )
            ),
        },
    }
    write_json(out_dir / "summary_table.json", final)

    # Human-readable report
    lines: list[str] = []
    lines.append("# Claim 2 Verification Report: Mic-Local Corruption Sweep\n")
    lines.append(f"Generated: {final['generated']}\n")
    lines.append(f"Run dir: `{out_dir}`\n")
    lines.append("")
    lines.append("## Grid\n")
    lines.append("| SNR (dB) | baseline fail_rate_ref(>5°) | LDV+MIC student fail_rate_ref(>5°) | LDV beats MIC-only | PASS |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for r in summary_rows:
        lines.append(
            f"| {r['snr_db']:.1f} | {r['baseline_fail_rate_ref_gt5deg']:.3f} | {r['ldv_mic_student_fail_rate_ref_gt5deg']:.3f} | {bool(r['claim2_checks']['ldv_beats_mic_only'])} | {bool(r['claim2_pass'])} |"
        )
    lines.append("")
    lines.append("## Final\n")
    lines.append(f"- claim2_supported_on_{{0,-5,-10}}: **{final['final']['claim2_supported_on_{0,-5,-10}']}**")
    lines.append(f"- near_fail_observed_on_{{0,-5,-10}}: {final['final']['near_fail_observed_on_{0,-5,-10}']}")
    lines.append(f"- note: {final['final']['note']}")
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info("Done. Results: %s", out_dir)
    logger.info("Final claim2_supported_on_{0,-5,-10}=%s", bool(final["final"]["claim2_supported_on_{0,-5,-10}"]))


if __name__ == "__main__":
    main()
