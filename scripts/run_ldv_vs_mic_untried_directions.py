#!/usr/bin/env python
"""
Batch runner for untried Stage4 directions from tau-collapse retrospective.

Directions covered:
- F: Welch-averaged GCC-PHAT
- A: Coherence-weighted GCC
- D: LDV velocity->pressure compensation
- H: Coherence-mask gating
- B: Narrowband TDOA fusion
- E: Full-segment evaluation (micl_micr)
- I: Dual-baseline joint estimation (LDV + Mic reliability fusion)
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

logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_fingerprint(files: list[Path]) -> str:
    entries = []
    for fp in sorted(files, key=lambda p: str(p).lower()):
        entries.append(f"{sha256_file(fp)} {fp}")
    joined = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def collect_subset_manifest(data_root: Path, speakers: list[str]) -> dict:
    files = []
    per_speaker = {}
    for sp in speakers:
        sp_dir = data_root / sp
        ldv = list(sp_dir.glob("*LDV*.wav"))
        mic_l = list(sp_dir.glob("*LEFT*.wav"))
        mic_r = list(sp_dir.glob("*RIGHT*.wav"))
        per_speaker[sp] = {
            "ldv": [str(p) for p in ldv],
            "mic_left": [str(p) for p in mic_l],
            "mic_right": [str(p) for p in mic_r],
        }
        files.extend(ldv + mic_l + mic_r)

    file_hashes = {str(p): sha256_file(p) for p in files}
    fingerprint = compute_dataset_fingerprint(files)
    return {
        "speakers": speakers,
        "files": per_speaker,
        "file_hashes": file_hashes,
        "total_files": len(files),
        "fingerprint_sha256": fingerprint,
    }


def write_code_state(output_dir: Path, script_paths: list[Path]) -> None:
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() != ""
    except Exception:
        git_head = None
        dirty = None

    payload = {
        "git_head": git_head,
        "dirty": dirty,
        "timestamp": datetime.now().isoformat(),
        "scripts": [
            {
                "path": str(p),
                "sha256": sha256_file(p),
            }
            for p in script_paths
        ],
    }
    (output_dir / "code_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_cmd(args: list[str], cwd: Path | None = None) -> int:
    logger.info("Running: %s", " ".join(args))
    proc = subprocess.run(args, cwd=str(cwd) if cwd else None)
    return int(proc.returncode)


def load_summary(summary_path: Path) -> dict | None:
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_summary_table(run_dir: Path, summaries: dict[str, dict | None]) -> None:
    lines = []
    lines.append(f"# Stage4 Untried Directions Summary: {run_dir.name}")
    lines.append("")
    lines.append("| speaker | tau_ref_ms | theta_ref_deg | theta_deg | theta_err_deg | psr_db | pass |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for sp, summary in summaries.items():
        if summary is None:
            lines.append(f"| {sp} | NA | NA | NA | NA | NA | False |")
            continue
        tau_ref = summary["truth_reference"]["tau_ref_ms"]
        theta_ref = summary["truth_reference"]["theta_ref_deg"]
        theta = summary["result"]["theta_median_deg"]
        theta_err = summary["result"]["theta_error_median_deg"]
        psr_db = summary["result"]["psr_median_db"]
        passed = summary["passed"]
        lines.append(
            f"| {sp} | {tau_ref} | {theta_ref} | {theta} | {theta_err} | {psr_db} | {passed} |"
        )
    lines.append("")
    pass_count = sum(1 for s in summaries.values() if s and s.get("passed"))
    lines.append(f"Pass count: {pass_count}/{len(summaries)}")
    (run_dir / "summary_table.md").write_text("\n".join(lines), encoding="utf-8")


def write_grid_summary(output_base: Path, grid_rows: list[dict]) -> None:
    (output_base / "grid_summary.json").write_text(
        json.dumps({"generated": datetime.now().isoformat(), "rows": grid_rows}, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("# Untried Directions Grid Summary")
    lines.append("")
    lines.append("| config | direction_tags | signal_pair | pass_mode | pass_count | failing_speakers | run_dir |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in grid_rows:
        lines.append(
            f"| {row['label']} | {row['direction_tags']} | {row['signal_pair']} | {row['pass_mode']} | "
            f"{row['pass_count']} | {', '.join(row['failing_speakers'])} | {row['run_dir']} |"
        )
    (output_base / "grid_summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_configs() -> list[dict]:
    return [
        {
            "label": "baseline_fft_band200_4000",
            "direction_tags": "baseline",
            "signal_pair": "ldv_micl",
            "pass_mode": "omp_vs_raw",
            "segment_mode": "scan",
            "analysis_slice_sec": 5.0,
            "eval_window_sec": 5.0,
            "extra_args": [
                "--gcc_method",
                "fft",
                "--gcc_bandpass_low",
                "200",
                "--gcc_bandpass_high",
                "4000",
                "--ldv_prealign",
                "gcc_phat",
            ],
        },
        {
            "label": "F_welch_phat_band200_4000",
            "direction_tags": "F",
            "signal_pair": "ldv_micl",
            "pass_mode": "omp_vs_raw",
            "segment_mode": "scan",
            "analysis_slice_sec": 5.0,
            "eval_window_sec": 5.0,
            "extra_args": [
                "--gcc_method",
                "welch_phat",
                "--gcc_bandpass_low",
                "200",
                "--gcc_bandpass_high",
                "4000",
                "--ldv_prealign",
                "gcc_phat",
            ],
        },
        {
            "label": "A_welch_coherence_weighted_band200_4000",
            "direction_tags": "A+F",
            "signal_pair": "ldv_micl",
            "pass_mode": "omp_vs_raw",
            "segment_mode": "scan",
            "analysis_slice_sec": 5.0,
            "eval_window_sec": 5.0,
            "extra_args": [
                "--gcc_method",
                "welch_coherence_weighted",
                "--gcc_bandpass_low",
                "200",
                "--gcc_bandpass_high",
                "4000",
                "--ldv_prealign",
                "gcc_phat",
            ],
        },
        {
            "label": "D_plus_F_welch_phat_ldv_comp",
            "direction_tags": "D+F",
            "signal_pair": "ldv_micl",
            "pass_mode": "omp_vs_raw",
            "segment_mode": "scan",
            "analysis_slice_sec": 5.0,
            "eval_window_sec": 5.0,
            "extra_args": [
                "--gcc_method",
                "welch_phat",
                "--gcc_bandpass_low",
                "200",
                "--gcc_bandpass_high",
                "4000",
                "--ldv_velocity_comp",
                "--ldv_velocity_comp_eps_hz",
                "20",
                "--ldv_prealign",
                "gcc_phat",
            ],
        },
        {
            "label": "D_plus_H_plus_F_ldv_comp_cohmask",
            "direction_tags": "D+H+F",
            "signal_pair": "ldv_micl",
            "pass_mode": "omp_vs_raw",
            "segment_mode": "scan",
            "analysis_slice_sec": 5.0,
            "eval_window_sec": 5.0,
            "extra_args": [
                "--gcc_method",
                "welch_coherence_mask",
                "--coherence_mask_min",
                "1e-12",
                "--gcc_bandpass_low",
                "200",
                "--gcc_bandpass_high",
                "4000",
                "--ldv_velocity_comp",
                "--ldv_velocity_comp_eps_hz",
                "20",
                "--ldv_prealign",
                "none",
            ],
        },
        {
            "label": "B_narrowband_fusion_welch_phat",
            "direction_tags": "B+F",
            "signal_pair": "ldv_micl",
            "pass_mode": "omp_vs_raw",
            "segment_mode": "scan",
            "analysis_slice_sec": 5.0,
            "eval_window_sec": 5.0,
            "extra_args": [
                "--gcc_method",
                "welch_phat",
                "--gcc_bandpass_low",
                "200",
                "--gcc_bandpass_high",
                "4000",
                "--narrowband_fusion",
                "--narrowband_width_hz",
                "200",
                "--ldv_prealign",
                "gcc_phat",
            ],
        },
        {
            "label": "I_dual_baseline_joint_weighted",
            "direction_tags": "I+A+F",
            "signal_pair": "ldv_micl",
            "pass_mode": "omp_vs_raw",
            "segment_mode": "scan",
            "analysis_slice_sec": 5.0,
            "eval_window_sec": 5.0,
            "extra_args": [
                "--gcc_method",
                "welch_coherence_weighted",
                "--gcc_bandpass_low",
                "200",
                "--gcc_bandpass_high",
                "4000",
                "--dual_baseline_fusion",
                "--ldv_prealign",
                "gcc_phat",
            ],
        },
        {
            "label": "E_full_segment_micl_micr_welch_weighted",
            "direction_tags": "E+A+F",
            "signal_pair": "micl_micr",
            "pass_mode": "theta_only",
            "segment_mode": "fixed",
            "analysis_slice_sec": 500.0,
            "eval_window_sec": 500.0,
            "extra_args": [
                "--gcc_method",
                "welch_coherence_weighted",
                "--gcc_bandpass_low",
                "200",
                "--gcc_bandpass_high",
                "4000",
            ],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage4 untried-direction experiments")
    parser.add_argument("--data_root", type=str, required=True, help="Speech dataset root")
    parser.add_argument(
        "--speakers",
        type=str,
        nargs="+",
        default=["18-0.1V", "19-0.1V", "20-0.1V", "21-0.1V", "22-0.1V"],
    )
    parser.add_argument("--output_base", type=str, required=True)
    parser.add_argument("--alignment_mode", type=str, choices=["omp", "dtmin"], default="omp")
    parser.add_argument("--dtmin_model_path", type=str, default=None)
    parser.add_argument("--max_k", type=int, default=3)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(output_base / "run.log"), logging.StreamHandler()],
    )

    data_root = Path(args.data_root)
    script_path = Path(__file__).parent / "stage4_doa_ldv_vs_mic_comparison.py"

    speakers = list(args.speakers)
    configs = build_configs()
    if args.debug:
        speakers = speakers[:2]
        configs = configs[:2]

    logger.info("Running %d configs x %d speakers", len(configs), len(speakers))
    grid_rows: list[dict] = []

    for cfg in configs:
        run_dir = output_base / cfg["label"]
        run_dir.mkdir(parents=True, exist_ok=True)

        summaries: dict[str, dict | None] = {}
        fail_runs: list[str] = []

        for sp in speakers:
            args_list = [
                sys.executable,
                "-u",
                str(script_path),
                "--data_root",
                str(data_root),
                "--speaker",
                sp,
                "--output_dir",
                str(run_dir),
                "--signal_pair",
                str(cfg["signal_pair"]),
                "--segment_mode",
                str(cfg["segment_mode"]),
                "--n_segments",
                "5" if cfg["segment_mode"] == "scan" else "1",
                "--analysis_slice_sec",
                str(cfg["analysis_slice_sec"]),
                "--eval_window_sec",
                str(cfg["eval_window_sec"]),
                "--alignment_mode",
                str(args.alignment_mode),
                "--max_k",
                str(int(args.max_k)),
                "--pass_theta_max_deg",
                "5.0",
                "--pass_mode",
                str(cfg["pass_mode"]),
                "--use_geometry_truth",
            ]

            if cfg["segment_mode"] == "scan":
                args_list += [
                    "--segment_offset_sec",
                    "100",
                    "--segment_spacing_sec",
                    "50",
                    "--scan_start_sec",
                    "100",
                    "--scan_end_sec",
                    "600",
                    "--scan_hop_sec",
                    "1",
                    "--scan_psr_min_db",
                    "-20",
                    "--scan_tau_err_max_ms",
                    "2.0",
                    "--scan_sort_by",
                    "tau_err",
                    "--scan_min_separation_sec",
                    "5",
                    "--gcc_guided_peak_radius_ms",
                    "0.3",
                ]

            args_list += cfg["extra_args"]
            if cfg["signal_pair"] == "ldv_micl":
                args_list += ["--pass_require_ldv_better_than_mic"]

            if cfg["signal_pair"] == "ldv_micl" and args.alignment_mode == "dtmin":
                if not args.dtmin_model_path:
                    raise ValueError("--alignment_mode dtmin requires --dtmin_model_path")
                args_list += ["--dtmin_model_path", str(args.dtmin_model_path)]

            ret = run_cmd(args_list)
            if ret != 0:
                fail_runs.append(sp)
                logger.warning("Run failed: config=%s speaker=%s", cfg["label"], sp)

            summary_path = run_dir / sp / "summary.json"
            summaries[sp] = load_summary(summary_path)

        write_summary_table(run_dir, summaries)
        subset_manifest = collect_subset_manifest(data_root, speakers)
        (run_dir / "subset_manifest.json").write_text(json.dumps(subset_manifest, indent=2), encoding="utf-8")
        (run_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "config": cfg,
                    "speakers": speakers,
                    "alignment_mode": args.alignment_mode,
                    "dtmin_model_path": args.dtmin_model_path,
                    "max_k": int(args.max_k),
                    "timestamp": datetime.now().isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        write_code_state(run_dir, [script_path, Path(__file__)])

        pass_count = sum(1 for s in summaries.values() if s and s.get("passed"))
        failing = [sp for sp, s in summaries.items() if not s or not s.get("passed")]
        grid_rows.append(
            {
                "label": cfg["label"],
                "direction_tags": cfg["direction_tags"],
                "signal_pair": cfg["signal_pair"],
                "pass_mode": cfg["pass_mode"],
                "pass_count": pass_count,
                "failing_speakers": failing,
                "exec_fail_speakers": fail_runs,
                "run_dir": str(run_dir),
            }
        )

    write_grid_summary(output_base, grid_rows)
    write_code_state(output_base, [script_path, Path(__file__)])
    logger.info("Completed. Summary: %s", output_base / "grid_summary.md")


if __name__ == "__main__":
    main()
