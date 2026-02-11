#!/usr/bin/env python
"""
Grid runner for LDV-vs-Mic DoA comparison (GCC-PHAT).

Runs a configuration matrix over:
- signal_pair: ldv_micl or micl_micr
- truth_type: chirp or geometry
- tau_err_max_ms: 2.0 (primary) and 0.3 (secondary)
- bandpass: none or 500-2000

Produces per-config run directories with summary_table.md and grid_summary.md/json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
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


def write_code_state(output_dir: Path, script_path: Path) -> None:
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() != ""
    except Exception:
        git_head = None
        dirty = None
    payload = {
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "git_head": git_head,
        "dirty": dirty,
        "timestamp": datetime.now().isoformat(),
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


def write_summary_table(run_dir: Path, summaries: dict[str, dict]) -> None:
    lines = []
    lines.append(f"# Stage4 Speech {run_dir.name} GCC-PHAT Summary")
    lines.append("")
    lines.append(f"Run dir: {run_dir}")
    lines.append("")
    lines.append("| Speaker | tau_ref_ms | theta_ref_deg | theta_omp_deg | theta_err_deg | psr_db | pass |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for sp, summary in summaries.items():
        if summary is None:
            lines.append(f"| {sp} | NA | NA | NA | NA | NA | False |")
            continue
        tau_ref = summary["truth_reference"]["tau_ref_ms"]
        theta_ref = summary["truth_reference"]["theta_ref_deg"]
        theta_omp = summary["result"]["theta_median_deg"]
        theta_err = summary["result"]["theta_error_median_deg"]
        psr_db = summary["result"]["psr_median_db"]
        passed = summary["passed"]
        lines.append(
            f"| {sp} | {tau_ref} | {theta_ref} | {theta_omp} | {theta_err} | {psr_db} | {passed} |"
        )
    lines.append("")
    pass_count = sum(1 for s in summaries.values() if s and s.get("passed"))
    lines.append(f"GCC-PHAT pass count: {pass_count}/{len(summaries)}")
    (run_dir / "summary_table.md").write_text("\n".join(lines), encoding="utf-8")


def write_grid_summary(output_base: Path, grid_rows: list[dict]) -> None:
    grid_summary = {
        "generated": datetime.now().isoformat(),
        "rows": grid_rows,
    }
    (output_base / "grid_summary.json").write_text(json.dumps(grid_summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# LDV-vs-Mic Grid Summary (GCC-PHAT)")
    lines.append("")
    lines.append("| truth_type | config | tau_err_max_ms | bandpass | pass_count | failing_speakers | run_dir |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in grid_rows:
        lines.append(
            f"| {row['truth_type']} | {row['label']} | {row['tau_err_max_ms']} | "
            f"{row['bandpass_low']}-{row['bandpass_high']} | {row['pass_count']} | "
            f"{', '.join(row['failing_speakers'])} | {row['run_dir']} |"
        )
    (output_base / "grid_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_grid_report(output_base: Path, grid_rows: list[dict], data_root: Path, chirp_truth_file: Path | None) -> None:
    lines = []
    lines.append("# LDV-vs-Mic Grid Comparison Report (GCC-PHAT Only)")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("Compare LDV-MicL and MicL-MicR under chirp and geometry guidance.")
    lines.append("")
    lines.append("## Data & Truth Sources")
    lines.append(f"- Speech data root: `{data_root}`")
    if chirp_truth_file:
        lines.append(f"- Chirp truth reference file: `{chirp_truth_file}`")
    lines.append("")
    lines.append("## Grid Summary")
    lines.append("")
    lines.append("| truth_type | config | tau_err_max_ms | bandpass | pass_count | failing_speakers | run_dir |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in grid_rows:
        lines.append(
            f"| {row['truth_type']} | {row['label']} | {row['tau_err_max_ms']} | "
            f"{row['bandpass_low']}-{row['bandpass_high']} | {row['pass_count']} | "
            f"{', '.join(row['failing_speakers'])} | {row['run_dir']} |"
        )
    lines.append("")
    best = max(grid_rows, key=lambda r: r["pass_count"]) if grid_rows else None
    if best:
        lines.append(f"Best pass count: {best['pass_count']}/5 (config {best['label']}).")
    lines.append("")
    (output_base / "grid_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid runner for LDV-vs-Mic DoA comparison")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for dataset")
    parser.add_argument("--speakers", type=str, nargs="+", default=["18-0.1V", "19-0.1V", "20-0.1V", "21-0.1V", "22-0.1V"])
    parser.add_argument("--output_base", type=str, required=True, help="Base output directory")
    parser.add_argument("--chirp_truth_file", type=str, default=None, help="Path to chirp truth JSON")
    parser.add_argument("--skip_smoke", action="store_true")
    parser.add_argument("--skip_guardrail", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    log_file = output_base / "grid_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    data_root = Path(args.data_root)
    chirp_truth = None
    if args.chirp_truth_file:
        chirp_truth = json.loads(Path(args.chirp_truth_file).read_text(encoding="utf-8"))

    configs = [
        {"label": "ldv_micl_chirp_tau2_band0", "signal_pair": "ldv_micl", "truth_type": "chirp", "tau_err_max_ms": 2.0, "bandpass_low": 0, "bandpass_high": 0},
        {"label": "ldv_micl_chirp_tau2_band500_2000", "signal_pair": "ldv_micl", "truth_type": "chirp", "tau_err_max_ms": 2.0, "bandpass_low": 500, "bandpass_high": 2000},
        {"label": "micl_micr_chirp_tau2_band500_2000", "signal_pair": "micl_micr", "truth_type": "chirp", "tau_err_max_ms": 2.0, "bandpass_low": 500, "bandpass_high": 2000},
        {"label": "ldv_micl_geometry_tau2_band500_2000", "signal_pair": "ldv_micl", "truth_type": "geometry", "tau_err_max_ms": 2.0, "bandpass_low": 500, "bandpass_high": 2000},
        {"label": "ldv_micl_chirp_tau0p3_band500_2000", "signal_pair": "ldv_micl", "truth_type": "chirp", "tau_err_max_ms": 0.3, "bandpass_low": 500, "bandpass_high": 2000},
    ]

    speakers = list(args.speakers)
    if args.debug:
        configs = configs[:1]
        speakers = speakers[:2]

    logger.info("Grid size: %d configs x %d speakers", len(configs), len(speakers))

    script_path = Path(__file__).parent / "stage4_doa_ldv_vs_mic_comparison.py"

    grid_rows = []

    for cfg in configs:
        run_dir = output_base / cfg["label"]
        run_dir.mkdir(parents=True, exist_ok=True)

        summaries = {}
        for sp in speakers:
            speaker_dir = run_dir / sp
            speaker_dir.mkdir(parents=True, exist_ok=True)

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
                cfg["signal_pair"],
                "--segment_mode",
                "scan",
                "--n_segments",
                "5",
                "--analysis_slice_sec",
                "5",
                "--eval_window_sec",
                "5",
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
                str(cfg["tau_err_max_ms"]),
                "--scan_sort_by",
                "tau_err",
                "--scan_min_separation_sec",
                "5",
                "--gcc_guided_peak_radius_ms",
                "0.3",
                "--gcc_bandpass_low",
                str(cfg["bandpass_low"]),
                "--gcc_bandpass_high",
                str(cfg["bandpass_high"]),
                "--ldv_prealign",
                "gcc_phat",
                "--pass_theta_max_deg",
                "5.0",
            ]

            if cfg["truth_type"] == "chirp":
                if not chirp_truth or "truth_ref" not in chirp_truth:
                    raise ValueError("chirp_truth_file missing or invalid")
                truth = chirp_truth["truth_ref"][sp]
                args_list += [
                    "--truth_tau_ms",
                    str(truth["tau_ms"]),
                    "--truth_theta_deg",
                    str(truth["theta_deg"]),
                    "--truth_label",
                    str(truth.get("label", "chirp_truth")),
                ]
            else:
                args_list.append("--use_geometry_truth")

            ret = run_cmd(args_list)
            if ret != 0:
                logger.warning("Run failed for %s (%s)", sp, cfg["label"])

            summary_path = run_dir / sp / "summary.json"
            summaries[sp] = load_summary(summary_path)

        write_summary_table(run_dir, summaries)

        run_config = {
            "config": cfg,
            "speakers": speakers,
            "segment_mode": "scan",
            "analysis_slice_sec": 5,
            "eval_window_sec": 5,
            "scan_start_sec": 100,
            "scan_end_sec": 600,
            "scan_hop_sec": 1,
            "scan_psr_min_db": -20,
            "scan_sort_by": "tau_err",
            "scan_min_separation_sec": 5,
            "gcc_guided_peak_radius_ms": 0.3,
            "ldv_prealign": "gcc_phat",
            "pass_theta_max_deg": 5.0,
        }
        (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

        subset_manifest = collect_subset_manifest(data_root, speakers)
        (run_dir / "subset_manifest.json").write_text(json.dumps(subset_manifest, indent=2), encoding="utf-8")
        write_code_state(run_dir, script_path)

        pass_count = sum(1 for s in summaries.values() if s and s.get("passed"))
        failing = [sp for sp, s in summaries.items() if not s or not s.get("passed")]
        grid_rows.append(
            {
                "truth_type": cfg["truth_type"],
                "label": cfg["label"],
                "tau_err_max_ms": cfg["tau_err_max_ms"],
                "bandpass_low": cfg["bandpass_low"],
                "bandpass_high": cfg["bandpass_high"],
                "pass_count": pass_count,
                "failing_speakers": failing,
                "run_dir": str(run_dir),
            }
        )

    write_grid_summary(output_base, grid_rows)
    write_grid_report(output_base, grid_rows, data_root, Path(args.chirp_truth_file) if args.chirp_truth_file else None)

    if not args.skip_smoke:
        logger.info("Running smoke tests")
        smoke_dir = output_base / "smoke_ldv_micl_chirp"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                sys.executable,
                "-u",
                str(script_path),
                "--data_root",
                str(data_root),
                "--speaker",
                "20-0.1V",
                "--output_dir",
                str(smoke_dir),
                "--signal_pair",
                "ldv_micl",
                "--segment_mode",
                "fixed",
                "--n_segments",
                "1",
                "--analysis_slice_sec",
                "2",
                "--eval_window_sec",
                "1",
                "--ldv_prealign",
                "gcc_phat",
            ]
        )

        smoke_dir = output_base / "smoke_micl_micr_chirp"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                sys.executable,
                "-u",
                str(script_path),
                "--data_root",
                str(data_root),
                "--speaker",
                "20-0.1V",
                "--output_dir",
                str(smoke_dir),
                "--signal_pair",
                "micl_micr",
                "--segment_mode",
                "fixed",
                "--n_segments",
                "1",
                "--analysis_slice_sec",
                "2",
                "--eval_window_sec",
                "1",
            ]
        )

        smoke_dir = output_base / "smoke_ldv_micl_geometry"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                sys.executable,
                "-u",
                str(script_path),
                "--data_root",
                str(data_root),
                "--speaker",
                "20-0.1V",
                "--output_dir",
                str(smoke_dir),
                "--signal_pair",
                "ldv_micl",
                "--segment_mode",
                "fixed",
                "--n_segments",
                "1",
                "--analysis_slice_sec",
                "2",
                "--eval_window_sec",
                "1",
                "--use_geometry_truth",
            ]
        )

    if not args.skip_guardrail:
        logger.info("Running guardrail test (expected fail)")
        guard_dir = output_base / "guardrail"
        guard_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                sys.executable,
                "-u",
                str(script_path),
                "--data_root",
                str(data_root),
                "--speaker",
                "21-0.1V",
                "--output_dir",
                str(guard_dir),
                "--signal_pair",
                "micl_micr",
                "--segment_mode",
                "scan",
                "--n_segments",
                "5",
                "--analysis_slice_sec",
                "1",
                "--eval_window_sec",
                "1",
                "--scan_start_sec",
                "100",
                "--scan_end_sec",
                "110",
                "--scan_hop_sec",
                "1",
                "--scan_psr_min_db",
                "20",
                "--scan_tau_err_max_ms",
                "0.01",
                "--scan_sort_by",
                "tau_err",
                "--scan_min_separation_sec",
                "1",
            ]
        )


if __name__ == "__main__":
    main()
