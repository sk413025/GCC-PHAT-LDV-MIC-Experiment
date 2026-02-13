#!/usr/bin/env python
"""
Grid runner for MSNF (Multi-Sensor Near-Field Fusion) DoA comparison.

Phases:
  0: tau2/tau3 feasibility diagnostic (Go/No-Go)
  1: Full-band MSNF (band=0)
  2: Bandpass MSNF (band=500-2000)
  3: guided_radius sweep for best MSNF method
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


def write_code_state(output_dir: Path, script_path: Path) -> None:
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ).strip()
            != ""
        )
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
    (output_dir / "code_state.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def run_cmd(args: list[str], cwd: Path | None = None) -> int:
    logger.info("Running: %s", " ".join(args))
    proc = subprocess.run(args, cwd=str(cwd) if cwd else None)
    return int(proc.returncode)


def load_summary(summary_path: Path) -> dict | None:
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_single_speaker(
    script_path: Path,
    data_root: Path,
    speaker: str,
    output_dir: Path,
    *,
    bandpass_low: float,
    bandpass_high: float,
    tau2_guided_radius_ms: float,
    extra_args: list[str] | None = None,
) -> int:
    """Run multi_sensor_fusion_doa.py for a single speaker."""
    args_list = [
        sys.executable,
        "-u",
        str(script_path),
        "--data_root",
        str(data_root),
        "--speaker",
        speaker,
        "--output_dir",
        str(output_dir),
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
        "--gcc_bandpass_low",
        str(bandpass_low),
        "--gcc_bandpass_high",
        str(bandpass_high),
        "--tau2_guided_radius_ms",
        str(tau2_guided_radius_ms),
        "--ldv_prealign",
        "gcc_phat",
        "--max_k",
        "3",
    ]
    if extra_args:
        args_list.extend(extra_args)
    return run_cmd(args_list)


def check_phase0_go(summaries: dict[str, dict | None]) -> dict:
    """Check Phase 0 Go/No-Go condition across speakers."""
    tau2_go_speakers = []
    tau3_go_speakers = []
    for sp, s in summaries.items():
        if s is None:
            continue
        p0 = s.get("phase0_diagnostic", {})
        if p0.get("tau2_go"):
            tau2_go_speakers.append(sp)
        if p0.get("tau3_go"):
            tau3_go_speakers.append(sp)

    n = len(summaries)
    tau2_go = len(tau2_go_speakers) >= 3
    tau3_go = len(tau3_go_speakers) >= 3

    return {
        "tau2_go": tau2_go,
        "tau2_go_count": len(tau2_go_speakers),
        "tau2_go_speakers": tau2_go_speakers,
        "tau3_go": tau3_go,
        "tau3_go_count": len(tau3_go_speakers),
        "tau3_go_speakers": tau3_go_speakers,
        "n_speakers": n,
        "overall_go": tau2_go,
    }


def write_phase_summary(
    run_dir: Path, summaries: dict[str, dict | None], phase_label: str
) -> None:
    """Write a markdown summary table for one phase."""
    lines = []
    lines.append(f"# MSNF {phase_label} Summary")
    lines.append("")
    lines.append(f"Run dir: {run_dir}")
    lines.append("")
    lines.append(
        "| Speaker | theta_true | A_err | B_err | C_err | D_err | E_err | F_err |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

    for sp, s in summaries.items():
        if s is None:
            lines.append(f"| {sp} | NA | NA | NA | NA | NA | NA | NA |")
            continue
        gt = s.get("ground_truth", {})
        theta_true = gt.get("theta_true_deg", "NA")
        r = s.get("result", {})

        def _err(mk):
            v = r.get(mk, {}).get("theta_error_median_deg")
            if v is None:
                return "NA"
            return f"{v:.2f}"

        lines.append(
            f"| {sp} | {theta_true:.2f} | "
            f"{_err('A_mic_mic')} | {_err('B_omp')} | "
            f"{_err('C_theta_fusion')} | {_err('D_msnf_2')} | "
            f"{_err('E_msnf_3')} | {_err('F_msnf_4')} |"
        )

    lines.append("")

    # Phase 0 check
    go_info = check_phase0_go(summaries)
    lines.append(f"Phase 0 Go: tau2={go_info['tau2_go']} ({go_info['tau2_go_count']}/{go_info['n_speakers']})")
    lines.append(f"Phase 0 Go: tau3={go_info['tau3_go']} ({go_info['tau3_go_count']}/{go_info['n_speakers']})")
    lines.append("")

    # Success criteria
    method_keys = ["A_mic_mic", "B_omp", "C_theta_fusion", "D_msnf_2", "E_msnf_3", "F_msnf_4"]
    for mk in method_keys:
        errors = []
        for s in summaries.values():
            if s is None:
                continue
            e = s.get("result", {}).get(mk, {}).get("theta_error_median_deg")
            if e is not None:
                errors.append(e)
        if errors:
            import numpy as np
            lines.append(f"{mk}: median_err={np.median(errors):.2f}, max={max(errors):.2f}, <2deg={sum(1 for e in errors if e < 2.0)}/{len(errors)}")

    (run_dir / "summary_table.md").write_text("\n".join(lines), encoding="utf-8")


def write_grid_report(output_base: Path, phase_results: dict) -> None:
    """Write overall grid report."""
    lines = []
    lines.append("# MSNF Grid Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for phase_label, info in phase_results.items():
        lines.append(f"## {phase_label}")
        lines.append("")
        lines.append(f"Run dir: {info.get('run_dir', 'N/A')}")
        go = info.get("phase0_go", {})
        if go:
            lines.append(f"Phase0 Go: tau2={go.get('tau2_go')}, tau3={go.get('tau3_go')}")
        lines.append("")

        method_keys = ["A_mic_mic", "B_omp", "C_theta_fusion", "D_msnf_2", "E_msnf_3", "F_msnf_4"]
        lines.append("| Method | median_err | beats_A | <2deg |")
        lines.append("| --- | --- | --- | --- |")

        a_median = info.get("method_medians", {}).get("A_mic_mic")
        for mk in method_keys:
            med = info.get("method_medians", {}).get(mk)
            if med is None:
                lines.append(f"| {mk} | NA | NA | NA |")
                continue
            beats_a = "Yes" if (a_median is not None and med < a_median) else "No"
            under_2 = "Yes" if med < 2.0 else "No"
            lines.append(f"| {mk} | {med:.2f} | {beats_a} | {under_2} |")
        lines.append("")

    (output_base / "grid_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_phase(
    script_path: Path,
    data_root: Path,
    speakers: list[str],
    output_dir: Path,
    phase_label: str,
    *,
    bandpass_low: float,
    bandpass_high: float,
    tau2_guided_radius_ms: float = 2.0,
    extra_args: list[str] | None = None,
) -> dict:
    """Run one phase for all speakers and return aggregated info."""
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict | None] = {}
    for sp in speakers:
        ret = run_single_speaker(
            script_path,
            data_root,
            sp,
            output_dir,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
            tau2_guided_radius_ms=tau2_guided_radius_ms,
            extra_args=extra_args,
        )
        if ret != 0:
            logger.warning("Run failed for %s (%s)", sp, phase_label)
        summary_path = output_dir / sp / "summary.json"
        summaries[sp] = load_summary(summary_path)

    write_phase_summary(output_dir, summaries, phase_label)
    write_code_state(output_dir, script_path)

    # Save run config
    run_config = {
        "phase": phase_label,
        "bandpass_low": bandpass_low,
        "bandpass_high": bandpass_high,
        "tau2_guided_radius_ms": tau2_guided_radius_ms,
        "speakers": speakers,
        "extra_args": extra_args,
        "timestamp": datetime.now().isoformat(),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    # Aggregate method errors
    method_keys = ["A_mic_mic", "B_omp", "C_theta_fusion", "D_msnf_2", "E_msnf_3", "F_msnf_4"]
    method_medians = {}
    for mk in method_keys:
        errors = []
        for s in summaries.values():
            if s is None:
                continue
            e = s.get("result", {}).get(mk, {}).get("theta_error_median_deg")
            if e is not None:
                errors.append(e)
        method_medians[mk] = float(np.median(errors)) if errors else None

    go_info = check_phase0_go(summaries)

    return {
        "run_dir": str(output_dir),
        "phase0_go": go_info,
        "method_medians": method_medians,
        "summaries": {sp: (s is not None) for sp, s in summaries.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid runner for MSNF DoA comparison"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Root directory for dataset"
    )
    parser.add_argument(
        "--speakers",
        type=str,
        nargs="+",
        default=["18-0.1V", "19-0.1V", "20-0.1V", "21-0.1V", "22-0.1V"],
    )
    parser.add_argument(
        "--output_base", type=str, required=True, help="Base output directory"
    )
    parser.add_argument(
        "--phases",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Which phases to run (0=diagnostic, 1=band0, 2=band500-2000, 3=radius sweep)",
    )
    parser.add_argument(
        "--phase3_radii",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 1.0, 2.0],
        help="guided_radius values for Phase 3 sweep",
    )
    parser.add_argument("--smoke_only", action="store_true", help="Run smoke test only (speaker 20, 1 segment)")
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
    speakers = list(args.speakers)
    script_path = Path(__file__).parent / "multi_sensor_fusion_doa.py"

    if args.debug:
        speakers = speakers[:2]

    if args.smoke_only:
        logger.info("=== SMOKE TEST ===")
        smoke_dir = output_base / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        ret = run_single_speaker(
            script_path,
            data_root,
            "20-0.1V",
            smoke_dir,
            bandpass_low=0,
            bandpass_high=0,
            tau2_guided_radius_ms=2.0,
            extra_args=["--segment_mode", "fixed", "--n_segments", "1"],
        )
        if ret == 0:
            s = load_summary(smoke_dir / "20-0.1V" / "summary.json")
            if s:
                logger.info("Smoke test passed. Phase0 tau2_go=%s", s.get("phase0_diagnostic", {}).get("tau2_go"))
                for mk in ["A_mic_mic", "D_msnf_2"]:
                    err = s.get("result", {}).get(mk, {}).get("theta_error_median_deg")
                    logger.info("  %s: theta_error = %s", mk, err)
        else:
            logger.error("Smoke test failed with return code %d", ret)
        return

    phases_to_run = set(args.phases)
    phase_results: dict[str, dict] = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Phase 0 & 1: Full-band (band=0)
    if 0 in phases_to_run or 1 in phases_to_run:
        label = f"msnf_band0_{timestamp}"
        logger.info("=== Phase 0+1: Full-band MSNF ===")
        result = run_phase(
            script_path,
            data_root,
            speakers,
            output_base / label,
            "Phase 0+1: band=0",
            bandpass_low=0,
            bandpass_high=0,
        )
        phase_results["Phase_0_1_band0"] = result

        go = result["phase0_go"]
        logger.info(
            "Phase 0 Go/No-Go: tau2=%s (%d/%d), tau3=%s (%d/%d)",
            go["tau2_go"],
            go["tau2_go_count"],
            go["n_speakers"],
            go["tau3_go"],
            go["tau3_go_count"],
            go["n_speakers"],
        )
        if not go["overall_go"]:
            logger.warning(
                "Phase 0 NO-GO: tau2 PSR>3dB in only %d/%d speakers. "
                "MSNF results may not be reliable.",
                go["tau2_go_count"],
                go["n_speakers"],
            )

    # Phase 2: Bandpass (500-2000 Hz)
    if 2 in phases_to_run:
        label = f"msnf_band500_2000_{timestamp}"
        logger.info("=== Phase 2: Bandpass MSNF (500-2000 Hz) ===")
        result = run_phase(
            script_path,
            data_root,
            speakers,
            output_base / label,
            "Phase 2: band=500-2000",
            bandpass_low=500,
            bandpass_high=2000,
        )
        phase_results["Phase_2_band500_2000"] = result

    # Phase 3: guided_radius sweep
    if 3 in phases_to_run:
        logger.info("=== Phase 3: guided_radius sweep ===")
        for radius in args.phase3_radii:
            label = f"msnf_band0_radius{radius}_{timestamp}"
            logger.info("Phase 3: tau2_guided_radius=%.1f ms", radius)
            result = run_phase(
                script_path,
                data_root,
                speakers,
                output_base / label,
                f"Phase 3: radius={radius}ms",
                bandpass_low=0,
                bandpass_high=0,
                tau2_guided_radius_ms=radius,
            )
            phase_results[f"Phase_3_radius{radius}"] = result

    # Write overall grid report
    write_grid_report(output_base, phase_results)

    grid_summary = {
        "generated": datetime.now().isoformat(),
        "phases": {k: {kk: vv for kk, vv in v.items() if kk != "summaries"} for k, v in phase_results.items()},
    }
    (output_base / "grid_summary.json").write_text(
        json.dumps(grid_summary, indent=2), encoding="utf-8"
    )

    logger.info("Grid run complete. Results in: %s", output_base)


if __name__ == "__main__":
    main()
