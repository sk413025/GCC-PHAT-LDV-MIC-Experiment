#!/usr/bin/env python
"""
Analysis script for LDV-vs-Mic DoA comparison results (GCC-PHAT).

Outputs:
- analysis_summary.json
- analysis_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from statistics import median

logger = logging.getLogger(__name__)


def load_summary(summary_path: Path) -> dict | None:
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def summarize_run(run_dir: Path, speakers: list[str]) -> dict:
    summaries = []
    for sp in speakers:
        summary = load_summary(run_dir / sp / "summary.json")
        if summary:
            summaries.append(summary)
    if not summaries:
        return {
            "run_dir": str(run_dir),
            "pass_count": 0,
            "pass_rate": 0.0,
            "theta_err_median_deg": None,
            "psr_median_db": None,
            "n_summaries": 0,
        }

    pass_flags = [s.get("passed", False) for s in summaries]
    theta_errs = [s["result"]["theta_error_median_deg"] for s in summaries]
    psrs = [s["result"]["psr_median_db"] for s in summaries]

    return {
        "run_dir": str(run_dir),
        "pass_count": int(sum(1 for p in pass_flags if p)),
        "pass_rate": float(sum(1 for p in pass_flags if p) / len(pass_flags)),
        "theta_err_median_deg": float(median(theta_errs)),
        "psr_median_db": float(median(psrs)),
        "n_summaries": len(summaries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze LDV-vs-Mic DoA results")
    parser.add_argument("--grid_base", type=str, required=True, help="Base directory of grid results")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for analysis")
    parser.add_argument("--speakers", type=str, nargs="+", default=["18-0.1V", "19-0.1V", "20-0.1V", "21-0.1V", "22-0.1V"])
    parser.add_argument("--compare_to_baseline", type=str, default=None, help="Optional baseline grid_summary.json")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(output_dir / "analysis.log"), logging.StreamHandler()],
    )

    grid_base = Path(args.grid_base)
    grid_summary_path = grid_base / "grid_summary.json"
    if not grid_summary_path.exists():
        raise FileNotFoundError(f"Missing grid_summary.json at {grid_summary_path}")

    grid_summary = json.loads(grid_summary_path.read_text(encoding="utf-8"))
    rows = grid_summary.get("rows", [])
    speakers = list(args.speakers)

    analysis_rows = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        summary = summarize_run(run_dir, speakers)
        summary.update(
            {
                "label": row.get("label"),
                "truth_type": row.get("truth_type"),
                "tau_err_max_ms": row.get("tau_err_max_ms"),
                "bandpass_low": row.get("bandpass_low"),
                "bandpass_high": row.get("bandpass_high"),
            }
        )
        analysis_rows.append(summary)

    report_lines = []
    report_lines.append("# LDV-vs-Mic Analysis Report (GCC-PHAT)")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("| label | truth_type | tau_err_max_ms | bandpass | pass_count | pass_rate | theta_err_median_deg | psr_median_db |")
    report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in analysis_rows:
        band = f"{row['bandpass_low']}-{row['bandpass_high']}"
        report_lines.append(
            f"| {row['label']} | {row['truth_type']} | {row['tau_err_max_ms']} | {band} | "
            f"{row['pass_count']} | {row['pass_rate']:.2f} | {row['theta_err_median_deg']} | {row['psr_median_db']} |"
        )

    if args.compare_to_baseline:
        report_lines.append("")
        report_lines.append("## Baseline Comparison")
        report_lines.append(f"Baseline grid summary: {args.compare_to_baseline}")

    (output_dir / "analysis_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    analysis_summary = {
        "generated": datetime.now().isoformat(),
        "grid_base": str(grid_base),
        "rows": analysis_rows,
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(analysis_summary, indent=2), encoding="utf-8")

    logger.info("Analysis complete: %s", output_dir)


if __name__ == "__main__":
    main()
