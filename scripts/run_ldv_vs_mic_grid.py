#!/usr/bin/env python
"""
Grid runner for LDV-vs-Mic DoA comparison experiment

Executes a 2×3 (or more) configuration matrix:
- Signal pairs: (ldv_micl, micl_micr)
- tau_err_max_ms: {2.0, 0.3}
- bandpass: {(0,0), (500,2000)}
- Optionally: ldv_micl + geometry (negative control)

Produces:
- Per-config run directories with per-speaker results
- Aggregated grid_summary.md and grid_summary.json
- Comparative analysis plots and tables

Reference: commit f9a30d7 (Stage 4-C grid)
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from itertools import product

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Grid runner for LDV-vs-Mic DoA comparison"
    )
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory for GCC-PHAT-LDV-MIC-Experiment")
    parser.add_argument("--speakers", type=str, nargs="+",
                       default=["18-0.1V", "19-0.1V", "20-0.1V", "21-0.1V", "22-0.1V"],
                       help="Speaker IDs to process")
    parser.add_argument("--output_base", type=str, required=True,
                       help="Base output directory for all grid runs")
    parser.add_argument("--chirp_truth_file", type=str, default=None,
                       help="Path to chirp truth JSON (optional, for validation)")
    parser.add_argument("--skip_smoke", action="store_true",
                       help="Skip smoke test")
    parser.add_argument("--skip_guardrail", action="store_true",
                       help="Skip guardrail test")
    parser.add_argument("--debug", action="store_true",
                       help="Run in debug mode (minimal grid)")

    args = parser.parse_args()

    # Setup logging
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    log_file = output_base / "grid_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger.info("=" * 80)
    logger.info("LDV-vs-Mic DoA Comparison - Grid Runner (Placeholder - Implementation in Phase 1)")
    logger.info("=" * 80)
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Speakers: {args.speakers}")
    logger.info(f"Output base: {args.output_base}")

    # Define grid configurations
    configs = [
        {"signal_pair": "ldv_micl", "tau_err_max": 2.0, "bandpass": (0, 0), "label": "ldv_micl_chirp_tau2_band0"},
        {"signal_pair": "ldv_micl", "tau_err_max": 2.0, "bandpass": (500, 2000), "label": "ldv_micl_chirp_tau2_band500_2000"},
        {"signal_pair": "micl_micr", "tau_err_max": 2.0, "bandpass": (500, 2000), "label": "micl_micr_chirp_tau2_band500_2000"},
        {"signal_pair": "ldv_micl", "tau_err_max": 2.0, "bandpass": (500, 2000), "label": "ldv_micl_geometry_tau2_band500_2000", "use_geometry": True},
        {"signal_pair": "ldv_micl", "tau_err_max": 0.3, "bandpass": (500, 2000), "label": "ldv_micl_chirp_tau0p3_band500_2000"},
    ]

    if args.debug:
        configs = configs[:1]  # Debug: single config

    logger.info(f"Grid size: {len(configs)} configurations × {len(args.speakers)} speakers = {len(configs) * len(args.speakers)} runs")
    logger.info("=" * 80)

    # Placeholder: log what would be executed
    logger.info("PLACEHOLDER: Following configurations will be executed in Phase 2:")
    for cfg in configs:
        logger.info(f"  - {cfg['label']}")
        logger.info(f"      signal_pair={cfg['signal_pair']}, tau_err_max={cfg['tau_err_max']}, bandpass={cfg['bandpass']}")

    # Smoke test (if enabled)
    if not args.skip_smoke:
        logger.info("=" * 80)
        logger.info("SMOKE TEST (Placeholder)")
        logger.info("Would run: single speaker (20-0.1V), fixed segment, 2s window")

    # Guardrail test (if enabled)
    if not args.skip_guardrail:
        logger.info("=" * 80)
        logger.info("GUARDRAIL TEST (Placeholder)")
        logger.info("Would run: high PSR gating, tight tau window (expected zero segments)")

    logger.info("=" * 80)
    logger.info("Phase 1 placeholder completed. Full grid execution in Phase 2.")

if __name__ == "__main__":
    main()
