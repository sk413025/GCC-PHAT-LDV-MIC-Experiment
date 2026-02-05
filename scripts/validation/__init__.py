"""
LDV Reorientation - Validation Test Suite

This module contains the systematic validation pipeline for τ stability
diagnosis and Stage 3 cross-mic TDoA validation.

Phases:
    1. phase1_tau_stability: Diagnose Speech τ stability across parameters
    2. phase2_guided_search: Implement and validate guided peak search
    3. phase3_stage3_revalidation: Re-run Stage 3 with improved methods
    4. phase4_final_validation: Generate final validation summary

Usage:
    # From command line:
    python -m scripts.validation.phase1_tau_stability
    python -m scripts.validation.phase2_guided_search
    python -m scripts.validation.phase3_stage3_revalidation
    python -m scripts.validation.phase4_final_validation

    # Or use the PowerShell runner:
    .\\run_validation_phases.ps1

Results are saved to:
    results/phase1_tau_stability/run_YYYYMMDD_HHMMSS/
    results/phase2_guided_search/run_YYYYMMDD_HHMMSS/
    results/phase3_stage3_revalidation/run_YYYYMMDD_HHMMSS/
    results/phase4_final_validation/run_YYYYMMDD_HHMMSS/
"""

__version__ = "1.0.0"
__author__ = "SBP Lab"

from pathlib import Path

# Package paths
PACKAGE_DIR = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
DATASET_ROOT = PROJECT_ROOT / "dataset" / "GCC-PHAT-LDV-MIC-Experiment"

# Phase result directories
PHASE1_RESULTS = RESULTS_ROOT / "phase1_tau_stability"
PHASE2_RESULTS = RESULTS_ROOT / "phase2_guided_search"
PHASE3_RESULTS = RESULTS_ROOT / "phase3_stage3_revalidation"
PHASE4_RESULTS = RESULTS_ROOT / "phase4_final_validation"
