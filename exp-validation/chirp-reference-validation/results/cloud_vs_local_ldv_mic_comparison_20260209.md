# Cloud vs Local LDV-MIC Comparison (Speech +0.8 Focus)

Date: 2026-02-09

## Scope
This report compares:
- Cloud commits on `exp-ldv-perfect-geometry-clean`:
  - 62a51617f62e08ff93f53d2be67ecd548b51cf30
  - f4c11ccfa2ec81e2c7eafef95639d67757adc9d7
  - 754a7e8f9e104e1f47ffa2dddbcb50086d7bf64f
  - 1edbd03e4c971212a28c13a5c027f27269247e00
  - 2d9bb654c96c75fbd4706f1f718b721b2ab8a478
  - 46eb1d25fded2b140582b1fcd42a0e25cf653edb
- Local chirp reference validation run:
  - `worktree/exp-chirp-reference-validation/results/chirp_ref_pipeline_20260207_173915`
- Local rerun of the cloud Stage 4-A pipeline using the exact cloud code:
  - `worktree/exp-ldv-perfect-geometry-cloud/exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation_cloud_rerun_20260209_000107`

Primary question: Does Speech +0.8 (18-0.1V) succeed in Stage 4-A (speech + geometry truth)?

## Executive Summary
- All six cloud commits show **FAIL** for Speech +0.8 (18-0.1V) in Stage 4-A (GCC-PHAT, OMP_LDV).
- The local rerun using the exact cloud code also shows **FAIL** for Speech +0.8.
- The local chirp reference validation run also **fails** overall (Phase 4 = VALIDATION_NEEDS_WORK), and chirp-based LDV delays produce larger residuals than legacy delays at +0.8.
- Conclusion: **Speech +0.8 does not succeed** in any of the listed cloud commits or in the local validation run.

## Plain-Language Description of Stage 4-A (Speech + Geometry Truth)
Stage 4-A answers one question: **If we replace a microphone with an OMP-aligned LDV signal, do we recover the correct DoA/TDoA for real speech?**

What it does, in simple terms:
- **Input**: Speech recordings (18-22) with three channels: LDV, Left Mic, Right Mic.
- **Reference truth**: Geometry-based expected TDoA/DoA for each speaker position.
- **Processing**:
  - Use OMP to align LDV so it should behave like the Left Mic.
  - Compute DoA/TDoA using multiple methods (GCC-PHAT, CC, NCC, MUSIC).
  - Use a fixed set of short time windows (1s) at spaced centers (e.g., 100s, 150s, 200s, 250s, 300s).
- **Pass criteria (per method)**:
  - OMP must be better than Raw LDV,
  - and the angle error must be small.
- **Stage result**: It passes only if the selected method meets those criteria.

Why this matters:
- If Stage 4-A succeeds, OMP-aligned LDV can effectively replace a microphone for speech TDoA/DoA.
- If Stage 4-A fails (as it does here), then **OMP alignment is not enough** for speech to recover the correct geometry-based delay/angle.

## Experiment Coverage (Cloud vs Local)

This section lists experiments and whether they were run in the cloud branch vs the local chirp-reference validation run.

| Experiment | Cloud Run? | Cloud Result | Local Run? | Local Result | Notes |
|---|---|---|---|---|---|
| Stage 4-A: Speech + Geometry truth | Yes | **FAIL (0/5)** | Yes (cloud-code rerun) | **FAIL (0/5)** | Speech 18–22 all fail; only speaker 20 may pass GCC-PHAT but overall fails. |
| Stage 1: Energy Reduction (OMP vs Random) | Yes | **PASS (5/5)** | No | N/A | OMP alignment improves energy reduction on speech. |
| Stage 2: Target Similarity (LDV → Mic) | Yes | **PASS (5/5)** | No | N/A | OMP-aligned LDV becomes closer to target mic. |
| Stage 3: Single-Segment TDoA (report baseline) | Yes | **PARTIAL (2/5)** | No | N/A | Pass rate depends on baseline definition. |
| Stage 3: Single-Segment TDoA (windowed, PSR>=10) | Yes | **PARTIAL (3/5)** | No | N/A | More stable baseline but still partial. |
| Stage 3: Multi-Segment TDoA (report, offset=100s) | Yes | **PARTIAL (1/5)** | No | N/A | Low pass rate; baseline sensitivity. |
| Stage 3: Multi-Segment TDoA (windowed, PSR>=10) | Yes | **PARTIAL (2/5)** | No | N/A | Still partial. |
| Stage 4-B: Chirp truth-ref (scan + prealign + PSR) | Yes | **PASS (2/2)** | No | N/A | `23-chirp(-0.8m)` and `24-chirp(-0.4m)` both pass GCC-PHAT. |
| Stage 4-B: Chirp truth-ref (prealign only, no scan) | Yes | **PASS (2/2)** | No | N/A | GCC-PHAT passes for both chirp positions. |
| Stage 4-B: Chirp truth-ref (no prealign) | Yes | **FAIL (2/2)** | No | N/A | Without prealign, GCC-PHAT fails. |
| Stage 4-C: Speech + Chirp truth-ref (scan, 1s) | Yes | **PARTIAL (1/2)** | No | N/A | `21-0.1V` pass, `22-0.1V` fail. |
| Stage 4-C: Speech + Chirp truth-ref (guided, 5s) | Yes | **PARTIAL (1/2)** | No | N/A | `21-0.1V` pass, `22-0.1V` fail. |
| Chirp Reference Validation (Step 0 + Phases 1–4) | No | N/A | Yes | **VALIDATION_NEEDS_WORK** | Different pipeline from Stage 1–4. |
| LDV Delay Re-evaluation (chirp vs old delay) | No | N/A | Yes | **Chirp delay worse than old delay** | +0.8 residuals are larger with chirp delay. |
| Negative Position Diagnosis (-0.4, -0.8) | No | N/A | Yes | **Gate failures dominate** | Most chirp events fail quality gates. |

Key takeaway: all non-Stage-4A experiments are **cloud-only** except the local chirp reference pipeline and diagnostics, which do not have cloud equivalents.

## Cloud Commit Results (Stage 4-A, Speech +0.8)
Source file (per commit):
`exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation/18-0.1V/summary.json`

| Commit | GCC-PHAT Pass | tau_true_ms | tau_omp_ms | theta_err_deg | Notes |
|---|---|---:|---:|---:|---|
| 62a51617 | False | 1.4504 | -0.0091 | 20.94 | OMP LDV near 0 ms, far from geometry truth |
| f4c11cc | False | 1.4504 | -0.0091 | 20.94 | Same as 62a51617 |
| 754a7e8 | False | 1.4504 | -0.0091 | 20.94 | Same as 62a51617 |
| 1edbd03 | False | 1.4504 | -0.0091 | 20.94 | Same as 62a51617 |
| 2d9bb65 | False | 1.4504 | -1.6667 | 44.92 | Legacy baseline locked to -81 samples sidelobe |
| 46eb1d2 | False | 1.4504 | -1.6667 | 44.92 | Same as 2d9bb65 |

Notes:
- For commits 2d9bb65 and 46eb1d2, the summary JSON is stored in Git LFS; the resolved content still reports FAIL.

## Local Rerun Using Cloud Code (Stage 4-A)
Rerun outputs:
`worktree/exp-ldv-perfect-geometry-cloud/exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation_cloud_rerun_20260209_000107/`

GCC-PHAT pass status (OMP_LDV):
- 18-0.1V (+0.8): FAIL
- 19-0.1V (+0.4): FAIL
- 20-0.1V (0.0): PASS
- 21-0.1V (-0.4): FAIL
- 22-0.1V (-0.8): FAIL

This matches the cloud Stage 4-A summary: Speech +0.8 fails.

## Local Chirp Reference Validation (Run 20260207_173915)
Source:
`worktree/exp-chirp-reference-validation/results/ldv_delay_reeval/run_20260207_173915/ldv_delay_reeval_report.json`

Key +0.8 (18-0.1V) residuals (LDV vs mic):
- LDV-LEFT: chirp abs_mean_ms = 3.7216 vs old_4.3ms abs_mean_ms = 0.2895
- LDV-RIGHT: chirp abs_mean_ms = 2.2461 vs old_3.8ms abs_mean_ms = 0.7188

Phase 4 decision: VALIDATION_NEEDS_WORK

Interpretation:
- Chirp-derived delays do not improve speech LDV-MIC alignment at +0.8.
- Legacy speech-derived delays (3.8-4.8 ms) yield smaller residuals for speech.

## Conclusion
Across all six cloud commits and the local rerun, **Speech +0.8 (18-0.1V) fails in Stage 4-A (speech + geometry truth)**. The local chirp reference validation also fails to produce a reliable speech LDV-MIC alignment at +0.8. Therefore, the evidence consistently indicates **no successful speech +0.8 case** in the specified cloud commits or in the local validation run.
