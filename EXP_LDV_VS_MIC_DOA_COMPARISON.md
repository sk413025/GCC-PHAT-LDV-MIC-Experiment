# Experiment: LDV-vs-Mic DoA Comparison (Guided Search)

**Status**: Plan (to be executed)
**Branch**: `exp/ldv-vs-mic-doa-comparison`
**Created**: 2026-02-11

---

## Executive Summary

Validate that LDV, when OMP-aligned to a reference microphone and guided by chirp truth calibration, achieves comparable or superior DoA performance to baseline Mic-Mic reference. This establishes the physical validity of using LDV as a virtual microphone substitute in guided acoustic localization pipelines.

---

## Experiment Context

### Background

- **Stage 1-2 (Prior)**: LDV→Mic OMP alignment confirmed (commit 62a51617); tau→0, PSR +6–8 dB
- **Stage 4-C (Prior)**: Mic-Mic GCC-PHAT with chirp truth guidance achieved 4/5 PASS (commit f9a30d7)
- **Limitation**: Have not yet directly compared LDV-Mic vs Mic-Mic DoA under identical guidance conditions

### Motivation

- Previous work validated OMP alignment (Stage 1-2) and Mic-Mic guided search (Stage 4-C) separately
- A direct comparison is needed to answer: **Can LDV replace MicL in a dual-mic setup without sacrificing DoA accuracy?**
- This is critical for sensor fusion and hybrid localization systems

### Purpose

Run a 2×3 matrix of DoA estimations:
1. **LDV-MicL** (OMP-aligned) with chirp truth guidance
2. **MicL-MicR** (baseline) with chirp truth guidance
3. **LDV-MicL** (OMP-aligned) with geometry guidance (negative control)

Measure:
- DoA error distribution and pass rate vs. chirp truth
- SNR/PSR stability across all three configurations
- Failure mode analysis (which speakers/windows fail and why)

### Expected Outcomes

- **LDV-MicL + chirp truth**: ≥4/5 PASS (matches or exceeds Mic-Mic)
- **MicL-MicR + chirp truth**: 4/5 PASS (baseline, from f9a30d7)
- **LDV-MicL + geometry**: ≤3/5 PASS (negative control; geometry insufficient)

**Physical intuition**: OMP alignment should preserve the coherence structure that chirp truth guidance exploits; therefore LDV-MicL should not underperform vs. Mic-MicL.

---

## Experimental Design

### Data and Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Speakers** | 18-0.1V, 19-0.1V, 20-0.1V, 21-0.1V, 22-0.1V | Real dataset subset (5 speakers) |
| **Signal pairs** | LDV-MicL, MicL-MicR, LDV-MicL | Compare all 3 configurations |
| **Segment mode** | scan (hop=1s) | Multi-segment analysis (from f9a30d7) |
| **n_segments** | 5 | Median aggregation for robustness |
| **analysis_slice_sec** | 5 | Window duration |
| **eval_window_sec** | 5 | Evaluation window |
| **scan_range** | 100–600 s | Standard scan window |
| **scan_tau_err_max_ms** | 2.0 | Loose tolerance (primary); 0.3 (secondary) |
| **bandpass** | None, 500–2000 Hz | Test both (500–2000 favored from f9a30d7) |
| **gcc_guided_peak_radius_ms** | 0.3 | Guided search tolerance |
| **ldv_prealign** | gcc_phat | OMP alignment via GCC-PHAT baseline |
| **truth_ref** | chirp (from validation) | Chirp truth guide (5s scan-derived) |

### Test Matrix

**Phase 1: Core Comparison (2×3 + 2 controls)**

```
Config Matrix:
┌─ Config A: LDV-MicL + chirp truth + tau2 + no bandpass
├─ Config B: LDV-MicL + chirp truth + tau2 + bandpass(500,2000)
├─ Config C: MicL-MicR + chirp truth + tau2 + bandpass(500,2000)  [BASELINE]
├─ Config D: LDV-MicL + geometry + tau2 + bandpass(500,2000)  [NEG CONTROL]
├─ Smoke (LDV-MicL + chirp, small data)
└─ Guardrail (zero segments expected)
```

**Phase 2: Sensitivity (if time permits)**

- tau_err_max_ms = 0.3 (tight gating)
- Alternative bandpass ranges (1000–4000 Hz)

### Scripts and Code

**New or Modified Files**:

1. **`worktree/exp-ldv-vs-mic-doa-comparison/scripts/stage4_doa_ldv_vs_mic_comparison.py`**
   - **Purpose**: Extend existing stage4_doa_validation.py to accept dual signal pairs (LDV-MicL, MicL-MicR)
   - **Key changes**:
     - Add `--signal_pair` argument: `ldv_micl`, `micl_micr`, or both
     - Load LDV waveform via OMP alignment (reuse Stage 1-2 alignment code)
     - Reuse GCC-PHAT + MUSIC + CC/NCC estimation code
     - Support both chirp truth and geometry truth guides
   - **Reference commits**:
     - Stage 4-C code: `a53c778412c7af34cc0e56b3553c6b81cc37c9ac` (exp-ldv-perfect-geometry-cloud)
     - Stage 1-2 OMP: commit 62a51617 (exp-ldv-perfect-geometry-cloud)

2. **`worktree/exp-ldv-vs-mic-doa-comparison/scripts/run_ldv_vs_mic_grid.py`**
   - **Purpose**: Grid runner (analog to f9a30d7's grid logic)
   - **Key features**:
     - 2×3 grid: (LDV-MicL, MicL-MicR) × (tau2, tau0p3) × (no bandpass, 500–2000)
     - Smoke and guardrail tests
     - Aggregated grid summary and per-speaker tables

3. **`worktree/exp-ldv-vs-mic-doa-comparison/scripts/analyze_ldv_vs_mic_results.py`**
   - **Purpose**: Comparative analysis across all 3 signal pairs
   - **Outputs**:
     - Per-pair DoA error histograms
     - Pass/fail comparison table
     - PSR/SNR stability plots
     - Failure mode breakdown

### Test Strategy

**Smoke Test** (10 min):
```bash
python scripts/stage4_doa_ldv_vs_mic_comparison.py \
  --data_root dataset/GCC-PHAT-LDV-MIC-Experiment \
  --speaker 20-0.1V \
  --signal_pair ldv_micl \
  --output_dir results/smoke_ldv_vs_mic_20260211 \
  --segment_mode fixed --n_segments 1 --analysis_slice_sec 2 \
  --eval_window_sec 1
```

**Functional Test** (positive path):
```bash
python scripts/stage4_doa_ldv_vs_mic_comparison.py \
  --data_root dataset/GCC-PHAT-LDV-MIC-Experiment \
  --speaker 21-0.1V \
  --signal_pair ldv_micl \
  --output_dir results/functional_ldv_vs_mic_20260211 \
  --segment_mode scan --n_segments 5 \
  --scan_start_sec 100 --scan_end_sec 200 --scan_hop_sec 1 \
  --scan_tau_err_max_ms 2.0 \
  --ldv_prealign gcc_phat \
  --truth_tau_ms -8.95 --truth_theta_deg -8.96 --truth_label "chirp 21 mic"
```

**Guardrail Test** (expected fail):
```bash
python scripts/stage4_doa_ldv_vs_mic_comparison.py \
  ... (same as functional but with scan_psr_min_db=20, scan_tau_err_max_ms=0.01)
```

---

## Execution Plan

### Phase 1: Code Setup (Day 1)
- [ ] Create worktree: `worktree/exp-ldv-vs-mic-doa-comparison/`
- [ ] Copy and adapt Stage 4-C script (`scripts/stage4_doa_ldv_vs_mic_comparison.py`)
- [ ] Implement `--signal_pair` logic (load LDV via OMP, compute tau/theta for both pairs)
- [ ] Implement grid runner and analysis script
- [ ] Smoke test (confirm code path works)
- [ ] **Commit**: Planning + code scaffold

### Phase 2: Grid Execution (Day 2–3)
- [ ] Run full 2×3 grid (LDV-MicL + chirp, MicL-MicR + chirp, LDV-MicL + geometry)
- [ ] Functional test (positive path for each pair)
- [ ] Guardrail test (expected fail)
- [ ] Aggregate results into grid summary
- [ ] **Commit**: Grid results + analysis

### Phase 3: Analysis & Documentation (Day 4)
- [ ] Compare DoA errors across all 3 pairs
- [ ] Extract failure modes (which speakers fail and why)
- [ ] Cross-reference with prior work (f9a30d7, 62a51617, a53c778)
- [ ] Formulate design principles (when/why to use LDV vs. Mic)
- [ ] **Commit**: Final analysis + principles

---

## Cross-Experiment References

This experiment builds on and compares against:

1. **Commit 62a51617f62e08ff93f53d2be67ecd548b51cf30** (Stage 1-2 OMP alignment)
   - OMP successfully aligns LDV to MicL (tau→0)
   - Reference for LDV loading and tau correction

2. **Commit f9a30d7eeae322a57d531003edf0c9c030fd9f87** (Stage 4-C Mic-Mic baseline)
   - Mic-MicL GCC-PHAT achieves 4/5 PASS with chirp truth guidance
   - Baseline for pass rate comparison

3. **Commit a53c778412c7af34cc0e56b3553c6b81cc37c9ac** (Stage 4-C local scan)
   - Chirp truth derivation for speech (5s scan windows)
   - Reference for scanning and truth-ref generation

4. **Commit c36dcebfd514bae88ad8a4e464505f49c94d2cb4** (Chirp reference validation)
   - Guided peak search methodology
   - Reference for guided search implementation

---

## Expected Deliverables

### Artifacts

```
results/
├── ldv_vs_mic_grid_20260211_XXXXXX/
│   ├── grid_summary.md
│   ├── grid_summary.json
│   ├── grid_report.md
│   ├── config_matrix.json
│   ├── ldv_micl_chirp_tau2_band0/
│   │   ├── {18,19,20,21,22}-0.1V/
│   │   │   ├── summary.json
│   │   │   └── run.log
│   │   ├── summary_table.md
│   │   ├── run_config.json
│   │   ├── subset_manifest.json
│   │   └── code_state.json
│   ├── ldv_micl_chirp_tau2_band500_2000/
│   │   └── ... (same structure)
│   ├── micl_micr_chirp_tau2_band500_2000/  [BASELINE]
│   │   └── ... (same structure)
│   ├── ldv_micl_geometry_tau2_band500_2000/  [NEG CONTROL]
│   │   └── ... (same structure)
│   ├── ldv_micl_chirp_tau0p3_band500_2000/  [SECONDARY]
│   │   └── ... (same structure)
│   ├── smoke_ldv_vs_mic/
│   │   └── run.log
│   └── guardrail_ldv_vs_mic/
│       └── run.log
```

### Analysis Outputs

- **grid_report.md**: Side-by-side comparison of pass rates, DoA errors, PSR/SNR
- **failure_mode_analysis.json**: Per-speaker failure breakdown
- **ldv_vs_mic_doa_comparison_plots.pdf**: Error histograms, PSR/SNR trends
- **principles_and_guidelines.md**: Design rules for LDV-vs-Mic selection

---

## Success Criteria

### Quantitative

- [ ] LDV-MicL + chirp truth: ≥4/5 PASS (≤1.0° avg error)
- [ ] MicL-MicR + chirp truth: 4/5 PASS (validate baseline)
- [ ] LDV-MicL + geometry: ≤3/5 PASS (negative control holds)
- [ ] Smoke test completes without error
- [ ] Functional test positive path all 3 pairs
- [ ] Guardrail test exhibits expected zero-segment behavior

### Qualitative

- [ ] Cross-experiment analysis: Explain success/failure BECAUSE of physical constraints
- [ ] Extracted principles: Clear design rules for future LDV-Mic pipelines
- [ ] Reproducibility: All commands documented; all data fingerprints recorded

---

## Failure Modes & Recovery

| Issue | Detection | Recovery |
|-------|-----------|----------|
| LDV loading fails | Script error on speaker #1 | Verify OMP alignment data exists; rerun Stage 1-2 if needed |
| Chirp truth unavailable | `truth_ref` dict missing speaker | Regenerate chirp truth using stage4_doa_validation.py |
| PSR too low for guided search | 0 segments after scan | Loosen scan_tau_err_max_ms or increase scan_psr_min_db cutoff |
| Inconsistent pass rates | Compare to f9a30d7 baseline | Check bandpass filter application; verify tau gating |
| MicL-MicR different from f9a30d7 | Baseline mismatch | Verify identical speakers, scans ranges, data files |

---

## Notes & Constraints

- **Data**: Use existing real dataset (GCC-PHAT-LDV-MIC-Experiment, 5 speakers)
- **Environment**: Python 3.10, numpy 2.2.6, scipy 1.15.3 (system Python)
- **Reproductibility**: All commands, data roots, fingerprints will be documented
- **Atomicity**: Code + results committed together per AGENTS.md
- **Language**: All code, docs, and logs in English

---

## Timeline

- **Phase 1 (Code)**: ~4 hours
- **Phase 2 (Execution)**: ~6–8 hours (depending on compute)
- **Phase 3 (Analysis)**: ~3–4 hours
- **Total**: ~13–16 hours of work across 3–4 days

---

## References

- **AGENTS.md**: Commit/testing/documentation standards
- **Commit 62a51617**: Stage 1-2 validation (OMP alignment)
- **Commit f9a30d7**: Stage 4-C grid (Mic-Mic baseline)
- **Commit a53c778**: Stage 4-C local scan (truth-ref generation)
- **Commit c36dcebfd514**: Chirp reference validation (guided search)

---

## Next Steps After Experiment

If LDV-MicL + chirp truth achieves ≥4/5 PASS:
- Propose LDV as a virtual microphone in hybrid sensor fusion
- Expand to multi-speaker scenarios
- Investigate real-time guided search for tracking

If LDV-MicL underperforms:
- Analyze coherence differences between LDV-MicL and Mic-MicL
- Explore alternative alignment methods (beyond OMP)
- Characterize failure modes in terms of reverberation patterns

---

**Document version**: 2026-02-11 (plan phase)
**Last updated by**: Jenner (automated plan)
