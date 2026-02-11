# LDV-vs-Mic DoA Comparison Experiment

**Branch**: `exp/ldv-vs-mic-doa-comparison`
**Worktree**: `worktree/exp-ldv-vs-mic-doa-comparison/`
**Status**: Planning & Code Scaffolding Complete (Phase 1 Ready)
**Created**: 2026-02-11

---

## Experiment Overview

This experiment validates whether **LDV (Laser Vibrometer), when OMP-aligned to a reference microphone and guided by chirp truth calibration, achieves comparable or superior DoA performance to baseline Mic-Mic reference systems**.

### Key Questions
1. ✅ Can LDV→Mic OMP alignment work? (YES - commit 62a51617)
2. ✅ Can Mic-Mic do DoA with chirp guidance? (YES - commit f9a30d7, 4/5 PASS)
3. **❓ Can LDV-Mic compete with Mic-Mic DoA?** ← THIS EXPERIMENT

### Physical Significance
If the answer is "yes," this enables:
- **Virtual microphone arrays** (hybrid sensor fusion)
- **LDV-as-Mic substitution** in acoustic localization
- **Multi-modal sensor integration** (optical + acoustic)

---

## Experiment Design

### Configuration Matrix (2×3 + Controls)

```
┌─────────────────────────────────────────────────────────────┐
│ Signal Pair     │ Truth Type │ Tau_err │ Bandpass │ Status  │
├─────────────────────────────────────────────────────────────┤
│ LDV-MicL        │ Chirp      │ 2.0 ms  │ None     │ PRIMARY │
│ LDV-MicL        │ Chirp      │ 2.0 ms  │ 500-2k   │ PRIMARY │
│ MicL-MicR       │ Chirp      │ 2.0 ms  │ 500-2k   │ BASELINE│
│ LDV-MicL        │ Geometry   │ 2.0 ms  │ 500-2k   │ NEG CTL │
│ LDV-MicL        │ Chirp      │ 0.3 ms  │ 500-2k   │ SECONDARY
└─────────────────────────────────────────────────────────────┘

Speakers: 18-0.1V, 19-0.1V, 20-0.1V, 21-0.1V, 22-0.1V (5 speakers)
Total: 5 speakers × 4-5 configurations = 20-25 runs
```

### Expected Results
| Configuration | Target Pass Rate | Rationale |
|---------------|------------------|-----------|
| **LDV-MicL + Chirp** | ≥4/5 | Should match Mic-MicL (commit f9a30d7) |
| **MicL-MicR + Chirp** | 4/5 | Baseline from f9a30d7 |
| **LDV-MicL + Geometry** | ≤3/5 | Negative control (geometry insufficient) |
| **LDV-MicL + Chirp (tau 0.3)** | 3-4/5 | Tight gating reduces available windows |

---

## Project Structure

```
worktree/exp-ldv-vs-mic-doa-comparison/
├── README_EXPERIMENT.md                  ← You are here
├── ../../../EXP_LDV_VS_MIC_DOA_COMPARISON.md       (Full plan)
├── ../../../IMPLEMENTATION_ROADMAP_LDV_VS_MIC.md   (Execution roadmap)
├── scripts/
│   ├── stage4_doa_ldv_vs_mic_comparison.py        (Main DoA estimation)
│   ├── run_ldv_vs_mic_grid.py                     (Grid orchestrator)
│   └── analyze_ldv_vs_mic_results.py              (Analysis tool)
└── results/                              (Will be populated during Phase 2)
    └── ldv_vs_mic_grid_TIMESTAMP/
        ├── grid_summary.md / .json
        ├── ldv_micl_chirp_tau2_band0/
        ├── ldv_micl_chirp_tau2_band500_2000/
        ├── micl_micr_chirp_tau2_band500_2000/
        └── ... (other configs)
```

---

## Phase Breakdown

### Phase 1: Code Implementation (Est. 4 hours) ⏳ NEXT
- [ ] Adapt stage4_doa_validation.py from commit a53c778
- [ ] Add `--signal_pair` support (ldv_micl / micl_micr)
- [ ] Integrate OMP alignment from commit 62a51617
- [ ] Implement guided search + both truth types
- [ ] Run smoke tests (all 3 paths)
- **Exit Gate**: All smoke tests PASS

### Phase 2: Grid Execution (Est. 6-8 hours) ⏳ PENDING PHASE 1
- [ ] Execute full 5×4-5 grid
- [ ] Generate grid_summary.md/json
- [ ] Per-speaker summary.json files
- [ ] Aggregate results tables
- **Exit Gate**: All 20-25 runs complete with valid outputs

### Phase 3: Analysis & Principles (Est. 3-4 hours) ⏳ PENDING PHASE 2
- [ ] Pass/fail comparison across all 3 signal pairs
- [ ] DoA error distribution analysis
- [ ] PSR/SNR stability trends
- [ ] Failure mode breakdown per speaker
- [ ] Cross-experiment analysis (BECAUSE/DUE TO/THEREFORE)
- [ ] Extracted design principles
- **Exit Gate**: Comprehensive analysis + principles document

---

## Critical References

| Commit | Purpose | Key Finding |
|--------|---------|-------------|
| 62a51617 | Stage 1-2 OMP alignment | LDV→Mic: tau→0, PSR +6–8 dB ✅ |
| f9a30d7 | Stage 4-C Mic-Mic baseline | GCC-PHAT 4/5 PASS w/ chirp guidance ✅ |
| a53c778 | Stage 4-C local scan | Chirp truth generation methodology |
| c36dcebfd514 | Chirp validation | Guided peak search implementation |
| 225aed7 | Geometry baseline | ~3/5 PASS (negative control target) |

---

## How to Execute

### Quick Start
```bash
# 1. Enter the worktree
cd worktree/exp-ldv-vs-mic-doa-comparison/

# 2. Run smoke test (LDV-MicL + chirp)
python ../../scripts/stage4_doa_ldv_vs_mic_comparison.py \
  --data_root ../../dataset/GCC-PHAT-LDV-MIC-Experiment \
  --speaker 20-0.1V \
  --signal_pair ldv_micl \
  --output_dir results/smoke_ldv_micl_20260211 \
  --segment_mode fixed --n_segments 1 --analysis_slice_sec 2

# 3. Run full grid (Phase 2)
python ../../scripts/run_ldv_vs_mic_grid.py \
  --data_root ../../dataset/GCC-PHAT-LDV-MIC-Experiment \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V \
  --output_base results/ldv_vs_mic_grid_$(date +%Y%m%d_%H%M%S)
```

### Key Parameters
- `--signal_pair`: `ldv_micl` (LDV aligned to MicL) or `micl_micr` (MicL-MicR baseline)
- `--truth_tau_ms / --truth_theta_deg`: Chirp truth values
- `--use_geometry_truth`: Enable geometry-based truth (negative control)
- `--scan_tau_err_max_ms`: Tau gating tolerance (2.0 or 0.3 ms)
- `--gcc_bandpass_low/high`: Bandpass filter (0 or 500-2000 Hz)

---

## AGENTS.md Compliance

✅ **Execution first**: Every commit will contain executed results + artifacts
✅ **Causal analysis**: Cross-experiment patterns use BECAUSE/DUE TO/THEREFORE
✅ **Real data only**: No synthetic data; all runs use GCC-PHAT-LDV-MIC-Experiment
✅ **Atomic commits**: Code + artifacts + analysis together
✅ **Comprehensive docs**: Background, motivation, purpose, setup, results, analysis, next steps
✅ **Reproducibility**: All commands, fingerprints, paths documented

---

## Success Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| LDV-MicL DoA PASS rate | ≥4/5 | ⏳ Pending execution |
| Baseline MicL-MicR | 4/5 | ✅ Known from f9a30d7 |
| Geometry control | ≤3/5 | ⏳ Pending execution |
| Smoke tests | All pass | ⏳ Phase 1 gate |
| Cross-experiment analysis | Complete | ⏳ Phase 3 |
| Design principles | Extracted | ⏳ Phase 3 |

---

## Next Immediate Actions

1. ✅ Create worktree (DONE)
2. ✅ Draft planning docs (DONE)
3. ✅ Create script scaffolds (DONE)
4. ⏳ **Implement Phase 1 code** (LDV-vs-Mic signal pair logic)
5. ⏳ Run smoke tests (confirm code paths work)
6. ⏳ Execute Phase 2 grid
7. ⏳ Phase 3 analysis + principles extraction

---

## Document References

- **Full Experiment Plan**: `../../EXP_LDV_VS_MIC_DOA_COMPARISON.md`
- **Implementation Roadmap**: `../../IMPLEMENTATION_ROADMAP_LDV_VS_MIC.md`
- **AGENTS.md Standards**: `../../AGENTS.md`

---

## Commits on This Branch

```
ea07b32 - Docs: Detailed implementation roadmap (Phases 1-3)
104ad22 - Code scaffold: LDV-vs-Mic DoA scripts (Phase 1 framework)
b68269a - Plan: LDV-vs-Mic DoA comparison (full specification)
```

---

**Status**: Ready for Phase 1 implementation
**Estimated Total Time**: 13-16 hours (3-4 days)
**Next Milestone**: Phase 1 smoke test completion
