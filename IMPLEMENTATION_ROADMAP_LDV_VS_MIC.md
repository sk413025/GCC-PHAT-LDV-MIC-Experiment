# Implementation Roadmap: LDV-vs-Mic DoA Comparison Experiment

**Branch**: `exp/ldv-vs-mic-doa-comparison`
**Status**: Phase 1 - Code Scaffolding Complete ✅
**Created**: 2026-02-11
**Last Updated**: 2026-02-11

---

## ✅ Completed (This Session)

### Commit b68269a: Planning
- Created `EXP_LDV_VS_MIC_DOA_COMPARISON.md` with full experiment specification
- Defined 2×3 (or 2×4) configuration matrix
- Documented cross-experiment references (62a51617, f9a30d7, a53c778, c36dcebfd514)
- Outlined success criteria and failure modes
- Estimated timeline: 13-16 hours across 3-4 days

### Commit 104ad22: Code Scaffolding
- **`scripts/stage4_doa_ldv_vs_mic_comparison.py`** (placeholder + interface)
  - Argument parser for all required parameters
  - Logging infrastructure in place
  - Will extend commit a53c778's stage4_doa_validation.py with:
    - `--signal_pair` support (ldv_micl / micl_micr)
    - LDV loading via OMP alignment (from commit 62a51617)
    - Support for both chirp truth and geometry truth guidance

- **`scripts/run_ldv_vs_mic_grid.py`** (grid orchestrator framework)
  - Configuration matrix definition (matching plan)
  - Will implement:
    - Parallel execution over speakers
    - Smoke test + guardrail test
    - Result aggregation and grid summary generation

- **`scripts/analyze_ldv_vs_mic_results.py`** (analysis framework)
  - Will implement:
    - Per-pair pass/fail rate comparison
    - DoA error distribution analysis
    - PSR/SNR stability plots
    - Failure mode breakdown
    - Cross-experiment analysis (BECAUSE/DUE TO/THEREFORE logic)
    - Extracted design principles

---

## 📋 Phase 1: Code Implementation (Next - Est. 4 hours)

### Task 1.1: Adapt Stage 4-C Main Script
**Reference**: Commit a53c778 (`exp-ldv-perfect-geometry-cloud/exp-validation/ldv-perfect-geometry/scripts/stage4_doa_validation.py`)

```bash
# Copy and adapt:
# - GCC-PHAT estimation logic
# - MUSIC estimation logic
# - CC/NCC estimation logic
# - Scanning window selection
# - Guided peak search logic
# - Per-speaker output structure (summary.json, run.log)
```

**Key modifications for stage4_doa_ldv_vs_mic_comparison.py**:
1. Add `--signal_pair` handling:
   ```python
   if args.signal_pair == "ldv_micl":
       # Load LDV waveform
       ldv_wav = load_ldv(args.speaker, args.data_root, apply_omp_alignment=True)
       # Load MicL waveform
       micl_wav = load_mic(args.speaker, args.data_root, channel="L")
       signal_pair = (ldv_wav, micl_wav)
   elif args.signal_pair == "micl_micr":
       # Load MicL and MicR
       signal_pair = (
           load_mic(args.speaker, args.data_root, channel="L"),
           load_mic(args.speaker, args.data_root, channel="R")
       )
   ```

2. OMP alignment for LDV:
   - Reference: Commit 62a51617's Stage 1-2 alignment code
   - Expected: tau → 0, PSR +6–8 dB improvement
   - Apply correction: `tau_ldv_corrected = tau_raw - tau_offset`

3. Support both truth types:
   ```python
   if args.use_geometry_truth:
       # Compute geometry-based truth
       truth_tau_ms = compute_geometry_tau(geometry_file, distance)
       truth_theta_deg = compute_geometry_theta(...)
   else:
       # Use provided chirp truth
       truth_tau_ms = args.truth_tau_ms
       truth_theta_deg = args.truth_theta_deg
   ```

**Smoke test for 1.1**:
```bash
python scripts/stage4_doa_ldv_vs_mic_comparison.py \
  --data_root dataset/GCC-PHAT-LDV-MIC-Experiment \
  --speaker 20-0.1V \
  --signal_pair ldv_micl \
  --output_dir results/smoke_ldv_micl_20260211 \
  --segment_mode fixed --n_segments 1 --analysis_slice_sec 2 \
  --eval_window_sec 1 \
  --ldv_prealign gcc_phat
```

**Acceptance criteria**:
- [ ] Script runs without error
- [ ] Outputs: `results/smoke_ldv_micl_20260211/{20-0.1V}/summary.json`
- [ ] summary.json contains: speaker, signal_pair, DoA estimate, error vs truth, pass/fail
- [ ] PSR value matches expected range from prior runs

---

### Task 1.2: Implement Grid Runner
**Reference**: Commit f9a30d7's grid logic (exp-chirp-reference-validation/results/stage4_speech_grid_compare_20260211_072624)

**Key implementation for run_ldv_vs_mic_grid.py**:
```python
configs = [
    # Primary comparisons
    {"signal_pair": "ldv_micl", "tau_err_max": 2.0, "bandpass": (0, 0), "truth": "chirp"},
    {"signal_pair": "ldv_micl", "tau_err_max": 2.0, "bandpass": (500, 2000), "truth": "chirp"},
    {"signal_pair": "micl_micr", "tau_err_max": 2.0, "bandpass": (500, 2000), "truth": "chirp"},
    {"signal_pair": "ldv_micl", "tau_err_max": 2.0, "bandpass": (500, 2000), "truth": "geometry"},
    # Secondary
    {"signal_pair": "ldv_micl", "tau_err_max": 0.3, "bandpass": (500, 2000), "truth": "chirp"},
]

speakers = ["18-0.1V", "19-0.1V", "20-0.1V", "21-0.1V", "22-0.1V"]

for truth_type in ["chirp", "geometry"]:
    for cfg in configs:
        run_dir = Path(f"results/ldv_vs_mic_grid_{cfg['label']}_{timestamp}")
        for speaker in speakers:
            args = [
                "python", "-u", "scripts/stage4_doa_ldv_vs_mic_comparison.py",
                "--data_root", data_root,
                "--speaker", speaker,
                "--signal_pair", cfg["signal_pair"],
                "--output_dir", run_dir,
                "--scan_tau_err_max_ms", str(cfg["tau_err_max"]),
                "--gcc_bandpass_low", str(cfg["bandpass"][0]),
                "--gcc_bandpass_high", str(cfg["bandpass"][1]),
                # ... other args
            ]
            subprocess.run(args, check=False)

# Aggregate results
aggregate_grid_summary(all_results, output_base)
```

**Smoke test for 1.2**:
```bash
python scripts/run_ldv_vs_mic_grid.py \
  --data_root dataset/GCC-PHAT-LDV-MIC-Experiment \
  --speakers 20-0.1V 21-0.1V \
  --output_base results/grid_smoke_20260211 \
  --debug  # Run only 1 config instead of full grid
```

**Acceptance criteria**:
- [ ] Runs 2 speakers × 1 config (debug mode) without error
- [ ] Produces: `results/grid_smoke_20260211/grid_summary.md` and `.json`
- [ ] Summary contains pass/fail counts and per-speaker tables
- [ ] Output structure matches f9a30d7 style

---

### Task 1.3: Phase 1 Smoke Test Suite
Run all three smoke tests to confirm Phase 1 completion:

```bash
# Smoke 1: LDV-MicL + chirp truth
python scripts/stage4_doa_ldv_vs_mic_comparison.py \
  --data_root dataset/GCC-PHAT-LDV-MIC-Experiment \
  --speaker 20-0.1V \
  --signal_pair ldv_micl \
  --output_dir results/smoke_ldv_micl_chirp_20260211 \
  --segment_mode fixed --n_segments 1 --analysis_slice_sec 2 \
  --ldv_prealign gcc_phat

# Smoke 2: MicL-MicR + chirp truth (baseline)
python scripts/stage4_doa_ldv_vs_mic_comparison.py \
  --data_root dataset/GCC-PHAT-LDV-MIC-Experiment \
  --speaker 20-0.1V \
  --signal_pair micl_micr \
  --output_dir results/smoke_micl_micr_chirp_20260211 \
  --segment_mode fixed --n_segments 1 --analysis_slice_sec 2

# Smoke 3: LDV-MicL + geometry (negative control)
python scripts/stage4_doa_ldv_vs_mic_comparison.py \
  --data_root dataset/GCC-PHAT-LDV-MIC-Experiment \
  --speaker 20-0.1V \
  --signal_pair ldv_micl \
  --output_dir results/smoke_ldv_micl_geometry_20260211 \
  --segment_mode fixed --n_segments 1 --analysis_slice_sec 2 \
  --use_geometry_truth
```

**Phase 1 Exit Commit**:
After all smoke tests pass:
```bash
git add scripts/ results/smoke_*/ \
        doa_rl/ nmf_localizer/  # if any modified
git commit -m "Phase 1 results: LDV-vs-Mic smoke tests (all paths)

Setup (REQUIRED):
- Python 3.10, numpy 2.2.6, scipy 1.15.3
- Data: GCC-PHAT-LDV-MIC-Experiment (5 speakers)
- Scripts: stage4_doa_ldv_vs_mic_comparison.py (main), run_ldv_vs_mic_grid.py (orchestrator)

Smoke tests executed (REQUIRED):
- Test 1 (LDV-MicL + chirp): results/smoke_ldv_micl_chirp_20260211/run.log
- Test 2 (MicL-MicR + chirp): results/smoke_micl_micr_chirp_20260211/run.log
- Test 3 (LDV-MicL + geometry): results/smoke_ldv_micl_geometry_20260211/run.log

Key findings:
- All three smoke tests confirm code paths work end-to-end
- Output structure (per-speaker summary.json, run.log) matches expected schema
- Ready for Phase 2 grid execution

Next steps:
- Phase 2: Run full 2×3 grid (5 speakers × 4-5 configs = 20-25 runs)
- Phase 3: Cross-experiment analysis + principles extraction"
```

---

## 🚀 Phase 2: Grid Execution (Est. 6-8 hours)

### Full Grid Configuration
```
5 speakers × 5 configurations = 25 total runs

Config matrix:
├─ ldv_micl_chirp_tau2_band0              (primary)
├─ ldv_micl_chirp_tau2_band500_2000       (primary + bandpass)
├─ micl_micr_chirp_tau2_band500_2000      (baseline from f9a30d7)
├─ ldv_micl_geometry_tau2_band500_2000    (negative control)
└─ ldv_micl_chirp_tau0p3_band500_2000     (secondary: tight gating)

Speakers: 18-0.1V, 19-0.1V, 20-0.1V, 21-0.1V, 22-0.1V

Run command:
python scripts/run_ldv_vs_mic_grid.py \
  --data_root dataset/GCC-PHAT-LDV-MIC-Experiment \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V \
  --output_base results/ldv_vs_mic_grid_TIMESTAMP \
  --chirp_truth_file worktree/exp-ldv-perfect-geometry-cloud/.../chirp_truthref_5s.json
```

### Expected Outputs
```
results/ldv_vs_mic_grid_TIMESTAMP/
├── grid_summary.md                 # Consolidated pass/fail summary
├── grid_summary.json               # Machine-readable grid results
├── grid_report.md                  # Detailed per-config analysis
├── config_matrix.json              # Full configuration specification
├── ldv_micl_chirp_tau2_band0/
│   ├── {18,19,20,21,22}-0.1V/
│   │   ├── summary.json            # Per-speaker DoA results
│   │   └── run.log                 # Execution log
│   ├── summary_table.md
│   ├── run_config.json
│   ├── subset_manifest.json
│   └── code_state.json
├── ldv_micl_chirp_tau2_band500_2000/
│   └── ... (same structure)
├── micl_micr_chirp_tau2_band500_2000/
│   └── ... (same structure) [BASELINE]
├── ldv_micl_geometry_tau2_band500_2000/
│   └── ... (same structure) [NEG CONTROL]
├── ldv_micl_chirp_tau0p3_band500_2000/
│   └── ... (same structure) [SECONDARY]
└── {smoke,guardrail}/
    └── run.log
```

### Expected Results Summary
```
Pass rates (target):
├─ ldv_micl_chirp_tau2_band0:         ≥4/5 PASS
├─ ldv_micl_chirp_tau2_band500_2000:  ≥4/5 PASS
├─ micl_micr_chirp_tau2_band500_2000: 4/5 PASS (baseline from f9a30d7)
├─ ldv_micl_geometry_tau2_band500_2000: ≤3/5 PASS (negative control)
└─ ldv_micl_chirp_tau0p3_band500_2000: 3-4/5 PASS (expected: tight gating reduces availability)
```

---

## 🔬 Phase 3: Analysis & Principles (Est. 3-4 hours)

### Analysis Tasks

1. **Pass/Fail Comparison**
   - Table: signal_pair × tau_err_max × bandpass → pass_count/5
   - Compare LDV-MicL to MicL-MicR baseline (should be comparable or better)
   - Validate negative control (geometry ≤3/5)

2. **DoA Error Analysis**
   - Per-pair error histograms (θ_estimated - θ_truth_deg)
   - Mean, std, percentiles (p25, p50, p75, p95)
   - Comparison: LDV-MicL vs MicL-MicR

3. **PSR/SNR Stability**
   - Trend plots: tau_err_max × bandpass → mean PSR/SNR
   - Which configurations maintain high SNR?
   - Which speakers are most sensitive to gating?

4. **Failure Mode Breakdown**
   - Which speakers fail for which configs?
   - Per-window tau/theta distributions for failed speakers
   - Hypothesis: geometry-based failures due to reverberation, chirp-based failures due to...?

5. **Cross-Experiment Analysis** (REQUIRED by AGENTS.md)
   ```
   Pattern recognition: Compare to commits f9a30d7 (Mic-MicR 4/5), 62a51617 (OMP alignment), 225aed7 (geometry ~3/5)
   - BECAUSE OMP alignment preserves coherence, THEREFORE LDV-MicL + chirp should match Mic-MicR
   - DUE TO geometry being insufficient for speech, THEREFORE geometry control should stay ≤3/5

   Success factors: Why does chirp guidance work better than geometry?
   - BECAUSE chirp-derived tau aligns with direct-path coherence peaks, THEREFORE window selection improves

   Failure modes: Which speakers consistently fail?
   - Identify speaker-specific reverberation patterns
   - Determine if LDV introduces new failure modes vs Mic-Mic
   ```

6. **Extracted Design Principles**
   ```
   - THEREFORE prioritize chirp truth guidance over geometry for speech DoA
   - GIVEN successful LDV-MicL performance, recommend LDV as virtual mic in hybrid systems
   - BECAUSE tight tau gating (0.3 ms) reduces available windows, allocate >= 2.0 ms for robustness
   - RISK: Bandpass (500-2000) helps some speakers; validate per-deployment
   ```

### Phase 3 Exit Commit
```bash
git add results/ldv_vs_mic_grid_TIMESTAMP/ scripts/analyze_ldv_vs_mic_results.py

git commit -m "Phase 3 results: LDV-vs-Mic DoA comparison analysis + principles

[Full AGENTS.md-compliant commit message with:]
- Physical/mathematical analysis of success/failure
- Cross-experiment patterns (f9a30d7, 62a51617, 225aed7)
- Extracted design principles for LDV-Mic systems
- Meta-reflection on experimental methodology
- Complete reproduction instructions
- Data lineage and fingerprints"
```

---

## 📊 Summary of Commits

| Commit | Phase | Type | Status |
|--------|-------|------|--------|
| b68269a | Plan | Planning | ✅ DONE |
| 104ad22 | Setup | Code scaffold | ✅ DONE |
| (TBD) | 1 | Phase 1 smoke tests | ⏳ NEXT |
| (TBD) | 2 | Phase 2 grid results | ⏳ QUEUE |
| (TBD) | 3 | Phase 3 analysis | ⏳ QUEUE |

---

## 🔗 Cross-Reference Index

| Reference | Commit | Purpose |
|-----------|--------|---------|
| OMP alignment | 62a51617f62e08ff93f53d2be67ecd548b51cf30 | Load LDV with tau correction |
| Mic-Mic baseline | f9a30d7eeae322a57d531003edf0c9c030fd9f87 | Comparison target (4/5 PASS) |
| Chirp truth generation | a53c778412c7af34cc0e56b3553c6b81cc37c9ac | Stage 4-C methodology |
| Guided search | c36dcebfd514bae88ad8a4e464505f49c94d2cb4 | Peak search implementation |
| Geometry baseline | 225aed7 (commit msg search) | Negative control (~3/5) |

---

## ✅ Next Immediate Actions

1. **Start Phase 1**:
   - Adapt stage4_doa_validation.py from exp-ldv-perfect-geometry-cloud
   - Copy to scripts/stage4_doa_ldv_vs_mic_comparison.py and implement --signal_pair logic
   - Extract OMP alignment code from commit 62a51617
   - Implement guided search logic

2. **Smoke test all three paths** (LDV-chirp, Mic-chirp, LDV-geometry)

3. **Commit Phase 1 results** with all smoke test artifacts

4. **Execute Phase 2 grid** (est. 6-8 hours of compute time)

5. **Phase 3 analysis** and final principles extraction

---

## 📝 Notes

- **Language**: All code, docs, and logs in English (per AGENTS.md)
- **Atomicity**: Each phase will be a single commit with code + artifacts (AGENTS.md requirement)
- **Real data only**: Using GCC-PHAT-LDV-MIC-Experiment (5 speakers)
- **No hypotheticals**: Every commit must have executed results and comprehensive analysis
- **Causal language**: Cross-experiment analysis MUST use BECAUSE/DUE TO/THEREFORE

---

**Document version**: 1.0 (2026-02-11)
**Author**: Claude Code (automated planning & scaffolding)
**Status**: Ready for Phase 1 implementation
