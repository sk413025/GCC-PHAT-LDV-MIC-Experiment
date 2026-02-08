# Experiment Design: Chirp Reference Validation

**Branch**: `exp/chirp-reference-validation`
**Date**: 2026-02-07
**Parent analysis**: `exp/tdoa-methods-validation` commit `9e9b74f`

---

## 1. Background

### 1.1 Previous Findings (exp/tdoa-methods-validation)

A 4-phase systematic validation of the GCC-PHAT LDV-Mic experiment revealed:

| Phase | Finding | Key Metric |
|-------|---------|------------|
| Phase 1: Tau Stability | Speech tau collapses to ~0 | 91.3% collapse rate |
| Phase 2: Guided Search | Guided peak search eliminates false peaks | 100% elimination |
| Phase 3: Stage 3 Revalidation | Marginal improvement with improved baseline | 0% -> 17% pass rate |
| Phase 4: Final Summary | Overall status | NEEDS_WORK |

### 1.2 Critical Limitation Identified

The previous validation used **geometric estimation** as chirp reference because the chirp data folders (`23-chirp(-0.8m)`, `24-chirp(-0.4m)`) were not found at the expected path in `dataset/GCC-PHAT-LDV-MIC-Experiment/`.

However, **two complete chirp calibration datasets** exist at different paths:
- `dataset/chirp/` — 5 positions, Speaker IDs 25-29
- `dataset/chirp_2/` — 5 positions, Speaker IDs 30-33 (independent re-recording)

Both datasets have pre-computed calibration results with per-sensor delay estimation and per-position MIC-MIC tau values.

### 1.3 Why This Matters

Using geometric estimation as reference introduces uncertainty:
- Assumes exact sensor positions (no measurement error)
- Assumes free-field propagation (no reflections)
- Assumes constant speed of sound (343 m/s, no temperature variation)

Real chirp measurements capture the actual acoustic environment, including room effects and exact hardware timing. The 91.3% speech tau collapse finding could be partially affected by reference error.

---

## 2. Objectives

### Objective 1: Re-run Validation with Real Chirp Reference

**Goal**: Replace geometric tau estimation with chirp-derived MIC-MIC tau measurements and re-run the 4-phase validation pipeline.

**Hypothesis**: The speech tau collapse (91.3%) is fundamentally a signal quality issue, not a reference error. We expect the collapse rate to remain high (~90%+) even with real chirp references, but deviation metrics will be more accurate and the analysis will be on firmer ground.

**Approach**:
- Extract MIC-MIC tau median from chirp and chirp_2 calibration results
- Cross-validate chirp vs chirp_2 for consistency
- Use chirp-derived tau as reference for positions with reliable data
- Re-run Phase 1 (stability), Phase 2 (guided search), Phase 3 (Stage 3), Phase 4 (summary)

### Objective 2: LDV Delay Re-evaluation

**Goal**: Use chirp-calibrated LDV sensor delay to re-evaluate LDV-MIC TDoA quality.

**Background**:
- Previous estimate (from speech data): LDV delay = 3.8 ~ 4.8 ms
- New chirp-based estimate: **chirp = 0.868 ms, chirp_2 = 0.683 ms**
- The difference is 4-7x, likely because the speech-derived estimate was corrupted by tau collapse

**Hypothesis**: The chirp-derived LDV delay (0.68-0.87 ms) is correct. Using this value for LDV-MIC TDoA compensation should produce dramatically smaller residuals compared to the old 3.8-4.8 ms estimate.

**Approach**:
- Load chirp-derived sensor delays (LDV, RIGHT-MIC)
- Compute LDV-MIC GCC-PHAT on speech segments
- Apply delay compensation: `tau_corrected = tau_meas - (delta_LDV - delta_MIC)`
- Compare residuals with old vs new delay values

### Objective 3: Negative Position Diagnosis

**Goal**: Understand why positions -0.4m and -0.8m consistently fail chirp event quality gates in both datasets.

**Evidence of systematic failure**:

| Position | chirp events_used | chirp_2 events_used |
|----------|-------------------|---------------------|
| -0.4m    | 0/4               | 0/7                 |
| -0.8m    | 1/6               | 0/4                 |

**Hypothesis**: The failure is caused by near-field geometry and/or room reflections.
- Speaker at -0.8m is only 0.1m horizontally from LEFT-MIC at (-0.7, 2.0)
- This near-perpendicular geometry means direct path and first reflections have very similar delays
- GCC-PHAT cannot distinguish between them, causing multi-peak confusion

**Approach**:
- Analyze full GCC-PHAT correlation curves (not just peaks) for failing events
- Systematically relax quality gates to find threshold where events pass
- Compute near-field geometric metrics
- Plot GCC waveforms for visual inspection

---

## 3. Data Inventory

### 3.1 Chirp Calibration Datasets

```
dataset/chirp/
  +0.0/  0128-{LDV,LEFT-MIC,RIGHT-MIC}-25-chirp(0.0m).wav
  +0.4/  0128-{LDV,LEFT-MIC,RIGHT-MIC}-28-chirp(+0.4m).wav
  +0.8/  0128-{LDV,LEFT-MIC,RIGHT-MIC}-29-chirp(+0.8m).wav
  -0.4/  0128-{LDV,LEFT-MIC,RIGHT-MIC}-26-chirp(-0.4m).wav
  -0.8/  0128-{LDV,LEFT-MIC,RIGHT-MIC}-27-chirp(-0.8m).wav
  results/chirp_calibration_summary.json   (pre-computed)

dataset/chirp_2/
  +0.0/  0128-{LDV,LEFT-MIC,RIGHT-MIC}-31-chirp(0.0m).wav
  +0.4/  0128-{LDV,LEFT-MIC,RIGHT-MIC}-33-chirp(+0.4m).wav
  +0.8/  0128-{LDV,LEFT-MIC,RIGHT-MIC}-30-chirp(+0.8m).wav
  -0.4/  0128-{LDV,LEFT-MIC,RIGHT-MIC}-33-chirp(-0.4m).wav
  -0.8/  0128-{LDV,LEFT-MIC,RIGHT-MIC}-32-chirp(-0.8m).wav
  results/chirp_calibration_summary.json   (pre-computed)
```

### 3.2 Speech Datasets (for validation re-run)

```
dataset/GCC-PHAT-LDV-MIC-Experiment/
  18-0.1V/  Speaker @ +0.8m, boy speech, 0.1V
  19-0.1V/  Speaker @ +0.4m
  20-0.1V/  Speaker @ +0.0m
  21-0.1V/  Speaker @ -0.4m
  22-0.1V/  Speaker @ -0.8m
```

### 3.3 Position Correspondence

| Speech Folder | Speech ID | Chirp Label | x (m) | chirp events | chirp_2 events | Reference Status |
|---------------|-----------|-------------|-------|--------------|----------------|-----------------|
| 18-0.1V | 18 | +0.8 | +0.8 | 5/5 | 4/4 | **Reliable** |
| 19-0.1V | 19 | +0.4 | +0.4 | 4/6 | 4/4 | **Reliable** |
| 20-0.1V | 20 | +0.0 | 0.0 | 2/3 | 5/5 | **Reliable** |
| 21-0.1V | 21 | -0.4 | -0.4 | 0/4 | 0/7 | Geometric fallback |
| 22-0.1V | 22 | -0.8 | -0.8 | 1/6 | 0/4 | Geometric fallback |

### 3.4 Pre-computed Calibration Results

#### Sensor Delays

| Parameter | chirp | chirp_2 | Agreement |
|-----------|-------|---------|-----------|
| LEFT-MIC delay | 0.000 ms (ref) | 0.000 ms (ref) | -- |
| RIGHT-MIC delay | -0.005 ms | -0.043 ms | ~0.04 ms diff |
| LDV delay | 0.868 ms | 0.683 ms | ~0.19 ms diff |

#### MIC-MIC Tau Reference Values

| Position | chirp tau_median (ms) | chirp_2 tau_median (ms) | Geometric tau (ms) |
|----------|-----------------------|-------------------------|---------------------|
| +0.8 | +1.5625 | +1.5208 | +1.4504 |
| +0.4 | +0.7500 | +0.8646 | +0.7585 |
| +0.0 | +0.0208 | +0.0000 | +0.0000 |
| -0.4 | -0.3125 (unreliable) | -0.3958 (unreliable) | -0.7585 |
| -0.8 | -1.1875 (1 event) | -1.0000 (unreliable) | -1.4504 |

#### Post-Calibration Residuals (chirp, all positions)

| Pair | n | median (ms) | std (ms) | max|.| (ms) |
|------|---|-------------|----------|-------------|
| MIC-MIC | 12 | 0.057 | 0.135 | 0.294 |
| LDV-LEFT | 12 | -0.076 | 0.332 | 0.809 |
| LDV-RIGHT | 12 | -0.056 | 0.247 | 0.448 |

---

## 4. Experiment Geometry

```
                    y (m)
                    ^
                    |
        LEFT MIC    |    RIGHT MIC
        (-0.7, 2.0) |    (+0.7, 2.0)
            o-------+-------o          y = 2.0 m
                    |
                    |
                +---+---+
                |  LDV  |              y = 0.5 m
                |  BOX  |
                +-------+
                (0, 0.5)
                    |
    ----------------+----------------  y = 0 m
    o       o       o       o       o
  -0.8    -0.4     0.0    +0.4    +0.8
  (22)    (21)    (20)    (19)    (18)   <- Speaker positions
```

Key distances for negative positions:
- Speaker -0.8m to LEFT-MIC: sqrt(0.01 + 4.0) = **2.002 m** (near-perpendicular)
- Speaker -0.4m to LEFT-MIC: sqrt(0.09 + 4.0) = **2.022 m**

---

## 5. Planned Commit Sequence

### Commit 1 (this commit): Experiment Design
- EXPERIMENT_DESIGN.md
- scripts/validation/chirp_reference.py (shared module)
- run_chirp_ref_validation.ps1 (orchestration)

### Commit 2: Chirp Cross-Validation (Step 0)
- Execute chirp_reference.py to cross-validate chirp vs chirp_2
- Produce results/chirp_cross_validation/
- **Blocks**: all subsequent commits

### Commit 3: Phase 1 Re-run with Chirp Reference (Objective 1a)
- scripts/validation/phase1_chirp_reference.py
- results/chirp_ref_phase1/
- **Key question**: Does 91.3% collapse change with real reference?

### Commit 4: Phase 2-4 Re-run (Objective 1b)
- Modify Phase 2/3/4 to accept external chirp reference
- results/chirp_ref_phase2/, chirp_ref_phase3/, chirp_ref_phase4/
- **Key question**: Does NEEDS_WORK status change?

### Commit 5: LDV Delay Re-evaluation (Objective 2)
- scripts/validation/ldv_delay_reeval.py
- results/ldv_delay_reeval/
- **Key question**: Does 0.68-0.87ms delay improve LDV-MIC quality?

### Commit 6: Negative Position Diagnosis (Objective 3)
- scripts/validation/negative_position_diagnosis.py
- results/negative_pos_diagnosis/
- **Key question**: Near-field or reflection cause?

---

## 6. Success Criteria

| Objective | Success | Partial Success | Failure |
|-----------|---------|-----------------|---------|
| Obj 1: Chirp ref validation | Collapse rate confirmed at >85% with real ref | Collapse rate shifts significantly (suggesting reference was an issue) | Cannot extract reliable chirp reference |
| Obj 2: LDV delay | Corrected LDV-MIC residuals < 0.5ms | Residuals improve but still >0.5ms | No improvement over old delay |
| Obj 3: Negative diagnosis | Root cause identified with evidence | Symptom characterized but cause unclear | Cannot reproduce the failure |

---

## 7. Dependencies and Reusable Code

### From dataset/chirp/validate_chirp_calibration.py
- `gcc_phat_guided()` — GCC-PHAT with guided peak search
- `detect_chirp_events()` — Event detection on MicL envelope
- `solve_sensor_delays_ms()` — Weighted least squares for sensor delays
- `DEFAULT_CONFIG` — Calibrated parameters

### From scripts/validation/ (exp/tdoa-methods-validation)
- `phase1_tau_stability.py` — Tau stability analysis pattern
- `phase2_guided_search.py` — Guided vs global peak comparison
- `phase3_stage3_revalidation.py` — Stage 3 re-validation
- `phase4_final_validation.py` — Final summary aggregation

### Pre-computed Results (read-only inputs)
- `dataset/chirp/results/chirp_calibration_summary.json`
- `dataset/chirp_2/results/chirp_calibration_summary.json`
