# Execution Plan: Extend Commit 530a63f with OMP->DTmin Learning

Generated: 2026-02-11
Branch: `exp/ldv-vs-mic-doa-comparison`
Baseline commit: `530a63fb84636cccfc1b67950a7eb54fe6ce397e`

## 1) Purpose

This document defines the complete design and execution plan to extend the strict GCC-PHAT baseline (`530a63f`) with an OMP-trajectory-to-DTmin learning pipeline, then re-evaluate under the same strict comparability rules.

Primary publication goal:
- Keep the Stage4 strict comparability protocol from `530a63f`.
- Add a learned DTmin policy trained from OMP teacher trajectories.
- Report whether DTmin preserves or improves strict-pass outcomes.

## 2) What Commit 530a63f Already Established

Commit `530a63f` is a strict-pass LDV-vs-Mic grid run with artifacts under:
- `results/ldv_vs_mic_grid_strict_20260211_103715/`

Main behavior:
- `ldv_micl`: pass mode `omp_vs_raw`
- `micl_micr`: pass mode `theta_only`

Key scripts in baseline:
- `scripts/stage4_doa_ldv_vs_mic_comparison.py`
- `scripts/run_ldv_vs_mic_grid.py`
- `scripts/analyze_ldv_vs_mic_results.py`

Strict baseline outputs include:
- `grid_summary.md`
- `grid_report.md`
- per-config `summary.json`, `run.log`, `run_config.json`, `subset_manifest.json`, `code_state.json`

## 3) Full OMP->DTmin Design (Target Architecture)

### 3.1 Teacher (OMP trajectory generation)

For each `(speaker, selected 5s segment, frequency bin)`:
- Build lag dictionary over `[-max_lag, +max_lag]`.
- Run OMP (or penalty-OMP with STOP) to produce step-wise lag selections.
- Save trajectory blocks with required fields:
  - `corrs`: per-step correlation features
  - `actions`: selected lag ids
  - `reductions` or `deltaE`
  - `valid_len`
  - optional RTG conditioning fields (`lambda_c`, `E0`)

Candidate implementation references:
- `worktree/exp-interspeech-GRU2/scripts/h_exploration/generate_lag_omp.py`
- `worktree/exp-interspeech-GRU2/scripts/h_exploration/generate_omp_trajectories.py`

### 3.2 Student (DTmin policy)

Train DTmin on teacher trajectories:
- Inputs: correlation state (and optional RTG channels)
- Outputs: lag action (plus optional STOP token)
- Loss: action cross-entropy (plus optional STOP supervision)

Candidate implementation references:
- `worktree/exp-interspeech-GRU2/scripts/h_exploration/train_dt_lag_seq_rtg.py`
- `worktree/exp-interspeech-GRU2/scripts/h_exploration/train_dtmin_h.py`

### 3.3 Integration into Stage4

In `stage4_doa_ldv_vs_mic_comparison.py` for `ldv_micl`:
- Replace pure OMP alignment path with DTmin policy inference path (`--alignment_mode dtmin`), while keeping OMP mode available (`--alignment_mode omp`) for A/B control.
- Keep strict pass rules unchanged:
  - `omp_vs_raw` logic equivalent for DTmin-vs-raw comparison
  - `theta_only` untouched for `micl_micr`

### 3.4 Evaluation contract

Do not change these controls versus `530a63f`:
- same speakers: `18-0.1V..22-0.1V`
- same scan windows / guided peak logic
- same chirp truth source file
- same strict pass definition and report structure

## 4) Does 530a63f Need Full Re-run?

Short answer for publication-level comparability: **Yes**.

Reason:
- OMP->DTmin changes the alignment decision function, so all previously reported strict metrics for `ldv_micl` become non-comparable unless re-executed with identical protocol.

Decision table:
- If only adding design documentation: no re-run required.
- If making a new performance claim (paper figure/table): full strict re-run required.

Required full re-run scope (minimum):
- All 5 baseline configs from `530a63f`.
- All 5 speakers per config.
- Smoke + guardrail runs.
- Updated `analysis/detailed_report.md` and summary tables.

## 5) Data Path and Format Registry

### 5.1 Stage4 strict speech dataset
- Root: `C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/dataset/GCC-PHAT-LDV-MIC-Experiment`
- Current strict speakers used:
  - `18-0.1V`
  - `19-0.1V`
  - `20-0.1V`
  - `21-0.1V`
  - `22-0.1V`
- File pattern inside each speaker dir:
  - `0128-LDV-<id>-boy-320.wav`
  - `0128-LEFT-MIC-<id>-boy-320.wav`
  - `0128-RIGHT-MIC-<id>-boy-320.wav`

### 5.2 Chirp truth reference
- File:
  - `C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/worktree/exp-ldv-perfect-geometry-cloud/exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation_speech_truthref_chirp_scan_guided_5s_local_20260209_170241/chirp_truthref_5s.json`
- Top-level keys:
  - `timestamp`
  - `method`
  - `truth_ref`
- `truth_ref` keys:
  - `18-0.1V`, `19-0.1V`, `20-0.1V`, `21-0.1V`, `22-0.1V`

### 5.3 OMP/DTmin paired speech dataset (existing E4-family reference)
- MIC root:
  - `C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/dataset/audio/boy1/MIC`
- LDV root:
  - `C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/dataset/audio/boy1/LDV`
- Pair naming pattern:
  - `boy1_papercup_MIC_XXX.wav` <-> `boy1_papercup_LDV_XXX.wav`

### 5.4 Existing strict baseline artifact layout
- Root:
  - `results/ldv_vs_mic_grid_strict_20260211_103715/`
- Core files:
  - `grid_report.md`
  - `grid_summary.md`
  - `grid_summary.json`
  - `grid_run.log`
- Per config directory schema:
  - `<config>/<speaker>/run.log`
  - `<config>/<speaker>/summary.json`
  - `<config>/summary_table.md`
  - `<config>/run_config.json`
  - `<config>/subset_manifest.json`
  - `<config>/code_state.json`

### 5.5 Proposed new artifact layout (OMP->DTmin extension)
- Root:
  - `results/ldv_vs_mic_grid_dtmin_strict_<timestamp>/`
- Subfolders:
  - `teacher_trajectories/`
  - `dtmin_model/`
  - `dtmin_eval/`
  - baseline-compatible per-config result dirs
- Required files:
  - `teacher_trajectories/lag_trajectories.pt`
  - `dtmin_model/model.pt` (or `.pth`)
  - `dtmin_eval/compute_matched_summary.json`
  - `grid_report.md`, `grid_summary.md`, `analysis/detailed_report.md`
  - per-config `code_state.json`, `subset_manifest.json`, run logs

## 6) Execution Plan (Phased)

### Phase A: Lock baseline and protocol
1. Freeze `530a63f` settings into a checked-in config preset file.
2. Add explicit `--alignment_mode {omp,dtmin}` flag.
3. Add strict invariants checks for identical scan settings.

### Phase B: Teacher trajectory generation
1. Build trajectory generator module in this branch (adapt from E4 scripts).
2. Generate trajectories on real speech subset first (smoke), then full 5-speaker strict subset.
3. Save trajectory dataset and diagnostics under `results/<run>/teacher_trajectories/`.

### Phase C: DTmin training
1. Train DTmin on generated trajectories.
2. Save model checkpoints and training logs.
3. Run functional checks (action validity, STOP behavior, no NaN, deterministic seed behavior).

### Phase D: Stage4 strict re-run with DTmin
1. Re-run all 5 configs from `530a63f` using `alignment_mode=dtmin` for `ldv_micl`.
2. Keep `micl_micr` as strict control.
3. Produce baseline-compatible summary and detailed report.

### Phase E: Paper-ready comparison package
1. Produce side-by-side table: `530a63f (OMP strict)` vs `DTmin strict`.
2. Include failure analysis per speaker with causal explanation.
3. Export final report md/pdf and attach reproducibility commands.

## 7) Command Skeleton (to execute later)

```bash
# 1) Teacher trajectories (smoke)
python scripts/generate_omp_dtmin_trajectories.py \
  --data_root "C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/dataset/GCC-PHAT-LDV-MIC-Experiment" \
  --speakers "20-0.1V" \
  --out_dir "results/ldv_vs_mic_dtmin_strict_<ts>/teacher_trajectories_smoke"

# 2) Teacher trajectories (full strict subset)
python scripts/generate_omp_dtmin_trajectories.py \
  --data_root "C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/dataset/GCC-PHAT-LDV-MIC-Experiment" \
  --speakers "18-0.1V,19-0.1V,20-0.1V,21-0.1V,22-0.1V" \
  --out_dir "results/ldv_vs_mic_dtmin_strict_<ts>/teacher_trajectories"

# 3) Train DTmin
python scripts/train_dtmin_from_omp_trajectories.py \
  --traj_path "results/ldv_vs_mic_dtmin_strict_<ts>/teacher_trajectories/lag_trajectories.pt" \
  --out_dir "results/ldv_vs_mic_dtmin_strict_<ts>/dtmin_model"

# 4) Strict grid re-run with DTmin
python scripts/run_ldv_vs_mic_grid.py \
  --data_root "C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/dataset/GCC-PHAT-LDV-MIC-Experiment" \
  --chirp_truth_file "C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/worktree/exp-ldv-perfect-geometry-cloud/exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation_speech_truthref_chirp_scan_guided_5s_local_20260209_170241/chirp_truthref_5s.json" \
  --alignment_mode dtmin \
  --output_base "results/ldv_vs_mic_dtmin_strict_<ts>"

# 5) Analyze
python scripts/analyze_ldv_vs_mic_results.py \
  --grid_base "results/ldv_vs_mic_dtmin_strict_<ts>" \
  --output_dir "results/ldv_vs_mic_dtmin_strict_<ts>/analysis"
```

## 8) Risks and Controls

Primary risks:
- DTmin learns degenerate constant policy.
- Train/eval leakage via shared trajectory windows.
- Non-comparable run settings versus `530a63f`.

Controls:
- Enforce clip/window identifiers in trajectory blocks.
- Keep baseline `530a63f` run-config lockfile and automatic config diff check.
- Add per-speaker diagnostics for OMP-vs-DTmin action divergence.

## 9) Immediate Progress Targets (next commit goals)

Target 1:
- Add `alignment_mode` plumbing and config lockfile.

Target 2:
- Add trajectory generator + schema validation + smoke run artifacts.

Target 3:
- Add DTmin training script + smoke training artifacts.

Target 4:
- Execute full strict re-run and publish side-by-side report (`OMP strict` vs `DTmin strict`).

## 10) Done Criteria for Paper Use

Done when all are true:
- Full strict re-run completed for all 5 configs and 5 speakers.
- DTmin path is reproducible from raw wav roots with exact commands.
- Report includes strict-pass table, failures, and causal interpretation.
- Artifacts include logs, manifests, fingerprints, and code_state snapshots.
