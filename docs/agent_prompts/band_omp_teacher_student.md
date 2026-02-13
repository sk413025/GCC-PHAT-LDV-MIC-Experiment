# Agent Prompt: Band-OMP Teacher → Band-DTmin Student (Speech Tail Robustness)

## Mission
Implement and run a **reproducible** teacher→student experiment that tries to make **LDV+MIC beat MIC-MIC** on **speech tail robustness** by learning a **sparse, OMP-like frequency-band selection** policy for MIC-MIC GCC-PHAT.

This experiment assumes:
- LDV↔MIC `τ₂/τ₃` is not currently usable as a geometry constraint (offset/coupling dominates).
- Chirp-based mic reference truth (`tau_ref_ms`, `theta_ref_deg`) is available per speaker.

Your job is to:
1) Generate teacher trajectories (band actions) using chirp truth.
2) Train a DTmin student to imitate the teacher’s band selections.
3) Evaluate student vs baseline MIC-MIC on held-out speech windows.
4) Record results and avoid known pitfalls.

## Success criteria (locked)
We claim “LDV+MIC beats MIC-MIC” if **student** improves vs **baseline MIC-MIC** on pooled **test** speech windows (vs chirp reference):
- `p95(theta_error_ref_deg)` improves by **≥ 15%**
- `fail_rate_ref(theta_error_ref_deg > 5°)` improves by **≥ 20%**
- Guardrail: `median(theta_error_ref_deg)` worsens by **≤ 5%**

## Quickstart (commands)

### 1) Teacher
```bash
python -u scripts/teacher_band_omp_micmic.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/band_omp_teacher_<YYYYMMDD_HHMMSS> \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V
```

### 2) Student
```bash
python -u scripts/train_dtmin_from_band_trajectories.py \
  --traj_path results/band_omp_teacher_<ts>/teacher_trajectories.npz \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/band_dtmin_student_<YYYYMMDD_HHMMSS>
```

## What to read to verify success

### Teacher sanity gate (must not be badly worse than baseline)
- File: `results/band_omp_teacher_<ts>/grid_report.md`
- Condition: `median(theta_error_ref)` worsening frac ≤ 0.05

### Student final decision (the real win)
- File: `results/band_dtmin_student_<ts>/eval_report.md`
- Look at:
  - `p95(theta_error_ref)` improvement
  - `fail_rate_ref(>5°)` improvement
  - `median(theta_error_ref)` worsening
  - `OVERALL: PASS/FAIL`

## Artifacts checklist (must exist)
Teacher run dir must include:
- `run.log`, `run_config.json`, `code_state.json`, `manifest.json`
- `per_speaker/<speaker>/coupling_mask.json`
- `per_speaker/<speaker>/windows.jsonl`
- `teacher_trajectories.npz`
- `summary.json`, `grid_report.md`

Student run dir must include:
- `run.log`, `code_state.json`, `manifest.json`
- `train_summary.json`, `model_dtmin_band_policy_k*.npz`
- `test_windows.jsonl`, `summary.json`, `eval_report.md`

## Pitfalls (do NOT do these)
1. Do not change the window grid or speech-active filter (100..600 step 1, RMS p50).
2. Do not optimize or gate on geometry truth for speech tails; use chirp-reference truth.
3. Do not remove the silence-window coupling mask; LDV energy can be dominated by coupling.
4. Do not silently resample or pad windows; fail fast on fs mismatch and bounds.
5. Do not write outputs to repo root; everything goes under `results/<run_name>/`.
6. Remember `results/` is gitignored; for commits you must `git add -f results/<run_name>/`.

## Troubleshooting hints
- If `forbidden_band_count == 64`, the raw hard-forbid rule has degenerated (coupling is high everywhere).
  In that case the implementation must disable the hard-forbid and use the coupling curve only as a penalty feature.
- If the teacher is worse than baseline, the scoring/greedy selection may be selecting bands that increase GCC sidelobes in the guided window. Do not “tune” heuristics silently—record failure and pivot to hardware isolation or offset calibration experiments.
- If the student is much worse than the teacher, the observation vector may be insufficient (information bottleneck) or the nearest-centroid student may be too weak. Record this clearly; do not change the model without a new plan and a new executed results commit.
