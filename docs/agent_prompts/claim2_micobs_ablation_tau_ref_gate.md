# Agent Prompt: Run Claim-2 Ablation v3 (Tau-Ref Support Gate)

## Goal
Run the Claim-2 mic-observation ablation under the same fixed near-fail occlusion setting, but with an additional **tau-ref support gate** to prevent selecting “coherent-but-wrong” bands.

This run is intended to eliminate **speaker-dependent negative transfer** (LDV+MIC hurting some speakers) while keeping pooled LDV benefit positive vs mic-only(control).

## Quickstart

### Smoke run (real data, fast)
```bash
python -u scripts/run_claim2_micobs_ablation_suite.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_micobs_ablation_tau_ref_gate_smoke_<YYYYMMDD_HHMMSS> \
  --smoke 1 \
  --coupling_hard_forbid_enable 0 \
  --dynamic_coh_gate_enable 1 \
  --dynamic_coh_min 0.05 \
  --tau_ref_gate_enable 1 \
  --tau_ref_gate_ratio_min 0.60
```

### Full run
```bash
python -u scripts/run_claim2_micobs_ablation_suite.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_micobs_ablation_tau_ref_gate_<YYYYMMDD_HHMMSS> \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V \
  --coupling_hard_forbid_enable 0 \
  --dynamic_coh_gate_enable 1 \
  --dynamic_coh_min 0.05 \
  --tau_ref_gate_enable 1 \
  --tau_ref_gate_ratio_min 0.60
```

## What to check (verification)
Open `results/<run_name>/report.md` and confirm:
1) **Policy gates** section lists:
   - `coupling_hard_forbid_enable: False`
   - `dynamic_coh_gate_enable: True (coh_min=0.05)`
   - `tau_ref_gate_enable: True (ratio_min=0.60)`
2) **Teacher identity checks**: PASS (actions/noise centers/forbidden mask identical across obs modes).
3) **Near-fail precondition**: baseline `fail_rate_ref(>5°) >= 0.40`.
4) **Negative transfer check (per speaker)**:
   - No speaker should be flagged.
5) **Pooled guardrail**:
   - `ldv_mic` should improve vs `mic_only_control` by:
     - `p95_improvement_frac >= 0.05`
     - `fail_rate_improvement_frac >= 0.10`

For definitions and acceptance rules, see:
- `docs/specs/claim2_micobs_ablation_tau_ref_gate_spec.md`

## Outputs (must exist)
Under `results/<run_name>/`:
- `run.log`, `run_config.json`, `code_state.json`
- `teacher_identity.json`
- `teacher/<obs_mode>/teacher_trajectories.npz`
- `student/<obs_mode>/test_windows.jsonl`
- `summary_table.json`, `report.md`

## Pitfalls (do not do this)
- Do not change the window grid or the speech-active RMS filter.
- Do not use truth to filter windows (no cherry-picking).
- Do not optimize against geometry truth; evaluation is vs chirp-reference truth.
- Do not write artifacts to repo root; everything goes under `results/<run_name>/`.
- Remember `results/` may be gitignored; commit artifacts with `git add -f results/<run_name>/`.

