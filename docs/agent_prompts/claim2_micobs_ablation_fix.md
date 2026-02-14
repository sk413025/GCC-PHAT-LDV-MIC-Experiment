# Agent Prompt: Run Claim-2 Ablation v2 (Soft Coupling + Dynamic Mic-Coherence Gate)

## Goal
Re-run the Claim-2 mic-observation ablation under the same fixed near-fail occlusion setting, but with:

1) **Soft coupling only** (no silence-coupling hard forbids), and  
2) **Dynamic per-window mic-coherence gating** (forbid very-low-coherence bands).

The purpose is to eliminate **speaker-dependent negative transfer** where LDV+MIC helps pooled tails but hurts some speakers (notably `21-0.1V` in the v1 run).

## Quickstart

### Smoke run (real data, fast)
```bash
python -u scripts/run_claim2_micobs_ablation_suite.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_micobs_ablation_fix_smoke_<YYYYMMDD_HHMMSS> \
  --smoke 1 \
  --coupling_hard_forbid_enable 0 \
  --dynamic_coh_gate_enable 1 \
  --dynamic_coh_min 0.05
```

### Full run
```bash
python -u scripts/run_claim2_micobs_ablation_suite.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_micobs_ablation_fix_<YYYYMMDD_HHMMSS> \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V \
  --coupling_hard_forbid_enable 0 \
  --dynamic_coh_gate_enable 1 \
  --dynamic_coh_min 0.05
```

## What to check (verification)
Open `results/<run_name>/report.md` and confirm:
1) **Teacher identity checks**: PASS (actions/noise centers/forbidden mask identical across obs modes).
2) **Near-fail precondition** (pooled test windows):
   - `fail_rate_theta_gt4deg >= 0.40` AND
   - `frac_psr_gt3db <= 0.10`
3) **Negative transfer check (per speaker)**:
   - No speaker should be flagged by:
     - `p95_ldv > 1.05 * p95_mic` OR
     - `fail_ldv > fail_mic + 0.02`
   - Pay special attention to `21-0.1V`.

For definitions and acceptance rules, see:
- `docs/specs/claim2_micobs_ablation_fix_spec.md`

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
