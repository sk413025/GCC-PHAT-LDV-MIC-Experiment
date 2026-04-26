# Agent Prompt: Run Claim-2 Mic-Observation Ablation (Coherence vs PSD vs LDV)

## Goal
Run a single fixed near-fail occlusion setting and ablate observation content to determine:
- Whether mic coherence alone is nearly sufficient to learn the band policy, and
- Whether LDV adds marginal predictive information beyond a strong mic-only observation.

This run must be fully reproducible and written to a dedicated results directory.

## Quickstart

### Smoke run (real data, fast)
```bash
python -u scripts/run_claim2_micobs_ablation_suite.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_micobs_ablation_smoke_<YYYYMMDD_HHMMSS> \
  --smoke 1
```

### Full run
```bash
python -u scripts/run_claim2_micobs_ablation_suite.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_micobs_ablation_<YYYYMMDD_HHMMSS> \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V
```

## What to check (verification)
Open `results/<run_name>/report.md` and confirm:
1) **Teacher identity checks**: PASS (actions/noise centers/forbidden mask identical across obs modes).
2) **Near-fail precondition** (pooled test windows):
   - `fail_rate_theta_gt4deg >= 0.40` AND
   - `frac_psr_gt3db <= 0.10`
3) Compare pooled `student p95` and `student fail` across obs modes:
   - `mic_only_coh_only` vs `mic_only_control` (is coherence nearly sufficient?)
   - `mic_only_psd_only` vs `mic_only_control` (does PSD help?)
   - `ldv_mic` vs `mic_only_control` (does LDV add marginal info?)

For exact definitions, see:
- `docs/specs/claim2_micobs_ablation_spec.md`

## Outputs (must exist)
Under `results/<run_name>/`:
- `run.log`, `run_config.json`, `code_state.json`
- `teacher_identity.json`
- `teacher/<obs_mode>/teacher_trajectories.npz`
- `student/<obs_mode>/test_windows.jsonl`
- `summary_table.json`, `report.md`

## Pitfalls (do not do this)
- Do not change the window grid or speech-active filter.
- Do not optimize or filter windows using truth (no cherry-picking).
- Do not optimize against geometry truth; evaluation is vs chirp-reference truth.
- Do not write artifacts to repo root; everything must go under `results/<run_name>/`.
- Remember `results/` may be gitignored; commit artifacts with `git add -f results/<run_name>/`.
