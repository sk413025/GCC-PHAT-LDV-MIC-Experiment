# Agent Prompt: Run Claim 2 Verification — “LDV Is Irreducible Under Mic-Local Degradation” (Existing WAVs Only)

## Goal
Verify (or falsify) the claim:

> Under mic-local degradations (noise + saturation/clipping affecting microphones but not LDV), an LDV+MIC student policy beats an otherwise identical MIC-only student on speech-tail robustness when stabilizing **MIC–MIC guided GCC-PHAT**.

This is an **existing-WAVs-only** experiment (no new recordings).

## Quickstart

### Smoke test (fast)
```bash
python -u scripts/run_claim2_ldv_irreducible_miclocal_sweep.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_irreducible_miclocal_smoke_<YYYYMMDD_HHMMSS> \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V \
  --smoke 1
```

### Full run (the real experiment)
```bash
python -u scripts/run_claim2_ldv_irreducible_miclocal_sweep.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_irreducible_miclocal_<YYYYMMDD_HHMMSS> \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V
```

## What to read for pass/fail
- Primary report: `results/<run>/report.md`
- Machine summary: `results/<run>/summary_table.json`
- Per-SNR compare reports:
  - `results/<run>/snr_*/compare/ablation_report.md`

## Acceptance (Claim 2 supported)
We declare Claim 2 supported if **at least one** severity in `{0, -5, -10} dB` satisfies:

1) **Near-fail baseline**: baseline MIC–MIC `fail_rate_ref(theta_error_ref_deg > 4°) >= 0.40`
2) **LDV feasibility**: LDV+MIC student `fail_rate_ref(>4°) <= 0.10`
3) **Irreducibility ablation**: LDV+MIC student beats MIC-only student:
   - p95 improvement ≥ 10% and fail-rate improvement ≥ 20% (relative)

If (1) never happens, record the negative result: the dataset/corruption proxy didn’t create sufficiently extreme windows.

## Important pitfalls (do not do these)
1) Do not redefine “extreme” using truth (no filtering by theta/tau error).
2) Do not change the window grid or RMS filter (avoid cherry-picking).
3) Do not inject low-frequency-only noise and then claim “wind” — the GCC band is 500–2000 Hz, so the stress must be **in-band**.
4) Do not use LDV-derived coupling masks in the primary ablation; use mic-only coupling so the test isolates **dynamic LDV features**.
5) Do not silently pad/resample/out-of-bounds windows; fail fast.

## Expected artifacts (sanity checklist)
Under `results/<run>/snr_*/teacher_*`:
- `teacher_trajectories.npz` includes `noise_center_sec_L/R`, `clip_frac_L/R`, and `corruption_config_json`.

Under `results/<run>/snr_*/student_*`:
- `test_windows.jsonl` includes a `corruption` block per window (noise centers + achieved SNR + clip_frac).

## If it fails (what it means)
- If baseline never near-fails: the current proxy is not strong enough; expand severity or add a new corruption mode (must be pre-registered).
- If near-fail occurs but LDV doesn’t beat MIC-only: LDV features are not providing unique predictive power under this dataset + student model.
