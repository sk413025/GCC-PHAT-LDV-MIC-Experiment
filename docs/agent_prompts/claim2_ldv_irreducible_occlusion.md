# Agent Prompt: Run Claim 2 Verification — Occlusion + Noise/Clipping Sweep (Existing WAVs Only)

## Goal
Test whether LDV is **irreducible** under **mic-local occlusion-style degradations**:

- MicL/MicR are degraded by in-band noise + saturation (gain→clip→de-gain).
- Additionally, **one mic only** (default MicR) is spectrally distorted (lowpass or spectral tilt).
- LDV remains unchanged.

Success is defined by the Claim-2 acceptance criteria in:
`docs/specs/claim2_ldv_irreducible_occlusion_spec.md`.

## Quickstart

### Smoke test (fast)
```bash
python -u scripts/run_claim2_ldv_irreducible_occlusion_sweep.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_irreducible_occlusion_smoke_<YYYYMMDD_HHMMSS> \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V \
  --smoke 1
```

### Full run
```bash
python -u scripts/run_claim2_ldv_irreducible_occlusion_sweep.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_irreducible_occlusion_<YYYYMMDD_HHMMSS> \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V
```

## What to read
- Primary report: `results/<run>/report.md`
- Machine summary: `results/<run>/summary_table.json`
- Per-grid-point ablation reports:
  - `results/<run>/<tag>/compare/ablation_report.md`

## Pitfalls
1) Do not change the window grid or RMS filter.
2) Do not define stress subsets using truth.
3) Ensure occlusion is mic-local (only one mic), and LDV stays clean.
4) Ensure evaluation replays the *exact* corruption (noise centers are saved in teacher trajectories).
5) Remember `results/` is gitignored: commit artifacts with `git add -f results/<run>/`.

