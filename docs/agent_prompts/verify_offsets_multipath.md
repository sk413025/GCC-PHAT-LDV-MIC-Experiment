# Agent Prompt: Verify "Offsets + Sparse Multipath" Measurement Chain

## Mission
Test whether the system is dominated by:
1) stable constant offsets / stable non-LOS paths, and
2) only a few dominant paths (sparse multipath),
using existing real WAVs only.

This is intended to support (or falsify) a physics narrative: "the measured GCC peaks are not geometry-driven direct-path TDOA under the current coupling/offset chain."

## Quickstart
```bash
python -u scripts/verify_offsets_and_sparse_multipath.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/offsets_multipath_verify_<YYYYMMDD_HHMMSS> \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V
```

## What to read
- `results/offsets_multipath_verify_<ts>/offsets_report.md`
- `results/offsets_multipath_verify_<ts>/pair_offset_estimates.json`
- Per-window peaks: `results/offsets_multipath_verify_<ts>/per_speaker/<speaker>/windows.jsonl`

## Acceptance (Claim 3)
Read the `Acceptance` section in `offsets_report.md`.
PASS/FAIL is recorded explicitly. If FAIL, do not tune thresholds; commit the negative result with analysis.

## Pitfalls
1) Do not interpret these taus as direct-path geometry TDOA unless offsets are removed and residual varies with speaker geometry.
2) Do not change window selection; keep the plan-locked grid + RMS rules.
3) Do not write artifacts to repo root; everything goes under `results/<run_name>/`.

