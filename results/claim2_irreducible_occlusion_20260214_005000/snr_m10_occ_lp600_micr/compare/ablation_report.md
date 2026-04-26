# Mic-only Ablation Comparison

Generated: 2026-02-14T03:47:51.243604

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m10_occ_lp600_micr/student_ldv_mic`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m10_occ_lp600_micr/student_mic_only`

## Acceptance (Claim 2)

- p95 improvement (A vs B): -0.049 (>= 0.10)
- fail-rate improvement (A vs B): -0.048 (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 318 | 2.604 | 4.660 | 5.621 | 0.069 |
| B (MIC-only) | 318 | 2.597 | 4.767 | 5.357 | 0.066 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 4.933 | 0.051 | 4.922 | 0.051 |
| 19-0.1V | 4.731 | 0.019 | 4.707 | 0.000 |
| 20-0.1V | 4.645 | 0.041 | 4.838 | 0.041 |
| 21-0.1V | 4.514 | 0.016 | 4.514 | 0.016 |
| 22-0.1V | 5.864 | 0.187 | 5.864 | 0.187 |
