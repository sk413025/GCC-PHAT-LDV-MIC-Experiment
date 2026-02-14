# Mic-only Ablation Comparison

Generated: 2026-02-14T03:22:37.405908

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m10_occ_lp800_micr/student_ldv_mic`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m10_occ_lp800_micr/student_mic_only`

## Acceptance (Claim 2)

- p95 improvement (A vs B): 0.034 (>= 0.10)
- fail-rate improvement (A vs B): 0.000 (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 318 | 2.579 | 4.655 | 5.506 | 0.072 |
| B (MIC-only) | 318 | 2.590 | 4.785 | 5.702 | 0.072 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 4.942 | 0.051 | 5.011 | 0.051 |
| 19-0.1V | 5.243 | 0.057 | 4.907 | 0.038 |
| 20-0.1V | 4.401 | 0.000 | 4.787 | 0.020 |
| 21-0.1V | 4.460 | 0.016 | 4.699 | 0.016 |
| 22-0.1V | 5.871 | 0.200 | 5.871 | 0.200 |
