# Mic-only Ablation Comparison

Generated: 2026-02-14T03:31:03.499886

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_p0_occ_lp600_micr/student_ldv_mic`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_p0_occ_lp600_micr/student_mic_only`

## Acceptance (Claim 2)

- p95 improvement (A vs B): -0.027 (>= 0.10)
- fail-rate improvement (A vs B): -0.043 (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 318 | 2.539 | 4.700 | 5.367 | 0.075 |
| B (MIC-only) | 318 | 2.324 | 4.664 | 5.225 | 0.072 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 5.732 | 0.115 | 5.391 | 0.090 |
| 19-0.1V | 4.846 | 0.038 | 4.419 | 0.019 |
| 20-0.1V | 4.613 | 0.041 | 5.091 | 0.082 |
| 21-0.1V | 5.344 | 0.063 | 5.368 | 0.079 |
| 22-0.1V | 5.182 | 0.093 | 5.208 | 0.080 |
