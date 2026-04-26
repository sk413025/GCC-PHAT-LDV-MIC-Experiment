# Mic-only Ablation Comparison

Generated: 2026-02-14T03:14:14.050902

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m5_occ_lp800_micr/student_ldv_mic`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m5_occ_lp800_micr/student_mic_only`

## Acceptance (Claim 2)

- p95 improvement (A vs B): 0.008 (>= 0.10)
- fail-rate improvement (A vs B): 0.095 (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 318 | 2.790 | 4.621 | 5.406 | 0.060 |
| B (MIC-only) | 318 | 2.716 | 4.623 | 5.447 | 0.066 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 4.835 | 0.038 | 4.816 | 0.038 |
| 19-0.1V | 4.700 | 0.038 | 4.632 | 0.019 |
| 20-0.1V | 4.465 | 0.000 | 4.652 | 0.041 |
| 21-0.1V | 4.729 | 0.016 | 4.565 | 0.048 |
| 22-0.1V | 5.789 | 0.173 | 5.789 | 0.160 |
