# Mic-only Ablation Comparison

Generated: 2026-02-14T04:13:14.296322

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m10_occ_tilt2_micr/student_ldv_mic`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m10_occ_tilt2_micr/student_mic_only`

## Acceptance (Claim 2)

- p95 improvement (A vs B): 0.015 (>= 0.10)
- fail-rate improvement (A vs B): 0.045 (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 318 | 2.232 | 4.556 | 5.152 | 0.066 |
| B (MIC-only) | 318 | 2.223 | 4.604 | 5.229 | 0.069 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 4.965 | 0.051 | 4.965 | 0.051 |
| 19-0.1V | 4.323 | 0.019 | 4.606 | 0.038 |
| 20-0.1V | 4.475 | 0.000 | 4.692 | 0.020 |
| 21-0.1V | 4.529 | 0.048 | 4.240 | 0.032 |
| 22-0.1V | 5.912 | 0.173 | 5.935 | 0.173 |
