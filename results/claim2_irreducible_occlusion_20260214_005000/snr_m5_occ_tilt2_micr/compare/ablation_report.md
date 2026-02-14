# Mic-only Ablation Comparison

Generated: 2026-02-14T04:04:45.615483

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m5_occ_tilt2_micr/student_ldv_mic`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m5_occ_tilt2_micr/student_mic_only`

## Acceptance (Claim 2)

- p95 improvement (A vs B): -0.019 (>= 0.10)
- fail-rate improvement (A vs B): -0.032 (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 318 | 2.832 | 4.937 | 5.719 | 0.101 |
| B (MIC-only) | 318 | 2.798 | 4.938 | 5.613 | 0.097 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 5.151 | 0.090 | 5.210 | 0.103 |
| 19-0.1V | 4.999 | 0.057 | 4.664 | 0.038 |
| 20-0.1V | 4.919 | 0.061 | 5.187 | 0.102 |
| 21-0.1V | 5.222 | 0.063 | 4.725 | 0.048 |
| 22-0.1V | 5.892 | 0.200 | 5.876 | 0.173 |
