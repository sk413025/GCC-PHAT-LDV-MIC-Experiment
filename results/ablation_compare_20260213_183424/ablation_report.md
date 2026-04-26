# Mic-only Ablation Comparison

Generated: 2026-02-13T18:47:49.167819

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/ablation_student_ldv_mic_20260213_183424`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/ablation_student_mic_only_20260213_183424`

## Acceptance (Claim 2)

- p95 improvement (A vs B): -0.200 (>= 0.10)
- fail-rate improvement (A vs B): -0.667 (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 318 | 0.294 | 1.574 | 3.825 | 0.016 |
| B (MIC-only) | 318 | 0.246 | 1.488 | 3.188 | 0.009 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 0.703 | 0.013 | 1.542 | 0.000 |
| 19-0.1V | 6.834 | 0.075 | 4.737 | 0.038 |
| 20-0.1V | 3.842 | 0.000 | 4.504 | 0.020 |
| 21-0.1V | 0.808 | 0.000 | 0.806 | 0.000 |
| 22-0.1V | 1.670 | 0.000 | 0.741 | 0.000 |
