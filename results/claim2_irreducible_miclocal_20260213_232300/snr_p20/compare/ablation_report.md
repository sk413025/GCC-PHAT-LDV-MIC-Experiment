# Mic-only Ablation Comparison

Generated: 2026-02-13T23:28:55.855700

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_p20/student_ldv_mic`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_p20/student_mic_only`

## Acceptance (Claim 2)

- p95 improvement (A vs B): -0.020 (>= 0.10)
- fail-rate improvement (A vs B): nan (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 318 | 0.907 | 4.055 | 4.422 | 0.006 |
| B (MIC-only) | 318 | 1.422 | 3.983 | 4.334 | 0.000 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 4.610 | 0.013 | 4.406 | 0.000 |
| 19-0.1V | 3.694 | 0.000 | 3.558 | 0.000 |
| 20-0.1V | 4.279 | 0.000 | 4.091 | 0.000 |
| 21-0.1V | 4.484 | 0.016 | 4.407 | 0.000 |
| 22-0.1V | 2.691 | 0.000 | 3.873 | 0.000 |
