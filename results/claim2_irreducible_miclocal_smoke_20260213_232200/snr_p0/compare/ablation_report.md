# Mic-only Ablation Comparison

Generated: 2026-02-13T23:20:47.149890

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_smoke_20260213_232200/snr_p0/student_ldv_mic`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_smoke_20260213_232200/snr_p0/student_mic_only`

## Acceptance (Claim 2)

- p95 improvement (A vs B): 0.000 (>= 0.10)
- fail-rate improvement (A vs B): nan (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 4 | 2.530 | 4.610 | 4.715 | 0.000 |
| B (MIC-only) | 4 | 2.530 | 4.610 | 4.715 | 0.000 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 20-0.1V | 4.715 | 0.000 | 4.715 | 0.000 |
