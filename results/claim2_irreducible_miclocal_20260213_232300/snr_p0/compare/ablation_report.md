# Mic-only Ablation Comparison

Generated: 2026-02-13T23:53:06.193462

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_p0/student_ldv_mic`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_p0/student_mic_only`

## Acceptance (Claim 2)

- p95 improvement (A vs B): -0.000 (>= 0.10)
- fail-rate improvement (A vs B): 0.143 (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 318 | 2.371 | 4.668 | 4.893 | 0.038 |
| B (MIC-only) | 318 | 2.612 | 4.560 | 4.893 | 0.044 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 5.086 | 0.051 | 5.506 | 0.115 |
| 19-0.1V | 5.060 | 0.075 | 5.028 | 0.057 |
| 20-0.1V | 4.327 | 0.000 | 4.298 | 0.000 |
| 21-0.1V | 4.859 | 0.032 | 4.617 | 0.016 |
| 22-0.1V | 4.786 | 0.027 | 4.416 | 0.013 |
