# Mic-only Ablation Comparison

Generated: 2026-02-14T00:09:14.002685

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_m10/student_ldv_mic`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_m10/student_mic_only`

## Acceptance (Claim 2)

- p95 improvement (A vs B): -0.044 (>= 0.10)
- fail-rate improvement (A vs B): -0.261 (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 318 | 2.424 | 4.892 | 5.657 | 0.091 |
| B (MIC-only) | 318 | 2.486 | 4.654 | 5.419 | 0.072 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 5.477 | 0.103 | 4.964 | 0.051 |
| 19-0.1V | 4.910 | 0.038 | 4.681 | 0.019 |
| 20-0.1V | 4.576 | 0.041 | 4.671 | 0.041 |
| 21-0.1V | 4.659 | 0.048 | 4.856 | 0.048 |
| 22-0.1V | 5.879 | 0.187 | 5.879 | 0.173 |
