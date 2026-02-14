# Mic-only Ablation Comparison

Generated: 2026-02-13T23:45:03.578337

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_p5/student_ldv_mic`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_p5/student_mic_only`

## Acceptance (Claim 2)

- p95 improvement (A vs B): -0.056 (>= 0.10)
- fail-rate improvement (A vs B): -11.000 (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 318 | 1.552 | 4.612 | 4.871 | 0.038 |
| B (MIC-only) | 318 | 1.163 | 4.463 | 4.612 | 0.003 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 4.884 | 0.038 | 4.765 | 0.000 |
| 19-0.1V | 5.416 | 0.132 | 4.607 | 0.000 |
| 20-0.1V | 4.365 | 0.000 | 4.273 | 0.000 |
| 21-0.1V | 4.882 | 0.016 | 4.739 | 0.016 |
| 22-0.1V | 4.032 | 0.013 | 4.066 | 0.000 |
