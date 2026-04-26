# Mic-only Ablation Comparison

Generated: 2026-02-13T23:37:00.714684

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_p10/student_ldv_mic`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_p10/student_mic_only`

## Acceptance (Claim 2)

- p95 improvement (A vs B): -0.063 (>= 0.10)
- fail-rate improvement (A vs B): -2.000 (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 318 | 2.332 | 4.427 | 4.816 | 0.038 |
| B (MIC-only) | 318 | 2.544 | 4.146 | 4.529 | 0.013 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 5.959 | 0.128 | 4.873 | 0.026 |
| 19-0.1V | 4.305 | 0.000 | 4.305 | 0.000 |
| 20-0.1V | 4.242 | 0.000 | 4.146 | 0.000 |
| 21-0.1V | 4.810 | 0.032 | 4.698 | 0.032 |
| 22-0.1V | 3.273 | 0.000 | 3.435 | 0.000 |
