# Mic-only Ablation Comparison

Generated: 2026-02-14T00:01:10.128947

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_m5/student_ldv_mic`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_m5/student_mic_only`

## Acceptance (Claim 2)

- p95 improvement (A vs B): 0.032 (>= 0.10)
- fail-rate improvement (A vs B): 0.000 (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 318 | 2.510 | 4.764 | 5.054 | 0.060 |
| B (MIC-only) | 318 | 2.511 | 4.681 | 5.220 | 0.060 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 5.054 | 0.064 | 5.006 | 0.051 |
| 19-0.1V | 4.918 | 0.038 | 4.918 | 0.019 |
| 20-0.1V | 4.801 | 0.020 | 4.681 | 0.020 |
| 21-0.1V | 4.832 | 0.016 | 4.441 | 0.016 |
| 22-0.1V | 5.816 | 0.133 | 5.816 | 0.160 |
