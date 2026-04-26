# Mic-only Ablation Comparison

Generated: 2026-02-14T03:39:27.853732

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m5_occ_lp600_micr/student_ldv_mic`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m5_occ_lp600_micr/student_mic_only`

## Acceptance (Claim 2)

- p95 improvement (A vs B): 0.022 (>= 0.10)
- fail-rate improvement (A vs B): -0.053 (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 318 | 2.487 | 4.622 | 5.512 | 0.063 |
| B (MIC-only) | 318 | 2.597 | 4.614 | 5.636 | 0.060 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 5.607 | 0.064 | 4.928 | 0.038 |
| 19-0.1V | 4.751 | 0.038 | 5.032 | 0.057 |
| 20-0.1V | 4.609 | 0.000 | 4.411 | 0.000 |
| 21-0.1V | 4.518 | 0.000 | 4.520 | 0.000 |
| 22-0.1V | 5.753 | 0.173 | 5.794 | 0.173 |
