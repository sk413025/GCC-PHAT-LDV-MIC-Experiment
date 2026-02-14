# Mic-only Ablation Comparison

Generated: 2026-02-14T03:05:49.542153

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_p0_occ_lp800_micr/student_ldv_mic`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_p0_occ_lp800_micr/student_mic_only`

## Acceptance (Claim 2)

- p95 improvement (A vs B): 0.088 (>= 0.10)
- fail-rate improvement (A vs B): 0.500 (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 318 | 2.591 | 4.444 | 4.782 | 0.035 |
| B (MIC-only) | 318 | 2.497 | 4.638 | 5.241 | 0.069 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 4.659 | 0.013 | 5.657 | 0.141 |
| 19-0.1V | 4.584 | 0.038 | 4.675 | 0.019 |
| 20-0.1V | 4.415 | 0.000 | 4.308 | 0.000 |
| 21-0.1V | 4.720 | 0.032 | 4.422 | 0.016 |
| 22-0.1V | 5.104 | 0.080 | 5.533 | 0.120 |
