# Mic-only Ablation Comparison

Generated: 2026-02-14T03:56:21.412659

Run A (LDV+MIC student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_p0_occ_tilt2_micr/student_ldv_mic`

Run B (MIC-only student): `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_p0_occ_tilt2_micr/student_mic_only`

## Acceptance (Claim 2)

- p95 improvement (A vs B): -0.052 (>= 0.10)
- fail-rate improvement (A vs B): -0.121 (>= 0.20)
- OVERALL: FAIL

## Teacher action identity check

- actions_identical: True
- diff_count: 0

## Pooled test metrics (vs chirp reference)

| Run | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (LDV+MIC) | 318 | 2.613 | 5.195 | 5.765 | 0.116 |
| B (MIC-only) | 318 | 2.694 | 5.019 | 5.478 | 0.104 |

## Per-speaker p95 / fail-rate (A vs B)

| speaker | A p95 | A fail | B p95 | B fail |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 5.836 | 0.141 | 5.239 | 0.103 |
| 19-0.1V | 5.477 | 0.094 | 5.309 | 0.094 |
| 20-0.1V | 4.951 | 0.061 | 5.104 | 0.082 |
| 21-0.1V | 4.838 | 0.048 | 4.849 | 0.048 |
| 22-0.1V | 5.890 | 0.200 | 5.890 | 0.173 |
