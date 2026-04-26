# Band-DTmin Student Report

Generated: 2026-02-14T03:53:58.775439
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_p0_occ_tilt2_micr/student_ldv_mic

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.362 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.532 (>= 0.200)
- median(theta_error_ref) worsening: 0.095 (<= 0.050)
- OVERALL: FAIL

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 2.386 | 6.706 | 9.036 | 0.248 |
| student | 318 | 2.613 | 5.195 | 5.765 | 0.116 |
