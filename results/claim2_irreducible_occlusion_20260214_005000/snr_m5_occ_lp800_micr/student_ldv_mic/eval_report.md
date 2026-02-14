# Band-DTmin Student Report

Generated: 2026-02-14T03:11:51.907357
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m5_occ_lp800_micr/student_ldv_mic

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.344 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.832 (>= 0.200)
- median(theta_error_ref) worsening: -0.375 (<= 0.050)
- OVERALL: PASS

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 4.465 | 6.964 | 8.236 | 0.355 |
| student | 318 | 2.790 | 4.621 | 5.406 | 0.060 |
