# Band-DTmin Student Report

Generated: 2026-02-14T03:45:28.373828
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m10_occ_lp600_micr/student_ldv_mic

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.105 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.614 (>= 0.200)
- median(theta_error_ref) worsening: -0.010 (<= 0.050)
- OVERALL: FAIL

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 2.631 | 5.742 | 6.278 | 0.179 |
| student | 318 | 2.604 | 4.660 | 5.621 | 0.069 |
