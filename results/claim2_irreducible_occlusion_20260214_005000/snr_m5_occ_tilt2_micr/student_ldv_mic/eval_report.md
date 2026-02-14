# Band-DTmin Student Report

Generated: 2026-02-14T04:02:23.076602
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m5_occ_tilt2_micr/student_ldv_mic

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.412 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.624 (>= 0.200)
- median(theta_error_ref) worsening: 0.001 (<= 0.050)
- OVERALL: PASS

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 2.828 | 6.916 | 9.721 | 0.267 |
| student | 318 | 2.832 | 4.937 | 5.719 | 0.101 |
