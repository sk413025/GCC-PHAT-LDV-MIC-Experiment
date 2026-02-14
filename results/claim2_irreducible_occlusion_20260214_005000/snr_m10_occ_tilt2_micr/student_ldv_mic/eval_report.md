# Band-DTmin Student Report

Generated: 2026-02-14T04:10:51.156725
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m10_occ_tilt2_micr/student_ldv_mic

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.241 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.667 (>= 0.200)
- median(theta_error_ref) worsening: -0.193 (<= 0.050)
- OVERALL: PASS

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 2.765 | 6.151 | 6.789 | 0.198 |
| student | 318 | 2.232 | 4.556 | 5.152 | 0.066 |
