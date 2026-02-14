# Band-DTmin Student Report

Generated: 2026-02-14T03:20:15.229579
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m10_occ_lp800_micr/student_ldv_mic

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.242 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.671 (>= 0.200)
- median(theta_error_ref) worsening: -0.202 (<= 0.050)
- OVERALL: PASS

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 3.232 | 5.959 | 7.266 | 0.220 |
| student | 318 | 2.579 | 4.655 | 5.506 | 0.072 |
