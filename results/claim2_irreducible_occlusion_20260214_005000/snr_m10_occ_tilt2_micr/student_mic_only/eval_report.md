# Band-DTmin Student Report

Generated: 2026-02-14T04:13:14.216863
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m10_occ_tilt2_micr/student_mic_only

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.230 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.651 (>= 0.200)
- median(theta_error_ref) worsening: -0.196 (<= 0.050)
- OVERALL: PASS

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 2.765 | 6.151 | 6.789 | 0.198 |
| student | 318 | 2.223 | 4.604 | 5.229 | 0.069 |
