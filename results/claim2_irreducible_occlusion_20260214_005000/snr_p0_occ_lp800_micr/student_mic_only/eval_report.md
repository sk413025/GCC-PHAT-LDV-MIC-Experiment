# Band-DTmin Student Report

Generated: 2026-02-14T03:05:49.460559
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_p0_occ_lp800_micr/student_mic_only

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.506 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.847 (>= 0.200)
- median(theta_error_ref) worsening: -0.483 (<= 0.050)
- OVERALL: PASS

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 4.835 | 6.919 | 10.606 | 0.453 |
| student | 318 | 2.497 | 4.638 | 5.241 | 0.069 |
