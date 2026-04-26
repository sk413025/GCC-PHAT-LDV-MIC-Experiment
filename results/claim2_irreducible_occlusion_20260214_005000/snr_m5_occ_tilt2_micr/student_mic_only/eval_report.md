# Band-DTmin Student Report

Generated: 2026-02-14T04:04:45.535602
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m5_occ_tilt2_micr/student_mic_only

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.423 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.635 (>= 0.200)
- median(theta_error_ref) worsening: -0.011 (<= 0.050)
- OVERALL: PASS

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 2.828 | 6.916 | 9.721 | 0.267 |
| student | 318 | 2.798 | 4.938 | 5.613 | 0.097 |
