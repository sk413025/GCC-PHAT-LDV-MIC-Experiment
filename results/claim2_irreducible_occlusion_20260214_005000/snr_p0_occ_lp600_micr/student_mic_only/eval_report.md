# Band-DTmin Student Report

Generated: 2026-02-14T03:31:03.419713
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_p0_occ_lp600_micr/student_mic_only

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.249 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.689 (>= 0.200)
- median(theta_error_ref) worsening: -0.212 (<= 0.050)
- OVERALL: PASS

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 2.950 | 5.993 | 6.961 | 0.233 |
| student | 318 | 2.324 | 4.664 | 5.225 | 0.072 |
