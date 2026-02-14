# Band-DTmin Student Report

Generated: 2026-02-14T00:06:53.530901
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_m10/student_ldv_mic

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.124 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.567 (>= 0.200)
- median(theta_error_ref) worsening: 0.058 (<= 0.050)
- OVERALL: FAIL

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 2.292 | 5.932 | 6.459 | 0.211 |
| student | 318 | 2.424 | 4.892 | 5.657 | 0.091 |
