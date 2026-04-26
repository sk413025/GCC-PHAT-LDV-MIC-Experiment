# Band-DTmin Student Report

Generated: 2026-02-13T23:50:45.790327
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_p0/student_ldv_mic

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.186 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.865 (>= 0.200)
- median(theta_error_ref) worsening: 0.376 (<= 0.050)
- OVERALL: FAIL

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 1.724 | 5.892 | 6.013 | 0.280 |
| student | 318 | 2.371 | 4.668 | 4.893 | 0.038 |
