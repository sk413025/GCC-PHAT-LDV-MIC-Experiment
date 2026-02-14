# Band-DTmin Student Report

Generated: 2026-02-13T23:37:00.638521
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_p10/student_mic_only

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.239 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.958 (>= 0.200)
- median(theta_error_ref) worsening: 0.350 (<= 0.050)
- OVERALL: FAIL

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 1.885 | 5.846 | 5.949 | 0.302 |
| student | 318 | 2.544 | 4.146 | 4.529 | 0.013 |
