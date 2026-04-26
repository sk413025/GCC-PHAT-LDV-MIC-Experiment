# Band-DTmin Student Report

Generated: 2026-02-13T23:26:36.480747
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_p20/student_ldv_mic

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.257 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.979 (>= 0.200)
- median(theta_error_ref) worsening: -0.574 (<= 0.050)
- OVERALL: PASS

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 2.130 | 5.850 | 5.950 | 0.296 |
| student | 318 | 0.907 | 4.055 | 4.422 | 0.006 |
