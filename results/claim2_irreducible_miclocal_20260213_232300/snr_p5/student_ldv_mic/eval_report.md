# Band-DTmin Student Report

Generated: 2026-02-13T23:42:43.070320
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_p5/student_ldv_mic

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.184 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.874 (>= 0.200)
- median(theta_error_ref) worsening: -0.175 (<= 0.050)
- OVERALL: PASS

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 1.881 | 5.844 | 5.966 | 0.299 |
| student | 318 | 1.552 | 4.612 | 4.871 | 0.038 |
