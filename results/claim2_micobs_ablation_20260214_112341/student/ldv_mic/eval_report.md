# Band-DTmin Student Report

Generated: 2026-02-14T11:33:20.144087
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_micobs_ablation_20260214_112341/student/ldv_mic

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.549 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.924 (>= 0.200)
- median(theta_error_ref) worsening: -0.464 (<= 0.050)
- OVERALL: PASS

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 4.835 | 6.919 | 10.606 | 0.453 |
| student | 318 | 2.591 | 4.444 | 4.782 | 0.035 |
