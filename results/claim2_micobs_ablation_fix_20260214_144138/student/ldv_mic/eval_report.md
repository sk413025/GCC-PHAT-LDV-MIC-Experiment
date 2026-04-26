# Band-DTmin Student Report

Generated: 2026-02-14T14:51:00.499553
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_micobs_ablation_fix_20260214_144138/student/ldv_mic

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.304 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.569 (>= 0.200)
- median(theta_error_ref) worsening: -0.370 (<= 0.050)
- OVERALL: PASS

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 4.835 | 6.919 | 10.606 | 0.453 |
| student | 318 | 3.046 | 5.911 | 7.387 | 0.195 |
