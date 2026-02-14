# Band-DTmin Student Report

Generated: 2026-02-14T14:55:44.050711
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_micobs_ablation_fix_20260214_144138/student/mic_only_coh_only

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.369 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.576 (>= 0.200)
- median(theta_error_ref) worsening: -0.390 (<= 0.050)
- OVERALL: PASS

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 4.835 | 6.919 | 10.606 | 0.453 |
| student | 318 | 2.948 | 5.494 | 6.691 | 0.192 |
