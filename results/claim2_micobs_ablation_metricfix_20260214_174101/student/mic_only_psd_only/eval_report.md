# Band-DTmin Student Report

Generated: 2026-02-14T17:57:24.156761
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_micobs_ablation_metricfix_20260214_174101/student/mic_only_psd_only

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.033 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.000 (>= 0.200)
- median(theta_error_ref) worsening: -0.424 (<= 0.050)
- OVERALL: FAIL

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 4.272 | 4.570 | 4.625 | 0.000 |
| student | 318 | 2.462 | 4.472 | 4.472 | 0.000 |
