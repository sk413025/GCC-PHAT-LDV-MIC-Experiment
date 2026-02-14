# Band-DTmin Student Report

Generated: 2026-02-14T17:40:37.045186
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_micobs_ablation_metricfix_smoke_20260214_174021/student/mic_only_psd_only

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.000 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.000 (>= 0.200)
- median(theta_error_ref) worsening: -0.260 (<= 0.050)
- OVERALL: FAIL

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 4 | 3.103 | 4.019 | 4.083 | 0.000 |
| student | 4 | 2.295 | 4.019 | 4.083 | 0.000 |
