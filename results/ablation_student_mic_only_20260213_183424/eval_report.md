# Band-DTmin Student Report

Generated: 2026-02-13T18:46:51.384108
Run dir: results/ablation_student_mic_only_20260213_183424

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.463 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.969 (>= 0.200)
- median(theta_error_ref) worsening: -0.887 (<= 0.050)
- OVERALL: PASS

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 2.179 | 5.845 | 5.934 | 0.308 |
| student | 318 | 0.246 | 1.488 | 3.188 | 0.009 |
