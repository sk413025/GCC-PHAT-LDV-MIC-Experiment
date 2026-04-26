# Band-DTmin Student Report

Generated: 2026-02-13T17:54:50.591594
Run dir: results/band_dtmin_student_20260213_175234

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.355 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.949 (>= 0.200)
- median(theta_error_ref) worsening: -0.865 (<= 0.050)
- OVERALL: PASS

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 2.179 | 5.845 | 5.934 | 0.308 |
| student | 318 | 0.294 | 1.574 | 3.825 | 0.016 |
