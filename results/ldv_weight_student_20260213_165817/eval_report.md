# Student LDV Weight Mask Report

Generated: 2026-02-13T16:58:30.020286
Run dir: results/ldv_weight_student_20260213_165817

## Acceptance (pooled test windows)

- teacher p95 improvement frac: -0.298
- student p95 improvement frac: -0.521
- student / teacher: nan
- student median worsening frac: 0.375
- OVERALL: FAIL

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 2.179 | 5.845 | 5.934 | 0.308 |
| teacher | 318 | 3.292 | 7.274 | 7.701 | 0.242 |
| student | 318 | 2.996 | 8.366 | 9.029 | 0.204 |

## Test Metrics (vs geometry truth)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 2.830 | 6.952 | 7.112 | 0.129 |
| teacher | 318 | 2.169 | 6.042 | 6.775 | 0.355 |
| student | 318 | 1.255 | 6.831 | 7.276 | 0.437 |