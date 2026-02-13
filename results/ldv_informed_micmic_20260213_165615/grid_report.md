# LDV-Informed MIC-MIC GCC Report

Generated: 2026-02-13T16:56:55.202395
Run dir: results/ldv_informed_micmic_20260213_165615

## Success Criteria (speech, pooled)

- p95(theta_error_ref) improvement: -0.355 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: -0.062 (>= 0.200)
- median(theta_error_ref) worsening: 0.628 (<= 0.050)
- OVERALL: FAIL

## Pooled Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1255 | 2.077 | 5.839 | 5.915 | 0.256 |
| teacher | 1255 | 3.381 | 7.261 | 8.017 | 0.272 |

## Pooled Metrics (vs geometry truth)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1255 | 2.685 | 6.775 | 7.075 | 0.104 |
| teacher | 1255 | 3.060 | 6.642 | 7.226 | 0.369 |
