# Band-OMP Teacher Report

Generated: 2026-02-13T18:38:17.869995
Run dir: results/ablation_teacher_ldv_mic_20260213_183424

## Teacher sanity gate

- median(theta_error_ref) worsening frac: -0.931 (<= 0.050)
- OVERALL: PASS

## Pooled Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1255 | 2.077 | 5.839 | 5.915 | 0.256 |
| teacher | 1255 | 0.142 | 0.467 | 0.569 | 0.000 |
