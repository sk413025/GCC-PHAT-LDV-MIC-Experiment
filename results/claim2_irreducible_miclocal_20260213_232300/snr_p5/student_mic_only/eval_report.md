# Band-DTmin Student Report

Generated: 2026-02-13T23:45:03.501230
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_p5/student_mic_only

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.227 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.989 (>= 0.200)
- median(theta_error_ref) worsening: -0.382 (<= 0.050)
- OVERALL: PASS

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 1.881 | 5.844 | 5.966 | 0.299 |
| student | 318 | 1.163 | 4.463 | 4.612 | 0.003 |
