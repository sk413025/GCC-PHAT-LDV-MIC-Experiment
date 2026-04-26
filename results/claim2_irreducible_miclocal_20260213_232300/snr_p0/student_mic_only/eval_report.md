# Band-DTmin Student Report

Generated: 2026-02-13T23:53:06.115458
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_p0/student_mic_only

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.186 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.843 (>= 0.200)
- median(theta_error_ref) worsening: 0.516 (<= 0.050)
- OVERALL: FAIL

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 1.724 | 5.892 | 6.013 | 0.280 |
| student | 318 | 2.612 | 4.560 | 4.893 | 0.044 |
