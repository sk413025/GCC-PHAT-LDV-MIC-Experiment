# Band-DTmin Student Report

Generated: 2026-02-14T00:01:10.052377
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_miclocal_20260213_232300/snr_m5/student_mic_only

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.186 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.740 (>= 0.200)
- median(theta_error_ref) worsening: 0.375 (<= 0.050)
- OVERALL: FAIL

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 1.826 | 5.992 | 6.416 | 0.230 |
| student | 318 | 2.511 | 4.681 | 5.220 | 0.060 |
