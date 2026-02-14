# Band-DTmin Student Report

Generated: 2026-02-14T14:41:27.686454
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_micobs_ablation_fix_smoke_20260214_144104/student/mic_only_psd_only

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.000 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.000 (>= 0.200)
- median(theta_error_ref) worsening: -0.260 (<= 0.050)
- OVERALL: FAIL

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 4 | 3.103 | 4.250 | 4.364 | 0.000 |
| student | 4 | 2.295 | 4.250 | 4.364 | 0.000 |
