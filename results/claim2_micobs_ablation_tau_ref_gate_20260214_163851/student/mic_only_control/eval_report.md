# Band-DTmin Student Report

Generated: 2026-02-14T16:50:28.822527
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_micobs_ablation_tau_ref_gate_20260214_163851/student/mic_only_control

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: 0.031 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.000 (>= 0.200)
- median(theta_error_ref) worsening: -0.290 (<= 0.050)
- OVERALL: FAIL

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 318 | 4.272 | 4.570 | 4.625 | 0.000 |
| student | 318 | 3.031 | 4.411 | 4.483 | 0.000 |
