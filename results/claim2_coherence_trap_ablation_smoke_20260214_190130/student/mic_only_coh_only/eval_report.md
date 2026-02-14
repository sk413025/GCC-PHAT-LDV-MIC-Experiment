# Band-DTmin Student Report

Generated: 2026-02-14T19:01:55.170307
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_coherence_trap_ablation_smoke_20260214_190130/student/mic_only_coh_only

## Success Criteria (pooled test windows, vs chirp reference)

- p95(theta_error_ref) improvement: -0.012 (>= 0.150)
- fail_rate_ref(theta_error_ref>5°) improvement: 0.000 (>= 0.200)
- median(theta_error_ref) worsening: -0.009 (<= 0.050)
- OVERALL: FAIL

## Test Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 4 | 3.129 | 3.141 | 3.142 | 0.000 |
| student | 4 | 3.101 | 3.174 | 3.180 | 0.000 |
