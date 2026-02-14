# Band-OMP Teacher Report

Generated: 2026-02-14T02:59:17.117788
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_p0_occ_lp800_micr/teacher_ldv_mic

## Teacher sanity gate

- median(theta_error_ref) worsening frac: -0.929 (<= 0.050)
- OVERALL: PASS

## Pooled Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1255 | 4.793 | 7.526 | 10.538 | 0.438 |
| teacher | 1255 | 0.343 | 1.026 | 1.354 | 0.000 |
