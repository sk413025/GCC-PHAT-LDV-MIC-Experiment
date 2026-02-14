# Band-OMP Teacher Report

Generated: 2026-02-14T03:32:54.033096
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m5_occ_lp600_micr/teacher_ldv_mic

## Teacher sanity gate

- median(theta_error_ref) worsening frac: -0.855 (<= 0.050)
- OVERALL: PASS

## Pooled Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1255 | 2.433 | 5.728 | 6.488 | 0.176 |
| teacher | 1255 | 0.354 | 1.062 | 1.372 | 0.000 |
