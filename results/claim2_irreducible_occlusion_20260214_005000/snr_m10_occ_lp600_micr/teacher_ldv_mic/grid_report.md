# Band-OMP Teacher Report

Generated: 2026-02-14T03:41:16.759575
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m10_occ_lp600_micr/teacher_ldv_mic

## Teacher sanity gate

- median(theta_error_ref) worsening frac: -0.854 (<= 0.050)
- OVERALL: PASS

## Pooled Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1255 | 2.377 | 5.716 | 6.388 | 0.158 |
| teacher | 1255 | 0.348 | 1.064 | 1.348 | 0.000 |
