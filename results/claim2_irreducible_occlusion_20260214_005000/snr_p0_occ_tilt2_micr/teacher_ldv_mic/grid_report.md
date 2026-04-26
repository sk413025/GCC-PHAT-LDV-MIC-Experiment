# Band-OMP Teacher Report

Generated: 2026-02-14T03:49:43.236663
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_p0_occ_tilt2_micr/teacher_ldv_mic

## Teacher sanity gate

- median(theta_error_ref) worsening frac: -0.855 (<= 0.050)
- OVERALL: PASS

## Pooled Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1255 | 2.239 | 6.868 | 8.705 | 0.233 |
| teacher | 1255 | 0.325 | 0.993 | 1.318 | 0.000 |
