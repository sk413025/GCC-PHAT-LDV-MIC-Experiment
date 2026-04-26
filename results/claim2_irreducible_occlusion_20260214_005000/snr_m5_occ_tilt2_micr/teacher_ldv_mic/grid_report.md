# Band-OMP Teacher Report

Generated: 2026-02-14T03:58:12.442797
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m5_occ_tilt2_micr/teacher_ldv_mic

## Teacher sanity gate

- median(theta_error_ref) worsening frac: -0.870 (<= 0.050)
- OVERALL: PASS

## Pooled Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1255 | 2.574 | 6.824 | 9.525 | 0.241 |
| teacher | 1255 | 0.335 | 0.987 | 1.258 | 0.000 |
