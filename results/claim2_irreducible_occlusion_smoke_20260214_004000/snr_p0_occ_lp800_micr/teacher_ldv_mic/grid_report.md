# Band-OMP Teacher Report

Generated: 2026-02-14T02:57:13.002867
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_smoke_20260214_004000/snr_p0_occ_lp800_micr/teacher_ldv_mic

## Teacher sanity gate

- median(theta_error_ref) worsening frac: -0.763 (<= 0.050)
- OVERALL: PASS

## Pooled Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 6 | 3.103 | 5.548 | 6.083 | 0.167 |
| teacher | 6 | 0.736 | 1.111 | 1.197 | 0.000 |
