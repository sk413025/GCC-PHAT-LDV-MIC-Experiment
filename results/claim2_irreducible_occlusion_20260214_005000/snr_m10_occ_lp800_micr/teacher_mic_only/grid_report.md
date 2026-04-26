# Band-OMP Teacher Report

Generated: 2026-02-14T03:17:53.541763
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m10_occ_lp800_micr/teacher_mic_only

## Teacher sanity gate

- median(theta_error_ref) worsening frac: -0.879 (<= 0.050)
- OVERALL: PASS

## Pooled Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1255 | 2.989 | 5.863 | 6.922 | 0.211 |
| teacher | 1255 | 0.361 | 1.048 | 1.367 | 0.000 |
