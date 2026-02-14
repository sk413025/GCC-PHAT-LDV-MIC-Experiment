# Band-OMP Teacher Report

Generated: 2026-02-14T03:09:30.232717
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m5_occ_lp800_micr/teacher_mic_only

## Teacher sanity gate

- median(theta_error_ref) worsening frac: -0.920 (<= 0.050)
- OVERALL: PASS

## Pooled Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1255 | 4.373 | 7.009 | 8.779 | 0.348 |
| teacher | 1255 | 0.349 | 1.076 | 1.374 | 0.000 |
