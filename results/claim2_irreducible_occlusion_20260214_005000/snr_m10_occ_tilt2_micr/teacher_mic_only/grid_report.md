# Band-OMP Teacher Report

Generated: 2026-02-14T04:08:28.605062
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_m10_occ_tilt2_micr/teacher_mic_only

## Teacher sanity gate

- median(theta_error_ref) worsening frac: -0.867 (<= 0.050)
- OVERALL: PASS

## Pooled Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1255 | 2.553 | 5.869 | 6.708 | 0.182 |
| teacher | 1255 | 0.340 | 1.107 | 1.384 | 0.000 |
