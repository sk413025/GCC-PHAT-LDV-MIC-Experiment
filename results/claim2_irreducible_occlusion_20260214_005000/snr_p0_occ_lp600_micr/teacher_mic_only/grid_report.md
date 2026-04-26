# Band-OMP Teacher Report

Generated: 2026-02-14T03:26:19.608423
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000/snr_p0_occ_lp600_micr/teacher_mic_only

## Teacher sanity gate

- median(theta_error_ref) worsening frac: -0.872 (<= 0.050)
- OVERALL: PASS

## Pooled Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1255 | 2.671 | 5.977 | 6.966 | 0.208 |
| teacher | 1255 | 0.343 | 1.025 | 1.362 | 0.000 |
