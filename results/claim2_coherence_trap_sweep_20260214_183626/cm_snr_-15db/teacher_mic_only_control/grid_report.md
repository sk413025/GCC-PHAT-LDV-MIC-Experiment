# Band-OMP Teacher Report

Generated: 2026-02-14T18:59:49.611674
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_coherence_trap_sweep_20260214_183626/cm_snr_-15db/teacher_mic_only_control

## Teacher sanity gate

- median(theta_error_ref) worsening frac: -0.985 (<= 0.050)
- OVERALL: PASS

## Pooled Metrics (vs chirp reference)

| Method | count | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1255 | 2.508 | 4.056 | 4.573 | 0.000 |
| teacher | 1255 | 0.037 | 0.688 | 1.264 | 0.000 |
