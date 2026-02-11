# OMP Strict vs DTmin Strict (K=3)

Generated: 2026-02-11 22:32:49

| config | baseline_omp_pass | dtmin_pass | delta |
| --- | ---: | ---: | ---: |
| ldv_micl_chirp_tau2_band0 | 4 | 4 | 0 |
| ldv_micl_chirp_tau2_band500_2000 | 4 | 4 | 0 |
| micl_micr_chirp_tau2_band500_2000 | 5 | 5 | 0 |
| ldv_micl_geometry_tau2_band500_2000 | 3 | 3 | 0 |
| ldv_micl_chirp_tau0p3_band500_2000 | 4 | 4 | 0 |

Notes:
- DTmin run fixed `max_k=3` end-to-end (teacher + student + stage4).
- Baseline anchor commit: `530a63f`.
- DTmin run root: `results/ldv_vs_mic_grid_dtmin_strict_20260211_215857/`.
