# 0223 Automated Peak-Pair Sweep

- Generated: 2026-03-17T13:57:12
- Data root: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223`
- Slice sec: `5.0`
- Lag min grid: `[3.5, 4.0, 4.4]`
- Lag max: `6.5` ms
- Top-k per side: `8`
- Delta gate: `|delta_tau| <= 1.0` ms

## Summary

| variant | strategy | lag_min_ms | delta_scale_ms | mean_tau_center_ms | mean_tau_scale_ms | valid_cases | physical_count | delta_tau_mae_ms | theta_v_mae_deg | max_delta_tau_err_ms | max_theta_v_err_deg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ldv_diff_bp500_2000 | amp_product_mean_tau_delta_quad | 4.400 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.199 | 2.806 | 0.358 | 5.043 |
| bp_500_2000 | amp_product | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.272 | 3.831 | 0.494 | 6.948 |
| bp_500_2000 | amp_product | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.277 | 3.904 | 0.494 | 6.948 |
| ldv_diff_bp500_2000 | balanced_mean_tau_delta_quad | 4.400 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.293 | 4.123 | 0.463 | 6.505 |
| ldv_preemph_bp500_2000 | amp_product | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.293 | 4.122 | 0.504 | 7.074 |
| raw_fullband | amp_product_delta_quad | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.298 | 4.198 | 0.421 | 5.920 |
| bp_500_2000 | amp_product_delta | 4.400 | 1.000 | 4.800 | 10.000 | 4 | 4 | 0.309 | 4.344 | 0.494 | 6.948 |
| bp_500_2000 | amp_product_delta_quad | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.309 | 4.344 | 0.494 | 6.948 |
| bp_500_2000 | amp_product_delta | 4.000 | 1.000 | 4.800 | 10.000 | 4 | 4 | 0.314 | 4.417 | 0.494 | 6.948 |
| bp_500_2000 | amp_product_delta_quad | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.314 | 4.417 | 0.494 | 6.948 |
| ldv_preemph_bp500_2000 | amp_product_mean_tau_delta_quad | 3.500 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.319 | 4.491 | 0.421 | 5.920 |
| ldv_preemph_bp500_2000 | balanced_mean_tau_delta_quad | 3.500 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.319 | 4.491 | 0.421 | 5.920 |
| ldv_preemph_bp500_2000 | amp_product_mean_tau_delta_quad | 4.000 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.319 | 4.491 | 0.421 | 5.920 |
| ldv_preemph_bp500_2000 | balanced_mean_tau_delta_quad | 4.000 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.319 | 4.491 | 0.421 | 5.920 |
| ldv_diff_bp500_2000 | amp_product_delta_quad | 3.500 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.324 | 4.564 | 0.410 | 5.778 |
| bp_500_2000 | amp_product_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.330 | 4.637 | 0.494 | 6.948 |
| bp_500_2000 | prom_product_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.330 | 4.637 | 0.494 | 6.948 |
| bp_500_2000 | balanced_product_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.330 | 4.637 | 0.494 | 6.948 |
| ldv_preemph_bp500_2000 | amp_product_mean_tau_delta_quad | 4.400 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.330 | 4.637 | 0.463 | 6.505 |
| ldv_preemph_bp500_2000 | balanced_mean_tau_delta_quad | 4.400 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.330 | 4.637 | 0.463 | 6.505 |
| raw_fullband | balanced_mean_tau_delta_quad | 3.500 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.335 | 4.710 | 0.463 | 6.505 |
| raw_fullband | balanced_mean_tau_delta_quad | 4.000 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.335 | 4.710 | 0.463 | 6.505 |
| raw_fullband | balanced_mean_tau_delta_quad | 4.400 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.335 | 4.710 | 0.463 | 6.505 |
| ldv_diff_bp500_2000 | amp_product_delta_quad | 4.000 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.340 | 4.783 | 0.421 | 5.920 |
| bp_500_2000 | amp_product_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.345 | 4.856 | 0.494 | 6.948 |
| bp_500_2000 | prom_product_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.345 | 4.856 | 0.494 | 6.948 |
| bp_500_2000 | balanced_product_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.345 | 4.856 | 0.494 | 6.948 |
| ldv_preemph_bp500_2000 | balanced_product_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.350 | 4.930 | 0.546 | 7.675 |
| ldv_diff_bp500_2000 | amp_product_mean_tau_delta_quad | 3.500 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.356 | 5.001 | 0.713 | 10.017 |
| ldv_diff_bp500_2000 | balanced_mean_tau_delta_quad | 3.500 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.356 | 5.001 | 0.713 | 10.017 |
| ldv_diff_bp500_2000 | amp_product_mean_tau_delta_quad | 4.000 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.356 | 5.001 | 0.713 | 10.017 |
| ldv_diff_bp500_2000 | balanced_mean_tau_delta_quad | 4.000 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.356 | 5.001 | 0.713 | 10.017 |
| raw_fullband | amp_product_delta_quad | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.356 | 5.003 | 0.504 | 7.090 |
| bp_1000_3000 | amp_product_mean_tau_delta_quad | 3.500 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.356 | 5.003 | 0.463 | 6.505 |
| bp_1000_3000 | amp_product_mean_tau_delta_quad | 4.000 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.356 | 5.003 | 0.463 | 6.505 |
| bp_1000_3000 | amp_product_mean_tau_delta_quad | 4.400 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.356 | 5.003 | 0.463 | 6.505 |
| raw_fullband | balanced_product_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.356 | 5.003 | 0.577 | 8.118 |
| ldv_diff_bp500_2000 | amp_product_delta | 3.500 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.361 | 5.076 | 0.421 | 5.920 |
| ldv_diff_bp500_2000 | amp_sum_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.361 | 5.076 | 0.421 | 5.920 |
| ldv_preemph_bp500_2000 | balanced_product_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.361 | 5.076 | 0.535 | 7.533 |
| ldv_preemph_bp500_2000 | amp_product_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.361 | 5.076 | 0.546 | 7.675 |
| ldv_preemph_bp500_2000 | prom_product_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.361 | 5.076 | 0.546 | 7.675 |
| bp_500_2000 | amp_product_mean_tau_delta_quad | 4.400 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.371 | 5.221 | 0.744 | 10.459 |
| bp_500_2000 | balanced_mean_tau_delta_quad | 4.400 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.371 | 5.221 | 0.744 | 10.459 |
| raw_fullband | amp_product_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.371 | 5.222 | 0.421 | 5.920 |
| raw_fullband | prom_product_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.371 | 5.222 | 0.421 | 5.920 |
| bp_500_2000 | amp_product_mean_tau_delta_quad | 3.500 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.371 | 5.221 | 0.744 | 10.459 |
| bp_500_2000 | balanced_mean_tau_delta_quad | 3.500 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.371 | 5.221 | 0.744 | 10.459 |
| bp_500_2000 | amp_product_mean_tau_delta_quad | 4.000 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.371 | 5.221 | 0.744 | 10.459 |
| bp_500_2000 | balanced_mean_tau_delta_quad | 4.000 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.371 | 5.221 | 0.744 | 10.459 |
| ldv_preemph_bp500_2000 | amp_product_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.371 | 5.222 | 0.535 | 7.533 |
| ldv_preemph_bp500_2000 | amp_product_delta | 4.000 | 1.000 | 4.800 | 10.000 | 4 | 4 | 0.371 | 5.222 | 0.535 | 7.533 |
| ldv_preemph_bp500_2000 | amp_product_delta_quad | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.371 | 5.222 | 0.535 | 7.533 |
| ldv_preemph_bp500_2000 | prom_product_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.371 | 5.222 | 0.535 | 7.533 |
| bp_1000_3000 | amp_product_delta | 3.500 | 1.000 | 4.800 | 10.000 | 4 | 4 | 0.371 | 5.222 | 0.494 | 6.948 |
| bp_1000_3000 | amp_product_delta_quad | 3.500 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.371 | 5.222 | 0.494 | 6.948 |
| bp_1000_3000 | amp_product_delta_quad | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.371 | 5.222 | 0.494 | 6.948 |
| ldv_diff_bp500_2000 | amp_product_delta | 4.000 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.376 | 5.295 | 0.421 | 5.920 |
| ldv_diff_bp500_2000 | amp_product_delta | 4.400 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.376 | 5.295 | 0.421 | 5.920 |
| ldv_diff_bp500_2000 | amp_product_delta_quad | 4.400 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.376 | 5.295 | 0.421 | 5.920 |
| ldv_preemph_bp500_2000 | amp_product | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.378 | 5.347 | 0.663 | 9.431 |
| bp_1000_3000 | amp_product_delta | 3.500 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.382 | 5.369 | 0.494 | 6.948 |
| bp_1000_3000 | amp_product_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.382 | 5.369 | 0.494 | 6.948 |
| bp_1000_3000 | amp_sum_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.382 | 5.369 | 0.494 | 6.948 |
| bp_1000_3000 | prom_product_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.382 | 5.369 | 0.494 | 6.948 |
| ldv_diff_bp500_2000 | amp_sum_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.387 | 5.442 | 0.421 | 5.920 |
| ldv_diff_bp500_2000 | amp_sum_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.392 | 5.515 | 0.421 | 5.920 |
| raw_fullband | amp_product_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.392 | 5.514 | 0.504 | 7.090 |
| raw_fullband | prom_product_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.392 | 5.514 | 0.504 | 7.090 |
| bp_1000_3000 | amp_product_delta | 4.000 | 1.000 | 4.800 | 10.000 | 4 | 4 | 0.392 | 5.515 | 0.494 | 6.948 |
| bp_1000_3000 | amp_product_delta_quad | 4.000 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.392 | 5.515 | 0.494 | 6.948 |
| bp_1000_3000 | amp_product_delta_quad | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.392 | 5.515 | 0.494 | 6.948 |
| bp_1000_3000 | balanced_product_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.392 | 5.515 | 0.494 | 6.948 |
| bp_1000_3000 | amp_product_delta | 4.400 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.397 | 5.588 | 0.514 | 7.240 |
| bp_1000_3000 | amp_product_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.397 | 5.588 | 0.514 | 7.240 |
| bp_1000_3000 | amp_product_delta | 4.400 | 1.000 | 4.800 | 10.000 | 4 | 4 | 0.397 | 5.588 | 0.514 | 7.240 |
| bp_1000_3000 | amp_product_delta_quad | 4.400 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.397 | 5.588 | 0.514 | 7.240 |
| bp_1000_3000 | amp_product_delta_quad | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.397 | 5.588 | 0.514 | 7.240 |
| bp_1000_3000 | prom_product_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.397 | 5.588 | 0.514 | 7.240 |
| bp_1000_3000 | amp_sum_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.397 | 5.588 | 0.514 | 7.240 |
| bp_500_2000 | amp_product_delta | 4.400 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.397 | 5.588 | 0.494 | 6.948 |
| bp_500_2000 | amp_product_delta_quad | 4.400 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.397 | 5.588 | 0.494 | 6.948 |
| bp_500_2000 | amp_sum_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.397 | 5.588 | 0.494 | 6.948 |
| bp_1000_3000 | balanced_product_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.397 | 5.588 | 0.494 | 6.948 |
| bp_1000_3000 | amp_product_delta | 4.000 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.402 | 5.661 | 0.494 | 6.948 |
| bp_1000_3000 | amp_product_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.402 | 5.661 | 0.494 | 6.948 |
| bp_1000_3000 | prom_product_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.402 | 5.661 | 0.494 | 6.948 |
| bp_1000_3000 | amp_sum_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.402 | 5.661 | 0.494 | 6.948 |
| bp_1000_3000 | balanced_product_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.408 | 5.734 | 0.514 | 7.240 |
| raw_fullband | amp_product_mean_tau_delta_quad | 3.500 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.413 | 5.807 | 0.619 | 8.703 |
| raw_fullband | amp_product_mean_tau_delta_quad | 4.000 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.413 | 5.807 | 0.619 | 8.703 |
| raw_fullband | amp_product_mean_tau_delta_quad | 4.400 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.413 | 5.807 | 0.619 | 8.703 |
| raw_fullband | amp_product_delta_quad | 3.500 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.413 | 5.807 | 0.577 | 8.118 |
| bp_500_2000 | amp_product_delta | 4.000 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.413 | 5.807 | 0.494 | 6.948 |
| bp_500_2000 | amp_product_delta_quad | 4.000 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.413 | 5.807 | 0.494 | 6.948 |
| bp_500_2000 | amp_sum_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.413 | 5.807 | 0.494 | 6.948 |
| bp_500_2000 | amp_product | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.413 | 5.807 | 0.494 | 6.948 |
| bp_500_2000 | amp_product_delta | 3.500 | 1.000 | 4.800 | 10.000 | 4 | 4 | 0.413 | 5.807 | 0.494 | 6.948 |
| bp_500_2000 | amp_product_delta_quad | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.413 | 5.807 | 0.494 | 6.948 |
| raw_fullband | amp_product_delta | 4.400 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.423 | 5.953 | 0.514 | 7.240 |
| raw_fullband | amp_sum_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.423 | 5.953 | 0.514 | 7.240 |
| ldv_diff_bp500_2000 | amp_product_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.423 | 5.954 | 0.744 | 10.459 |
| ldv_diff_bp500_2000 | prom_product_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.423 | 5.954 | 0.744 | 10.459 |
| ldv_diff_bp500_2000 | balanced_product_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.423 | 5.954 | 0.744 | 10.459 |
| raw_fullband | amp_product_delta | 3.500 | 1.000 | 4.800 | 10.000 | 4 | 4 | 0.423 | 5.953 | 0.577 | 8.118 |
| raw_fullband | amp_product_delta_quad | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.423 | 5.953 | 0.577 | 8.118 |
| raw_fullband | amp_product_delta | 3.500 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.428 | 6.027 | 0.577 | 8.118 |
| raw_fullband | amp_product_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.428 | 6.027 | 0.577 | 8.118 |
| raw_fullband | prom_product_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.428 | 6.027 | 0.577 | 8.118 |
| ldv_preemph_bp500_2000 | amp_product_delta | 4.000 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.434 | 6.100 | 0.535 | 7.533 |
| ldv_preemph_bp500_2000 | amp_product_delta_quad | 4.000 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.434 | 6.100 | 0.535 | 7.533 |
| ldv_preemph_bp500_2000 | amp_sum_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.434 | 6.100 | 0.535 | 7.533 |
| raw_fullband | amp_product_delta | 4.000 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.444 | 6.246 | 0.514 | 7.240 |
| raw_fullband | amp_sum_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.444 | 6.246 | 0.514 | 7.240 |
| bp_500_2000 | amp_product_delta | 3.500 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.444 | 6.246 | 0.494 | 6.948 |
| bp_500_2000 | amp_product_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.444 | 6.246 | 0.494 | 6.948 |
| bp_500_2000 | amp_product_delta_quad | 3.500 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.444 | 6.246 | 0.494 | 6.948 |
| bp_500_2000 | amp_sum_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.444 | 6.246 | 0.494 | 6.948 |
| bp_500_2000 | prom_product_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.444 | 6.246 | 0.494 | 6.948 |
| bp_500_2000 | balanced_product_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.444 | 6.246 | 0.494 | 6.948 |
| raw_fullband | amp_product_delta_quad | 4.400 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.449 | 6.319 | 0.619 | 8.703 |
| raw_fullband | amp_product_delta | 4.000 | 1.000 | 4.800 | 10.000 | 4 | 4 | 0.449 | 6.322 | 0.879 | 12.366 |
| raw_fullband | amp_product_delta | 4.400 | 1.000 | 4.800 | 10.000 | 4 | 4 | 0.449 | 6.322 | 0.879 | 12.366 |
| ldv_preemph_bp500_2000 | amp_product_delta | 4.400 | 1.000 | 4.800 | 10.000 | 4 | 4 | 0.449 | 6.319 | 0.723 | 10.166 |
| ldv_preemph_bp500_2000 | amp_product_delta_quad | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.449 | 6.319 | 0.723 | 10.166 |
| ldv_diff_bp500_2000 | amp_product_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.460 | 6.466 | 0.744 | 10.459 |
| ldv_diff_bp500_2000 | prom_product_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.460 | 6.466 | 0.744 | 10.459 |
| ldv_diff_bp500_2000 | balanced_product_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.460 | 6.466 | 0.744 | 10.459 |
| raw_fullband | amp_sum_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.460 | 6.465 | 0.577 | 8.118 |
| bp_1000_3000 | balanced_mean_tau_delta_quad | 3.500 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.460 | 6.465 | 0.577 | 8.118 |
| bp_1000_3000 | balanced_mean_tau_delta_quad | 4.000 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.460 | 6.465 | 0.577 | 8.118 |
| ldv_preemph_bp500_2000 | amp_sum_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.460 | 6.465 | 0.546 | 7.675 |
| ldv_preemph_bp500_2000 | amp_product_delta | 4.400 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.465 | 6.538 | 0.546 | 7.675 |
| raw_fullband | amp_product_delta_quad | 4.000 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.470 | 6.612 | 0.619 | 8.703 |
| ldv_diff_bp500_2000 | amp_product_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.475 | 6.685 | 0.744 | 10.459 |
| ldv_diff_bp500_2000 | amp_product_delta_quad | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.475 | 6.685 | 0.744 | 10.459 |
| ldv_diff_bp500_2000 | prom_product_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.475 | 6.685 | 0.744 | 10.459 |
| ldv_diff_bp500_2000 | balanced_product_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.475 | 6.685 | 0.744 | 10.459 |
| ldv_diff_bp500_2000 | amp_product_delta_quad | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.475 | 6.685 | 0.744 | 10.459 |
| ldv_preemph_bp500_2000 | amp_product_delta_quad | 4.400 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.475 | 6.685 | 0.577 | 8.118 |
| ldv_preemph_bp500_2000 | amp_product_delta | 3.500 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.486 | 6.831 | 0.535 | 7.533 |
| ldv_preemph_bp500_2000 | amp_product_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.486 | 6.831 | 0.535 | 7.533 |
| ldv_preemph_bp500_2000 | amp_product_delta | 3.500 | 1.000 | 4.800 | 10.000 | 4 | 4 | 0.486 | 6.831 | 0.535 | 7.533 |
| ldv_preemph_bp500_2000 | amp_product_delta_quad | 3.500 | 0.250 | 4.800 | 10.000 | 4 | 4 | 0.486 | 6.831 | 0.535 | 7.533 |
| ldv_preemph_bp500_2000 | amp_product_delta_quad | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.486 | 6.831 | 0.535 | 7.533 |
| ldv_preemph_bp500_2000 | amp_sum_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.486 | 6.831 | 0.535 | 7.533 |
| ldv_preemph_bp500_2000 | prom_product_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.486 | 6.831 | 0.535 | 7.533 |
| ldv_preemph_bp500_2000 | balanced_product_delta | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.486 | 6.831 | 0.535 | 7.533 |
| bp_1000_3000 | balanced_mean_tau_delta_quad | 4.400 | 0.600 | 4.800 | 0.450 | 4 | 4 | 0.496 | 6.977 | 0.723 | 10.166 |
| raw_fullband | balanced_product_delta | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.507 | 7.126 | 0.879 | 12.366 |
| ldv_diff_bp500_2000 | amp_product_delta_quad | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.512 | 7.197 | 0.744 | 10.459 |
| ldv_diff_bp500_2000 | amp_product | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.522 | 7.345 | 0.931 | 13.100 |
| ldv_diff_bp500_2000 | amp_product_delta | 3.500 | 1.000 | 4.800 | 10.000 | 4 | 4 | 0.522 | 7.345 | 0.931 | 13.100 |
| ldv_diff_bp500_2000 | amp_product_delta | 4.000 | 1.000 | 4.800 | 10.000 | 4 | 4 | 0.522 | 7.345 | 0.931 | 13.100 |
| raw_fullband | balanced_product_delta | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.522 | 7.345 | 0.879 | 12.366 |
| ldv_diff_bp500_2000 | amp_product_delta | 4.400 | 1.000 | 4.800 | 10.000 | 4 | 4 | 0.559 | 7.857 | 0.931 | 13.100 |
| bp_1000_3000 | amp_product | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.593 | 8.374 | 1.400 | 19.792 |
| ldv_preemph_bp500_2000 | amp_product | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.599 | 8.445 | 0.754 | 10.604 |
| raw_fullband | amp_product | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.690 | 9.739 | 1.129 | 15.910 |
| raw_fullband | amp_product | 3.500 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.758 | 10.700 | 1.244 | 17.531 |
| ldv_diff_bp500_2000 | amp_product | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.791 | 11.173 | 1.338 | 18.891 |
| bp_1000_3000 | amp_product | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.809 | 11.407 | 1.400 | 19.792 |
| ldv_diff_bp500_2000 | amp_product | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.841 | 11.897 | 1.338 | 18.891 |
| bp_1000_3000 | amp_product | 4.400 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.888 | 12.559 | 1.264 | 17.829 |
| raw_fullband | amp_product | 4.000 | 0.500 | 4.800 | 10.000 | 4 | 4 | 0.932 | 13.136 | 1.244 | 17.531 |

## Per-Case Selections

### block4_p08_17 / raw_fullband / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.729 ms, tau_vr=3.854 ms, delta_tau=-0.875 ms, theta_v=-12.379 deg, score=0.000222
- errors: delta_tau_abs_err=0.433 ms, theta_v_abs_err=6.166 deg

### block5_p00_18 / raw_fullband / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.646 ms, tau_vr=6.083 ms, delta_tau=0.438 ms, theta_v=6.153 deg, score=0.000581
- errors: delta_tau_abs_err=0.879 ms, theta_v_abs_err=12.366 deg

### block6_n04_19 / raw_fullband / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.979 ms, tau_vr=4.250 ms, delta_tau=-0.729 ms, theta_v=-10.291 deg, score=0.000154
- errors: delta_tau_abs_err=1.244 ms, theta_v_abs_err=17.531 deg

### block7_n08_20 / raw_fullband / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.542 ms, tau_vr=4.771 ms, delta_tau=-0.771 ms, theta_v=-10.886 deg, score=0.000269
- errors: delta_tau_abs_err=0.476 ms, theta_v_abs_err=6.738 deg

### block4_p08_17 / raw_fullband / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=4, vr_rank=2, tau_vl=3.875 ms, tau_vr=3.854 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000174
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=4, tau_vl=3.854 ms, tau_vr=3.792 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000309
- errors: delta_tau_abs_err=0.379 ms, theta_v_abs_err=5.335 deg

### block6_n04_19 / raw_fullband / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=3.833 ms, tau_vr=3.771 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000097
- errors: delta_tau_abs_err=0.577 ms, theta_v_abs_err=8.118 deg

### block7_n08_20 / raw_fullband / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=5, tau_vl=5.542 ms, tau_vr=5.583 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000206
- errors: delta_tau_abs_err=0.337 ms, theta_v_abs_err=4.733 deg

### block4_p08_17 / raw_fullband / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=4, vr_rank=2, tau_vl=3.875 ms, tau_vr=3.854 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000182
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=4, tau_vl=3.854 ms, tau_vr=3.792 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000350
- errors: delta_tau_abs_err=0.379 ms, theta_v_abs_err=5.335 deg

### block6_n04_19 / raw_fullband / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=3.833 ms, tau_vr=3.771 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000110
- errors: delta_tau_abs_err=0.577 ms, theta_v_abs_err=8.118 deg

### block7_n08_20 / raw_fullband / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=5, tau_vl=5.542 ms, tau_vr=5.583 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000224
- errors: delta_tau_abs_err=0.337 ms, theta_v_abs_err=4.733 deg

### block4_p08_17 / raw_fullband / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=3.938 ms, tau_vr=3.854 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000196
- errors: delta_tau_abs_err=0.358 ms, theta_v_abs_err=5.043 deg

### block5_p00_18 / raw_fullband / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=4, tau_vl=3.667 ms, tau_vr=3.792 ms, delta_tau=0.125 ms, theta_v=1.755 deg, score=0.000387
- errors: delta_tau_abs_err=0.567 ms, theta_v_abs_err=7.967 deg

### block6_n04_19 / raw_fullband / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=3.833 ms, tau_vr=3.771 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000117
- errors: delta_tau_abs_err=0.577 ms, theta_v_abs_err=8.118 deg

### block7_n08_20 / raw_fullband / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.542 ms, tau_vr=5.438 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000236
- errors: delta_tau_abs_err=0.191 ms, theta_v_abs_err=2.686 deg

### block4_p08_17 / raw_fullband / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=3.938 ms, tau_vr=3.854 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000191
- errors: delta_tau_abs_err=0.358 ms, theta_v_abs_err=5.043 deg

### block5_p00_18 / raw_fullband / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=4, tau_vl=3.854 ms, tau_vr=3.792 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000372
- errors: delta_tau_abs_err=0.379 ms, theta_v_abs_err=5.335 deg

### block6_n04_19 / raw_fullband / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=3.833 ms, tau_vr=3.771 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000117
- errors: delta_tau_abs_err=0.577 ms, theta_v_abs_err=8.118 deg

### block7_n08_20 / raw_fullband / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=5, tau_vl=5.542 ms, tau_vr=5.583 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000237
- errors: delta_tau_abs_err=0.337 ms, theta_v_abs_err=4.733 deg

### block4_p08_17 / raw_fullband / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=3.938 ms, tau_vr=3.854 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000207
- errors: delta_tau_abs_err=0.358 ms, theta_v_abs_err=5.043 deg

### block5_p00_18 / raw_fullband / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=4, tau_vl=3.667 ms, tau_vr=3.792 ms, delta_tau=0.125 ms, theta_v=1.755 deg, score=0.000411
- errors: delta_tau_abs_err=0.567 ms, theta_v_abs_err=7.967 deg

### block6_n04_19 / raw_fullband / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=3.833 ms, tau_vr=3.771 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000122
- errors: delta_tau_abs_err=0.577 ms, theta_v_abs_err=8.118 deg

### block7_n08_20 / raw_fullband / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.542 ms, tau_vr=5.438 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000251
- errors: delta_tau_abs_err=0.191 ms, theta_v_abs_err=2.686 deg

### block4_p08_17 / raw_fullband / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=4, vr_rank=2, tau_vl=3.875 ms, tau_vr=3.854 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.026421
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=7, vr_rank=1, tau_vl=4.229 ms, tau_vr=4.292 ms, delta_tau=0.062 ms, theta_v=0.877 deg, score=0.035487
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.090 deg

### block6_n04_19 / raw_fullband / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=3.833 ms, tau_vr=3.771 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.019696
- errors: delta_tau_abs_err=0.577 ms, theta_v_abs_err=8.118 deg

### block7_n08_20 / raw_fullband / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=5, tau_vl=5.542 ms, tau_vr=5.583 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.030141
- errors: delta_tau_abs_err=0.337 ms, theta_v_abs_err=4.733 deg

### block4_p08_17 / raw_fullband / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=4, vr_rank=2, tau_vl=3.875 ms, tau_vr=3.854 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000182
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=4, tau_vl=3.854 ms, tau_vr=3.792 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000350
- errors: delta_tau_abs_err=0.379 ms, theta_v_abs_err=5.335 deg

### block6_n04_19 / raw_fullband / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=3.833 ms, tau_vr=3.771 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000110
- errors: delta_tau_abs_err=0.577 ms, theta_v_abs_err=8.118 deg

### block7_n08_20 / raw_fullband / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=5, tau_vl=5.542 ms, tau_vr=5.583 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000224
- errors: delta_tau_abs_err=0.337 ms, theta_v_abs_err=4.733 deg

### block4_p08_17 / raw_fullband / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=3.938 ms, tau_vr=3.854 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000177
- errors: delta_tau_abs_err=0.358 ms, theta_v_abs_err=5.043 deg

### block5_p00_18 / raw_fullband / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=4, tau_vl=3.854 ms, tau_vr=3.792 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000331
- errors: delta_tau_abs_err=0.379 ms, theta_v_abs_err=5.335 deg

### block6_n04_19 / raw_fullband / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=3.833 ms, tau_vr=3.771 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000098
- errors: delta_tau_abs_err=0.577 ms, theta_v_abs_err=8.118 deg

### block7_n08_20 / raw_fullband / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=3.854 ms, tau_vr=3.667 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000157
- errors: delta_tau_abs_err=0.108 ms, theta_v_abs_err=1.515 deg

### block4_p08_17 / raw_fullband / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=8, tau_vl=4.729 ms, tau_vr=4.708 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000170
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=8, vr_rank=7, tau_vl=4.729 ms, tau_vr=4.750 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000215
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block6_n04_19 / raw_fullband / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=7, tau_vl=4.979 ms, tau_vr=4.875 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000076
- errors: delta_tau_abs_err=0.619 ms, theta_v_abs_err=8.703 deg

### block7_n08_20 / raw_fullband / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=5, vr_rank=2, tau_vl=4.917 ms, tau_vr=4.771 ms, delta_tau=-0.146 ms, theta_v=-2.048 deg, score=0.000160
- errors: delta_tau_abs_err=0.149 ms, theta_v_abs_err=2.101 deg

### block4_p08_17 / raw_fullband / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=8, tau_vl=4.729 ms, tau_vr=4.708 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000125
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=8, vr_rank=7, tau_vl=4.729 ms, tau_vr=4.750 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000213
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block6_n04_19 / raw_fullband / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.979 ms, tau_vr=5.188 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000069
- errors: delta_tau_abs_err=0.306 ms, theta_v_abs_err=4.315 deg

### block7_n08_20 / raw_fullband / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=5, vr_rank=2, tau_vl=4.917 ms, tau_vr=4.771 ms, delta_tau=-0.146 ms, theta_v=-2.048 deg, score=0.000147
- errors: delta_tau_abs_err=0.149 ms, theta_v_abs_err=2.101 deg

### block4_p08_17 / raw_fullband / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=5.646 ms, tau_vr=6.333 ms, delta_tau=0.688 ms, theta_v=9.697 deg, score=0.000222
- errors: delta_tau_abs_err=1.129 ms, theta_v_abs_err=15.910 deg

### block5_p00_18 / raw_fullband / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.646 ms, tau_vr=6.083 ms, delta_tau=0.438 ms, theta_v=6.153 deg, score=0.000581
- errors: delta_tau_abs_err=0.879 ms, theta_v_abs_err=12.366 deg

### block6_n04_19 / raw_fullband / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.979 ms, tau_vr=4.250 ms, delta_tau=-0.729 ms, theta_v=-10.291 deg, score=0.000154
- errors: delta_tau_abs_err=1.244 ms, theta_v_abs_err=17.531 deg

### block7_n08_20 / raw_fullband / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.542 ms, tau_vr=4.771 ms, delta_tau=-0.771 ms, theta_v=-10.886 deg, score=0.000269
- errors: delta_tau_abs_err=0.476 ms, theta_v_abs_err=6.738 deg

### block4_p08_17 / raw_fullband / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=4.729 ms, tau_vr=4.708 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000161
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=1, tau_vl=4.229 ms, tau_vr=4.292 ms, delta_tau=0.062 ms, theta_v=0.877 deg, score=0.000301
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.090 deg

### block6_n04_19 / raw_fullband / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=4, vr_rank=7, tau_vl=4.771 ms, tau_vr=4.771 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.000062
- errors: delta_tau_abs_err=0.514 ms, theta_v_abs_err=7.240 deg

### block7_n08_20 / raw_fullband / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.542 ms, tau_vr=5.583 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000206
- errors: delta_tau_abs_err=0.337 ms, theta_v_abs_err=4.733 deg

### block4_p08_17 / raw_fullband / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=4.729 ms, tau_vr=4.708 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000168
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=1, tau_vl=4.229 ms, tau_vr=4.292 ms, delta_tau=0.062 ms, theta_v=0.877 deg, score=0.000341
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.090 deg

### block6_n04_19 / raw_fullband / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=4.979 ms, tau_vr=5.188 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000083
- errors: delta_tau_abs_err=0.306 ms, theta_v_abs_err=4.315 deg

### block7_n08_20 / raw_fullband / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.542 ms, tau_vr=5.583 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000224
- errors: delta_tau_abs_err=0.337 ms, theta_v_abs_err=4.733 deg

### block4_p08_17 / raw_fullband / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=4.729 ms, tau_vr=4.708 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000172
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.646 ms, tau_vr=6.083 ms, delta_tau=0.438 ms, theta_v=6.153 deg, score=0.000375
- errors: delta_tau_abs_err=0.879 ms, theta_v_abs_err=12.366 deg

### block6_n04_19 / raw_fullband / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=4.979 ms, tau_vr=5.188 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000103
- errors: delta_tau_abs_err=0.306 ms, theta_v_abs_err=4.315 deg

### block7_n08_20 / raw_fullband / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.542 ms, tau_vr=5.438 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000236
- errors: delta_tau_abs_err=0.191 ms, theta_v_abs_err=2.686 deg

### block4_p08_17 / raw_fullband / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=4.729 ms, tau_vr=4.708 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000174
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=1, tau_vl=4.229 ms, tau_vr=4.292 ms, delta_tau=0.062 ms, theta_v=0.877 deg, score=0.000363
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.090 deg

### block6_n04_19 / raw_fullband / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=5, tau_vl=4.979 ms, tau_vr=4.875 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000071
- errors: delta_tau_abs_err=0.619 ms, theta_v_abs_err=8.703 deg

### block7_n08_20 / raw_fullband / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.542 ms, tau_vr=5.583 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000237
- errors: delta_tau_abs_err=0.337 ms, theta_v_abs_err=4.733 deg

### block4_p08_17 / raw_fullband / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=4.729 ms, tau_vr=4.708 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000175
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=1, tau_vl=4.229 ms, tau_vr=4.292 ms, delta_tau=0.062 ms, theta_v=0.877 deg, score=0.000380
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.090 deg

### block6_n04_19 / raw_fullband / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=4.979 ms, tau_vr=5.188 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000106
- errors: delta_tau_abs_err=0.306 ms, theta_v_abs_err=4.315 deg

### block7_n08_20 / raw_fullband / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.542 ms, tau_vr=5.438 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000251
- errors: delta_tau_abs_err=0.191 ms, theta_v_abs_err=2.686 deg

### block4_p08_17 / raw_fullband / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=4.729 ms, tau_vr=4.708 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.025695
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=1, tau_vl=4.229 ms, tau_vr=4.292 ms, delta_tau=0.062 ms, theta_v=0.877 deg, score=0.035487
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.090 deg

### block6_n04_19 / raw_fullband / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=4, vr_rank=7, tau_vl=4.771 ms, tau_vr=4.771 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.016079
- errors: delta_tau_abs_err=0.514 ms, theta_v_abs_err=7.240 deg

### block7_n08_20 / raw_fullband / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.542 ms, tau_vr=5.583 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.030141
- errors: delta_tau_abs_err=0.337 ms, theta_v_abs_err=4.733 deg

### block4_p08_17 / raw_fullband / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=4.729 ms, tau_vr=4.708 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000168
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=1, tau_vl=4.229 ms, tau_vr=4.292 ms, delta_tau=0.062 ms, theta_v=0.877 deg, score=0.000341
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.090 deg

### block6_n04_19 / raw_fullband / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=4.979 ms, tau_vr=5.188 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000083
- errors: delta_tau_abs_err=0.306 ms, theta_v_abs_err=4.315 deg

### block7_n08_20 / raw_fullband / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.542 ms, tau_vr=5.583 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000224
- errors: delta_tau_abs_err=0.337 ms, theta_v_abs_err=4.733 deg

### block4_p08_17 / raw_fullband / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=3, tau_vl=4.042 ms, tau_vr=4.021 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000162
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.646 ms, tau_vr=6.083 ms, delta_tau=0.438 ms, theta_v=6.153 deg, score=0.000228
- errors: delta_tau_abs_err=0.879 ms, theta_v_abs_err=12.366 deg

### block6_n04_19 / raw_fullband / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=4.979 ms, tau_vr=5.188 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000076
- errors: delta_tau_abs_err=0.306 ms, theta_v_abs_err=4.315 deg

### block7_n08_20 / raw_fullband / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=5, vr_rank=1, tau_vl=4.646 ms, tau_vr=4.771 ms, delta_tau=0.125 ms, theta_v=1.755 deg, score=0.000123
- errors: delta_tau_abs_err=0.420 ms, theta_v_abs_err=5.903 deg

### block4_p08_17 / raw_fullband / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=4.729 ms, tau_vr=4.708 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000170
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=4, vr_rank=5, tau_vl=4.729 ms, tau_vr=4.750 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000215
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block6_n04_19 / raw_fullband / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=5, tau_vl=4.979 ms, tau_vr=4.875 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000076
- errors: delta_tau_abs_err=0.619 ms, theta_v_abs_err=8.703 deg

### block7_n08_20 / raw_fullband / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.917 ms, tau_vr=4.771 ms, delta_tau=-0.146 ms, theta_v=-2.048 deg, score=0.000160
- errors: delta_tau_abs_err=0.149 ms, theta_v_abs_err=2.101 deg

### block4_p08_17 / raw_fullband / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=4.729 ms, tau_vr=4.708 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000125
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=4, vr_rank=5, tau_vl=4.729 ms, tau_vr=4.750 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000213
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block6_n04_19 / raw_fullband / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=4.979 ms, tau_vr=5.188 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000069
- errors: delta_tau_abs_err=0.306 ms, theta_v_abs_err=4.315 deg

### block7_n08_20 / raw_fullband / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.917 ms, tau_vr=4.771 ms, delta_tau=-0.146 ms, theta_v=-2.048 deg, score=0.000147
- errors: delta_tau_abs_err=0.149 ms, theta_v_abs_err=2.101 deg

### block4_p08_17 / raw_fullband / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=1, tau_vl=5.646 ms, tau_vr=6.333 ms, delta_tau=0.688 ms, theta_v=9.697 deg, score=0.000222
- errors: delta_tau_abs_err=1.129 ms, theta_v_abs_err=15.910 deg

### block5_p00_18 / raw_fullband / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.646 ms, tau_vr=6.083 ms, delta_tau=0.438 ms, theta_v=6.153 deg, score=0.000581
- errors: delta_tau_abs_err=0.879 ms, theta_v_abs_err=12.366 deg

### block6_n04_19 / raw_fullband / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.979 ms, tau_vr=5.771 ms, delta_tau=0.792 ms, theta_v=11.184 deg, score=0.000129
- errors: delta_tau_abs_err=0.277 ms, theta_v_abs_err=3.943 deg

### block7_n08_20 / raw_fullband / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.542 ms, tau_vr=4.771 ms, delta_tau=-0.771 ms, theta_v=-10.886 deg, score=0.000269
- errors: delta_tau_abs_err=0.476 ms, theta_v_abs_err=6.738 deg

### block4_p08_17 / raw_fullband / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=7, vr_rank=1, tau_vl=6.354 ms, tau_vr=6.333 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000173
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=7, vr_rank=1, tau_vl=6.104 ms, tau_vr=6.083 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000279
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / raw_fullband / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=4, vr_rank=6, tau_vl=4.771 ms, tau_vr=4.771 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.000062
- errors: delta_tau_abs_err=0.514 ms, theta_v_abs_err=7.240 deg

### block7_n08_20 / raw_fullband / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.542 ms, tau_vr=5.583 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000206
- errors: delta_tau_abs_err=0.337 ms, theta_v_abs_err=4.733 deg

### block4_p08_17 / raw_fullband / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=7, vr_rank=1, tau_vl=6.354 ms, tau_vr=6.333 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000180
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=7, vr_rank=1, tau_vl=6.104 ms, tau_vr=6.083 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000291
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / raw_fullband / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.188 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000083
- errors: delta_tau_abs_err=0.306 ms, theta_v_abs_err=4.315 deg

### block7_n08_20 / raw_fullband / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.542 ms, tau_vr=5.583 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000224
- errors: delta_tau_abs_err=0.337 ms, theta_v_abs_err=4.733 deg

### block4_p08_17 / raw_fullband / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=7, vr_rank=1, tau_vl=6.354 ms, tau_vr=6.333 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000184
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.646 ms, tau_vr=6.083 ms, delta_tau=0.438 ms, theta_v=6.153 deg, score=0.000375
- errors: delta_tau_abs_err=0.879 ms, theta_v_abs_err=12.366 deg

### block6_n04_19 / raw_fullband / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.188 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000103
- errors: delta_tau_abs_err=0.306 ms, theta_v_abs_err=4.315 deg

### block7_n08_20 / raw_fullband / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.542 ms, tau_vr=5.438 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000236
- errors: delta_tau_abs_err=0.191 ms, theta_v_abs_err=2.686 deg

### block4_p08_17 / raw_fullband / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=7, vr_rank=1, tau_vl=6.354 ms, tau_vr=6.333 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000187
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=7, vr_rank=1, tau_vl=6.104 ms, tau_vr=6.083 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000301
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / raw_fullband / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.979 ms, tau_vr=4.875 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000071
- errors: delta_tau_abs_err=0.619 ms, theta_v_abs_err=8.703 deg

### block7_n08_20 / raw_fullband / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.542 ms, tau_vr=5.583 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000237
- errors: delta_tau_abs_err=0.337 ms, theta_v_abs_err=4.733 deg

### block4_p08_17 / raw_fullband / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=7, vr_rank=1, tau_vl=6.354 ms, tau_vr=6.333 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000188
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=5.646 ms, tau_vr=5.479 ms, delta_tau=-0.167 ms, theta_v=-2.340 deg, score=0.000313
- errors: delta_tau_abs_err=0.275 ms, theta_v_abs_err=3.872 deg

### block6_n04_19 / raw_fullband / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.188 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000106
- errors: delta_tau_abs_err=0.306 ms, theta_v_abs_err=4.315 deg

### block7_n08_20 / raw_fullband / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.542 ms, tau_vr=5.438 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000251
- errors: delta_tau_abs_err=0.191 ms, theta_v_abs_err=2.686 deg

### block4_p08_17 / raw_fullband / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=7, vr_rank=1, tau_vl=6.354 ms, tau_vr=6.333 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.027404
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=7, vr_rank=1, tau_vl=6.104 ms, tau_vr=6.083 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.034853
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / raw_fullband / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=4, vr_rank=6, tau_vl=4.771 ms, tau_vr=4.771 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.016079
- errors: delta_tau_abs_err=0.514 ms, theta_v_abs_err=7.240 deg

### block7_n08_20 / raw_fullband / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.542 ms, tau_vr=5.583 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.030141
- errors: delta_tau_abs_err=0.337 ms, theta_v_abs_err=4.733 deg

### block4_p08_17 / raw_fullband / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=7, vr_rank=1, tau_vl=6.354 ms, tau_vr=6.333 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000180
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=7, vr_rank=1, tau_vl=6.104 ms, tau_vr=6.083 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000291
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / raw_fullband / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.188 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000083
- errors: delta_tau_abs_err=0.306 ms, theta_v_abs_err=4.315 deg

### block7_n08_20 / raw_fullband / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.542 ms, tau_vr=5.583 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000224
- errors: delta_tau_abs_err=0.337 ms, theta_v_abs_err=4.733 deg

### block4_p08_17 / raw_fullband / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=5.646 ms, tau_vr=5.688 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000136
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.797 deg

### block5_p00_18 / raw_fullband / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.646 ms, tau_vr=6.083 ms, delta_tau=0.438 ms, theta_v=6.153 deg, score=0.000228
- errors: delta_tau_abs_err=0.879 ms, theta_v_abs_err=12.366 deg

### block6_n04_19 / raw_fullband / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.188 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000076
- errors: delta_tau_abs_err=0.306 ms, theta_v_abs_err=4.315 deg

### block7_n08_20 / raw_fullband / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=5, vr_rank=1, tau_vl=4.646 ms, tau_vr=4.771 ms, delta_tau=0.125 ms, theta_v=1.755 deg, score=0.000123
- errors: delta_tau_abs_err=0.420 ms, theta_v_abs_err=5.903 deg

### block4_p08_17 / raw_fullband / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.729 ms, tau_vr=4.708 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000170
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.729 ms, tau_vr=4.750 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000215
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block6_n04_19 / raw_fullband / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.979 ms, tau_vr=4.875 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000076
- errors: delta_tau_abs_err=0.619 ms, theta_v_abs_err=8.703 deg

### block7_n08_20 / raw_fullband / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.917 ms, tau_vr=4.771 ms, delta_tau=-0.146 ms, theta_v=-2.048 deg, score=0.000160
- errors: delta_tau_abs_err=0.149 ms, theta_v_abs_err=2.101 deg

### block4_p08_17 / raw_fullband / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.729 ms, tau_vr=4.708 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000125
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / raw_fullband / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.729 ms, tau_vr=4.750 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000213
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block6_n04_19 / raw_fullband / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.188 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000069
- errors: delta_tau_abs_err=0.306 ms, theta_v_abs_err=4.315 deg

### block7_n08_20 / raw_fullband / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.917 ms, tau_vr=4.771 ms, delta_tau=-0.146 ms, theta_v=-2.048 deg, score=0.000147
- errors: delta_tau_abs_err=0.149 ms, theta_v_abs_err=2.101 deg

### block4_p08_17 / bp_500_2000 / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.688 ms, tau_vr=3.667 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000055
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / bp_500_2000 / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.250 ms, tau_vr=4.104 ms, delta_tau=-0.146 ms, theta_v=-2.048 deg, score=0.000075
- errors: delta_tau_abs_err=0.296 ms, theta_v_abs_err=4.165 deg

### block6_n04_19 / bp_500_2000 / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000029
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.854 ms, tau_vr=4.000 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000062
- errors: delta_tau_abs_err=0.441 ms, theta_v_abs_err=6.196 deg

### block4_p08_17 / bp_500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.688 ms, tau_vr=3.667 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000051
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / bp_500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000056
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000027
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.854 ms, tau_vr=4.000 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000035
- errors: delta_tau_abs_err=0.441 ms, theta_v_abs_err=6.196 deg

### block4_p08_17 / bp_500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.688 ms, tau_vr=3.667 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000053
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / bp_500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000058
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000028
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.854 ms, tau_vr=4.000 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000047
- errors: delta_tau_abs_err=0.441 ms, theta_v_abs_err=6.196 deg

### block4_p08_17 / bp_500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.688 ms, tau_vr=3.667 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000054
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / bp_500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.250 ms, tau_vr=4.104 ms, delta_tau=-0.146 ms, theta_v=-2.048 deg, score=0.000065
- errors: delta_tau_abs_err=0.296 ms, theta_v_abs_err=4.165 deg

### block6_n04_19 / bp_500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000028
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.854 ms, tau_vr=4.000 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000054
- errors: delta_tau_abs_err=0.441 ms, theta_v_abs_err=6.196 deg

### block4_p08_17 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.688 ms, tau_vr=3.667 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000055
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000060
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000029
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.854 ms, tau_vr=4.000 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000044
- errors: delta_tau_abs_err=0.441 ms, theta_v_abs_err=6.196 deg

### block4_p08_17 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.688 ms, tau_vr=3.667 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000055
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.250 ms, tau_vr=4.104 ms, delta_tau=-0.146 ms, theta_v=-2.048 deg, score=0.000069
- errors: delta_tau_abs_err=0.296 ms, theta_v_abs_err=4.165 deg

### block6_n04_19 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000029
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.854 ms, tau_vr=4.000 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000057
- errors: delta_tau_abs_err=0.441 ms, theta_v_abs_err=6.196 deg

### block4_p08_17 / bp_500_2000 / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.688 ms, tau_vr=3.667 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.014358
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / bp_500_2000 / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.015045
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.010555
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.854 ms, tau_vr=4.000 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.011896
- errors: delta_tau_abs_err=0.441 ms, theta_v_abs_err=6.196 deg

### block4_p08_17 / bp_500_2000 / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.688 ms, tau_vr=3.667 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000053
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / bp_500_2000 / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000058
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000028
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.854 ms, tau_vr=4.000 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000047
- errors: delta_tau_abs_err=0.441 ms, theta_v_abs_err=6.196 deg

### block4_p08_17 / bp_500_2000 / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.688 ms, tau_vr=3.667 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000040
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / bp_500_2000 / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000048
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000018
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.854 ms, tau_vr=4.000 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000036
- errors: delta_tau_abs_err=0.441 ms, theta_v_abs_err=6.196 deg

### block4_p08_17 / bp_500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=5.000 ms, tau_vr=4.604 ms, delta_tau=-0.396 ms, theta_v=-5.565 deg, score=0.000027
- errors: delta_tau_abs_err=0.046 ms, theta_v_abs_err=0.647 deg

### block5_p00_18 / bp_500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000043
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=3, vr_rank=4, tau_vl=5.208 ms, tau_vr=4.979 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000010
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / bp_500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=3, vr_rank=3, tau_vl=4.375 ms, tau_vr=4.354 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000009
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / bp_500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=5.000 ms, tau_vr=4.604 ms, delta_tau=-0.396 ms, theta_v=-5.565 deg, score=0.000026
- errors: delta_tau_abs_err=0.046 ms, theta_v_abs_err=0.647 deg

### block5_p00_18 / bp_500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000035
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=3, vr_rank=4, tau_vl=5.208 ms, tau_vr=4.979 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000008
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / bp_500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=3, vr_rank=3, tau_vl=4.375 ms, tau_vr=4.354 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000009
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / bp_500_2000 / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.000 ms, tau_vr=4.604 ms, delta_tau=-0.396 ms, theta_v=-5.565 deg, score=0.000042
- errors: delta_tau_abs_err=0.046 ms, theta_v_abs_err=0.647 deg

### block5_p00_18 / bp_500_2000 / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.250 ms, tau_vr=4.104 ms, delta_tau=-0.146 ms, theta_v=-2.048 deg, score=0.000075
- errors: delta_tau_abs_err=0.296 ms, theta_v_abs_err=4.165 deg

### block6_n04_19 / bp_500_2000 / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000029
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.375 ms, tau_vr=4.354 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000023
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=5.000 ms, tau_vr=5.021 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000016
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000056
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000027
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.375 ms, tau_vr=4.354 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000021
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=5.729 ms, tau_vr=5.479 ms, delta_tau=-0.250 ms, theta_v=-3.512 deg, score=0.000023
- errors: delta_tau_abs_err=0.192 ms, theta_v_abs_err=2.701 deg

### block5_p00_18 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000058
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000028
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.375 ms, tau_vr=4.354 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000022
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=5.729 ms, tau_vr=5.479 ms, delta_tau=-0.250 ms, theta_v=-3.512 deg, score=0.000030
- errors: delta_tau_abs_err=0.192 ms, theta_v_abs_err=2.701 deg

### block5_p00_18 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.250 ms, tau_vr=4.104 ms, delta_tau=-0.146 ms, theta_v=-2.048 deg, score=0.000065
- errors: delta_tau_abs_err=0.296 ms, theta_v_abs_err=4.165 deg

### block6_n04_19 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000028
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.375 ms, tau_vr=4.354 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000022
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=5.000 ms, tau_vr=5.021 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000017
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000060
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000029
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.375 ms, tau_vr=4.354 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000023
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=5.729 ms, tau_vr=5.479 ms, delta_tau=-0.250 ms, theta_v=-3.512 deg, score=0.000030
- errors: delta_tau_abs_err=0.192 ms, theta_v_abs_err=2.701 deg

### block5_p00_18 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.250 ms, tau_vr=4.104 ms, delta_tau=-0.146 ms, theta_v=-2.048 deg, score=0.000069
- errors: delta_tau_abs_err=0.296 ms, theta_v_abs_err=4.165 deg

### block6_n04_19 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000029
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.375 ms, tau_vr=4.354 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000023
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / bp_500_2000 / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=5.000 ms, tau_vr=5.021 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.008840
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_500_2000 / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.015045
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.010555
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.375 ms, tau_vr=4.354 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.009149
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / bp_500_2000 / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=5.729 ms, tau_vr=5.479 ms, delta_tau=-0.250 ms, theta_v=-3.512 deg, score=0.000023
- errors: delta_tau_abs_err=0.192 ms, theta_v_abs_err=2.701 deg

### block5_p00_18 / bp_500_2000 / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000058
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000028
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.375 ms, tau_vr=4.354 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000022
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / bp_500_2000 / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=5.729 ms, tau_vr=5.479 ms, delta_tau=-0.250 ms, theta_v=-3.512 deg, score=0.000021
- errors: delta_tau_abs_err=0.192 ms, theta_v_abs_err=2.701 deg

### block5_p00_18 / bp_500_2000 / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000048
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000018
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.375 ms, tau_vr=4.354 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000022
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / bp_500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.000 ms, tau_vr=4.604 ms, delta_tau=-0.396 ms, theta_v=-5.565 deg, score=0.000027
- errors: delta_tau_abs_err=0.046 ms, theta_v_abs_err=0.647 deg

### block5_p00_18 / bp_500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000043
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=3, vr_rank=4, tau_vl=5.208 ms, tau_vr=4.979 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000010
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / bp_500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.375 ms, tau_vr=4.354 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000009
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / bp_500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.000 ms, tau_vr=4.604 ms, delta_tau=-0.396 ms, theta_v=-5.565 deg, score=0.000026
- errors: delta_tau_abs_err=0.046 ms, theta_v_abs_err=0.647 deg

### block5_p00_18 / bp_500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000035
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=3, vr_rank=4, tau_vl=5.208 ms, tau_vr=4.979 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000008
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / bp_500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.375 ms, tau_vr=4.354 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000009
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / bp_500_2000 / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.000 ms, tau_vr=4.604 ms, delta_tau=-0.396 ms, theta_v=-5.565 deg, score=0.000042
- errors: delta_tau_abs_err=0.046 ms, theta_v_abs_err=0.647 deg

### block5_p00_18 / bp_500_2000 / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=6.125 ms, tau_vr=6.021 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000067
- errors: delta_tau_abs_err=0.338 ms, theta_v_abs_err=4.750 deg

### block6_n04_19 / bp_500_2000 / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000029
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.562 ms, tau_vr=5.479 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000018
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=5.000 ms, tau_vr=5.021 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000016
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000056
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000027
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.562 ms, tau_vr=5.479 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000013
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=5.729 ms, tau_vr=5.479 ms, delta_tau=-0.250 ms, theta_v=-3.512 deg, score=0.000023
- errors: delta_tau_abs_err=0.192 ms, theta_v_abs_err=2.701 deg

### block5_p00_18 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000058
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000028
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.562 ms, tau_vr=5.479 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000015
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=5.729 ms, tau_vr=5.479 ms, delta_tau=-0.250 ms, theta_v=-3.512 deg, score=0.000030
- errors: delta_tau_abs_err=0.192 ms, theta_v_abs_err=2.701 deg

### block5_p00_18 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=6.125 ms, tau_vr=6.021 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000060
- errors: delta_tau_abs_err=0.338 ms, theta_v_abs_err=4.750 deg

### block6_n04_19 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000028
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.562 ms, tau_vr=5.479 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000017
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=5.000 ms, tau_vr=5.021 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000017
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000060
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000029
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.562 ms, tau_vr=5.479 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000016
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=5.729 ms, tau_vr=5.479 ms, delta_tau=-0.250 ms, theta_v=-3.512 deg, score=0.000030
- errors: delta_tau_abs_err=0.192 ms, theta_v_abs_err=2.701 deg

### block5_p00_18 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=6.125 ms, tau_vr=6.021 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000064
- errors: delta_tau_abs_err=0.338 ms, theta_v_abs_err=4.750 deg

### block6_n04_19 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000029
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.562 ms, tau_vr=5.479 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000017
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_500_2000 / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=5.000 ms, tau_vr=5.021 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.008840
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_500_2000 / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.015045
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.010555
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.562 ms, tau_vr=5.479 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.007876
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_500_2000 / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=5.729 ms, tau_vr=5.479 ms, delta_tau=-0.250 ms, theta_v=-3.512 deg, score=0.000023
- errors: delta_tau_abs_err=0.192 ms, theta_v_abs_err=2.701 deg

### block5_p00_18 / bp_500_2000 / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000058
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000028
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.562 ms, tau_vr=5.479 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000015
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_500_2000 / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=5.729 ms, tau_vr=5.479 ms, delta_tau=-0.250 ms, theta_v=-3.512 deg, score=0.000021
- errors: delta_tau_abs_err=0.192 ms, theta_v_abs_err=2.701 deg

### block5_p00_18 / bp_500_2000 / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000048
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.667 ms, tau_vr=5.688 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000018
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_500_2000 / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.562 ms, tau_vr=5.479 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000006
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.000 ms, tau_vr=4.604 ms, delta_tau=-0.396 ms, theta_v=-5.565 deg, score=0.000027
- errors: delta_tau_abs_err=0.046 ms, theta_v_abs_err=0.647 deg

### block5_p00_18 / bp_500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000043
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=4, tau_vl=5.208 ms, tau_vr=4.979 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000010
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / bp_500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=5, vr_rank=3, tau_vl=4.917 ms, tau_vr=4.896 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000007
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / bp_500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.000 ms, tau_vr=4.604 ms, delta_tau=-0.396 ms, theta_v=-5.565 deg, score=0.000026
- errors: delta_tau_abs_err=0.046 ms, theta_v_abs_err=0.647 deg

### block5_p00_18 / bp_500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.542 ms, tau_vr=4.521 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000035
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / bp_500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=4, tau_vl=5.208 ms, tau_vr=4.979 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000008
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / bp_500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=5, vr_rank=3, tau_vl=4.917 ms, tau_vr=4.896 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000005
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / bp_1000_3000 / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=3.688 ms, tau_vr=3.583 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000060
- errors: delta_tau_abs_err=0.338 ms, theta_v_abs_err=4.750 deg

### block5_p00_18 / bp_1000_3000 / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.208 ms, tau_vr=5.167 ms, delta_tau=0.958 ms, theta_v=13.579 deg, score=0.000066
- errors: delta_tau_abs_err=1.400 ms, theta_v_abs_err=19.792 deg

### block6_n04_19 / bp_1000_3000 / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000042
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.125 ms, tau_vr=3.688 ms, delta_tau=-0.438 ms, theta_v=-6.153 deg, score=0.000067
- errors: delta_tau_abs_err=0.142 ms, theta_v_abs_err=2.005 deg

### block4_p08_17 / bp_1000_3000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=3.688 ms, tau_vr=3.583 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000039
- errors: delta_tau_abs_err=0.338 ms, theta_v_abs_err=4.750 deg

### block5_p00_18 / bp_1000_3000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.208 ms, tau_vr=4.208 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.000057
- errors: delta_tau_abs_err=0.442 ms, theta_v_abs_err=6.213 deg

### block6_n04_19 / bp_1000_3000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000038
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=3.729 ms, tau_vr=3.688 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000051
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / bp_1000_3000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=3.688 ms, tau_vr=3.583 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000048
- errors: delta_tau_abs_err=0.338 ms, theta_v_abs_err=4.750 deg

### block5_p00_18 / bp_1000_3000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.208 ms, tau_vr=4.208 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.000057
- errors: delta_tau_abs_err=0.442 ms, theta_v_abs_err=6.213 deg

### block6_n04_19 / bp_1000_3000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000040
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=3.729 ms, tau_vr=3.688 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000055
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / bp_1000_3000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=3.688 ms, tau_vr=3.583 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000054
- errors: delta_tau_abs_err=0.338 ms, theta_v_abs_err=4.750 deg

### block5_p00_18 / bp_1000_3000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=6.104 ms, tau_vr=6.062 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000058
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block6_n04_19 / bp_1000_3000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000041
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=3.729 ms, tau_vr=3.688 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000058
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=3.688 ms, tau_vr=3.583 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000050
- errors: delta_tau_abs_err=0.338 ms, theta_v_abs_err=4.750 deg

### block5_p00_18 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=6.104 ms, tau_vr=6.062 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000058
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block6_n04_19 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000041
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=3.729 ms, tau_vr=3.688 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000058
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=3.688 ms, tau_vr=3.583 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000057
- errors: delta_tau_abs_err=0.338 ms, theta_v_abs_err=4.750 deg

### block5_p00_18 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=6.104 ms, tau_vr=6.062 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000060
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block6_n04_19 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000042
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=3.729 ms, tau_vr=3.688 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000060
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / bp_1000_3000 / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=3.688 ms, tau_vr=3.583 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.012909
- errors: delta_tau_abs_err=0.338 ms, theta_v_abs_err=4.750 deg

### block5_p00_18 / bp_1000_3000 / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.208 ms, tau_vr=4.208 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.015117
- errors: delta_tau_abs_err=0.442 ms, theta_v_abs_err=6.213 deg

### block6_n04_19 / bp_1000_3000 / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.012473
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=3.729 ms, tau_vr=3.688 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.015037
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / bp_1000_3000 / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=3.688 ms, tau_vr=3.583 ms, delta_tau=-0.104 ms, theta_v=-1.462 deg, score=0.000048
- errors: delta_tau_abs_err=0.338 ms, theta_v_abs_err=4.750 deg

### block5_p00_18 / bp_1000_3000 / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.208 ms, tau_vr=4.208 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.000057
- errors: delta_tau_abs_err=0.442 ms, theta_v_abs_err=6.213 deg

### block6_n04_19 / bp_1000_3000 / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000040
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=3.729 ms, tau_vr=3.688 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000055
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / bp_1000_3000 / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=4, vr_rank=2, tau_vl=3.542 ms, tau_vr=3.583 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000034
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.797 deg

### block5_p00_18 / bp_1000_3000 / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=6.104 ms, tau_vr=6.062 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000054
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block6_n04_19 / bp_1000_3000 / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000031
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=4.125 ms, tau_vr=4.042 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000033
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=3, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000035
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=6, vr_rank=3, tau_vl=4.854 ms, tau_vr=4.792 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000032
- errors: delta_tau_abs_err=0.379 ms, theta_v_abs_err=5.335 deg

### block6_n04_19 / bp_1000_3000 / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=5, vr_rank=1, tau_vl=5.042 ms, tau_vr=5.229 ms, delta_tau=0.188 ms, theta_v=2.633 deg, score=0.000015
- errors: delta_tau_abs_err=0.327 ms, theta_v_abs_err=4.607 deg

### block7_n08_20 / bp_1000_3000 / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=8, vr_rank=5, tau_vl=4.688 ms, tau_vr=4.646 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000009
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / bp_1000_3000 / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=3, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000027
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=6, vr_rank=3, tau_vl=4.854 ms, tau_vr=4.792 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000018
- errors: delta_tau_abs_err=0.379 ms, theta_v_abs_err=5.335 deg

### block6_n04_19 / bp_1000_3000 / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=5, vr_rank=7, tau_vl=5.042 ms, tau_vr=4.979 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000013
- errors: delta_tau_abs_err=0.577 ms, theta_v_abs_err=8.118 deg

### block7_n08_20 / bp_1000_3000 / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=8, vr_rank=8, tau_vl=4.688 ms, tau_vr=4.812 ms, delta_tau=0.125 ms, theta_v=1.755 deg, score=0.000008
- errors: delta_tau_abs_err=0.420 ms, theta_v_abs_err=5.903 deg

### block4_p08_17 / bp_1000_3000 / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=5.667 ms, tau_vr=6.354 ms, delta_tau=0.688 ms, theta_v=9.697 deg, score=0.000052
- errors: delta_tau_abs_err=1.129 ms, theta_v_abs_err=15.910 deg

### block5_p00_18 / bp_1000_3000 / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.208 ms, tau_vr=5.167 ms, delta_tau=0.958 ms, theta_v=13.579 deg, score=0.000066
- errors: delta_tau_abs_err=1.400 ms, theta_v_abs_err=19.792 deg

### block6_n04_19 / bp_1000_3000 / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000042
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.125 ms, tau_vr=4.042 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000054
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000039
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.208 ms, tau_vr=4.208 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.000057
- errors: delta_tau_abs_err=0.442 ms, theta_v_abs_err=6.213 deg

### block6_n04_19 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000038
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.125 ms, tau_vr=4.042 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000039
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000040
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.208 ms, tau_vr=4.208 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.000057
- errors: delta_tau_abs_err=0.442 ms, theta_v_abs_err=6.213 deg

### block6_n04_19 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000040
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.125 ms, tau_vr=4.042 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000046
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000041
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=6.104 ms, tau_vr=6.062 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000058
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block6_n04_19 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000041
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.125 ms, tau_vr=4.042 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000050
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000042
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=6.104 ms, tau_vr=6.062 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000058
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block6_n04_19 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000041
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.125 ms, tau_vr=4.042 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000048
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000042
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=6.104 ms, tau_vr=6.062 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000060
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block6_n04_19 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000042
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.125 ms, tau_vr=4.042 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000053
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=1, tau_vl=6.333 ms, tau_vr=6.354 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.012571
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.208 ms, tau_vr=4.208 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.015117
- errors: delta_tau_abs_err=0.442 ms, theta_v_abs_err=6.213 deg

### block6_n04_19 / bp_1000_3000 / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.012473
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.125 ms, tau_vr=4.042 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.012623
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000040
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.208 ms, tau_vr=4.208 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.000057
- errors: delta_tau_abs_err=0.442 ms, theta_v_abs_err=6.213 deg

### block6_n04_19 / bp_1000_3000 / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000040
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.125 ms, tau_vr=4.042 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000046
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000031
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=6.104 ms, tau_vr=6.062 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000054
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block6_n04_19 / bp_1000_3000 / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.208 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000031
- errors: delta_tau_abs_err=0.494 ms, theta_v_abs_err=6.948 deg

### block7_n08_20 / bp_1000_3000 / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=4.125 ms, tau_vr=4.042 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000033
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000035
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=6, vr_rank=3, tau_vl=4.854 ms, tau_vr=4.792 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000032
- errors: delta_tau_abs_err=0.379 ms, theta_v_abs_err=5.335 deg

### block6_n04_19 / bp_1000_3000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=5, vr_rank=1, tau_vl=5.042 ms, tau_vr=5.229 ms, delta_tau=0.188 ms, theta_v=2.633 deg, score=0.000015
- errors: delta_tau_abs_err=0.327 ms, theta_v_abs_err=4.607 deg

### block7_n08_20 / bp_1000_3000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=6, vr_rank=3, tau_vl=4.688 ms, tau_vr=4.646 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000009
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / bp_1000_3000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000027
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=6, vr_rank=3, tau_vl=4.854 ms, tau_vr=4.792 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000018
- errors: delta_tau_abs_err=0.379 ms, theta_v_abs_err=5.335 deg

### block6_n04_19 / bp_1000_3000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=5, vr_rank=6, tau_vl=5.042 ms, tau_vr=4.979 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000013
- errors: delta_tau_abs_err=0.577 ms, theta_v_abs_err=8.118 deg

### block7_n08_20 / bp_1000_3000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=6, vr_rank=6, tau_vl=4.688 ms, tau_vr=4.812 ms, delta_tau=0.125 ms, theta_v=1.755 deg, score=0.000008
- errors: delta_tau_abs_err=0.420 ms, theta_v_abs_err=5.903 deg

### block4_p08_17 / bp_1000_3000 / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=5.667 ms, tau_vr=6.354 ms, delta_tau=0.688 ms, theta_v=9.697 deg, score=0.000052
- errors: delta_tau_abs_err=1.129 ms, theta_v_abs_err=15.910 deg

### block5_p00_18 / bp_1000_3000 / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=6.104 ms, tau_vr=5.167 ms, delta_tau=-0.938 ms, theta_v=-13.279 deg, score=0.000064
- errors: delta_tau_abs_err=0.496 ms, theta_v_abs_err=7.066 deg

### block6_n04_19 / bp_1000_3000 / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.979 ms, tau_vr=5.229 ms, delta_tau=-0.750 ms, theta_v=-10.588 deg, score=0.000040
- errors: delta_tau_abs_err=1.264 ms, theta_v_abs_err=17.829 deg

### block7_n08_20 / bp_1000_3000 / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.604 ms, tau_vr=4.646 ms, delta_tau=-0.958 ms, theta_v=-13.579 deg, score=0.000025
- errors: delta_tau_abs_err=0.663 ms, theta_v_abs_err=9.431 deg

### block4_p08_17 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000039
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=6.104 ms, tau_vr=6.062 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000051
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block6_n04_19 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.979 ms, tau_vr=5.979 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.000028
- errors: delta_tau_abs_err=0.514 ms, theta_v_abs_err=7.240 deg

### block7_n08_20 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=5.604 ms, tau_vr=5.521 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000012
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000040
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=6.104 ms, tau_vr=6.062 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000055
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block6_n04_19 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.979 ms, tau_vr=5.979 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.000028
- errors: delta_tau_abs_err=0.514 ms, theta_v_abs_err=7.240 deg

### block7_n08_20 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=5.604 ms, tau_vr=5.521 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000014
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000041
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=6.104 ms, tau_vr=6.062 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000058
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block6_n04_19 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.979 ms, tau_vr=5.979 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.000028
- errors: delta_tau_abs_err=0.514 ms, theta_v_abs_err=7.240 deg

### block7_n08_20 / bp_1000_3000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=5.604 ms, tau_vr=5.521 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000016
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000042
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=6.104 ms, tau_vr=6.062 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000058
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block6_n04_19 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.979 ms, tau_vr=5.979 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.000028
- errors: delta_tau_abs_err=0.514 ms, theta_v_abs_err=7.240 deg

### block7_n08_20 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=5.604 ms, tau_vr=5.521 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000015
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000042
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=6.104 ms, tau_vr=6.062 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000060
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block6_n04_19 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.979 ms, tau_vr=5.979 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.000028
- errors: delta_tau_abs_err=0.514 ms, theta_v_abs_err=7.240 deg

### block7_n08_20 / bp_1000_3000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=5.604 ms, tau_vr=5.521 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000016
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=1, tau_vl=6.333 ms, tau_vr=6.354 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.012571
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=6.104 ms, tau_vr=6.062 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.014270
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block6_n04_19 / bp_1000_3000 / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.979 ms, tau_vr=5.979 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.010556
- errors: delta_tau_abs_err=0.514 ms, theta_v_abs_err=7.240 deg

### block7_n08_20 / bp_1000_3000 / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=5.604 ms, tau_vr=5.521 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.007871
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000040
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=6.104 ms, tau_vr=6.062 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000055
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block6_n04_19 / bp_1000_3000 / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.979 ms, tau_vr=5.979 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.000028
- errors: delta_tau_abs_err=0.514 ms, theta_v_abs_err=7.240 deg

### block7_n08_20 / bp_1000_3000 / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=6, tau_vl=5.604 ms, tau_vr=5.521 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000014
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / bp_1000_3000 / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000031
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=6.104 ms, tau_vr=6.062 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000054
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block6_n04_19 / bp_1000_3000 / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.979 ms, tau_vr=5.979 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.000026
- errors: delta_tau_abs_err=0.514 ms, theta_v_abs_err=7.240 deg

### block7_n08_20 / bp_1000_3000 / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.646 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000007
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / bp_1000_3000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000035
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=5, vr_rank=3, tau_vl=4.854 ms, tau_vr=4.792 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000032
- errors: delta_tau_abs_err=0.379 ms, theta_v_abs_err=5.335 deg

### block6_n04_19 / bp_1000_3000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=5.042 ms, tau_vr=5.229 ms, delta_tau=0.188 ms, theta_v=2.633 deg, score=0.000015
- errors: delta_tau_abs_err=0.327 ms, theta_v_abs_err=4.607 deg

### block7_n08_20 / bp_1000_3000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.646 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000009
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / bp_1000_3000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.979 ms, tau_vr=5.000 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000027
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block5_p00_18 / bp_1000_3000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=5, vr_rank=3, tau_vl=4.854 ms, tau_vr=4.792 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000018
- errors: delta_tau_abs_err=0.379 ms, theta_v_abs_err=5.335 deg

### block6_n04_19 / bp_1000_3000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=4, vr_rank=8, tau_vl=5.042 ms, tau_vr=4.833 ms, delta_tau=-0.208 ms, theta_v=-2.926 deg, score=0.000013
- errors: delta_tau_abs_err=0.723 ms, theta_v_abs_err=10.166 deg

### block7_n08_20 / bp_1000_3000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=4, tau_vl=4.688 ms, tau_vr=4.812 ms, delta_tau=0.125 ms, theta_v=1.755 deg, score=0.000008
- errors: delta_tau_abs_err=0.420 ms, theta_v_abs_err=5.903 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=3.604 ms, tau_vr=3.646 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000014
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.797 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.396 ms, tau_vr=3.979 ms, delta_tau=-0.417 ms, theta_v=-5.859 deg, score=0.000083
- errors: delta_tau_abs_err=0.025 ms, theta_v_abs_err=0.353 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=3.812 ms, tau_vr=4.167 ms, delta_tau=0.354 ms, theta_v=4.978 deg, score=0.000007
- errors: delta_tau_abs_err=0.160 ms, theta_v_abs_err=2.263 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.667 ms, tau_vr=3.875 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000106
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.074 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=3.604 ms, tau_vr=3.646 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000012
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.797 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000074
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.167 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000005
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.667 ms, tau_vr=3.875 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000046
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.074 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=3.604 ms, tau_vr=3.646 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000013
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.797 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000077
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.167 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000005
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.667 ms, tau_vr=3.875 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000070
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.074 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=3.604 ms, tau_vr=3.646 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000014
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.797 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000079
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.167 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000005
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.667 ms, tau_vr=3.875 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000086
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.074 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=3.604 ms, tau_vr=3.646 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000014
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.797 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000080
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.167 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000005
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.667 ms, tau_vr=3.875 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000053
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.074 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=3.604 ms, tau_vr=3.646 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000014
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.797 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000080
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.167 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000005
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.667 ms, tau_vr=3.875 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000090
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.074 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=3.604 ms, tau_vr=3.646 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.006991
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.797 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.017221
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.167 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.004375
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.667 ms, tau_vr=3.875 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.013611
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.074 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=3.604 ms, tau_vr=3.646 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000013
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.797 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000077
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.167 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000005
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.667 ms, tau_vr=3.875 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000070
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.074 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=3.604 ms, tau_vr=3.646 ms, delta_tau=0.042 ms, theta_v=0.585 deg, score=0.000010
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.797 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000076
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.167 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000003
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.667 ms, tau_vr=3.875 ms, delta_tau=0.208 ms, theta_v=2.926 deg, score=0.000066
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.074 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000009
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000034
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=6, vr_rank=1, tau_vl=4.729 ms, tau_vr=4.875 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000004
- errors: delta_tau_abs_err=0.369 ms, theta_v_abs_err=5.193 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=3, vr_rank=4, tau_vl=4.562 ms, tau_vr=4.500 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000013
- errors: delta_tau_abs_err=0.233 ms, theta_v_abs_err=3.271 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000008
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000034
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=6, vr_rank=1, tau_vl=4.729 ms, tau_vr=4.875 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000004
- errors: delta_tau_abs_err=0.369 ms, theta_v_abs_err=5.193 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=3, vr_rank=4, tau_vl=4.562 ms, tau_vr=4.500 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000010
- errors: delta_tau_abs_err=0.233 ms, theta_v_abs_err=3.271 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000013
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000081
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.188 ms, tau_vr=4.875 ms, delta_tau=0.688 ms, theta_v=9.697 deg, score=0.000005
- errors: delta_tau_abs_err=0.173 ms, theta_v_abs_err=2.457 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.458 ms, tau_vr=4.500 ms, delta_tau=-0.958 ms, theta_v=-13.579 deg, score=0.000026
- errors: delta_tau_abs_err=0.663 ms, theta_v_abs_err=9.431 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.688 ms, tau_vr=4.750 ms, delta_tau=0.062 ms, theta_v=0.877 deg, score=0.000007
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.090 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000074
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.167 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000005
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=5.458 ms, tau_vr=5.438 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000019
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000009
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000077
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.167 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000005
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=5.458 ms, tau_vr=5.438 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000020
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000011
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000079
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.167 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000005
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=5.458 ms, tau_vr=5.438 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000020
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.688 ms, tau_vr=4.750 ms, delta_tau=0.062 ms, theta_v=0.877 deg, score=0.000009
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.090 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000080
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.167 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000005
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=5.458 ms, tau_vr=5.438 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000020
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000011
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000080
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.167 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000005
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=5.458 ms, tau_vr=5.438 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000020
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.688 ms, tau_vr=4.750 ms, delta_tau=0.062 ms, theta_v=0.877 deg, score=0.005461
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.090 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.017221
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.167 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.004375
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=5.458 ms, tau_vr=5.438 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.009492
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000009
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000077
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.167 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000005
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=5.458 ms, tau_vr=5.438 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000020
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000008
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000076
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.188 ms, tau_vr=4.167 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000003
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=3, tau_vl=4.562 ms, tau_vr=4.500 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000013
- errors: delta_tau_abs_err=0.233 ms, theta_v_abs_err=3.271 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000009
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000034
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=5, vr_rank=1, tau_vl=4.729 ms, tau_vr=4.875 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000004
- errors: delta_tau_abs_err=0.369 ms, theta_v_abs_err=5.193 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=3, tau_vl=4.562 ms, tau_vr=4.500 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000013
- errors: delta_tau_abs_err=0.233 ms, theta_v_abs_err=3.271 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000008
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.396 ms, tau_vr=4.375 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000034
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=5, vr_rank=1, tau_vl=4.729 ms, tau_vr=4.875 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000004
- errors: delta_tau_abs_err=0.369 ms, theta_v_abs_err=5.193 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=3, tau_vl=4.562 ms, tau_vr=4.500 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000010
- errors: delta_tau_abs_err=0.233 ms, theta_v_abs_err=3.271 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000013
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.958 ms, tau_vr=6.271 ms, delta_tau=0.312 ms, theta_v=4.391 deg, score=0.000062
- errors: delta_tau_abs_err=0.754 ms, theta_v_abs_err=10.604 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.083 ms, tau_vr=4.875 ms, delta_tau=-0.208 ms, theta_v=-2.926 deg, score=0.000005
- errors: delta_tau_abs_err=0.723 ms, theta_v_abs_err=10.166 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.458 ms, tau_vr=4.500 ms, delta_tau=-0.958 ms, theta_v=-13.579 deg, score=0.000026
- errors: delta_tau_abs_err=0.663 ms, theta_v_abs_err=9.431 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.688 ms, tau_vr=4.750 ms, delta_tau=0.062 ms, theta_v=0.877 deg, score=0.000007
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.090 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.958 ms, tau_vr=6.062 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.000035
- errors: delta_tau_abs_err=0.546 ms, theta_v_abs_err=7.675 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=8, vr_rank=1, tau_vl=4.896 ms, tau_vr=4.875 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000003
- errors: delta_tau_abs_err=0.535 ms, theta_v_abs_err=7.533 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.458 ms, tau_vr=5.438 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000019
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000009
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.958 ms, tau_vr=6.062 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.000043
- errors: delta_tau_abs_err=0.546 ms, theta_v_abs_err=7.675 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.729 ms, tau_vr=4.875 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000003
- errors: delta_tau_abs_err=0.369 ms, theta_v_abs_err=5.193 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.458 ms, tau_vr=5.438 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000020
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000011
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.958 ms, tau_vr=6.062 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.000047
- errors: delta_tau_abs_err=0.546 ms, theta_v_abs_err=7.675 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.083 ms, tau_vr=4.875 ms, delta_tau=-0.208 ms, theta_v=-2.926 deg, score=0.000004
- errors: delta_tau_abs_err=0.723 ms, theta_v_abs_err=10.166 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.458 ms, tau_vr=5.438 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000020
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.688 ms, tau_vr=4.750 ms, delta_tau=0.062 ms, theta_v=0.877 deg, score=0.000009
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.090 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.958 ms, tau_vr=6.062 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.000044
- errors: delta_tau_abs_err=0.546 ms, theta_v_abs_err=7.675 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=5.083 ms, tau_vr=5.021 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000003
- errors: delta_tau_abs_err=0.577 ms, theta_v_abs_err=8.118 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.458 ms, tau_vr=5.438 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000020
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000011
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.958 ms, tau_vr=6.062 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.000050
- errors: delta_tau_abs_err=0.546 ms, theta_v_abs_err=7.675 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.083 ms, tau_vr=4.875 ms, delta_tau=-0.208 ms, theta_v=-2.926 deg, score=0.000004
- errors: delta_tau_abs_err=0.723 ms, theta_v_abs_err=10.166 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.458 ms, tau_vr=5.438 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000020
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=4, tau_vl=4.688 ms, tau_vr=4.750 ms, delta_tau=0.062 ms, theta_v=0.877 deg, score=0.005461
- errors: delta_tau_abs_err=0.504 ms, theta_v_abs_err=7.090 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.958 ms, tau_vr=6.062 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.012008
- errors: delta_tau_abs_err=0.546 ms, theta_v_abs_err=7.675 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=7, tau_vl=5.396 ms, tau_vr=5.396 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.003438
- errors: delta_tau_abs_err=0.514 ms, theta_v_abs_err=7.240 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.458 ms, tau_vr=5.438 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.009492
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000009
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.958 ms, tau_vr=6.062 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.000043
- errors: delta_tau_abs_err=0.546 ms, theta_v_abs_err=7.675 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.729 ms, tau_vr=4.875 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000003
- errors: delta_tau_abs_err=0.369 ms, theta_v_abs_err=5.193 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.458 ms, tau_vr=5.438 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000020
- errors: delta_tau_abs_err=0.274 ms, theta_v_abs_err=3.856 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000008
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.958 ms, tau_vr=6.062 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.000028
- errors: delta_tau_abs_err=0.546 ms, theta_v_abs_err=7.675 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.729 ms, tau_vr=4.875 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000003
- errors: delta_tau_abs_err=0.369 ms, theta_v_abs_err=5.193 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=4.562 ms, tau_vr=4.500 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000013
- errors: delta_tau_abs_err=0.233 ms, theta_v_abs_err=3.271 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000009
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.750 ms, tau_vr=4.771 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000022
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.729 ms, tau_vr=4.875 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000004
- errors: delta_tau_abs_err=0.369 ms, theta_v_abs_err=5.193 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=4.562 ms, tau_vr=4.500 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000013
- errors: delta_tau_abs_err=0.233 ms, theta_v_abs_err=3.271 deg

### block4_p08_17 / ldv_preemph_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.688 ms, tau_vr=4.500 ms, delta_tau=-0.188 ms, theta_v=-2.633 deg, score=0.000008
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.580 deg

### block5_p00_18 / ldv_preemph_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=4.750 ms, tau_vr=4.771 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000010
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block6_n04_19 / ldv_preemph_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.729 ms, tau_vr=4.875 ms, delta_tau=0.146 ms, theta_v=2.048 deg, score=0.000004
- errors: delta_tau_abs_err=0.369 ms, theta_v_abs_err=5.193 deg

### block7_n08_20 / ldv_preemph_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=2, vr_rank=2, tau_vl=4.562 ms, tau_vr=4.500 ms, delta_tau=-0.062 ms, theta_v=-0.877 deg, score=0.000010
- errors: delta_tau_abs_err=0.233 ms, theta_v_abs_err=3.271 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.583 ms, tau_vr=3.542 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000061
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.417 ms, tau_vr=4.250 ms, delta_tau=-0.167 ms, theta_v=-2.340 deg, score=0.000096
- errors: delta_tau_abs_err=0.275 ms, theta_v_abs_err=3.872 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.833 ms, tau_vr=5.417 ms, delta_tau=-0.417 ms, theta_v=-5.859 deg, score=0.000051
- errors: delta_tau_abs_err=0.931 ms, theta_v_abs_err=13.100 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.667 ms, tau_vr=3.854 ms, delta_tau=0.188 ms, theta_v=2.633 deg, score=0.000075
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.781 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.583 ms, tau_vr=3.542 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000051
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=4, tau_vl=5.979 ms, tau_vr=5.958 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000051
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=5, tau_vl=5.833 ms, tau_vr=5.938 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.000017
- errors: delta_tau_abs_err=0.410 ms, theta_v_abs_err=5.778 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=3.667 ms, tau_vr=3.583 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000037
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.583 ms, tau_vr=3.542 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000056
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.417 ms, tau_vr=4.250 ms, delta_tau=-0.167 ms, theta_v=-2.340 deg, score=0.000069
- errors: delta_tau_abs_err=0.275 ms, theta_v_abs_err=3.872 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.833 ms, tau_vr=5.604 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000026
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.667 ms, tau_vr=3.854 ms, delta_tau=0.188 ms, theta_v=2.633 deg, score=0.000051
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.781 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.583 ms, tau_vr=3.542 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000058
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.417 ms, tau_vr=4.250 ms, delta_tau=-0.167 ms, theta_v=-2.340 deg, score=0.000081
- errors: delta_tau_abs_err=0.275 ms, theta_v_abs_err=3.872 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.833 ms, tau_vr=5.417 ms, delta_tau=-0.417 ms, theta_v=-5.859 deg, score=0.000034
- errors: delta_tau_abs_err=0.931 ms, theta_v_abs_err=13.100 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=3.500, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.667 ms, tau_vr=3.854 ms, delta_tau=0.188 ms, theta_v=2.633 deg, score=0.000062
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.781 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.583 ms, tau_vr=3.542 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000059
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.417 ms, tau_vr=4.250 ms, delta_tau=-0.167 ms, theta_v=-2.340 deg, score=0.000061
- errors: delta_tau_abs_err=0.275 ms, theta_v_abs_err=3.872 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=5, tau_vl=5.833 ms, tau_vr=5.938 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.000022
- errors: delta_tau_abs_err=0.410 ms, theta_v_abs_err=5.778 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=3.667 ms, tau_vr=3.583 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000046
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.583 ms, tau_vr=3.542 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000060
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.417 ms, tau_vr=4.250 ms, delta_tau=-0.167 ms, theta_v=-2.340 deg, score=0.000086
- errors: delta_tau_abs_err=0.275 ms, theta_v_abs_err=3.872 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.833 ms, tau_vr=5.604 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000033
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.667 ms, tau_vr=3.854 ms, delta_tau=0.188 ms, theta_v=2.633 deg, score=0.000065
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.781 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.583 ms, tau_vr=3.542 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.014581
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=4, tau_vl=5.979 ms, tau_vr=5.958 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.014309
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=5, tau_vl=5.833 ms, tau_vr=5.938 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.008378
- errors: delta_tau_abs_err=0.410 ms, theta_v_abs_err=5.778 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_sum_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=3.667 ms, tau_vr=3.583 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.012496
- errors: delta_tau_abs_err=0.212 ms, theta_v_abs_err=2.978 deg

### block4_p08_17 / ldv_diff_bp500_2000 / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.583 ms, tau_vr=3.542 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000056
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block5_p00_18 / ldv_diff_bp500_2000 / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.417 ms, tau_vr=4.250 ms, delta_tau=-0.167 ms, theta_v=-2.340 deg, score=0.000069
- errors: delta_tau_abs_err=0.275 ms, theta_v_abs_err=3.872 deg

### block6_n04_19 / ldv_diff_bp500_2000 / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.833 ms, tau_vr=5.604 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000026
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / ldv_diff_bp500_2000 / prom_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.667 ms, tau_vr=3.854 ms, delta_tau=0.188 ms, theta_v=2.633 deg, score=0.000051
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.781 deg

### block4_p08_17 / ldv_diff_bp500_2000 / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.583 ms, tau_vr=3.542 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000039
- errors: delta_tau_abs_err=0.400 ms, theta_v_abs_err=5.628 deg

### block5_p00_18 / ldv_diff_bp500_2000 / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.417 ms, tau_vr=4.250 ms, delta_tau=-0.167 ms, theta_v=-2.340 deg, score=0.000068
- errors: delta_tau_abs_err=0.275 ms, theta_v_abs_err=3.872 deg

### block6_n04_19 / ldv_diff_bp500_2000 / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.833 ms, tau_vr=5.604 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000023
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / ldv_diff_bp500_2000 / balanced_product_delta

- lag_min_ms=3.500, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=3.667 ms, tau_vr=3.854 ms, delta_tau=0.188 ms, theta_v=2.633 deg, score=0.000048
- errors: delta_tau_abs_err=0.483 ms, theta_v_abs_err=6.781 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=7, vr_rank=5, tau_vl=4.812 ms, tau_vr=4.729 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000017
- errors: delta_tau_abs_err=0.358 ms, theta_v_abs_err=5.043 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.417 ms, tau_vr=4.688 ms, delta_tau=0.271 ms, theta_v=3.805 deg, score=0.000044
- errors: delta_tau_abs_err=0.713 ms, theta_v_abs_err=10.017 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=3, tau_vl=4.438 ms, tau_vr=4.854 ms, delta_tau=0.417 ms, theta_v=5.859 deg, score=0.000015
- errors: delta_tau_abs_err=0.098 ms, theta_v_abs_err=1.381 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=6, vr_rank=5, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000010
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=7, vr_rank=5, tau_vl=4.812 ms, tau_vr=4.729 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000014
- errors: delta_tau_abs_err=0.358 ms, theta_v_abs_err=5.043 deg

### block5_p00_18 / ldv_diff_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.417 ms, tau_vr=4.688 ms, delta_tau=0.271 ms, theta_v=3.805 deg, score=0.000034
- errors: delta_tau_abs_err=0.713 ms, theta_v_abs_err=10.017 deg

### block6_n04_19 / ldv_diff_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=3, tau_vl=4.438 ms, tau_vr=4.854 ms, delta_tau=0.417 ms, theta_v=5.859 deg, score=0.000014
- errors: delta_tau_abs_err=0.098 ms, theta_v_abs_err=1.381 deg

### block7_n08_20 / ldv_diff_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=3.500, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=6, vr_rank=5, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000007
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=5.542 ms, tau_vr=6.438 ms, delta_tau=0.896 ms, theta_v=12.678 deg, score=0.000038
- errors: delta_tau_abs_err=1.338 ms, theta_v_abs_err=18.891 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.417 ms, tau_vr=4.250 ms, delta_tau=-0.167 ms, theta_v=-2.340 deg, score=0.000096
- errors: delta_tau_abs_err=0.275 ms, theta_v_abs_err=3.872 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.833 ms, tau_vr=5.417 ms, delta_tau=-0.417 ms, theta_v=-5.859 deg, score=0.000051
- errors: delta_tau_abs_err=0.931 ms, theta_v_abs_err=13.100 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.438 ms, tau_vr=4.521 ms, delta_tau=-0.917 ms, theta_v=-12.978 deg, score=0.000031
- errors: delta_tau_abs_err=0.621 ms, theta_v_abs_err=8.830 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.167 ms, tau_vr=5.146 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000024
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=4, tau_vl=5.979 ms, tau_vr=5.958 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000051
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=5, tau_vl=5.833 ms, tau_vr=5.938 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.000017
- errors: delta_tau_abs_err=0.410 ms, theta_v_abs_err=5.778 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=3, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000012
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.167 ms, tau_vr=5.146 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000025
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.417 ms, tau_vr=4.250 ms, delta_tau=-0.167 ms, theta_v=-2.340 deg, score=0.000069
- errors: delta_tau_abs_err=0.275 ms, theta_v_abs_err=3.872 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.833 ms, tau_vr=5.604 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000026
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=3, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000013
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=1, tau_vl=6.250 ms, tau_vr=6.438 ms, delta_tau=0.188 ms, theta_v=2.633 deg, score=0.000028
- errors: delta_tau_abs_err=0.629 ms, theta_v_abs_err=8.845 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.417 ms, tau_vr=4.250 ms, delta_tau=-0.167 ms, theta_v=-2.340 deg, score=0.000081
- errors: delta_tau_abs_err=0.275 ms, theta_v_abs_err=3.872 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.833 ms, tau_vr=5.417 ms, delta_tau=-0.417 ms, theta_v=-5.859 deg, score=0.000034
- errors: delta_tau_abs_err=0.931 ms, theta_v_abs_err=13.100 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.000, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=3, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000014
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.167 ms, tau_vr=5.146 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000026
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.417 ms, tau_vr=4.250 ms, delta_tau=-0.167 ms, theta_v=-2.340 deg, score=0.000061
- errors: delta_tau_abs_err=0.275 ms, theta_v_abs_err=3.872 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=5, tau_vl=5.833 ms, tau_vr=5.938 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.000022
- errors: delta_tau_abs_err=0.410 ms, theta_v_abs_err=5.778 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=3, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000014
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=1, tau_vl=6.250 ms, tau_vr=6.438 ms, delta_tau=0.188 ms, theta_v=2.633 deg, score=0.000029
- errors: delta_tau_abs_err=0.629 ms, theta_v_abs_err=8.845 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.417 ms, tau_vr=4.250 ms, delta_tau=-0.167 ms, theta_v=-2.340 deg, score=0.000086
- errors: delta_tau_abs_err=0.275 ms, theta_v_abs_err=3.872 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.833 ms, tau_vr=5.604 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000033
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=3, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000014
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.167 ms, tau_vr=5.146 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.009801
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=4, tau_vl=5.979 ms, tau_vr=5.958 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.014309
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=5, tau_vl=5.833 ms, tau_vr=5.938 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.008378
- errors: delta_tau_abs_err=0.410 ms, theta_v_abs_err=5.778 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_sum_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=5, vr_rank=2, tau_vl=4.271 ms, tau_vr=4.271 ms, delta_tau=0.000 ms, theta_v=0.000 deg, score=0.007394
- errors: delta_tau_abs_err=0.295 ms, theta_v_abs_err=4.148 deg

### block4_p08_17 / ldv_diff_bp500_2000 / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.167 ms, tau_vr=5.146 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000025
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / ldv_diff_bp500_2000 / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.417 ms, tau_vr=4.250 ms, delta_tau=-0.167 ms, theta_v=-2.340 deg, score=0.000069
- errors: delta_tau_abs_err=0.275 ms, theta_v_abs_err=3.872 deg

### block6_n04_19 / ldv_diff_bp500_2000 / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.833 ms, tau_vr=5.604 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000026
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / ldv_diff_bp500_2000 / prom_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=3, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000013
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.167 ms, tau_vr=5.146 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000023
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / ldv_diff_bp500_2000 / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=4.417 ms, tau_vr=4.250 ms, delta_tau=-0.167 ms, theta_v=-2.340 deg, score=0.000068
- errors: delta_tau_abs_err=0.275 ms, theta_v_abs_err=3.872 deg

### block6_n04_19 / ldv_diff_bp500_2000 / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.833 ms, tau_vr=5.604 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000023
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / ldv_diff_bp500_2000 / balanced_product_delta

- lag_min_ms=4.000, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=3, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000008
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=4, vr_rank=4, tau_vl=4.812 ms, tau_vr=4.729 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000017
- errors: delta_tau_abs_err=0.358 ms, theta_v_abs_err=5.043 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.417 ms, tau_vr=4.688 ms, delta_tau=0.271 ms, theta_v=3.805 deg, score=0.000044
- errors: delta_tau_abs_err=0.713 ms, theta_v_abs_err=10.017 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=3, tau_vl=4.438 ms, tau_vr=4.854 ms, delta_tau=0.417 ms, theta_v=5.859 deg, score=0.000015
- errors: delta_tau_abs_err=0.098 ms, theta_v_abs_err=1.381 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=3, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000010
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=4, vr_rank=4, tau_vl=4.812 ms, tau_vr=4.729 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000014
- errors: delta_tau_abs_err=0.358 ms, theta_v_abs_err=5.043 deg

### block5_p00_18 / ldv_diff_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=4.417 ms, tau_vr=4.688 ms, delta_tau=0.271 ms, theta_v=3.805 deg, score=0.000034
- errors: delta_tau_abs_err=0.713 ms, theta_v_abs_err=10.017 deg

### block6_n04_19 / ldv_diff_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=3, tau_vl=4.438 ms, tau_vr=4.854 ms, delta_tau=0.417 ms, theta_v=5.859 deg, score=0.000014
- errors: delta_tau_abs_err=0.098 ms, theta_v_abs_err=1.381 deg

### block7_n08_20 / ldv_diff_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.000, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=3, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000007
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=2, vr_rank=1, tau_vl=5.542 ms, tau_vr=6.438 ms, delta_tau=0.896 ms, theta_v=12.678 deg, score=0.000038
- errors: delta_tau_abs_err=1.338 ms, theta_v_abs_err=18.891 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.979 ms, tau_vr=5.062 ms, delta_tau=-0.917 ms, theta_v=-12.978 deg, score=0.000055
- errors: delta_tau_abs_err=0.475 ms, theta_v_abs_err=6.766 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.833 ms, tau_vr=5.417 ms, delta_tau=-0.417 ms, theta_v=-5.859 deg, score=0.000051
- errors: delta_tau_abs_err=0.931 ms, theta_v_abs_err=13.100 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.438 ms, tau_vr=4.521 ms, delta_tau=-0.917 ms, theta_v=-12.978 deg, score=0.000031
- errors: delta_tau_abs_err=0.621 ms, theta_v_abs_err=8.830 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.167 ms, tau_vr=5.146 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000024
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.979 ms, tau_vr=5.958 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000051
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=5, tau_vl=5.833 ms, tau_vr=5.938 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.000017
- errors: delta_tau_abs_err=0.410 ms, theta_v_abs_err=5.778 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000012
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.167 ms, tau_vr=5.146 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000025
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.979 ms, tau_vr=5.958 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000053
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.833 ms, tau_vr=5.604 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000026
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000013
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=1, tau_vl=6.250 ms, tau_vr=6.438 ms, delta_tau=0.188 ms, theta_v=2.633 deg, score=0.000028
- errors: delta_tau_abs_err=0.629 ms, theta_v_abs_err=8.845 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.979 ms, tau_vr=5.958 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000054
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=1, tau_vl=5.833 ms, tau_vr=5.417 ms, delta_tau=-0.417 ms, theta_v=-5.859 deg, score=0.000034
- errors: delta_tau_abs_err=0.931 ms, theta_v_abs_err=13.100 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_delta

- lag_min_ms=4.400, delta_scale_ms=1.000, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000014
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.167 ms, tau_vr=5.146 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000026
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.979 ms, tau_vr=5.958 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000055
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=5, tau_vl=5.833 ms, tau_vr=5.938 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.000022
- errors: delta_tau_abs_err=0.410 ms, theta_v_abs_err=5.778 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.250, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000014
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=1, tau_vl=6.250 ms, tau_vr=6.438 ms, delta_tau=0.188 ms, theta_v=2.633 deg, score=0.000029
- errors: delta_tau_abs_err=0.629 ms, theta_v_abs_err=8.845 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.979 ms, tau_vr=5.958 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000055
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.833 ms, tau_vr=5.604 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000033
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000014
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.167 ms, tau_vr=5.146 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.009801
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.979 ms, tau_vr=5.958 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.014309
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=5, tau_vl=5.833 ms, tau_vr=5.938 ms, delta_tau=0.104 ms, theta_v=1.462 deg, score=0.008378
- errors: delta_tau_abs_err=0.410 ms, theta_v_abs_err=5.778 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_sum_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=1, vr_rank=7, tau_vl=5.438 ms, tau_vr=5.458 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.007444
- errors: delta_tau_abs_err=0.316 ms, theta_v_abs_err=4.441 deg

### block4_p08_17 / ldv_diff_bp500_2000 / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.167 ms, tau_vr=5.146 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000025
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / ldv_diff_bp500_2000 / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.979 ms, tau_vr=5.958 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000053
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_diff_bp500_2000 / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.833 ms, tau_vr=5.604 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000026
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / ldv_diff_bp500_2000 / prom_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000013
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.167 ms, tau_vr=5.146 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000023
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block5_p00_18 / ldv_diff_bp500_2000 / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=1, vr_rank=3, tau_vl=5.979 ms, tau_vr=5.958 ms, delta_tau=-0.021 ms, theta_v=-0.292 deg, score=0.000043
- errors: delta_tau_abs_err=0.421 ms, theta_v_abs_err=5.920 deg

### block6_n04_19 / ldv_diff_bp500_2000 / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=1, vr_rank=2, tau_vl=5.833 ms, tau_vr=5.604 ms, delta_tau=-0.229 ms, theta_v=-3.219 deg, score=0.000023
- errors: delta_tau_abs_err=0.744 ms, theta_v_abs_err=10.459 deg

### block7_n08_20 / ldv_diff_bp500_2000 / balanced_product_delta

- lag_min_ms=4.400, delta_scale_ms=0.500, mean_tau_center_ms=4.800, mean_tau_scale_ms=10.000
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000008
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=4, vr_rank=4, tau_vl=4.812 ms, tau_vr=4.729 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000017
- errors: delta_tau_abs_err=0.358 ms, theta_v_abs_err=5.043 deg

### block5_p00_18 / ldv_diff_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=1, tau_vl=5.042 ms, tau_vr=4.688 ms, delta_tau=-0.354 ms, theta_v=-4.978 deg, score=0.000026
- errors: delta_tau_abs_err=0.088 ms, theta_v_abs_err=1.235 deg

### block6_n04_19 / ldv_diff_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=3, tau_vl=4.438 ms, tau_vr=4.854 ms, delta_tau=0.417 ms, theta_v=5.859 deg, score=0.000015
- errors: delta_tau_abs_err=0.098 ms, theta_v_abs_err=1.381 deg

### block7_n08_20 / ldv_diff_bp500_2000 / amp_product_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000010
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg

### block4_p08_17 / ldv_diff_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=4, vr_rank=4, tau_vl=4.812 ms, tau_vr=4.729 ms, delta_tau=-0.083 ms, theta_v=-1.170 deg, score=0.000014
- errors: delta_tau_abs_err=0.358 ms, theta_v_abs_err=5.043 deg

### block5_p00_18 / ldv_diff_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=5.070 ms, tau_vr=4.629 ms, delta_tau=-0.442 ms, theta_v=-6.213 deg
- selected: vl_rank=3, vr_rank=2, tau_vl=5.042 ms, tau_vr=5.062 ms, delta_tau=0.021 ms, theta_v=0.292 deg, score=0.000019
- errors: delta_tau_abs_err=0.463 ms, theta_v_abs_err=6.505 deg

### block6_n04_19 / ldv_diff_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.601 ms, tau_vr=5.115 ms, delta_tau=0.514 ms, theta_v=7.240 deg
- selected: vl_rank=2, vr_rank=3, tau_vl=4.438 ms, tau_vr=4.854 ms, delta_tau=0.417 ms, theta_v=5.859 deg, score=0.000014
- errors: delta_tau_abs_err=0.098 ms, theta_v_abs_err=1.381 deg

### block7_n08_20 / ldv_diff_bp500_2000 / balanced_mean_tau_delta_quad

- lag_min_ms=4.400, delta_scale_ms=0.600, mean_tau_center_ms=4.800, mean_tau_scale_ms=0.450
- reference: tau_vl=4.984 ms, tau_vr=4.689 ms, delta_tau=-0.295 ms, theta_v=-4.148 deg
- selected: vl_rank=4, vr_rank=1, tau_vl=4.562 ms, tau_vr=4.521 ms, delta_tau=-0.042 ms, theta_v=-0.585 deg, score=0.000007
- errors: delta_tau_abs_err=0.254 ms, theta_v_abs_err=3.563 deg
