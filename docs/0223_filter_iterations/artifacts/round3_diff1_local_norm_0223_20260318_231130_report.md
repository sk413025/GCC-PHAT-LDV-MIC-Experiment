# Round 3 Diff1 Local Normalization Sweep

- Generated: 2026-03-18T23:12:50
- Data root: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223`
- Window sec: `5.0`
- Window offsets sec: `[-1.0, -0.5, 0.0, 0.5, 1.0]`

## Summary

| variant | central_valid | physical | central_dt_mae_ms | central_theta_mae_deg | central_max_dt_ms | oracle_rank_mean | oracle_margin_mean | oracle_psr_mean | window_dt_mae_ms | window_std_mean_ms | window_std_max_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| diff_len80_ldvonly_bp700_1800 | 4 | 4 | 0.142 | 2.000 | 0.254 | 8.250 | 0.516 | -39.053 | 0.207 | 0.110 | 0.337 |
| diff_len40_ldvonly_sqrtgain_bp700_1800 | 4 | 4 | 0.142 | 2.000 | 0.254 | 11.250 | 0.469 | -39.251 | 0.257 | 0.136 | 0.360 |
| diff_len40_ldvonly_clip0p5_2p0_bp700_1800 | 4 | 4 | 0.147 | 2.073 | 0.254 | 8.500 | 0.478 | -38.150 | 0.212 | 0.109 | 0.334 |
| diff_len40_ldvonly_bp700_1800 | 4 | 4 | 0.147 | 2.073 | 0.254 | 8.500 | 0.493 | -38.142 | 0.211 | 0.111 | 0.334 |
| diff_len20_ldvonly_bp700_1800 | 4 | 4 | 0.147 | 2.073 | 0.254 | 14.000 | 0.216 | -38.610 | 0.225 | 0.131 | 0.345 |
| baseline_diff_bp700_1800 | 4 | 4 | 0.152 | 2.147 | 0.264 | 11.000 | 0.236 | -40.498 | 0.262 | 0.136 | 0.360 |
| diff_len40_both_bp700_1800 | 4 | 4 | 0.177 | 2.490 | 0.275 | 9.750 | 0.311 | -40.313 | 0.325 | 0.185 | 0.294 |
| diff_len40_ldvonly_robustmedian_bp700_1800 | 4 | 4 | 0.345 | 4.856 | 0.838 | 6.750 | 0.680 | -41.822 | 0.225 | 0.112 | 0.313 |