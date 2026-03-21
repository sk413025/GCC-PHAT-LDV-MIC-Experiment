# Round 3 Blind Proxy Scoring Sweep

- Generated: 2026-03-21T11:34:06
- Data root: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223`
- Window sec: `5.0`
- Fixed front-end: `diff_len80_ldvonly_bp700_1800`

## Summary

| variant | central_valid | physical | central_dt_mae_ms | central_theta_mae_deg | central_max_dt_ms | correct_rank_mean | correct_margin_mean | window_dt_mae_ms | window_std_mean_ms | window_std_max_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| len80_window_support_rank_guard | 4 | 4 | 0.142 | 2.000 | 0.254 | 1.000 | 2.322 | 0.193 | 0.083 | 0.232 |
| len80_current_score | 4 | 4 | 0.142 | 2.000 | 0.254 | 1.000 | 1.522 | 0.207 | 0.110 | 0.337 |
| len80_balance_guard | 4 | 4 | 0.142 | 2.000 | 0.254 | 1.000 | 1.733 | 0.207 | 0.110 | 0.337 |
| len80_psr_like_boost | 4 | 4 | 0.142 | 2.000 | 0.254 | 1.000 | 1.628 | 0.207 | 0.110 | 0.337 |
| len80_window_support | 4 | 4 | 0.142 | 2.000 | 0.254 | 1.000 | 1.873 | 0.207 | 0.110 | 0.337 |
| len80_window_support_balance | 4 | 4 | 0.142 | 2.000 | 0.254 | 1.000 | 2.138 | 0.207 | 0.110 | 0.337 |
| len80_window_support_psr | 4 | 4 | 0.142 | 2.000 | 0.254 | 1.000 | 1.960 | 0.207 | 0.110 | 0.337 |
| len80_hybrid_proxy | 4 | 4 | 0.142 | 2.000 | 0.254 | 1.000 | 2.524 | 0.207 | 0.110 | 0.337 |
| len80_tight_mean_tau | 4 | 4 | 0.142 | 2.000 | 0.254 | 1.000 | 1.461 | 0.219 | 0.127 | 0.337 |