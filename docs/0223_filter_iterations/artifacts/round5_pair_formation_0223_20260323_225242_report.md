# Round 5 Pair Formation Sweep

- Generated: 2026-03-23T22:52:48
- Data root: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223`
- Fixed front-end: `diff_len80_ldvonly_bp700_1800`
- Fixed scorer: `len80_current_score`

## Summary

| variant | central_dt_mae_ms | hard_dt_mae_ms | hard_win_rate | central_max_dt_ms | block6_pair_rank | block7_pair_rank | block6_dt | block7_dt | block6_hit5 | block7_hit5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base_control | 0.142 | 0.249 | 0.000 | 0.254 | 7 | 4 | 0.271 | -0.042 | 2 | 2 |
| cond_prune_soft | 0.257 | 0.478 | 0.000 | 0.712 | 5 | 3 | 0.271 | 0.417 | 4 | 4 |
| cond_prune_medium | 0.402 | 0.769 | 0.000 | 0.827 | 3 | 3 | -0.312 | 0.417 | 5 | 4 |
| cond_prune_medium_mutual2 | 0.402 | 0.769 | 0.000 | 0.827 | 3 | 3 | -0.312 | 0.417 | 5 | 4 |
| cond_prune_medium_mutual3 | 0.402 | 0.769 | 0.000 | 0.827 | 3 | 3 | -0.312 | 0.417 | 5 | 4 |
| cond_prune_hard_mutual2 | 0.428 | 0.822 | 0.000 | 0.931 | 2 | NA | -0.417 | 0.417 | 5 | 3 |