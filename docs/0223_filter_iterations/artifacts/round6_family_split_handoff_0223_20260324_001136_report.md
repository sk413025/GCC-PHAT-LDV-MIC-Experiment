# Round 6 Family-Split Handoff

- Generated: 2026-03-24T00:15:10
- Data root: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223`

## Summary

| variant | central_dt_mae_ms | hard_dt_mae_ms | hard_win_rate | window_mean_std | window_max_std | pass_rate | block6_hit@1 | block7_hit@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control_sign_veto_poslift_r0p10 | 0.055 | 0.074 | 1.000 | 0.106 | 0.200 | 0.400 | 3 | 1 |
| split_neg_target_t0p18_poslift_r0p10 | 0.055 | 0.074 | 1.000 | 0.069 | 0.100 | 0.400 | 3 | 0 |
| split_neg_target_t0p25_poslift_r0p10 | 0.055 | 0.074 | 1.000 | 0.069 | 0.100 | 0.400 | 3 | 0 |
| split_neg_target_t0p18_poslift_r0p15 | 0.055 | 0.074 | 1.000 | 0.078 | 0.135 | 0.333 | 2 | 0 |

## Winner

- variant: `split_neg_target_t0p18_poslift_r0p10`
- central_delta_tau_mae_ms: `0.05488870500594145`
- hard_case_delta_tau_mae_ms: `0.07432639873206037`
- window_stability_mean_std_ms: `0.06894544061866734`
- window_stability_max_std_ms: `0.10034662148993555`
- setting_pass_rate: `0.4`