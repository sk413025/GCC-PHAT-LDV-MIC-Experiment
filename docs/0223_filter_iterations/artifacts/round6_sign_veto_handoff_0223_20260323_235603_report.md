# Round 6 Sign-Veto And In-Family Handoff

- Generated: 2026-03-23T23:59:41
- Data root: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223`
- Temporal reference: `subwin60_vote2_ratio0p10_lift0p15`

## Summary

| variant | central_dt_mae_ms | hard_dt_mae_ms | hard_win_rate | window_mean_std | window_max_std | pass_rate | block6_hit@1 | block7_hit@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control_temporal_ref | 0.055 | 0.074 | 1.000 | 0.066 | 0.157 | 0.333 | 2 | 1 |
| sign_veto_only | 0.099 | 0.162 | 0.500 | 0.113 | 0.200 | 0.267 | 1 | 1 |
| sign_veto_poslift_r0p15 | 0.055 | 0.074 | 1.000 | 0.114 | 0.200 | 0.333 | 2 | 1 |
| sign_veto_poslift_r0p10 | 0.055 | 0.074 | 1.000 | 0.106 | 0.200 | 0.400 | 3 | 1 |

## Winner

- variant: `sign_veto_poslift_r0p10`
- central_delta_tau_mae_ms: `0.055`
- hard_case_delta_tau_mae_ms: `0.074`
- window_stability_mean_std_ms: `0.106`
- window_stability_max_std_ms: `0.200`
- setting_pass_rate: `0.400`