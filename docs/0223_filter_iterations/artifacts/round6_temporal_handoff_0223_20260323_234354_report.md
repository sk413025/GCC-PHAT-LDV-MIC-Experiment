# Round 6 Temporal Handoff Stabilization

- Generated: 2026-03-23T23:48:21
- Data root: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223`
- Frozen baseline: `cond_prune_soft + same_vl_replace_w6_p3 + handoff_ratio_0p90`

## Summary

| variant | central_dt_mae_ms | hard_dt_mae_ms | hard_win_rate | window_mean_std | window_max_std | pass_rate | block6_hit@1 | block7_hit@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control_frozen | 0.055 | 0.074 | 1.000 | 0.232 | 0.404 | 0.200 | 2 | 1 |
| subwin70_vote2_ratio0p10_lift0p15 | 0.055 | 0.074 | 1.000 | 0.146 | 0.335 | 0.333 | 3 | 1 |
| subwin70_vote2_ratio0p10_lift0p20 | 0.055 | 0.074 | 1.000 | 0.146 | 0.335 | 0.333 | 3 | 1 |
| subwin60_vote2_ratio0p10_lift0p15 | 0.055 | 0.074 | 1.000 | 0.066 | 0.157 | 0.333 | 2 | 1 |
| subwin70_vote2_ratio0p15_lift0p15 | 0.055 | 0.074 | 1.000 | 0.216 | 0.335 | 0.267 | 2 | 1 |

## Winner

- variant: `subwin60_vote2_ratio0p10_lift0p15`
- central_delta_tau_mae_ms: `0.055`
- hard_case_delta_tau_mae_ms: `0.074`
- window_stability_mean_std_ms: `0.066`
- window_stability_max_std_ms: `0.157`
- setting_pass_rate: `0.333`