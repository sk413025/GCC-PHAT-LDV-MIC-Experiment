# Round 7 Block7 Same-VR Negative-Family Persistence

- Generated: 2026-03-24T23:18:43
- Data root: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223`
- Frozen reference: `sign_veto_poslift_r0p10`

## Summary

| variant | central_dt_mae_ms | hard_dt_mae_ms | pass_rate | window_mean_std | block6_hit@1 | block7_hit@1 | block7_hit@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control_sign_veto_poslift_r0p10 | 0.055 | 0.074 | 0.400 | 0.106 | 3 | 1 | 2 |
| b7_samevr_neg_promote_vr0p15_dt0p15_r0p10 | 0.055 | 0.074 | 0.400 | 0.072 | 3 | 0 | 2 |
| b7_samevr_neg_promote_vr0p20_dt0p15_r0p10 | 0.039 | 0.043 | 0.400 | 0.074 | 3 | 1 | 2 |
| b7_samevr_neg_veto_vr0p15_dt0p15_r0p10 | 0.055 | 0.074 | 0.400 | 0.072 | 3 | 0 | 2 |

## Winner

- variant: `b7_samevr_neg_promote_vr0p20_dt0p15_r0p10`
- block7 hit@1: `1`
- block7 hit@3: `2`
- block6 hit@1: `3`
- pass_rate: `0.400`