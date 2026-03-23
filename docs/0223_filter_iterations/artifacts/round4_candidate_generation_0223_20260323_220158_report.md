# Round 4 Candidate Generation Sweep

- Generated: 2026-03-23T22:02:31
- Data root: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223`
- Fixed scorer: `len80_current_score` on base curve

## Summary

| variant | central_dt_mae_ms | hard_dt_mae_ms | central_max_dt_ms | window_dt_mae_ms | oracle_vl_hits | oracle_vr_hits | oracle_pair_recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base_only | 0.142 | 0.249 | 0.254 | 0.207 | 4 | 4 | 4 |
| base_plus_diff2_500_2000 | 0.142 | 0.249 | 0.254 | 0.207 | 4 | 4 | 4 |
| base_plus_diff2_700_1800 | 0.272 | 0.249 | 0.567 | 0.233 | 4 | 4 | 4 |
| base_plus_mic_flatten_700_1800 | 0.314 | 0.249 | 0.733 | 0.202 | 4 | 4 | 4 |
| base_plus_diff2_700_1800_and_mic_flatten | 0.314 | 0.249 | 0.733 | 0.202 | 4 | 4 | 4 |
| base_plus_oracle_bundle | 0.314 | 0.249 | 0.733 | 0.202 | 4 | 4 | 4 |