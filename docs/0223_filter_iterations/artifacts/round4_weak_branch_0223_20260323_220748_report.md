# Round 4 Weak-Branch Rescue Sweep

- Generated: 2026-03-23T22:08:23
- Data root: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223`
- Fixed scorer: `len80_current_score`

## Summary

| variant | central_dt_mae_ms | hard_dt_mae_ms | central_max_dt_ms | oracle_vl_hits | oracle_vr_hits | oracle_pair_recall | weak_vl_cases | weak_vr_cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base_only | 0.142 | 0.249 | 0.254 | 4 | 4 | 4 | 2 | 2 |
| weak_local_zscore_mic | 0.142 | 0.249 | 0.254 | 4 | 4 | 4 | 2 | 2 |
| weak_clip_rms_mic | 0.142 | 0.249 | 0.254 | 4 | 4 | 4 | 2 | 2 |
| weak_diff2_union | 0.142 | 0.249 | 0.254 | 4 | 4 | 4 | 2 | 2 |
| weak_residual_second_pass | 0.142 | 0.249 | 0.254 | 4 | 4 | 4 | 2 | 2 |
| weak_bundle | 0.142 | 0.249 | 0.254 | 4 | 4 | 4 | 2 | 2 |