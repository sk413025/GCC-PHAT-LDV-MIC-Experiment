# 2026-03-24 Round 6 Supervisor Memo

## Scope

Round 6 evaluated whether the promoted round5 baseline is robust beyond the
exact central `5.0 s` slice, then ran three narrow recovery lanes.

Tracked artifact bundles:

- `artifacts/round6_baseline_robustness_0223_20260323_233741_report.md`
- `artifacts/round6_baseline_robustness_0223_20260323_233741_summary.json`
- `artifacts/round6_temporal_handoff_0223_20260323_234354_report.md`
- `artifacts/round6_temporal_handoff_0223_20260323_234354_summary.json`
- `artifacts/round6_sign_veto_handoff_0223_20260323_235603_report.md`
- `artifacts/round6_sign_veto_handoff_0223_20260323_235603_summary.json`
- `artifacts/round6_family_split_handoff_0223_20260324_001136_report.md`
- `artifacts/round6_family_split_handoff_0223_20260324_001136_summary.json`

## Lane 1 Verdict

`REJECT FOR ROBUSTNESS`

The promoted baseline remains a valid central-window winner, but not a
robustness-qualified reference stack.

Key evidence:

- central metrics preserved exactly: `0.055 / 0.074`
- window stability fails: mean/std `0.232`, max/std `0.404`
- hard-case persistence fails:
  - `block6 hit@1 = 2/5`
  - `block7 hit@1 = 1/5`
- perturbation suite fails hard:
  - `pass_rate = 0.117`
  - aggregate hard-case win `= 0.258`
  - simultaneous hard-failure settings `= 36`

## Lane 2 Verdict

`PARTIAL RECOVERY`

The temporal wrapper `subwin60_vote2_ratio0p10_lift0p15` stabilizes the
decision trajectory, especially for wrong-sign drift, but does not solve
strict hard-case persistence.

Measured:

- central unchanged: `0.055 / 0.074`
- window mean/std improves `0.232 -> 0.066`
- window max/std improves `0.404 -> 0.157`
- setting pass rate improves `0.117 -> 0.333`

But:

- `block6 hit@1` only rises to `2/5`
- `block7 hit@1` stays `1/5`

## Lane 3 Verdict

`BEST ROUND6 RECOVERY SO FAR`

The best lane3 variant is:

- `sign_veto_poslift_r0p10`

This keeps the central deployment exactly intact while improving the hard-case
recovery structure over lane2.

Measured:

- central unchanged: `0.055 / 0.074`
- window mean/std: `0.106`
- window max/std: `0.200`
- setting pass rate: `0.400`
- `block6 hit@1 = 3/5`
- `block7 hit@1 = 1/5`

Interpretation:

- the split between temporal sign control and positive-family lift is useful
- it materially helps the `block6` same-sign family failure
- but it does not yet solve the `block7` strict top1 persistence problem

## Lane 4 Verdict

`NO IMPROVEMENT OVER LANE 3`

The best lane4 variant:

- `split_neg_target_t0p18_poslift_r0p10`

keeps central metrics intact and drives window max/std down further to `0.100`,
but does not improve the decisive robustness target:

- setting pass rate remains `0.400`
- `block6 hit@1` stays `3/5`
- `block7 hit@1` drops to `0/5`

So lane4 is not the new winner.

## Supervisor Conclusion

Round 6 best-so-far remains:

- `sign_veto_poslift_r0p10`

This is not robustness-qualified for promotion, but it is the strongest
recovery lane discovered in Round 6.

Current status:

- keep round5 lane4 as the promoted central-window deployed baseline
- treat round6 lane3 as the best robustness-recovery branch
- do not promote any round6 variant yet

## Next Step

The next justified lane is narrow:

- preserve lane3
- focus only on the remaining `block7` family persistence gap
- avoid reopening preprocessing, candidate generation, or broad scorer search
