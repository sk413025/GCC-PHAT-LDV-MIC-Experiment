# 2026-03-25 Round 8 Validator Note

## Validation Scope

Validated result bundles:

- `results/round8_block6_samevl_temporal_persistence_0223_supervisor_20260325_0029/`
- `results/round8_block6_positive_family_target_hold_0223_20260325_001724/`
- `results/round8_block6_sameanchor_temporal_handoff_0223_20260325_1/`

## Lane 8.1 Judgment

`REJECT AS NEW WINNER`

Reason:

- all tracked metrics reduce to control
- `block6 hit@1` stays `3/5`
- `pass_rate` stays `0.400`

## Lane 8.2 Judgment

`ACCEPT AS BEST ROUND8 RECOVERY`

The best lane 8.2 variant is:

- `b6_family_hold_span2_support2_floor0p08`

Reason:

- central metrics remain intact: `0.055 / 0.074`
- `block6 hit@1` rises from `3/5` to `5/5`
- `block6 hit@3` rises from `3/5` to `5/5`
- `pass_rate` rises from `0.400` to `0.467`
- `block7 hit@1` does not regress below `1/5`

This is a real robustness recovery because it repairs the exact `block6`
persistence gap left open by round6 lane3.

## Lane 8.3 Judgment

`REJECT AS NEW WINNER`

Reason:

- same-anchor temporal handoff shows no measurable gain over control
- `block6 hit@1` stays `3/5`
- `pass_rate` stays `0.400`

## Validator Conclusion

Best validated round8 status is:

- freeze round6 lane3 no longer as the leading branch
- replace it with round8 lane2 winner
- carry forward `b6_family_hold_span2_support2_floor0p08` as the strongest
  current robustness branch

Remaining missing evidence:

- `pass_rate >= 0.50`
- stronger persistence outside the repaired `block6` `5.0 s` window family
- explicit recovery for the remaining `block7` persistence gap
