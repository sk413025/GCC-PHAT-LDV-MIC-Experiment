# 2026-03-25 Round 8 Supervisor Memo

## Scope

Round 8 tested three narrow `block6`-only persistence lanes on top of the
frozen round6 lane3 recovery branch:

- base reference: `sign_veto_poslift_r0p10`
- lane 8.1: same-`VL` temporal persistence
- lane 8.2: positive-family target hold
- lane 8.3: same-anchor temporal handoff

Tracked result bundles:

- `results/round8_block6_samevl_temporal_persistence_0223_supervisor_20260325_0029/`
- `results/round8_block6_positive_family_target_hold_0223_20260325_001724/`
- `results/round8_block6_sameanchor_temporal_handoff_0223_20260325_1/`

## Lane 8.1 Verdict

`NO IMPROVEMENT`

The same-`VL` temporal persistence lane collapses to the frozen control.

Measured:

- central unchanged: `0.055 / 0.074`
- `pass_rate = 0.400`
- `block6 hit@1 = 3/5`
- `block6 hit@3 = 3/5`

Interpretation:

- a fixed anchor around the central rescue family is too rigid
- the failing `block6` windows do not recover from same-`VL` hold alone

## Lane 8.2 Verdict

`BEST ROUND8 RESULT`

The best lane 8.2 variant is:

- `b6_family_hold_span2_support2_floor0p08`

Measured:

- central unchanged: `0.055 / 0.074`
- `pass_rate = 0.467`
- window mean/std improves `0.106 -> 0.092`
- `block6 hit@1 = 5/5`
- `block6 hit@3 = 5/5`
- `block7 hit@1 = 1/5`

Interpretation:

- the missing `block6` mechanism is not extra sign control
- it is a positive-family hold that tracks the local supported family across
  nearby windows
- this fully repairs the `5.0 s` hard-window persistence for `block6`
  without regressing the deployed central solution

## Lane 8.3 Verdict

`NO IMPROVEMENT`

The same-anchor temporal handoff lane also collapses to control.

Measured:

- central unchanged: `0.055 / 0.074`
- `pass_rate = 0.400`
- `block6 hit@1 = 3/5`
- `block6 hit@3 = 3/5`

Interpretation:

- the old round5 same-anchor handoff rule remains useful at the central window
- but adding temporal support to that takeover rule does not extend persistence
  in the failing `block6` offsets

## Supervisor Conclusion

Round 8 has a single real winner:

- `b6_family_hold_span2_support2_floor0p08`

What round8 proves:

- `block6` persistence can be repaired on the round6 branch
- the correct axis is local positive-family retention
- same-`VL` freeze and same-anchor temporal handoff are both too weak on their
  own

What round8 does not solve:

- `pass_rate` is still only `0.467`
- `block7 hit@1` remains `1/5`
- the broader `4.0 s` and `6.0 s` failure pockets still exist

## Next Step

The next justified move is narrow again:

- keep `b6_family_hold_span2_support2_floor0p08` as the new robustness branch
- do not reopen the failed lane 8.1 or lane 8.3 axes
- run a follow-up round that keeps the lane 8.2 `block6` repair fixed while
  targeting the remaining `block7` persistence gap or the low-pass-rate
  `4.0 s` / `6.0 s` settings
