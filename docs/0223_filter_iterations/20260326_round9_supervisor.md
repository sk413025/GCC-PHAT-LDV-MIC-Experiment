# 2026-03-26 Round 9 Supervisor Memo

## Scope

Round 9 starts from the round8 winner:

- frozen reference: `b6_family_hold_span2_support2_floor0p08`

This lane does not reopen `block6`.
It only tests a narrow `block7` negative-family persistence rule.

Tracked result bundles:

- `results/round9_block7_negative_family_target_hold_0223_20260326_0001/`
- `results/round9_block7_nearest_negative_family_0223_20260326_0002/`

## Lane 9.1 Verdict

`USEFUL IMPROVEMENT, NOT A FULL PASS`

The best lane 9.1 variant is:

- `b7_neg_hold_span1_support2_floor0p08`

Measured:

- central improves `0.055 / 0.074 -> 0.039 / 0.043`
- `block6 hit@1` stays `5/5`
- `block7 hit@1` improves `1/5 -> 2/5`
- `block7 hit@3` stays `2/5`
- `pass_rate` stays `0.467`
- window mean/std improves `0.092 -> 0.076`

Interpretation:

- the round8 `block6` repair survives intact
- a short-span negative-family hold can recover the correct block7 central
  family again
- but the remaining off-center block7 windows are still not stable enough to
  lift overall pass rate

## Rejected Variants

The broader span2 variants are not winners.

What they do:

- push `pass_rate` up to `0.533`

Why they fail:

- central regresses to `0.091 / 0.147`
- `block7 hit@1` collapses to `0/5`

So those variants are smoothing the average while destroying the decisive hard
case signal.

## Lane 9.2 Verdict

`NO IMPROVEMENT`

Round 9 lane 2 tested nearest-neighbor negative-family retention on top of the
lane 9.1 winner.

Measured:

- control remains the winner
- `block7 hit@1` stays `2/5`
- `block7 hit@3` stays `2/5`
- `pass_rate` stays `0.467`

Interpretation:

- nearest-neighbor anchoring alone is too greedy
- it does not repair the remaining off-center `block7` failures
- stronger magnitude bias without a better family identity just harms the lane

## Supervisor Conclusion

Current best robustness branch is now:

- round8 lane2 `block6` hold
- plus round9 lane1 `block7` negative-family target hold

What round9 proves:

- the block7 central correction still exists on top of the repaired block6
  branch
- the safe search axis is still narrow local family retention

What round9 does not solve:

- `block7 hit@1` is only `2/5`
- `block7 hit@3` is still `2/5`
- `pass_rate` is still `0.467`

## Next Step

The next justified move is:

- freeze `b7_neg_hold_span1_support2_floor0p08`
- stop trying nearest-neighbor negative retention as a standalone fix
- target only the remaining block7 off-center failures at `offset = -1.0`,
  `-0.5`, and `0.5` with a more selective asymmetric family rule
- avoid span2 broadening, because it improves averages while breaking the hard
  case gate
