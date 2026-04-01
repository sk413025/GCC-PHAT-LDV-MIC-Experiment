# 2026-03-26 Round 9 Validator Note

## Validation Scope

Validated result bundles:

- `results/round9_block7_negative_family_target_hold_0223_20260326_0001/`
- `results/round9_block7_nearest_negative_family_0223_20260326_0002/`

## Lane 9.1 Judgment

`ACCEPT AS BEST ROUND9 BRANCH`

The best variant is:

- `b7_neg_hold_span1_support2_floor0p08`

Reason:

- central metrics improve from `0.055 / 0.074` to `0.039 / 0.043`
- `block6 hit@1` remains `5/5`
- `block7 hit@1` rises from `1/5` to `2/5`
- window mean/std improves
- no regression appears on the repaired block6 core signal

## Rejected Variants

The span2 variants are rejected as new winners because:

- they trade away the decisive hard-case target
- `block7 hit@1` falls to `0/5`
- central hard-case metrics regress sharply

Average-only pass-rate gains do not count.

## Lane 9.2 Judgment

`REJECT AS NEW WINNER`

Reason:

- nearest-negative retention does not beat lane 9.1 control
- `block7 hit@1` remains `2/5`
- `pass_rate` remains `0.467`

So this axis should not replace the current round9 leader.

## Validator Conclusion

Best validated status is:

- keep the round8 block6 repair
- add the round9 span1 block7 negative-family hold

Remaining missing evidence:

- `block7 hit@1 >= 3/5`
- `block7 hit@3 >= 4/5`
- `pass_rate >= 0.50` without central regression
