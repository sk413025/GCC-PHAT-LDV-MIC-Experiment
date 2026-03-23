# 2026-03-24 Round 6 Validator Note

## Validation Scope

Validated artifact bundles:

- `artifacts/round6_baseline_robustness_0223_20260323_233741_report.md`
- `artifacts/round6_temporal_handoff_0223_20260323_234354_report.md`
- `artifacts/round6_sign_veto_handoff_0223_20260323_235603_report.md`
- `artifacts/round6_family_split_handoff_0223_20260324_001136_report.md`

## Lane 1 Judgment

`FAIL`

Reason:

- central preservation passes
- hard-case central win passes
- window stability gate fails
- perturbation gate fails decisively

Lane 1 therefore cannot certify the promoted baseline as robust.

## Lane 2 Judgment

`ACCEPT AS STABILITY IMPROVEMENT`

Reason:

- central metrics remain intact
- temporal variance drops sharply
- several wrong-sign settings are repaired

But lane2 still fails the Round 6 recovery gate because:

- `block6 hit@1 = 2/5`
- `block7 hit@1 = 1/5`
- perturbation-level persistence remains too weak

## Lane 3 Judgment

`ACCEPT AS BEST ROUND6 RECOVERY`

The best lane3 variant is:

- `sign_veto_poslift_r0p10`

It keeps the promoted central regime and improves the hard-case recovery
structure further than lane2:

- setting pass rate rises to `0.400`
- `block6 hit@1` rises to `3/5`
- central metrics remain unchanged

This is still below the promotion bar, but it is a real robustness recovery.

## Lane 4 Judgment

`REJECT AS NEW WINNER`

The family-split lane does not beat lane3 on the decisive metrics:

- setting pass rate does not improve
- `block7 hit@1` degrades to `0/5`

So lane4 should not replace lane3 as the round6 recovery leader.

## Validator Conclusion

No round6 lane is promotable as a robustness-confirmed baseline.

Best validated status is:

- round5 lane4 remains the promoted central solver
- round6 lane3 is the best recovery branch for robustness work

The remaining missing evidence is explicit hard-case persistence:

- `block6 hit@1 >= 4/5`
- `block7 hit@1 >= 3/5`
- higher perturbation pass rate without central regression
