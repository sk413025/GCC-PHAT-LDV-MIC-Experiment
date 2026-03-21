# 2026-03-21 Round 3 Lane 3 Validator Note

## Validation Scope

Validated artifact bundle:

- `artifacts/round3_competition_gap_0223_20260321_121335_report.md`
- `artifacts/round3_competition_gap_0223_20260321_121335_summary.json`

Lane 3 reference:

- scorer: `lane2_control`
- fixed front-end: `diff_len80_ldvonly_bp700_1800`

Reference metrics:

- central `delta_tau_mae_ms = 0.142`
- hard-case `delta_tau_mae_ms = 0.249`
- hard-case win rate `= 0.000`
- window `delta_tau_mae_ms = 0.207`
- window `stability_mean_std_ms = 0.110`

## Validation Result

No lane 3 scorer achieved a deployment promotion.

The strongest competition-gap families only tied the reference on central
metrics:

- `worst_constraint_margin`
- `bilateral_ownership_gate`
- `corridor_relative_margin`
- `local_consensus_gap`

The hybrid ablation family also failed to create a central gain:

- `hybrid_gap_zero0`
- `hybrid_gap_zero05`
- `hybrid_gap_zero10`

The only clear change from the ablation was a larger penalty share, not better
deployment accuracy.

## Zero-Penalty Check

Lane 3 passed an important negative test:

- adding more zero-avoid weight did not improve central deployment metrics

Observed penalty share:

- `hybrid_gap_zero0`: median `0.000`
- `hybrid_gap_zero05`: median `0.208`
- `hybrid_gap_zero10`: median `0.341`

Validator interpretation:

- the lane is not winning because of away-from-zero forcing
- but it is also not winning because the tested competition-gap features do not
  change the hard-case top-1 choice

## Failure Modes Observed

### `delta_shift_per_cost`

Rejected:

- central `delta_tau_mae_ms = 0.314`
- central `max_delta_tau_abs_err_ms = 0.400`

Reason:

- too aggressive
- damages central deployment performance

### Plateau Families

Rejected for promotion:

- all families that tied the control

Reason:

- no hard-case win
- no central worst-case reduction
- no promotion-level deployment improvement

## Validator Conclusion

Lane 3 is a valid negative result.

It narrows the search space:

- pair-level competition-gap features on the current candidate pool are not
  sufficient
- stronger zero-avoid pressure is not justified
- the next meaningful change must happen before or during candidate generation

No lane 3 scorer should be promoted.
