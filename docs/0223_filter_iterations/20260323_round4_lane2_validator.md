# 2026-03-23 Round 4 Lane 2 Validator Note

## Validation Scope

Validated artifact bundle:

- `artifacts/round4_weak_branch_0223_20260323_220748_report.md`
- `artifacts/round4_weak_branch_0223_20260323_220748_summary.json`
- `artifacts/round4_weak_branch_0223_20260323_220748_strict_rescue.md`

Validation rule for this lane:

- weak-branch rescue must improve strict rescue rank first
- then that improvement must survive the fixed blind scorer

Strict rescue targets:

- `block6`: `VL ~= 4.854 ms`, `VR ~= 5.438 ms`, `delta_tau ~= +0.583 ms`
- `block7`: `VL ~= 4.833 ms`, `VR ~= 4.521 ms`, `delta_tau ~= -0.312 ms`

Strict match rule:

- branch-wise tolerance `= +/- 0.20 ms`

## Reference

Reference central metrics:

- `delta_tau_mae_ms = 0.142`
- hard-case `delta_tau_mae_ms = 0.249`
- central `max_delta_tau_abs_err_ms = 0.254`

Reference rescue ranks:

- `block6` rescue `VR` rank `= 3`, pair rank `= 7`
- `block7` rescue `VL` rank `= 3`, pair rank `= 4`

## Validation Result

No weak-branch family passed the lane-local rescue test.

For every tested family:

- rescue branch ranks were unchanged
- rescue pair ranks were unchanged
- central deployment metrics were unchanged

This means the lane did not produce either:

- candidate-pool improvement
- or downstream blind gain

## Validator Judgment

Verdict:

- `REJECT ALL WEAK-BRANCH RESCUE FAMILIES FOR PROMOTION`

Reason:

- strict rescue-rank did not improve
- rescue pair did not move closer to `top3`
- hard-case deployment error stayed frozen
- selected hard-case `delta_tau` stayed on the same incorrect branch

## Important Negative Finding

This lane fails in a stronger way than lane 4.1.

Lane 4.1 at least changed the proposal pool.
Lane 4.2 left both the strict rescue ranks and the final deployment outcome
unchanged.

That means these weak-branch rescue families are currently functionally
equivalent to the control under the fixed scorer.

The most important validator conclusion is:

- the rescue branches already exist in the candidate pool
- the bottleneck is not weak-branch visibility by itself
- the bottleneck remains pair-level competition and blind selection

## Validator Conclusion

Round 4 lane 2 is a valid negative result.

It narrows the path again:

- more weak-branch rescue variants should not be promoted
- next work should target pair formation, corridor pruning, or branch-aware
  competition features that can exploit rescue branches already present in the
  pool
