# 2026-03-23 Round 5 Lane 4 Supervisor Memo

## Scope

Round 5 lane 4 tested a same-anchor handoff selector on top of the lane 3
scored soft pool.

Tracked artifact bundle:

- `artifacts/round5_same_anchor_handoff_0223_20260323_231341_report.md`
- `artifacts/round5_same_anchor_handoff_0223_20260323_231341_summary.json`

## Result

The winner is:

- `handoff_ratio_0p90`

Equivalent alternative:

- `handoff_ratio_0p93`

Measured deployment:

- central `delta_tau_mae_ms = 0.055`
- hard-case `delta_tau_mae_ms = 0.074`
- hard-case `win_rate = 1.0`
- central `max_delta_tau_abs_err_ms = 0.080`
- `block6` selected `delta_tau_ms = +0.583`
- `block7` selected `delta_tau_ms = -0.375`

## Supervisor Judgment

Decision:

- promote `handoff_ratio_0p90` to the new deployed baseline

Reason:

- it closes the last remaining `block6` failure mode
- it preserves the `block7` fix
- it materially improves every tracked deployment metric over the previous
  winner

## Main Lesson

The final missing piece was not another filter or another broad scorer.

It was a family-level decision rule:

- once a same-anchor rescue pair is already nearly tied
- allow it to take over from the monopoly top1

This converts the round 5 structural gains into the first fully corrected blind
selection regime.

## Supervisor Conclusion

The search objective for this task is complete.

If future work continues, it should treat:

- `cond_prune_soft`
- `same_vl_replace_w6_p3`
- `handoff_ratio_0p90`

as the new reference stack.
