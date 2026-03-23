# 2026-03-23 Round 4 Lane 2 Supervisor Memo

## Scope

Round 4 lane 2 tested weak-branch-specific rescue only.

Fixed components:

- base front-end: `diff_len80_ldvonly_bp700_1800`
- downstream scorer: `len80_current_score`

Weak-branch rescue families:

- `weak_local_zscore_mic`
- `weak_clip_rms_mic`
- `weak_diff2_union`
- `weak_residual_second_pass`
- `weak_bundle`

Tracked artifact bundle:

- `artifacts/round4_weak_branch_0223_20260323_220748_report.md`
- `artifacts/round4_weak_branch_0223_20260323_220748_summary.json`
- `artifacts/round4_weak_branch_0223_20260323_220748_strict_rescue.md`

## Reference

Reference pool:

- `base_only`

Reference central metrics:

- `delta_tau_mae_ms = 0.142`
- hard-case `delta_tau_mae_ms = 0.249`
- central `max_delta_tau_abs_err_ms = 0.254`

Reference strict rescue ranks on central window:

- `block6`: rescue `VR` rank `= 3`, rescue pair rank `= 7`
- `block7`: rescue `VL` rank `= 3`, rescue pair rank `= 4`

## Lane 2 Result

All weak-branch rescue families tied the control exactly.

Observed across every tested family:

- central `delta_tau_mae_ms = 0.142`
- hard-case `delta_tau_mae_ms = 0.249`
- central `max_delta_tau_abs_err_ms = 0.254`
- oracle hit counts remained saturated at `4 / 4`

This is not a near tie.
It is an identical deployment outcome.

## Hard-Case Rescue Check

The strict rescue-rank table did not move at all.

Observed central-window rescue structure for every variant:

- `block6`
  - rescue `VR` rank: `3`
  - rescue pair rank: `7`
  - selected `delta_tau_ms`: `+0.271`
- `block7`
  - rescue `VL` rank: `3`
  - rescue pair rank: `4`
  - selected `delta_tau_ms`: `-0.042`

Supervisor interpretation:

- weak-side-only preprocessing did not create a new effective candidate path
- the weak branch was already present in the pool
- the tested rescue families did not alter the ranked competition enough to
  change pair selection

## Supervisor Judgment

Decision:

- accept the lane as a clean negative result
- do not promote any weak-branch rescue family

Reason:

- no deployment gain
- no strict rescue-rank gain
- no directional movement toward the target hard-case `delta_tau`

## Main Lesson

Weak-side preprocessing is not sufficient if it only re-expresses candidates
that are already in the base pool.

This lane shows:

- `block6` is not failing because the rescue `VR` is invisible
- `block7` is not failing because the rescue `VL` is absent
- both failures remain pair-competition failures under the current scoring and
  pair-formation regime

## Next Supervisor Instruction

The next lane should stop adding more weak-side rescue families and move one
level deeper into pair formation.

Priority direction:

- branch-aware pair gating
- selective pruning of the near-zero hijack corridor
- hard-case-specific pair construction that can surface the existing rescue
  branch as a more competitive pair
