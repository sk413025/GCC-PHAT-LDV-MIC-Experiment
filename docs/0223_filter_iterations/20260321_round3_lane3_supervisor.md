# 2026-03-21 Round 3 Lane 3 Supervisor Memo

## Scope

Lane 3 fixed:

- front-end: `diff_len80_ldvonly_bp700_1800`
- candidate generation: same as Round 3 lane 2

This lane tested competition-gap scorers only.
The main goal was to break the central-window hijack pairs on:

- `block6_n04_19`
- `block7_n08_20`

Tracked artifact bundle:

- `artifacts/round3_competition_gap_0223_20260321_121335_report.md`
- `artifacts/round3_competition_gap_0223_20260321_121335_summary.json`

## Reference for Lane 3

Reference scorer:

- `lane2_control`

Reference metrics:

- central `delta_tau_mae_ms = 0.142`
- hard-case `delta_tau_mae_ms = 0.249`
- hard-case win rate `= 0.000`
- central `max_delta_tau_abs_err_ms = 0.254`
- window `delta_tau_mae_ms = 0.207`
- window `stability_mean_std_ms = 0.110`

## Lane 3 Result

No competition-gap candidate beat the reference on the central deployment
metrics.

Best central metrics remained identical to the reference for:

- `worst_constraint_margin`
- `bilateral_ownership_gate`
- `corridor_relative_margin`
- `local_consensus_gap`
- `hybrid_gap_zero05`
- `hybrid_gap_zero10`

Observed plateau:

- central `delta_tau_mae_ms = 0.142`
- hard-case `delta_tau_mae_ms = 0.249`
- hard-case win rate `= 0.000`
- central `max_delta_tau_abs_err_ms = 0.254`

One candidate worsened deployment behavior:

- `delta_shift_per_cost`
- central `delta_tau_mae_ms = 0.314`
- central `max_delta_tau_abs_err_ms = 0.400`

## Supervisor Judgment

Decision:

- accept the lane as a negative but useful result
- do not promote any lane 3 scorer

Reason:

- same-anchor competition-gap features did not change the central top-1 pair
- the current hard-case top pairs appear locally self-consistent under the
  current front-end
- this means the next bottleneck is no longer simple pair competition inside
  the present candidate pool

## Main Technical Lesson

Lane 3 falsified a tempting hypothesis:

- the hard cases are not currently failing because the top-1 pair is obviously
  weak in same-`VL`, same-`VR`, or local `mean_tau` sub-competitions

What lane 3 actually showed is:

- `block6` top-1 remains locally strong under same-anchor margin tests
- `block7` top-1 also retains positive local margins under the tested
  competition rules
- mild zero-avoid terms do not create a real breakthrough

This means the current candidate pool likely lacks a blind feature that truly
separates the rescue pair from the hijack pair.

## Zero-Avoid Interpretation

The zero-avoid ablation was useful:

- `hybrid_gap_zero0`, `hybrid_gap_zero05`, and `hybrid_gap_zero10` all keep the
  same central deployment error
- penalty share grows from `0.000` to about `0.341` median without producing a
  central win

Supervisor conclusion:

- zero-avoid is not the missing ingredient
- pushing harder on away-from-zero heuristics is not justified

## Next Supervisor Instruction

Do not keep expanding pair-level scorer algebra on the same candidate pool.

The next lane should instead target one of these:

- candidate generation changes that create better hard-case alternatives
- side-specific preprocessing for the weak branch in `block6` / `block7`
- local branch credibility features before pair formation, not after

The lane 3 outcome says the next win is unlikely to come from a better
reweighting of the current pair rows alone.
