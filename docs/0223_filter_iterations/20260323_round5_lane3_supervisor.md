# 2026-03-23 Round 5 Lane 3 Supervisor Memo

## Scope

Round 5 lane 3 tested a narrow same-anchor replacement scorer on top of the
current soft-pool winner.

Fixed components:

- front-end: `diff_len80_ldvonly_bp700_1800`
- pair formation pool: `cond_prune_soft`
- base scorer: `soft_anchor_pow_1p0`

Tracked artifact bundle:

- `artifacts/round5_same_anchor_replacement_0223_20260323_231143_report.md`
- `artifacts/round5_same_anchor_replacement_0223_20260323_231143_summary.json`

## Result

Lane 3 improved `block6` rescue pair competitiveness but not final selection.

Best structural movement:

- `block6` rescue pair rank: `4 -> 2`
- `block7` stayed stable at rescue pair rank `= 3`

But deployment stayed identical to the lane 2 winner:

- central `delta_tau_mae_ms = 0.099`
- hard-case `delta_tau_mae_ms = 0.162`
- hard-case `win_rate = 0.5`
- `block6` still selected `delta_tau_ms = +0.271`

## Supervisor Judgment

Decision:

- accept as a targeted structural positive
- reject as a deployment promotion by itself

Reason:

- lane 3 proves the same-anchor replacement idea is correct
- but the rescue pair still stops at rank 2 instead of taking top1

## Main Lesson

This lane isolates the final remaining issue:

- ranking is now close enough
- but a final family-level handoff is still missing

That turns the next lane from a score search into a selector search.

## Next Supervisor Instruction

The next lane should apply a narrow final handoff rule:

- only in same-anchor double-top contexts
- only when the rescue pair is already nearly tied
- without touching the already-correct `block7` regime
