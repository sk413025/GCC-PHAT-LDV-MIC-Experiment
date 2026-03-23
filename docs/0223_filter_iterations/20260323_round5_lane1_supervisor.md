# 2026-03-23 Round 5 Lane 1 Supervisor Memo

## Scope

Round 5 lane 1 tested pair formation only.

Fixed components:

- front-end: `diff_len80_ldvonly_bp700_1800`
- branch pool: `base_only`
- downstream scorer: `len80_current_score`

Search target:

- improve hard-case rescue-pair competitiveness before blind scoring

Tracked artifact bundle:

- `artifacts/round5_pair_formation_0223_20260323_225242_report.md`
- `artifacts/round5_pair_formation_0223_20260323_225242_summary.json`

## Reference

Reference deployment:

- central `delta_tau_mae_ms = 0.142`
- hard-case `delta_tau_mae_ms = 0.249`
- central `max_delta_tau_abs_err_ms = 0.254`

Reference strict rescue ranks:

- `block6` rescue pair rank `= 7`
- `block7` rescue pair rank `= 4`

## Lane 1 Result

Pair formation changed the rescue structure, but not the final deployment in the
right direction.

Best structural movement came from:

- `cond_prune_medium`
- `block6` rescue pair rank: `7 -> 3`
- `block7` rescue pair rank: `4 -> 3`
- `block6` hit@5: `2 -> 5`
- `block7` hit@5: `2 -> 4`

But final blind selections became worse:

- `block6` selected `delta_tau_ms = -0.312`
- `block7` selected `delta_tau_ms = +0.417`
- hard-case `delta_tau_mae_ms = 0.769`

## Supervisor Judgment

Decision:

- accept the lane as a structural positive result
- reject it as a deployment baseline

Reason:

- pair formation successfully moved rescue pairs upward
- but the fixed blind scorer still selected the wrong surviving pair families
- this proves the bottleneck has shifted from pair availability to post-pruning
  ranking

## Main Lesson

Round 5 lane 1 is the first strong evidence that pair formation matters.

It does two important things:

- confirms rescue pairs can be pulled into the top competition set
- falsifies the idea that pair formation alone is enough

This is a classic:

- formation positive
- scorer negative

result.

## Next Supervisor Instruction

The next lane should keep the improved pool and redesign the scorer around
single-anchor ownership.

Priority direction:

- reward one strong anchor plus one viable replacement branch
- compare `soft` and `medium` pruning pools
- promote only if the new scorer preserves easy cases while keeping the
  hard-case rescue movement
