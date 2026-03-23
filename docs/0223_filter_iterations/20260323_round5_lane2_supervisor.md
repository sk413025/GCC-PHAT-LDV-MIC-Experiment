# 2026-03-23 Round 5 Lane 2 Supervisor Memo

## Scope

Round 5 lane 2 tested anchor-replacement scoring on top of the round 5 pruned
pair pools.

Compared pools:

- `cond_prune_soft`
- `cond_prune_medium`

Tracked artifact bundle:

- `artifacts/round5_anchor_replacement_0223_20260323_225819_report.md`
- `artifacts/round5_anchor_replacement_0223_20260323_225819_summary.json`

## Reference

Current deployed baseline:

- central `delta_tau_mae_ms = 0.142`
- hard-case `delta_tau_mae_ms = 0.249`
- central `max_delta_tau_abs_err_ms = 0.254`

Round 5 lane 1 best structural baseline:

- `cond_prune_medium + current score`
- central `delta_tau_mae_ms = 0.402`
- hard-case `delta_tau_mae_ms = 0.769`

## Lane 2 Result

The clear winner is:

- `soft_anchor_pow_1p0`

Observed deployment:

- central `delta_tau_mae_ms = 0.099`
- hard-case `delta_tau_mae_ms = 0.162`
- hard-case `win_rate = 0.5`
- central `max_delta_tau_abs_err_ms = 0.244`

Case-level interpretation:

- `block7` is corrected to the right negative branch
  - selected `delta_tau_ms = -0.375`
- `block6` remains frozen on the old branch
  - selected `delta_tau_ms = +0.271`

## Supervisor Judgment

Decision:

- accept `soft_anchor_pow_1p0` as the new provisional round 5 winner
- do not promote it to the deployed baseline yet

Reason:

- it materially beats the deployed baseline on both central and hard-case MAE
- it preserves easy cases much better than the `medium` pool families
- but it still only breaks one of the two hard-case failure modes

## Main Lesson

This is the first genuinely positive deployment lane after the round 4 pivot.

The winning structure is:

- softer corridor pruning
- then an anchor-aware scorer

Notably:

- the `soft` pool is safer than the `medium` pool
- the anchor scorer is useful only when the pool remains conservative enough to
  preserve easy cases

## Next Supervisor Instruction

Keep `soft_anchor_pow_1p0` as the leading candidate and run one more focused
lane aimed only at `block6`.

Priority direction:

- preserve the `soft` pool
- preserve the `block7` fix
- target the still-frozen `block6` branch competition without reopening the
  regressions seen in `medium` pruning
