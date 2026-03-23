# 2026-03-23 Round 5 Lane 4 Validator Note

## Validation Scope

Validated artifact bundle:

- `artifacts/round5_same_anchor_handoff_0223_20260323_231341_report.md`
- `artifacts/round5_same_anchor_handoff_0223_20260323_231341_summary.json`

## Validator Reference

The locked pre-lane baseline was:

- central `delta_tau_mae_ms = 0.099`
- hard-case `delta_tau_mae_ms = 0.162`
- hard-case `win_rate = 0.5`
- central `max_delta_tau_abs_err_ms = 0.244`

## Validation Result

`handoff_ratio_0p90` and `handoff_ratio_0p93` both satisfy the promotion gate.

Measured:

- central `delta_tau_mae_ms = 0.055`
- hard-case `delta_tau_mae_ms = 0.074`
- hard-case `win_rate = 1.0`
- central `max_delta_tau_abs_err_ms = 0.080`
- `block6` rescue pair rank `= 1`
- `block6` selected `delta_tau_ms = +0.583`
- `block7` rescue pair rank `= 3`
- `block7` selected `delta_tau_ms = -0.375`

## Validator Judgment

Verdict:

- `FULL PROMOTION`

Reason:

- all tracked deployment metrics improve
- both hard cases are now inside the blind-correct regime
- no regression appears on the easy cases

## Validator Conclusion

This lane closes the open gap left by the provisional winner.

Recommended new baseline:

- pool: `cond_prune_soft`
- scorer: `same_vl_replace_w6_p3`
- handoff selector: `handoff_ratio_0p90`
