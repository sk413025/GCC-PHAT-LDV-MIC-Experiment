# 2026-03-23 Round 5 Lane 2 Validator Note

## Validation Scope

Validated artifact bundle:

- `artifacts/round5_anchor_replacement_0223_20260323_225819_report.md`
- `artifacts/round5_anchor_replacement_0223_20260323_225819_summary.json`

## Reference

Current deployed baseline:

- central `delta_tau_mae_ms = 0.142`
- hard-case `delta_tau_mae_ms = 0.249`
- central `max_delta_tau_abs_err_ms = 0.254`
- hard-case `win_rate = 0.0`

## Validation Result

Best candidate:

- `soft_anchor_pow_1p0`

Measured improvement over deployed baseline:

- central `delta_tau_mae_ms`: `0.142 -> 0.099`
- hard-case `delta_tau_mae_ms`: `0.249 -> 0.162`
- central `max_delta_tau_abs_err_ms`: `0.254 -> 0.244`
- hard-case `win_rate`: `0.0 -> 0.5`

## Validator Judgment

Verdict:

- `PROVISIONAL WINNER`

This is a real deployment improvement.
It is not a false average-only gain.

Why it is not full promotion yet:

- `block6` is still not rescued
  - rescue pair rank `= 4`
  - selected `delta_tau_ms = +0.271`
- `block7` improves materially
  - rescue pair rank `= 3`
  - selected `delta_tau_ms = -0.375`

So the method closes only one of the two hard-case failure modes.

## Exact Remaining Gap

To become a clean promotion candidate, the next lane still needs to:

- preserve the `soft_anchor_pow_1p0` gains on `block7`
- improve `block6` rescue competitiveness or selection
- keep central `max_delta_tau_abs_err_ms` from regressing above the current
  `0.244`

## Validator Conclusion

Round 5 lane 2 is the strongest result so far after the round 4 pivot.

It should replace the previous search direction as the working candidate, but it
should still be labeled:

- provisional

until `block6` is also brought under control.
