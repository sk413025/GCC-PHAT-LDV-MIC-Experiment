# 2026-03-21 Round 3 Lane 2 Validator Note

## Validation Scope

Validated artifact bundle:

- `artifacts/round3_blind_proxy_0223_20260321_113400_report.md`
- `artifacts/round3_blind_proxy_0223_20260321_113400_summary.json`

Lane reference:

- fixed front-end: `diff_len80_ldvonly_bp700_1800`
- reference scorer: `len80_current_score`

Reference metrics:

- central `delta_tau_mae_ms = 0.142`
- central `max_delta_tau_abs_err_ms = 0.254`
- window `delta_tau_mae_ms = 0.207`
- window `stability_mean_std_ms = 0.110`
- window `stability_max_std_ms = 0.337`

## Validation Result

Best lane-local candidate:

- `len80_window_support_rank_guard`

Measured change relative to `len80_current_score`:

- central `delta_tau_mae_ms`: unchanged at `0.142`
- central `theta_v_mae_deg`: unchanged at `2.000`
- central `max_delta_tau_abs_err_ms`: unchanged at `0.254`
- window `delta_tau_mae_ms`: `0.207 -> 0.193`
- window `stability_mean_std_ms`: `0.110 -> 0.083`
- window `stability_max_std_ms`: `0.337 -> 0.232`

Physical validity:

- central `valid_cases = 4`
- central `physical_count = 4`

## Validator Judgment

Verdict:

- `ACCEPT AS STABILITY WINNER`
- `REJECT FOR PROMOTION`

Reason:

- the candidate clearly improves multi-window stability
- but it does not improve the central-window deployment error
- worst-case central error is still `0.254`, above the practical target needed
  for real promotion

## Leakage Check

The lane stayed inside blind-observable features:

- `mean_tau`
- local candidate ranking
- `amp` and pair product
- window support
- rank discount

No reference-derived feature was used in the deployed score.

## Remaining Failure

The lane did not solve the core hard-case issue:

- `block6` is still hijacked by the strongest `VR` branch
- `block7` is still hijacked by the near-zero `delta_tau` pair

So this lane improved `stability`, not `selection`.

## Validator Conclusion

Lane 2 produced a useful secondary result:

- better off-center robustness

But it did not produce a new deployed scorer.

The next lane must directly target central-pair competition on the hard cases,
especially:

- same-`VL` alternative comparisons for `block6`
- same-`VR` alternative comparisons for `block7`

Without that, the central worst-case error will stay stuck.
