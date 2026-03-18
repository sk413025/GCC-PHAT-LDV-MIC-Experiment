# 2026-03-18 Round 3 Lane 1 Validator Note

## Validation Scope

Validated artifact bundle:

- `artifacts/round3_diff1_local_norm_0223_20260318_231130_report.md`
- `artifacts/round3_diff1_local_norm_0223_20260318_231130_summary.json`

Reference baseline before Lane 1:

- `baseline_diff_bp700_1800`
- central `delta_tau_mae_ms = 0.152`
- central `theta_v_mae_deg = 2.147`
- central `max_delta_tau_abs_err_ms = 0.264`
- window `delta_tau_mae_ms = 0.262`
- window `stability_mean_std_ms = 0.136`

## Validation Result

Best candidate in Lane 1:

- `diff_len80_ldvonly_bp700_1800`

Measured improvement relative to the reference baseline:

- central `delta_tau_mae_ms`: `0.152 -> 0.142`
- central `theta_v_mae_deg`: `2.147 -> 2.000`
- central `max_delta_tau_abs_err_ms`: `0.264 -> 0.254`
- window `delta_tau_mae_ms`: `0.262 -> 0.207`
- window `stability_mean_std_ms`: `0.136 -> 0.110`
- oracle `correct_pair_rank_mean`: `11.000 -> 8.250`
- oracle `correct_pair_margin_mean`: `0.236 -> 0.516`

Physical validity:

- central `valid_cases = 4`
- central `physical_count = 4`

## Validator Judgment

Verdict:

- `ACCEPT AS LANE-LOCAL WINNER`
- `REJECT FOR GLOBAL PROMOTION`

Why promotion is rejected:

- central blind improvement is real but small
- worst-case improvement is small
- the hard promotion gate from the Round 3 roadmap is not met
- `window_stability_max_std_ms = 0.337` stays below the warning ceiling but is
  still close to it

Promotion gate check:

- blind `delta_tau_mae_ms <= 0.18`: pass
- blind `max_delta_tau_abs_err_ms <= 0.35`: pass
- `valid_cases = 4`: pass
- `physical_count = 4`: pass
- strong enough promotion margin over the old winner: fail
- enough evidence against teacher-student gap: fail

## Key Risk

Lane 1 still does not close the teacher-student gap.

Evidence:

- `diff_len40_ldvonly_robustmedian_bp700_1800` achieves strong oracle structure
  metrics
- but blind deployment degrades badly to central `delta_tau_mae_ms = 0.345`

This means oracle rank and oracle margin are useful diagnostics, but they are
not by themselves promotion criteria.

## Validator Conclusion

Lane 1 produced a useful front-end:

- `diff_len80_ldvonly_bp700_1800`

It is good enough to carry into the next blind-scoring lane, but not good
enough to replace the deployed baseline yet.

The next lane must test whether a blind scorer can convert the better front-end
landscape into a stronger deployed result.
