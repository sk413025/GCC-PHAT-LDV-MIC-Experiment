# 2026-03-18 Round 3 Lane 1 Supervisor Memo

## Scope

Lane 1 executed the first Round 3 front-end search:

- fixed pair scorer: accepted `mean_tau` blind pairing rule
- fixed band family: `700-1800 Hz`
- search axis: `diff1 + local normalization`
- extra validation: multi-window stability using offsets
  `[-1.0, -0.5, 0.0, 0.5, 1.0] s`

Tracked artifact bundle:

- `artifacts/round3_diff1_local_norm_0223_20260318_231130_report.md`
- `artifacts/round3_diff1_local_norm_0223_20260318_231130_summary.json`

## Reference Baseline

Current accepted blind baseline before Lane 1:

- `baseline_diff_bp700_1800`
- central `delta_tau_mae_ms = 0.152`
- central `theta_v_mae_deg = 2.147`
- central `max_delta_tau_abs_err_ms = 0.264`
- oracle `correct_pair_rank_mean = 11.000`
- oracle `correct_pair_margin_mean = 0.236`
- window `delta_tau_mae_ms = 0.262`
- window `stability_mean_std_ms = 0.136`

## Lane 1 Winner

Local winner:

- `diff_len80_ldvonly_bp700_1800`

Observed metrics:

- central `delta_tau_mae_ms = 0.142`
- central `theta_v_mae_deg = 2.000`
- central `max_delta_tau_abs_err_ms = 0.254`
- oracle `correct_pair_rank_mean = 8.250`
- oracle `correct_pair_margin_mean = 0.516`
- window `delta_tau_mae_ms = 0.207`
- window `stability_mean_std_ms = 0.110`
- window `stability_max_std_ms = 0.337`

Per-case central-window behavior:

- `block4`: improved from `0.046` to `0.025 ms`
- `block5`: held at `0.046 ms`
- `block6`: improved from `0.264` to `0.244 ms`
- `block7`: held at `0.254 ms`

## Supervisor Judgment

Decision:

- accept `diff_len80_ldvonly_bp700_1800` as the `lane1 local winner`
- do not yet promote it to the global deployed baseline

Reason:

- it improves the current baseline on central blind metrics
- it also improves oracle structure metrics and multi-window stability
- this is a real front-end gain, not only a teacher-side illusion
- however, the gain is still modest and the deployment story is incomplete

The strongest signal from Lane 1 is:

- longer-window LDV-only normalization helps
- the useful effect is not just pointwise MAE reduction
- the more important change is a better correct-pair landscape:
  lower oracle rank and larger oracle margin

The lane also showed what not to chase:

- `diff_len40_ldvonly_sqrtgain_bp700_1800` matches the central blind MAE but
  does not improve transfer structure enough
- `diff_len40_ldvonly_robustmedian_bp700_1800` is an oracle-friendly trap:
  stronger oracle rank and margin, but blind deployment collapses

## Next Supervisor Instruction

Round 3 should now move to the next lane:

- keep `diff_len80_ldvonly_bp700_1800` as the provisional front-end
- do not open another broad filter sweep yet
- focus on blind proxy scoring that tries to exploit the improved front-end

Priority next lane:

- `mean_tau prior`
- `PSR-weighted pair scoring`
- `pair balance weight`
- `pair margin proxy`
- `window consistency weight`

The next question is no longer:

- can a filter make the correct peak more visible

The next question is:

- can a blind scorer exploit that visibility without oracle access
