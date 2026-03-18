# 2026-03-18 Round 1 Validator Note

## Validation Scope

Validated artifacts:

- `artifacts/filter_family_time_domain_0223_20260318_212521_summary.json`
- `artifacts/filter_family_spectral_0223_20260318_212521_summary.json`
- `artifacts/peak_pair_family_0223_20260318_212521_summary.json`

Reference baseline:

- `ldv_diff_bp500_2000`
- `amp_product_mean_tau_delta_quad`
- `delta_tau_mae_ms = 0.199`
- `theta_v_mae_deg = 2.806`

## Validation Result

### Time-Domain Lane

Best candidate:

- `ldv_diff2_bp500_2000`
- `delta_tau_mae_ms = 0.158`
- `theta_v_mae_deg = 2.220`
- `max_delta_tau_abs_err_ms = 0.295`
- verdict: `PASS`

### Spectral Lane

Best candidate:

- `ldv_diff_bp700_1800`
- `delta_tau_mae_ms = 0.152`
- `theta_v_mae_deg = 2.147`
- `max_delta_tau_abs_err_ms = 0.264`
- verdict: `PASS`

### Pairing Lane

Best candidate:

- `single_band_baseline`
- `delta_tau_mae_ms = 0.199`
- `theta_v_mae_deg = 2.806`
- verdict: `NO IMPROVEMENT`

## Independent Cross-Check

An additional local cross-check was run after Round 1:

- combine the best time-domain and spectral ideas
- retest pairing logic on the new spectral winner

Result:

- no combined candidate beat `ldv_diff_bp700_1800`
- no pairing variant improved on the accepted pairing rule when retested on the
  `700-1800 Hz` winner

## Accepted Baseline After Validation

The validator accepts the following as the new baseline:

- preprocessing: `ldv_diff`
- band: `700-1800 Hz`
- pairing rule: current accepted `mean_tau` pairing rule

Accepted metrics:

- `delta_tau_mae_ms = 0.152`
- `theta_v_mae_deg = 2.147`
- `physical_count = 4 / 4`
- `max_delta_tau_abs_err_ms = 0.264`
- `max_theta_v_abs_err_deg = 3.729`

## Commit Recommendation

The validator recommends committing:

- the multi-agent team contract
- the three engineer family scripts
- the supervisor and validator notes
- the tracked artifact summaries for Round 1

The validator does not recommend committing raw `results/` directories.
