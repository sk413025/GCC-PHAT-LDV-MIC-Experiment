# 2026-03-18 Round 1 Supervisor Memo

## Team Assignment

- Engineer A: time-domain filter families
- Engineer B: spectral filter families
- Engineer C: pairing and scoring families
- Validator: compare all candidates against the current accepted baseline

## Hard Blocking Metrics

Any candidate must satisfy:

- `valid_cases = 4`
- `physical_count = 4`
- `delta_tau_mae_ms <= 0.22`
- `theta_v_mae_deg <= 3.20`
- `max_delta_tau_abs_err_ms <= 0.40`

## Baseline Replacement Rule

A candidate replaces the current baseline only if it beats:

- current baseline `delta_tau_mae_ms = 0.199`
- or stays within `0.205` while reducing `theta_v_mae_deg`

## Round 1 Outcomes

### Engineer A

Best candidate:

- `ldv_diff2_bp500_2000`
- `delta_tau_mae_ms = 0.158`
- `theta_v_mae_deg = 2.220`
- `max_delta_tau_abs_err_ms = 0.295`

Decision:

- accepted as a valid candidate
- promoted above the old baseline
- kept as backup, not final winner

### Engineer B

Best candidate:

- `ldv_diff_bp700_1800`
- `delta_tau_mae_ms = 0.152`
- `theta_v_mae_deg = 2.147`
- `max_delta_tau_abs_err_ms = 0.264`

Decision:

- accepted
- replaces the current baseline
- becomes the new leader after Round 1

### Engineer C

Best candidate:

- `single_band_baseline`
- `delta_tau_mae_ms = 0.199`
- `theta_v_mae_deg = 2.806`

Decision:

- rejected as a baseline replacement
- no pairing family beat the existing accepted rule

## Supervisor Decision

Round 1 winner:

- `Engineer B / ldv_diff_bp700_1800`

Reason:

- best average `delta_tau` error
- best average `theta_v` error
- best worst-case error among accepted candidates

## Next-Step Instruction

If another iteration is opened later, the next task should start from:

- preprocessing: `ldv_diff`
- band: `700-1800 Hz`
- pairing: current accepted `mean_tau` rule

Further work should focus on:

- testing the new baseline on a broader 0223 subset
- only revisiting pairing if the broader subset exposes new failure modes
