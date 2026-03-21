# 2026-03-21 Round 3 Lane 2 Supervisor Memo

## Scope

Lane 2 fixed the Round 3 lane 1 provisional front-end:

- front-end: `diff_len80_ldvonly_bp700_1800`
- candidate policy: positive-lag top-k pairs with `|delta_tau| <= 1.0 ms`
- objective: improve blind pair ranking only

Tracked artifact bundle:

- `artifacts/round3_blind_proxy_0223_20260321_113400_report.md`
- `artifacts/round3_blind_proxy_0223_20260321_113400_summary.json`

## Reference for Lane 2

Reference scorer on the fixed front-end:

- `len80_current_score`
- central `delta_tau_mae_ms = 0.142`
- central `theta_v_mae_deg = 2.000`
- central `max_delta_tau_abs_err_ms = 0.254`
- window `delta_tau_mae_ms = 0.207`
- window `stability_mean_std_ms = 0.110`
- window `stability_max_std_ms = 0.337`

## Lane 2 Winner

Local winner:

- `len80_window_support_rank_guard`

Observed metrics:

- central `delta_tau_mae_ms = 0.142`
- central `theta_v_mae_deg = 2.000`
- central `max_delta_tau_abs_err_ms = 0.254`
- central `correct_pair_rank_mean = 1.000`
- central `correct_pair_margin_mean = 2.322`
- window `delta_tau_mae_ms = 0.193`
- window `stability_mean_std_ms = 0.083`
- window `stability_max_std_ms = 0.232`

## Supervisor Judgment

Decision:

- accept `len80_window_support_rank_guard` as the `lane2 local winner`
- do not promote it as the new deployed scorer

Reason:

- it improves window-level robustness clearly
- it does not improve the central-window blind error at all
- the hard cases remain unchanged

This lane therefore succeeded at one thing:

- it converted the provisional front-end into a more stable multi-window scorer

But it failed at the more important thing:

- it did not dethrone the central-window hijack pairs on the hard cases

## What the Lane Revealed

The stability signal is real:

- plain window support and rank guarding reduce off-center failures
- the best lane 2 variant lowers window `delta_tau_mae_ms` from `0.207` to
  `0.193`
- it lowers window `stability_mean_std_ms` from `0.110` to `0.083`

However, the central-window selections stayed frozen:

- `block4` remained `delta_tau_ms = -0.417`
- `block5` remained `delta_tau_ms = -0.396`
- `block6` remained `delta_tau_ms = +0.271`
- `block7` remained `delta_tau_ms = -0.042`

That means the currently tested proxy families were not strong enough to change
the top-1 central pair ordering.

## Hard-Case Interpretation

### `block6`

The current top pair still uses the strongest `VL` and strongest `VR`, giving:

- `tau_vl_ms = 4.854`
- `tau_vr_ms = 5.125`
- `delta_tau_ms = +0.271`

But a more plausible competitor exists with:

- `tau_vl_ms = 4.854`
- `tau_vr_ms = 5.438`
- `delta_tau_ms = +0.583`

This suggests the next proxy should punish the strongest local `VR` hijack when
another `VR` candidate paired with the same `VL` is more globally plausible.

### `block7`

The current top pair remains:

- `tau_vl_ms = 4.562`
- `tau_vr_ms = 4.521`
- `delta_tau_ms = -0.042`

But a more plausible competitor exists with:

- `tau_vl_ms = 4.833`
- `tau_vr_ms = 4.521`
- `delta_tau_ms = -0.312`

This suggests the next proxy should compare competitors that share the same
`VR` branch and reward the pair whose `VL` creates a more plausible
`mean_tau / delta_tau` geometry.

## Next Supervisor Instruction

Do not open another generic proxy sweep.

The next lane should explicitly target:

- `pair-margin proxy`
- `same-VL competition gap`
- `same-VR competition gap`
- stronger blind penalties for central zero-delta hijack pairs

The next scorer family should be designed to answer:

- when two candidate pairs share one side, which pair is more globally
  self-consistent

Not:

- which pair simply has the biggest local amplitude product
