# 0223 Peak-Pair Validation

This note records the validated `delta_tau` recovery recipe for the 0223
blocked LDV-MIC cases.

## Scope

- Worktree: `exp/ldv-vs-mic-doa-comparison`
- Dataset root used: `worktree/doc-interspeech-2026-repro/dataset/0223`
- Cases: `block4_p08_17`, `block5_p00_18`, `block6_n04_19`, `block7_n08_20`

## Commands

```powershell
python scripts\filter_sweep_0223_delta_tau.py --slice_sec 5.0
python scripts\peak_pair_sweep_0223_delta_tau.py
python -m py_compile scripts\peak_pair_sweep_0223_delta_tau.py scripts\filter_sweep_0223_delta_tau.py
```

## Best Validated Recipe

- Variant: `ldv_diff_bp500_2000`
- Strategy: `amp_product_mean_tau_delta_quad`
- `lag_min_ms = 4.4`
- `lag_max_ms = 6.5`
- `delta_limit_ms = 1.0`
- `delta_scale_ms = 0.6`
- `mean_tau_center_ms = 4.8`
- `mean_tau_scale_ms = 0.45`
- `top_k = 8`

## Outcome

The formal sweep result is stored under:

- `results/peak_pair_sweep_0223_20260317_135546/`

Best summary row:

- `delta_tau_mae_ms = 0.199`
- `theta_v_mae_deg = 2.806`
- `physical_count = 4 / 4`
- `max_delta_tau_abs_err_ms = 0.358`
- `max_theta_v_abs_err_deg = 5.043`

Best preprocessing-only baseline from the centered 5-second sweep:

- Variant: `bp_1000_3000`
- `global_delta_tau_mae_ms = 0.414`
- `global_theta_mae_deg = 5.820`
- `global_physical_count = 0 / 4`

This means the validated peak-pair recipe improves `delta_tau` MAE by about
`51.8%` over the best preprocessing-only global argmax baseline, while also
restoring physically valid picks on all four cases.

## Interpretation

The result supports the following claim:

- fixed preprocessing alone is not sufficient
- positive-lag top-k pairing is necessary
- adding a physical prior on the expected mean lag is what pushes the method
  into a reliable regime on this 0223 subset

The result does not prove full free-space `XY` localization. It validates a
more modest point: for the calibrated 0223 blocked cases, constrained
cross-modal peak pairing can recover accurate `delta_tau`.
