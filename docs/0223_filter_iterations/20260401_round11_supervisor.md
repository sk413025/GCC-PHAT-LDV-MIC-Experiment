# Round 11 Supervisor Memo

## Objective
Assess the targeted block7 offset=-0.5 follow-up and decide whether it is cleared enough to become the next frozen baseline.

## Round 10 Baseline Reference
Both round11 sweeps were measured against the frozen round10 winner `b7_offset_family_hold_gate1_span1_r0p08_f0p06_c0p18_dt0p04` from `dataset/0223`.

## Primary Round 11 Sweep Result
- `results/round11_block7_minus0p5_targeted_family_hold_0223_20260331_233600/`
- Winner: `b7_m05_bridge_mean3_r0p15_vl0p20_vr0p22_dt0p10`
- `non_regressing=True`
- `blocker_dt_err_ms=0.059` vs round10 control `0.163`
- `pass_rate=0.467`
- `block6_hit@1=5`, `block7_hit@1=3`, `block7_hit@3=3`
- `stability=0.056`

This is the cleanest improvement on the blocker itself, but it does not move the block7 hit metrics beyond the round10 baseline pattern.

## Backup Round 11 Sweep Result
- `results/round11_block7_minus0p5_targeted_family_hold_b_0223_20260331_234702/`
- Winner: `b7_m05_negctr_blend50_floor0p12_f0p40_c0p40_dt0p22`
- `non_regressing=True`
- `pass_rate=0.533`
- `block6_hit@1=5`, `block7_hit@1=2`, `block7_hit@3=2`
- `window_stability_mean_std_ms=0.074`

The backup sweep is more stable on pass rate, but it is weaker on block7 retrieval and does not provide a better blocker resolution than the primary sweep.

## Conclusion
`block7 offset=-0.5` is only partially improved. The primary sweep narrows the blocker error, but the round11 evidence does not show a fully cleared block7 offset=-0.5 case.

## Recommendation
Do not freeze round11 as the new baseline yet. Keep the primary sweep as the best provisional continuation, but treat block7 offset=-0.5 as still open until a follow-up improves both blocker error and block7 hit quality.
