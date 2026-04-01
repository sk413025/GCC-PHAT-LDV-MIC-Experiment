# Round 10 Supervisor Memo

## Objective
Compare the five round10 lane families on block7 and identify the cleanest continuation path for the next iteration.

## Baseline Reference
All lanes were evaluated against the frozen round9 lane1 winner `b7_neg_hold_span1_support2_floor0p08` from `dataset/0223`.

## Per-Lane Result Summary

- `results/round10_block7_asym_negative_family_hold_0223_20260331_230339/`  
  Winner: `b7_carry_asym_left0p25_dt0p30`  
  `block7_hit@1=4`, `block7_hit@3=4`, `pass_rate=0.400`  
  Best raw block7 lift in the set, but it trades away pass rate and is less aligned with the narrow selective path.

- `results/round10_block7_offset_selective_family_hold_0223_20260331_225154/`  
  Winner: `b7_offset_family_hold_gate1_span1_r0p08_f0p06_c0p18_dt0p04`  
  `block7_hit@1=3`, `block7_hit@3=3`, `pass_rate=0.467`, `promoted_count=1`  
  Strongest fit for selective family continuity with a narrow gate.

- `results/round10_block7_temporal_family_promotion_0223_20260331_230326/`  
  Winner: `b7_central_decay_r0p95_floor0p20_vr0p08_vl0p35_dt0p16_cap1p40`  
  `block7_hit@1=3`, `block7_hit@3=3`, `pass_rate=0.467`, `promoted_neighbors=1`  
  Solid, but the mechanism is broader temporal promotion rather than the narrower family continuation we want.

- `results/round10_block7_track_before_select_0223_20260331_225103/`  
  Winner: `control_round9a_ref`  
  `block7_hit@1=2`, `block7_hit@3=2`, `pass_rate=0.467`  
  Track-first did not beat the frozen reference on block7, so it is not the carry-forward path.

- `results/round10_block7_family_plus_track_hybrid_0223_20260331_224906/`  
  Winner: `b7_hybrid_span1_track1_floor0p08`  
  `block7_hit@1=3`, `block7_hit@3=3`, `pass_rate=0.467`, `promoted_cases=0`  
  Better than control on block7, but it mixes in broader tracking and is not the preferred next step.

## Recommendation
Carry forward the `offset selective family hold` lane, specifically the narrow `gate1/span1` variant from `results/round10_block7_offset_selective_family_hold_0223_20260331_225154/`. It is the cleanest continuation of the selective family line and avoids drifting into broad tracking behavior.

## Remaining Blocker
The remaining blocker is `block7 offset=-0.5`. That case still needs to be resolved before promoting the lane further.

## Next Step
Keep the next round focused on narrow selective family continuity around `block7 offset=-0.5`, and do not switch the effort to broad track-first or hybrid tracking strategies.
