# 2026-03-24 Round 7 Supervisor Memo

## Scope

Round 7 followed the fixed plan:

- lane 7.1: block7-first same-`VR` negative-family persistence
- lane 7.2: block7 temporal family persistence on top of the lane 7.1 winner

Tracked artifact bundles:

- `artifacts/round7_block7_samevr_0223_20260324_231357_report.md`
- `artifacts/round7_block7_samevr_0223_20260324_231357_summary.json`
- `artifacts/round7_block7_temporal_persistence_0223_20260324_232156_report.md`
- `artifacts/round7_block7_temporal_persistence_0223_20260324_232156_summary.json`

## Lane 7.1 Verdict

`USEFUL LOCAL WIN, NOT A LANE PASS`

Best variant:

- `b7_samevr_neg_promote_vr0p20_dt0p15_r0p10`

What it proves:

- the block7 central mechanism is correct
- central `block7` moves to `-0.3125`
- central `block7` rescue rank reaches `1`
- central aggregate improves to `0.039 / 0.043`

What it does not solve:

- `block7 hit@1` stays `1/5`
- `block7 hit@3` stays `2/5`
- setting `pass_rate` stays `0.400`

So lane 7.1 is a mechanism signal, not a persistence recovery.

## Lane 7.2 Verdict

`NO IMPROVEMENT`

The temporal-hold variants do not beat the lane 7.1 winner.

Best result remains the control:

- `control_lane7a_ref`

Measured:

- `block7 hit@1 = 1/5`
- `block7 hit@3 = 2/5`
- `pass_rate = 0.400`

Interpretation:

- the correct block7 negative-family can be found at the center
- but temporal wrapping alone does not make that family persist across nearby
  windows

## Supervisor Conclusion

Current round7 status:

- no round7 lane passes the block7-persistence gate
- lane 7.1 provides a valid central-correction mechanism
- lane 7.2 shows temporal hold is not enough

Best round7 artifact to carry forward:

- `b7_samevr_neg_promote_vr0p20_dt0p15_r0p10`

This should be treated as a block7 central-improvement branch, not as a robust
lane winner.

## Next Step

Round7 should stop block7 temporal wrapping here.

The next justified move is:

- pivot away from block7 temporal hold
- move to block6 persistence next
- reserve the block7 central mechanism for a later hybrid lane
