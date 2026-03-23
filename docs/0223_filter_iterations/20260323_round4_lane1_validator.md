# 2026-03-23 Round 4 Lane 1 Validator Note

## Validation Scope

Validated artifact bundle:

- `artifacts/round4_candidate_generation_0223_20260323_220158_report.md`
- `artifacts/round4_candidate_generation_0223_20260323_220158_summary.json`

Validation rule for this lane:

- candidate pool must improve rescue availability first
- then that improvement must survive the fixed downstream scorer

## Reference

Reference pool:

- `base_only`

Reference central metrics:

- `delta_tau_mae_ms = 0.142`
- hard-case `delta_tau_mae_ms = 0.249`
- central `max_delta_tau_abs_err_ms = 0.254`

Reference rescue ranks:

- `block6` rescue `VR` rank `= 7`, pair rank `= 7`
- `block7` rescue `VL` rank `= 4`, pair rank `= 4`

## Validation Result

No family passed the lane-local rescue test.

The best deployment tie was:

- `base_plus_diff2_500_2000`

But it did not improve the rescue structure:

- `block6` rescue `VR` rank worsened from `7` to `9`
- `block7` rescue `VL` rank stayed at `4`

This means the tie on average metrics is not meaningful for promotion.

## Validator Judgment

Verdict:

- `REJECT ALL CANDIDATE-GENERATION FAMILIES FOR PROMOTION`

Reason:

- hard-case rescue ranks did not improve
- hard-case deployment error did not improve
- several families worsened central deployment without creating rescue gains

## Important Negative Finding

The broad proposal families failed in exactly the way the validator wanted to
catch:

- they changed the pool
- but they did not improve rescue `rank / hit@K`
- therefore they are closer to noisy candidate inflation than true branch
  repair

The most direct example is:

- `base_plus_diff2_500_2000`
- average deployment ties the control
- but rescue availability in `block6` gets worse

That is not a pass.

## Validator Conclusion

Round 4 lane 1 is a valid negative result.

It narrows the path:

- global candidate union is not enough
- next lane must use weak-branch-specific preprocessing or rescue logic

Without improving rescue ranks on the weak branch itself, candidate-generation
changes should not be promoted.
