# 2026-03-23 Round 4 Lane 1 Supervisor Memo

## Scope

Round 4 lane 1 tested candidate-generation changes only.

Fixed components:

- base front-end: `diff_len80_ldvonly_bp700_1800`
- downstream scorer: `len80_current_score`

Search target:

- improve hard-case rescue-pair availability before pair scoring

Tracked artifact bundle:

- `artifacts/round4_candidate_generation_0223_20260323_220158_report.md`
- `artifacts/round4_candidate_generation_0223_20260323_220158_summary.json`

## Reference

Reference candidate pool:

- `base_only`

Reference deployment metrics:

- central `delta_tau_mae_ms = 0.142`
- hard-case `delta_tau_mae_ms = 0.249`
- central `max_delta_tau_abs_err_ms = 0.254`
- window `delta_tau_mae_ms = 0.207`

Reference rescue ranks on central window:

- `block6`: rescue `VR` rank `= 7`, rescue pair rank `= 7`
- `block7`: rescue `VL` rank `= 4`, rescue pair rank `= 4`

## Lane 4 Result

No candidate-generation family improved deployment metrics.

Best result only tied the control:

- `base_plus_diff2_500_2000`
- central `delta_tau_mae_ms = 0.142`
- hard-case `delta_tau_mae_ms = 0.249`

Other proposal families degraded central deployment:

- `base_plus_diff2_700_1800`
- `base_plus_mic_flatten_700_1800`
- `base_plus_diff2_700_1800_and_mic_flatten`
- `base_plus_oracle_bundle`

## Hard-Case Rescue Check

The decisive result is not the average.
It is the rescue rank.

Observed central-window rescue ranks:

- `block6`
  - control rescue `VR` rank: `7`
  - `diff2_500_2000` rescue `VR` rank: `9`
- `block7`
  - control rescue `VL` rank: `4`
  - `diff2_500_2000` rescue `VL` rank: `4`

Supervisor interpretation:

- the tested proposal families did not pull the weak branch upward
- some even pushed the rescue branch farther down
- this means the current proposal strategy is still too global and not weak-side
  specific enough

## Supervisor Judgment

Decision:

- accept the lane as a negative result
- do not promote any candidate-generation family

Reason:

- no improvement in hard-case rescue rank
- no improvement in hard-case deployment error
- several families worsen central deployment while keeping the same rescue
  structure

## Main Lesson

Teacher-style proposal sources are not automatically useful as candidate
generators.

In particular:

- `diff2_500_2000` can be neutral as a proposal source
- `diff2_700_1800` and flatten-style proposals are too blunt when injected
  directly into the candidate pool

This strongly suggests that the next lane should move from:

- global candidate union

to:

- weak-branch-specific preprocessing or weak-branch candidate rescue

## Next Supervisor Instruction

The next lane should focus on branch-specific rescue, not broader proposal
bundles.

Priority direction:

- target the weak side directly before candidate extraction
- especially the `VR` branch in `block6` and the `VL` branch in `block7`

Most likely next experiments:

- weak-side local normalization
- weak-side transient gating
- weak-side branch-only cleanup before GCC candidate extraction
