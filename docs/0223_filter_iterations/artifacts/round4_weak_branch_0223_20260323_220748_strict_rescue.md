# Round 4 Weak-Branch Rescue Strict Rescue Check

Strict rescue targets on the central window:

- `block6_n04_19`: `VL ~= 4.854 ms`, `VR ~= 5.438 ms`, `delta_tau ~= +0.583 ms`
- `block7_n08_20`: `VL ~= 4.833 ms`, `VR ~= 4.521 ms`, `delta_tau ~= -0.312 ms`

Branch-wise tolerance:

- `+/- 0.20 ms`

## Central Strict Rescue Ranks

| variant | case | rescue_vl_rank | rescue_vr_rank | rescue_pair_rank | selected_delta_tau_ms |
| --- | --- | ---: | ---: | ---: | ---: |
| base_only | block6_n04_19 | 1 | 3 | 7 | 0.271 |
| weak_local_zscore_mic | block6_n04_19 | 1 | 3 | 7 | 0.271 |
| weak_clip_rms_mic | block6_n04_19 | 1 | 3 | 7 | 0.271 |
| weak_diff2_union | block6_n04_19 | 1 | 3 | 7 | 0.271 |
| weak_residual_second_pass | block6_n04_19 | 1 | 3 | 7 | 0.271 |
| weak_bundle | block6_n04_19 | 1 | 3 | 7 | 0.271 |
| base_only | block7_n08_20 | 3 | 1 | 4 | -0.042 |
| weak_local_zscore_mic | block7_n08_20 | 3 | 1 | 4 | -0.042 |
| weak_clip_rms_mic | block7_n08_20 | 3 | 1 | 4 | -0.042 |
| weak_diff2_union | block7_n08_20 | 3 | 1 | 4 | -0.042 |
| weak_residual_second_pass | block7_n08_20 | 3 | 1 | 4 | -0.042 |
| weak_bundle | block7_n08_20 | 3 | 1 | 4 | -0.042 |

## Reading

The rescue branches are already present:

- `block6` rescue `VR` is single-side rank `3`
- `block7` rescue `VL` is single-side rank `3`

But the rescue pair still loses:

- `block6` rescue pair rank stays at `7`
- `block7` rescue pair rank stays at `4`

This is the decisive negative finding for lane 4.2:

- weak-branch rescue families did not improve candidate visibility
- and they also did not improve pair competitiveness
