# Offsets + Sparse Multipath Verification Report

Generated: 2026-02-13T18:53:05.053519

Run dir: `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/offsets_multipath_verify_20260213_185224`

## Offsets (silence-derived, global GCC peak)

| pair | across-speaker mean (ms) | across-speaker std (ms) |
| --- | ---: | ---: |
| micl_micr | -1.650 | 0.000 |
| ldv_micl | 5.210 | 1.666 |
| ldv_micr | 1.950 | 0.000 |

## MIC-MIC guided tau error vs chirp reference (before/after offset subtraction)

- median |tau_guided - tau_ref| before: 0.1478 ms
- median |(tau_guided - tau_offset_silence) - tau_ref| after: 1.5635 ms
- improvement frac: -9.575 (>= 0.20 required by acceptance)

## Acceptance (Claim 3)

- OVERALL: FAIL
- micl_micr_median_abs_tauerr_ref_improve_ge_0p20: False
- offset_std_le_0p3ms_ldv_micl: False
- offset_std_le_0p3ms_ldv_micr: True
- sparse_top1_bin_mass_ge_0p6_any_ldv_pair: True

## Notes

- If the MIC-MIC 'after' error worsens, this suggests the silence-derived mode does not represent a pure channel offset compatible with chirp reference; record as a negative result rather than tuning thresholds.
- Per-window peak lists are stored under `per_speaker/<speaker>/windows.jsonl`.
