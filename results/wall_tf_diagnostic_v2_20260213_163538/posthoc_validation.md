# Post-hoc validation (auto-generated)

**Run dir**: `results/wall_tf_diagnostic_v2_20260213_163538/`  
**Generated**: 2026-02-13T16:38:37.228971  
**Purpose**: Sanity-check narrowband sweep outputs and summarize wideband GCC behavior.

## Reproduction

```bash
python -u scripts/diagnose_wall_transfer_function.py \
  --data_root /home/sbplab/jiawei/data \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V \
  --segment_source /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/stage4_speech_chirp_tau2_band0_20260211_072624 \
  --output_dir /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/wall_tf_diagnostic_v2
```

## 1) Narrowband sweep sanity

- Total band estimates: **11400**
- `psr_db` NaN: **0**
- `|tau_ms| == MAX_LAG_MS (7.0 ms)`: **236**
- `|tau_ms| < 0.3 ms`: **429**

Median narrowband PSR: **0.054 dB**

## 2) Pair-to-pair τ(f) correlation (median)

- `LDV_MicL__vs__MicL_MicR`: **0.008**
- `LDV_MicL__vs__LDV_MicR`: **0.290**
- `LDV_MicR__vs__MicL_MicR`: **0.141**

## 3) Wideband GCC-PHAT τ (500–2000 Hz) median

- `LDV_MicL`: **+3.851 ms**
- `LDV_MicR`: **+1.965 ms**
- `MicL_MicR`: **-1.690 ms**

## Note

Per-band τ(f) estimates are typically low-confidence when PSR is near 0 dB. Interpret τ physically only after accounting for constant per-channel offsets.
