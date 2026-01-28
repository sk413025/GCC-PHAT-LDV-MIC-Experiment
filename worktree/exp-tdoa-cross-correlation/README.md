# exp-tdoa-cross-correlation

Cross-Correlation methods for TDoA estimation in MIC-LDV speech data.

## Overview

This experiment compares different cross-correlation methods for Time Difference of Arrival (TDoA) estimation:

1. **Standard Cross-Correlation (CC)**: Basic cross-correlation with amplitude weighting
2. **Normalized Cross-Correlation (NCC)**: Energy-normalized, range [-1, +1]
3. **GCC-PHAT**: Phase transform, equal weighting across frequencies (baseline from E4m)

## Quick Start

### Smoke Test (1 pair)

```bash
# Activate environment
conda activate trl-training

# Set paths
ROOT="C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/worktree/exp-tdoa-cross-correlation"
MIC_ROOT="C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/dataset/audio/boy1/MIC"
LDV_ROOT="C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/dataset/audio/boy1/LDV"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run smoke test
python -u "$ROOT/scripts/run_cross_correlation_tdoa.py" \
    --mic_root "$MIC_ROOT" \
    --ldv_root "$LDV_ROOT" \
    --out_dir "$ROOT/results/smoke_$TIMESTAMP" \
    --mode smoke \
    --fs 16000 \
    --hop_length 160 \
    --n_fft 2048 \
    --freq_min 300 \
    --freq_max 3000 \
    2>&1 | tee "$ROOT/results/smoke_$TIMESTAMP/run.log"
```

### Scale Test (48 pairs)

```bash
python -u "$ROOT/scripts/run_cross_correlation_tdoa.py" \
    --mic_root "$MIC_ROOT" \
    --ldv_root "$LDV_ROOT" \
    --out_dir "$ROOT/results/scale_$TIMESTAMP" \
    --mode scale \
    --num_pairs 48 \
    2>&1 | tee "$ROOT/results/scale_$TIMESTAMP/run.log"
```

### Full Dataset (416 pairs)

```bash
python -u "$ROOT/scripts/run_cross_correlation_tdoa.py" \
    --mic_root "$MIC_ROOT" \
    --ldv_root "$LDV_ROOT" \
    --out_dir "$ROOT/results/full_$TIMESTAMP" \
    --mode full \
    2>&1 | tee "$ROOT/results/full_$TIMESTAMP/run.log"
```

## Array Simulation + Beamforming/MUSIC (Simulated)

This path creates a synthetic linear array from mono MIC WAVs, then runs DS
beamforming and MUSIC on the simulated multi-channel audio.

### 1) Simulate array WAVs

```bash
python -u "$ROOT/scripts/simulate_array_dataset.py" \
    --in_dir "$MIC_ROOT" \
    --out_dir "$ROOT/results/array_sim_$TIMESTAMP" \
    --fs 16000 \
    --num_mics 4 \
    --spacing_m 0.035 \
    --angle_mode random \
    --angle_min -60 \
    --angle_max 60 \
    --snr_db 30
```

### 2) Run Beamforming + MUSIC

```bash
python -u "$ROOT/scripts/run_array_methods.py" \
    --array_root "$ROOT/results/array_sim_$TIMESTAMP" \
    --out_dir "$ROOT/results/array_eval_$TIMESTAMP" \
    --fs 16000 \
    --num_mics 4 \
    --spacing_m 0.035 \
    --freq_min 300 \
    --freq_max 3000 \
    --angle_min -90 \
    --angle_max 90 \
    --angle_step 2
```

### New Scripts
- `scripts/simulate_array_dataset.py`
- `scripts/run_array_methods.py`

## Output Files

Each run produces:
- `manifest.json`: Configuration and file list
- `detailed_results.json`: Per-window results for all methods
- `summary.json`: Aggregate statistics
- `run.log`: Console output

## Metrics

### Per-Method
- `tau_ms`: Estimated delay in milliseconds
- `psr`: Peak-to-sidelobe ratio (confidence)

### Cross-Method Comparison
- `cc_vs_gcc_phat.abs_diff_ms`: Absolute difference between CC and GCC-PHAT
- `cc_vs_gcc_phat.correlation`: Correlation of estimates

## Directory Structure

```
exp-tdoa-cross-correlation/
├── README.md                 # This file
├── docs/
│   └── EXPERIMENT_PLAN.md   # Detailed plan
├── scripts/
│   └── run_cross_correlation_tdoa.py
└── results/
    ├── smoke_YYYYMMDD_HHMMSS/
    ├── scale_YYYYMMDD_HHMMSS/
    └── full_YYYYMMDD_HHMMSS/
```

## References

- E4m baseline: `worktree/exp-interspeech-GRU2/results/rtgomp_dispersion_E4m_speech_full_dataset_paired_conda_20260126_191837/`
- GCC-PHAT implementation: `worktree/exp-interspeech-GRU2/scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`

## Dependencies

- numpy
- scipy
- Python 3.10+

## Author

Created: 2026-01-27
