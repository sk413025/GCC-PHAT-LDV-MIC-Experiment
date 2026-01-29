# TDOA Methods Validation - Ablation Study

**Branch**: `exp/tdoa-methods-validation`
**Created**: 2026-01-29
**Updated**: 2026-01-29

---

## 1. Objective

Perform an **ablation study** comparing three delay compensation methods on GCC-PHAT LDV-MIC experiment data:

| Method | Description | Training Required |
|--------|-------------|-------------------|
| **A: Baseline** | No compensation, raw signals | No |
| **B: Direct Phase Estimation** | Per-frequency phase difference calculation | No |
| **C: DTmin (Resampled)** | Decision Transformer with 48kHz→16kHz resampling | Uses pre-trained model |

---

## 2. Data Sources

### 2.1 Experiment Data (48kHz)

**Location**: `dataset/GCC-PHAT-LDV-MIC-Experiment/`

| Folder | Content | Speaker Position | Signal Type |
|--------|---------|------------------|-------------|
| 03-0.3V | MIC-MIC only | +0.8m | Chirp |
| 04-0.1V | MIC-MIC only | +0.8m | Chirp |
| 05-0.01V | MIC-MIC only | +0.8m | Chirp |
| 06-0.001V | MIC-MIC only | +0.8m | Chirp |
| 07-0.005V | MIC-MIC only | +0.8m | Chirp |
| **18-0.1V** | LDV + 2 MICs | +0.8m | Speech |
| **19-0.1V** | LDV + 2 MICs | +0.4m | Speech |
| **20-0.1V** | LDV + 2 MICs | 0.0m | Speech |
| **21-0.1V** | LDV + 2 MICs | -0.4m | Speech |
| **22-0.1V** | LDV + 2 MICs | -0.8m | Speech |

**Focus**: Folders 18-22 (have LDV + MIC data for all three methods)

### 2.2 Pre-trained DTmin Model (16kHz)

**Location**: `worktree/exp-interspeech-GRU2/worktrees/exp-h-exploration-speech/results/exp_h_full/`

| File | Size | Description |
|------|------|-------------|
| `model.pt` | 2.2 MB | Trained Decision Transformer |
| `omp_trajectories.pt` | 53.5 MB | Training trajectories |

---

## 3. Method Details

### 3.1 Method A: Baseline (No Compensation)

```python
# Direct GCC-PHAT on raw signals
X_mic = stft(mic_signal)
X_ldv = stft(ldv_signal)
tau = gcc_phat(X_mic, X_ldv)
```

### 3.2 Method B: Direct Phase Estimation

```python
# Per-frequency phase difference estimation
X_mic = stft(mic_signal)
X_ldv = stft(ldv_signal)

for f in frequencies:
    phase_diff = np.angle(np.conj(X_mic[f]) * X_ldv[f])
    tau_f[f] = -phase_diff / (2 * np.pi * f)

# Apply phase compensation
X_ldv_comp = X_ldv * np.exp(1j * 2 * np.pi * freqs * tau_f)
tau = gcc_phat(X_mic, X_ldv_comp)
```

### 3.3 Method C: DTmin (Resampled)

```python
# Resample 48kHz -> 16kHz
mic_16k = resample(mic_signal, 48000, 16000)
ldv_16k = resample(ldv_signal, 48000, 16000)

# STFT at 16kHz
X_mic = stft(mic_16k, n_fft=2048, hop_length=256)
X_ldv = stft(ldv_16k, n_fft=2048, hop_length=256)

# Use pre-trained DTmin model to predict lags
model = load_model("model.pt")
predicted_lags = model.predict(X_mic, X_ldv)

# Apply phase compensation based on predicted lags
X_ldv_comp = apply_lag_compensation(X_ldv, predicted_lags)
tau = gcc_phat(X_mic, X_ldv_comp)

# Convert tau back to 48kHz time scale if needed
```

---

## 4. Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **TDOA Error** | \|τ_measured - τ_theory\| | Absolute accuracy |
| **Relative Error** | 100 × error / \|τ_theory\| | Percentage accuracy |
| **PSR** | 20·log₁₀(peak / max(sidelobes)) | Peak quality (dB) |
| **tau_std** | std(τ estimates across frames) | Estimation stability |
| **Peak** | max(GCC-PHAT correlation) | Correlation strength |

---

## 5. Experiment Protocol

### Phase 1: Data Preparation

1. Load WAV files from folders 18-22
2. For Method C: Resample to 16kHz
3. Compute STFT for all methods

### Phase 2: Run Three Methods

For each folder (18-22) and each sensor pair (MIC-MIC, LDV-LEFT, LDV-RIGHT):

```
┌─────────────────────────────────────────────────────────────┐
│  Input: MIC WAV, LDV WAV                                    │
│                                                             │
│  ┌─────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │Baseline │  │Direct Phase Est.│  │DTmin (Resampled)    │ │
│  │   (A)   │  │      (B)        │  │       (C)           │ │
│  └────┬────┘  └────────┬────────┘  └──────────┬──────────┘ │
│       │                │                      │            │
│       ▼                ▼                      ▼            │
│    GCC-PHAT        GCC-PHAT              GCC-PHAT          │
│    MUSIC           MUSIC                 MUSIC             │
│       │                │                      │            │
│       └────────────────┼──────────────────────┘            │
│                        ▼                                   │
│              Compute metrics for each                      │
└─────────────────────────────────────────────────────────────┘
```

### Phase 3: Analysis

1. Compare metrics across methods A, B, C
2. Analyze by speaker position
3. Analyze by sensor pair (MIC-MIC vs LDV-MIC)

---

## 6. Expected Outcomes

| Comparison | Expected Result | Rationale |
|------------|-----------------|-----------|
| B vs A | B better | Phase compensation reduces dispersion |
| C vs A | C better | DTmin learns optimal lag selection |
| C vs B | C ≥ B | DTmin should be at least as good |
| C on 48kHz data | May degrade | Domain shift from 16kHz training |

---

## 7. File Structure

```
worktree/exp-tdoa-methods-validation/
├── EXPERIMENT_PLAN.md              # This file
├── scripts/
│   ├── run_ablation_study.py       # Main experiment script
│   ├── method_baseline.py          # Method A implementation
│   ├── method_direct_phase.py      # Method B implementation
│   ├── method_dtmin.py             # Method C implementation
│   └── utils/
│       ├── audio_io.py             # WAV loading, resampling
│       ├── stft.py                 # STFT computation
│       ├── gcc_phat.py             # GCC-PHAT implementation
│       └── metrics.py              # PSR, tau_std, etc.
├── results/
│   └── ablation_study/
│       ├── results_baseline.json
│       ├── results_direct_phase.json
│       ├── results_dtmin.json
│       └── comparison_summary.json
└── docs/
    └── ABLATION_STUDY_REPORT.md    # Final report
```

---

## 8. Resource Paths

### Input Data
```
dataset/GCC-PHAT-LDV-MIC-Experiment/
├── 18-0.1V/
│   ├── 0128-LDV-18-boy-320.wav
│   ├── 0128-LEFT-MIC-18-boy-320.wav
│   └── 0128-RIGHT-MIC-18-boy-320.wav
├── 19-0.1V/
│   └── ...
├── 20-0.1V/
│   └── ...
├── 21-0.1V/
│   └── ...
└── 22-0.1V/
    └── ...
```

### Pre-trained Model
```
worktree/exp-interspeech-GRU2/worktrees/exp-h-exploration-speech/results/exp_h_full/model.pt
```

---

## 9. Theoretical TDOA Reference

From original experiment report:

| Speaker X | MIC-MIC τ (ms) | LDV-LEFT τ (ms) | LDV-RIGHT τ (ms) |
|-----------|---------------|-----------------|------------------|
| +0.8m (18) | +1.450 | -4.538 | -3.088 |
| +0.4m (19) | +0.759 | -4.788 | -4.029 |
| 0.0m (20) | 0.000 | -4.720 | -4.720 |
| -0.4m (21) | -0.759 | -4.029 | -4.788 |
| -0.8m (22) | -1.450 | -3.088 | -4.538 |

**Note**: LDV has ~4.5ms equipment delay that needs calibration.

---

## 10. Next Steps

- [ ] User approval of this plan
- [ ] Implement `run_ablation_study.py`
- [ ] Run experiments on folders 18-22
- [ ] Generate comparison report

---

*Plan updated: 2026-01-29*
