# Experiment Plan: TDoA via Cross-Correlation

> Created: 2026-01-27
> Branch: exp-tdoa-cross-correlation
> Status: Planning

---

## 1. Background and Motivation

### 1.1 Current State (E4m/E4o)

From previous experiments (E4m, E4o):
- GCC-PHAT provides stable delay estimates (within_clip_tau_mad ~0.01 ms)
- Phase-slope diagnostics reveal dispersion (tau_band_spread ~10 ms)
- E4o's output-driven phase equalization reduces tau_band_spread by 30%
- **BUT**: We still lack geometry-grounded validation (does estimated tau correspond to physical distance?)

### 1.2 Why Cross-Correlation?

| Method | Pros | Cons |
|--------|------|------|
| **GCC-PHAT** | Sharp peaks, robust to noise | Phase Transform may lose amplitude info |
| **Standard Cross-Correlation** | Preserves amplitude weighting | Peaks may be broader |
| **Generalized CC (GCC)** | Flexible weighting | Need to choose weighting function |

**Goal**: Compare Cross-Correlation variants against GCC-PHAT to understand:
1. Do they give similar tau estimates?
2. Which is more robust under different SNR conditions?
3. Can we get better absolute accuracy for geometry validation?

---

## 2. Research Questions

### RQ1: Consistency
Do Cross-Correlation and GCC-PHAT produce consistent tau estimates on paired MIC-LDV data?

### RQ2: Stability
Which method has lower within-clip variance (tau_mad)?

### RQ3: Dispersion Sensitivity
Does Cross-Correlation show different dispersion characteristics than GCC-PHAT?

### RQ4: Geometry Grounding (if calibration data available)
Can either method produce tau that corresponds to known physical distances?

### RQ5: Array-Based Methods (if array data available)
Do beamforming/MUSIC provide more stable or accurate DoA/TDoA than GCC-based methods?

---

## 3. Methods

### 3.1 Standard Cross-Correlation

```
R_xy[tau] = sum_n { x[n] * y[n + tau] }

Frequency domain (faster):
R_xy = IFFT( conj(FFT(x)) * FFT(y) )
```

No normalization, amplitude-weighted.

### 3.2 Normalized Cross-Correlation (NCC)

```
NCC[tau] = R_xy[tau] / sqrt(R_xx[0] * R_yy[0])

Range: [-1, +1]
```

Energy-normalized, useful for comparing across clips.

### 3.3 GCC-PHAT (Baseline)

```
R_phat = IFFT( conj(X) * Y / |conj(X) * Y| )
```

Phase-only, equal weighting across frequencies.

### 3.4 GCC with ML Weighting (Optional)

```
R_ml = IFFT( W(f) * conj(X) * Y )

W(f) = coherence-based weighting
```

### 3.5 Beamforming (Array-based)

Beamforming converts multichannel array data to a spatial scan and can be used
to estimate direction-of-arrival (DoA) or infer TDoA via steering delays.

**Delay-and-Sum (DS):**
```
Y(theta) = sum_m x_m(t + tau_m(theta))
```
Scan theta to maximize output energy or peak power.

**MVDR/Capon (optional):**
```
P_MVDR(theta) = 1 / (a(theta)^H R_xx^{-1} a(theta))
```

Notes:
- Requires array geometry and synchronized channels.
- With only 2 sensors, DS beamforming reduces to a TDoA scan and is close to GCC.

### 3.6 MUSIC (Array-based)

MUSIC estimates DoA by separating signal/noise subspaces of the spatial covariance.

```
P_MUSIC(theta) = 1 / (a(theta)^H E_n E_n^H a(theta))
```

Notes:
- Requires M >= 2 sensors and known steering vector a(theta).
- Wideband MUSIC: apply per-band and aggregate (e.g., incoherent sum).

### 3.7 Data Requirements and Variants

Beamforming/MUSIC require multichannel array data with known geometry. The
current MIC-LDV dataset appears to be single-channel per file, so array methods
will need one of:
- A separate multi-mic dataset with known sensor positions, or
- A simulated array (synthetic multi-channel using room impulse responses),
- Or a special recording where MIC has multiple synchronized channels.

---

## 4. Dataset

### 4.1 Primary Dataset
- Same as E4m: boy1 speech WAV (416 pairs)
- mic_root: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\dataset\audio\boy1\MIC`
- ldv_root: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\dataset\audio\boy1\LDV`

### 4.2 Parameters
- fs = 16000 Hz
- hop_length = 160 samples (10 ms)
- n_fft = 2048
- freq_band = [300, 3000] Hz

### 4.3 Array Dataset (Required for Beamforming/MUSIC)
- Number of sensors (M), geometry (positions), and spacing (d)
- Synchronization and channel alignment
- Optional calibration: array orientation and coordinate frame

### 4.4 Simulation Protocol (If No Array Data)
- Linear array, M=4, spacing=3.5 cm (baseline)
- DOA angles: uniform random in [-60, 60] deg or grid step 5 deg
- Additive noise: SNR=30 dB (sweep 10/20/30 dB optional)
- Optional sensor gain jitter: 0-1 dB per channel

---

## 5. Evaluation Metrics

### 5.1 Delay Estimation Metrics
- `tau_hat_ms`: Estimated delay in milliseconds
- `within_clip_tau_mad_ms`: Median absolute deviation of tau within same clip
- `psr` or `peak_prominence`: Peak quality measure

### 5.2 Dispersion Metrics (from E4m)
- `tau_band_spread_ms`: max(tau_by_band) - min(tau_by_band)
- `phase_slope_r2`: Linearity of phase vs frequency

### 5.3 Cross-Method Comparison
- `tau_cc_vs_gcc_agreement_ms`: |tau_cc - tau_gcc|
- `correlation_of_estimates`: corr(tau_cc, tau_gcc) across windows

### 5.4 Array/DoA Metrics (If Array Data Available)
- `doa_error_deg`: |theta_hat - theta_gt| if geometry + ground truth exists
- `beamwidth_deg`: mainlobe width at -3 dB
- `spatial_psr`: peak-to-sidelobe ratio in spatial spectrum
- `snr_gain_db`: beamformer output SNR improvement

---

## 6. Experiment Phases

### Phase 0: Array Data Audit
- Verify whether multichannel MIC data exists
- Confirm array geometry and synchronization

### Phase 1: Smoke Test
- 1 pair, verify cross-correlation implementation
- Compare CC vs GCC-PHAT on same window
- Output: basic metrics, visualization

### Phase 2: Scale Test (48 pairs)
- Run on scale_check_subset
- Compare CC, NCC, GCC-PHAT
- Output: summary statistics, comparison table

### Phase 3: Full Dataset (416 pairs)
- Run on full dataset
- Comprehensive comparison
- Output: final report

### Phase 4: Array Methods (Beamforming/MUSIC)
- Run on array dataset or simulated array
- Compare DS/MVDR/MUSIC vs GCC-based TDoA (if comparable)
- Output: spatial spectra, DoA error, robustness vs noise

---

## 7. Expected Outcomes

### Hypothesis 1
Cross-Correlation and GCC-PHAT will give similar tau_hat (within ~1 ms) because they both find the same peak, just with different sharpness.

### Hypothesis 2
GCC-PHAT will have higher PSR (sharper peaks) due to phase transform.

### Hypothesis 3
Both methods will show similar dispersion (tau_band_spread ~10 ms) because dispersion is a physical property, not an algorithm artifact.

### Hypothesis 4 (Array Methods)
If array data is available, MUSIC should provide sharper spatial peaks than DS,
and MVDR should show better interference suppression than DS.

---

## 8. Success Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| CC implementation correctness | tau_cc matches scipy reference | Sanity check |
| CC vs GCC-PHAT agreement | mean(|tau_cc - tau_gcc|) < 1 ms | Methods should be consistent |
| Within-clip stability | tau_mad < 0.1 ms | Similar to E4m GCC-PHAT |

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| CC peaks too broad to localize | Use parabolic interpolation |
| Numerical issues at high lags | Limit search range to ±50 frames |
| Results differ from GCC-PHAT | Document and analyze differences |
| No multichannel array data | Use simulated array or postpone array methods |

---

## 10. File Structure

```
exp-tdoa-cross-correlation/
├── docs/
│   ├── EXPERIMENT_PLAN.md          # This file
│   └── RESULTS_REPORT.md           # To be created after experiments
├── scripts/
│   └── run_cross_correlation_tdoa.py
└── results/
    ├── smoke_<timestamp>/
    ├── scale_<timestamp>/
    └── full_<timestamp>/
```

---

## 11. Timeline

| Phase | Target |
|-------|--------|
| Implementation | Day 1 |
| Smoke Test | Day 1 |
| Scale Test | Day 2 |
| Full Dataset | Day 2-3 |
| Analysis & Report | Day 3-4 |

---

## 12. References

- E4m Results: `worktree/exp-interspeech-GRU2/results/rtgomp_dispersion_E4m_speech_full_dataset_paired_conda_20260126_191837/`
- GCC-PHAT Implementation: `worktree/exp-interspeech-GRU2/scripts/h_exploration/run_rtgomp_e4h_paper_eval.py:64-136`
