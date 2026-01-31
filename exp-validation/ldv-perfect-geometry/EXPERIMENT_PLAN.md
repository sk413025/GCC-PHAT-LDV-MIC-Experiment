# LDV-as-Mic Calibration for TDoA/DoA (Geometry-Driven)

**Branch**: `exp/ldv-perfect-geometry`
**Created**: 2026-01-29
**Updated**: 2026-01-29

---

## 1. Objective

Keep the **array positions fixed** at LEFT=(-0.7, 2.0) and RIGHT=(+0.7, 2.0). Replace **one mic** with a **calibrated LDV-derived signal** and test whether DoA/TDoA improves (especially when replacing the farther mic).

Pipeline overview:
1) Use geometry to synthesize an **ideal mic signal** from LDV for a target side (LEFT or RIGHT).
2) Use **OMP** to learn the lag-based mapping from **actual LDV** to the **ideal mic** (per frequency band).
3) Train **DTmin** to imitate OMP.
4) Compare **OMP vs DTmin vs Random vs Raw LDV vs Mic-Mic** on DoA/TDoA metrics.

We will train **10 DTmin models** total: 5 speaker positions (18-22) ? 2 target sides (LDV->LEFT, LDV->RIGHT).

---

## 2. Geometry (Source of Truth)

Sensor positions (fixed):
- LEFT MIC: **(-0.7, 2.0) m**
- RIGHT MIC: **(+0.7, 2.0) m**
- LDV BOX: **(0.0, 0.5) m**
- Speaker line: **y = 0 m**, positions:
  - (22) x=-0.8, (21) x=-0.4, (20) x=0.0, (19) x=+0.4, (18) x=+0.8

Constants:
- Mic spacing: 1.4 m
- Speed of sound: 343 m/s

Distances (from the GCC-PHAT report) are the validation reference.

---

## 3. Data Sources (Real Only)

Dataset (48 kHz): `dataset/GCC-PHAT-LDV-MIC-Experiment/`
- 18-0.1V, 19-0.1V, 20-0.1V, 21-0.1V, 22-0.1V
- Each contains LDV + LEFT MIC + RIGHT MIC WAVs

No synthetic data. If dataset is missing, **fail fast** and document prerequisites.

STFT configuration (native 48 kHz, full band):
- `fs=48000`, `n_fft=4096`, `hop_length=1024`
- **No bandpass** (full-band processing), aligned with the V1 phase estimator report.

---

## 4. Ideal Mic Synthesis from LDV (Geometry-Driven)

Assume free-field propagation with 1/r attenuation:

- x_ldv(t) = s(t - d_ldv / c) / d_ldv
- x_mic(t) = s(t - d_mic / c) / d_mic

Solve for the ideal mic signal using LDV only:

- x_mic_ideal(t) = (d_ldv / d_mic) * x_ldv(t + (d_ldv - d_mic) / c)

Implementation (sub-sample shift):
- X_mic_ideal(f) = X_ldv(f) * exp(+j * 2*pi*f*delta_t) * (d_ldv / d_mic)
- delta_t = (d_ldv - d_mic) / c

We compute **two ideal targets** per speaker position:
- LDV->LEFT (target mic = LEFT)
- LDV->RIGHT (target mic = RIGHT)

These are the **geometry-optimal** references for OMP/DTmin.

---

## 5. OMP Teacher (LDV -> Ideal Mic)

For each speaker position (18-22) and each target side (LEFT/RIGHT):

- **Input**: LDV STFT windows
- **Target**: Ideal mic STFT windows (from Section 4)
- **Dictionary**: time-lagged LDV windows

OMP outputs lag trajectories + energy reduction metrics.

Constraints:
- No duplicate lags.
- No STFT grid mismatch (F, N, fs, n_fft must match).
- Use Gain=100 where energy ratios are evaluated to avoid epsilon-floor artifacts.

---

## 6. DTmin Student (10 Models)

Train 10 DTmin models:
- 5 speaker positions ? 2 target sides (LDV->LEFT, LDV->RIGHT)

Each model learns to predict the OMP lag sequence for its specific (position, side) dataset.

Evaluation:
- Lag prediction accuracy vs OMP
- Energy capture vs OMP
- Random baseline (same lag budget)

---

## 7. DoA/TDoA Evaluation (Ablation Matrix)

For each speaker position:

**Baselines**
1) MIC-LEFT vs MIC-RIGHT (true mic pair)
2) Raw LDV vs MIC-LEFT
3) Raw LDV vs MIC-RIGHT

**Calibrated LDV replacements**
4) OMP-calibrated LDV as LEFT + MIC-RIGHT
5) DTmin-calibrated LDV as LEFT + MIC-RIGHT
6) Random-calibrated LDV as LEFT + MIC-RIGHT

7) OMP-calibrated LDV as RIGHT + MIC-LEFT
8) DTmin-calibrated LDV as RIGHT + MIC-LEFT
9) Random-calibrated LDV as RIGHT + MIC-LEFT

Methods:
- GCC-PHAT
- CC
- NCC
- MUSIC

Metrics:
- tau_median, tau_std, PSR, Peak
- DoA error vs theoretical angle (from geometry)

Frequency policy:
- **Full-band evaluation** (no bandpass), matching the V1 phase estimator setup.

Hypothesis focus:
- Replacing the **farther mic** with calibrated LDV should improve DoA/TDoA stability.
- Replacing the nearer mic may show smaller gains or regressions.

---

## 8. Acceptance Criteria

1) OMP must outperform Random in energy capture (non-zero delta, meaningful spread).
2) DTmin should be within 5-10% of OMP capture on the same subset.
3) Calibrated LDV should reduce tau_std and/or improve PSR vs raw LDV for the same pairing.
4) For the farther-mic replacement, DoA error should improve or remain stable vs raw LDV.

---

## 9. Output Structure (Mandatory)

All artifacts under `results/<run_name>/`:
- run.log (unbuffered)
- subset_manifest.json + dataset fingerprint
- lag_trajectories.pt (OMP)
- DTmin checkpoints
- comparison summaries
- numeric_diagnostics.jsonl
- code_state.json

---

## 10. Risks / Open Questions

- Free-field 1/r + pure-delay assumption may be violated by room reflections.
- LDV dispersion may require per-band phase compensation beyond simple delay.
- Training 10 separate models may be expensive; consider a conditioned single model if needed.
- Explicit resampling (48k -> 16k) vs native 48k training must be chosen and audited.

---

## 11. Lessons Learned & Plan Revision

### 問題回顧

原計畫直接從幾何模型 → OMP → DTmin → TDoA 評估，跳了太多步驟。當最終結果失敗時（OMP 無法改進 GCC-PHAT PSR），很難定位問題出在哪裡：

- OMP 本身沒學到東西？
- OMP 學到了但 ISTFT 重建有問題？
- 重建正確但 TDoA 評估設定錯誤？

### 新策略：階段性驗證

改採**逐層驗證**策略，每一層都有明確的「通過條件」，失敗時可以精確定位問題。

詳見 `STAGE_VALIDATION_PLAN.md`。

| 階段 | 驗證目標 | 通過條件 |
|------|---------|---------|
| 1 | Energy Reduction | OMP > Random + 10% |
| 2 | Target Similarity | GCC-PHAT τ → 0, PSR ↑ |
| 3 | Cross-Mic TDoA | τ_error < 0.5 ms |

---

## 12. Next Steps (Revised)

1) **階段 1**：實作 `stage1_energy_reduction.py`，單一 segment 快速驗證 OMP > Random
2) **階段 1 通過後**：實作 `stage2_target_similarity.py`，驗證對齊品質
3) **階段 2 通過後**：實作 `stage3_tdoa_evaluation.py`，完整 TDoA 評估
4) 全部通過後，再考慮 DTmin 訓練和多 speaker 擴展

