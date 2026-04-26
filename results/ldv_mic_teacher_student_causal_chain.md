# Causal Chain Memo: From MSNF τ₂ Failure → Wall-TF Diagnostics → A Teacher→Student Path That Can Beat MIC–MIC on Speech Tails

**Generated**: 2026-02-13  
**Repo / branch**: `exp-tdoa-cross-correlation`  
**Primary evidence**: `results/wall_tf_diagnostic_v2_20260213_163538/`  
**Primary goal (current)**: Make an LDV-assisted method beat the MIC–MIC baseline **on speech tail robustness** (p95 / failure-rate), while keeping a physically interpretable teacher→student decomposition.

This memo records (1) what we originally wanted, (2) what we observed, (3) what hypotheses were tested and falsified, and (4) why the “fixed offset + a few dominant paths (sparse multipath)” model is consistent with the teacher→student plan for speech tails.

---

## 0) What we wanted (ideal state)

The original MSNF design intent was:

- Use **MicL–MicR** to measure `τ₁` (a robust TDOA constraint).
- Use **LDV–MicL** / **LDV–MicR** to measure `τ₂` / `τ₃` (additional geometric constraints).
- Fuse multiple constraints with WLS (MSNF-D/E/F) to reduce angle-error variance vs MIC–MIC (Method A).

In the ideal state, the measured `τ₂`/`τ₃` would:

1) have a clear, high-PSR peak near the geometry-predicted delay (or at least vary with speaker position as geometry predicts), and  
2) add Fisher information rather than being down-weighted to ~0.

---

## 1) What actually happened (Phase 0 failure mode)

Empirically, the LDV↔MIC delay estimates did **not** behave like geometry-driven direct-path TDOA:

- Guided searches near the geometry `τ₂` often had **no usable peak** (very low or negative PSR).
- A strong peak could appear at an “implausible” delay (e.g., near 0 in early analyses), which the pre-alignment step would lock onto.
- MSNF’s PSR→weight guardrail then correctly suppressed the unreliable LDV constraint, causing MSNF to **degenerate back to MIC–MIC** (no harm, but no gain).

So the immediate problem was not “optimization”; it was “the τ₂ observable is not there (or not dominant) under the current measurement chain.”

---

## 2) Hypothesis A (plausible, but now secondary): dense wall-panel modal phase flips

### A.1 Why it was plausible
If LDV measures **surface velocity** and a microphone measures **pressure**, the relationship is not guaranteed to be a pure delay. A frequency-dependent structural transfer function can add strong phase terms, and GCC-PHAT can fail if phase is not approximately linear in frequency.

### A.2 Why it is no longer the working root cause
The wall-transfer-function diagnostics (v1 + v2) do **not** show the signature of “dense resonant phase flips in 500–2000 Hz” being the dominant failure mechanism:

- Phase vs frequency is **highly linear** in 500–2000 Hz (R² ≈ 0.95–0.97).
- The number of phase jumps in-band is **small** (typically ~5–7, not dozens).
- A narrowband sweep almost never recovers the geometry-expected `τ₂` (fraction correct ~1% for LDV–MicL in v2).

This is inconsistent with a “residual phase nonlinearity compensation should recover τ₂” story.

References:
- v2 report: `results/wall_tf_diagnostic_v2_20260213_163538/diagnostic_report.md`
- v2 post-hoc: `results/wall_tf_diagnostic_v2_20260213_163538/posthoc_validation.md`

---

## 3) What diagnostics *did* confirm (and why it matters)

### 3.1 Narrowband sweep is numerically sane (v2)
The v2 post-hoc validation shows the sweep is no longer dominated by numerical artifacts:

- No NaN PSR values.
- Only a small fraction of estimates are boundary-pinned at max lag.
- “Near-zero” τ is no longer ubiquitous.

This means the narrowband sweep is usable as a **diagnostic instrument**.

Reference: `results/wall_tf_diagnostic_v2_20260213_163538/posthoc_validation.md`

### 3.2 But narrowband confidence is still very low
Even when stable, most narrow bands have extremely low PSR (median around ~0.05 dB in v2). Therefore:

- Per-band `τ(f)` should be treated as **low-confidence** unless gated by PSR (or replaced with a higher-SNR estimator).
- “Recoverability in narrow bands” is not currently a reliable path to recover `τ₂` for MSNF.

### 3.3 Wideband GCC yields stable *constant offsets* (key observation)
In v2 post-hoc wideband GCC-PHAT (500–2000 Hz), the dominant peak delay is:

- **not near zero**, and
- **very stable across speakers / segments**, e.g.:
  - LDV–MicL: ~ +3.85 ms
  - LDV–MicR: ~ +1.96 ms
  - MicL–MicR: ~ −1.69 ms

If the dominant peak were the geometric direct-path TDOA, it should vary with speaker position.

The best interpretation is:
> the observed GCC peak is dominated by **fixed channel/system delays** and/or a **stable non-LOS path**, not the geometry-dependent direct-path TDOA.

Reference: `results/wall_tf_diagnostic_v2_20260213_163538/posthoc_validation.md`

---

## 4) Minimal physics model that matches the observations

The simplest model consistent with the above is:

1) Each channel has a **constant time offset** (DAQ / driver / channel chain), and  
2) the measured signals are dominated by a **small number of paths** (direct path + a few strong reflections / structure-borne paths), i.e. sparse multipath.

In time domain, for a pair:

`y(t) ≈ delay_offset + Σ_k a_k x(t − τ_k) + noise`

Consequences:

- If a stable path dominates (or if constant offsets dominate), GCC produces a stable peak that does not move with speaker geometry.
- The true geometry-dependent direct path can become “non-observable” under GCC-PHAT if it is weaker than the stable path in the analysis band.

This is exactly the kind of model that is naturally connected to sparse convolution / multipath inference.

---

## 5) Truth definitions: why we report both “geometry truth” and “mic-reference truth”

Given constant offsets and stable non-LOS dominance, **geometry truth is not guaranteed to equal the dominant measured delay**.

Therefore, we separate two truths:

### (A) Geometry truth
Computed from the known rig geometry + speed of sound.
- Useful to evaluate “are we actually measuring direct-path physics?”
- Can be misleading as a calibration target if channel offsets are not removed.

### (B) Mic-reference truth (chirp-based)
Estimated from chirp calibration (session-calibrated reference).
- Speech-independent.
- Captures constant offsets and the dominant system path as seen by the signal chain.
- Appropriate as the *primary reference* when the goal is “robust speech performance under this measurement chain.”

In this repo, the chirp truth-reference is stored per speaker as:
- `truth_reference.tau_ref_ms`
- `truth_reference.theta_ref_deg`

Example path:
- `results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000/<speaker>/summary.json`

---

## 6) Which story maps cleanly to OMP (physics → algorithm)

### 6.1 OMP matches sparse multipath (few dominant delays)
OMP is naturally justified when the relationship can be approximated by a small number of delayed atoms:

`y(t) ≈ Σ_{k=1..K} a_k x(t − τ_k)`

This corresponds to:
- a small set of physical paths (direct + a few reflections / couplings),
- plus (optionally) a constant channel offset handled by calibration.

### 6.2 Wall-modal transfer-function equalization is *not* an OMP-friendly derivation
A dense, frequency-dependent structural transfer function is not sparse in time; it tends to be long-memory / high-order. You can model it, but it is not the clean “few atoms” story that OMP embodies.

Therefore:
- If the goal is “derive an OMP-like decision process from physics,” the **fixed offset + sparse multipath** framing is the more consistent choice.

---

## 7) Is “fixed offset + sparse multipath” consistent with “teacher→student wins on speech tails”?

Yes—because the teacher→student plan does **not** require τ₂ to be directly measurable.

Speech tail failures typically happen when MIC–MIC GCC:
- has weak direct-path evidence (low SNR / low coherence in-band), and
- latches onto an unstable or wrong peak.

Under the sparse-multipath + offset model, the teacher’s job is:
- not to “force τ₂ into existence,”
- but to **select / weight the frequency content that is reliable for MIC–MIC τ₁** in hard windows.

LDV is still useful because it can serve as a near-field source reference for:
- “source-present” detection in frequency (a mask),
- and a principled way to downweight frequency regions that are dominated by coupling / noise.

That is exactly the rationale behind the implemented Path B scripts:

- Teacher: `scripts/ldv_informed_gcc_micmic.py`
  - Builds a Wiener-like frequency weight `W(f)` using LDV PSD and silence-estimated noise PSD.
  - Applies `W(f)` to MIC–MIC GCC-PHAT and evaluates vs chirp-reference truth on speech windows.
- Student: `scripts/train_student_ldv_weight_mask.py`
  - Learns a simplified bandmask predictor so the inference pipeline becomes easy to learn (DTmin-friendly).

This keeps the teacher “physics-shaped” (mask from PSD + noise) and gives the student a low-complexity target.

---

## 8) What “good” looks like next (decision-complete)

### 8.1 For the tail-robustness goal (primary)
Declare “LDV+MIC beats MIC–MIC” on speech if, pooled across speakers/windows:
- `p95(theta_error_ref_deg)` improves by ≥ 15%, and
- `fail_rate_ref(theta_error_ref_deg > 5°)` improves by ≥ 20%,
with the guardrail that median error does not worsen by more than +5%.

### 8.2 For the original MSNF τ₂ goal (secondary / later)
Do not interpret measured τ as physical direct-path TDOA until:
- constant channel offsets are estimated and removed, and
- residual τ varies with speaker position according to geometry.

Hardware isolation A/B is still the most likely way to make τ₂ observable later.

---

## 9) Reproduction commands (as of 2026-02-13)

### 9.1 Wall-TF diagnostic v2
```bash
python -u scripts/diagnose_wall_transfer_function.py \
  --data_root /home/sbplab/jiawei/data \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V \
  --segment_source results/stage4_speech_chirp_tau2_band0_20260211_072624 \
  --output_dir results/wall_tf_diagnostic_v2
```

### 9.2 Teacher: LDV-informed MIC–MIC weighting (speech tail eval)
```bash
python -u scripts/ldv_informed_gcc_micmic.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/ldv_informed_micmic_20260213_<ts>
```

### 9.3 Student: learn the teacher bandmask and evaluate (held-out time windows)
```bash
python -u scripts/train_student_ldv_weight_mask.py \
  --teacher_dataset results/ldv_informed_micmic_20260213_<ts>/teacher_dataset.npz \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/ldv_weight_student_20260213_<ts>
```

---

## 10) Bottom line

The wall-modal “dense residual phase” mechanism was a plausible explanation for τ₂ failure, but the v1/v2 transfer-function diagnostics show phase is mostly linear and narrowband recoverability of geometry τ₂ is ~1%, so residual-phase compensation is unlikely to unlock τ₂ without hardware changes. The v2 post-hoc instead points to a system dominated by constant per-channel offsets and stable non-LOS / coupling paths. That physics maps cleanly to a sparse multipath framing (OMP-friendly), and it is fully consistent with a teacher→student strategy that uses LDV to produce a frequency reliability mask to improve MIC–MIC robustness in the speech tail.
