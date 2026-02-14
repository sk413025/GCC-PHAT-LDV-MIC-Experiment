# Spec: Claim 2 “Coherence Trap” — Common-Mode Coherent Interference (Existing WAVs Only)

## Purpose
This experiment creates a **mic-only coherence trap**: a condition where **MicL–MicR coherence is high**, yet **MIC–MIC guided GCC-PHAT becomes near-failing** vs the chirp-reference truth.

Why this matters:
- In ideal conditions, **high coherence** often implies “good information.”
- With **common-mode coherent interference** (electrical/common-path or crosstalk-like contamination), coherence can be **artificially inflated** while the phase/delay evidence relevant to the true `tau_ref_ms` becomes unreliable.
- This is a plausible regime where **LDV becomes irreducible** as side information: LDV can indicate source/band reliability when mic-only cues become misleading.

This spec defines:
1) A sweep that selects a common-mode interference severity that meets **near-fail + high-coherence** criteria.
2) A follow-up **mic-observation ablation** run at the chosen level to measure whether LDV adds marginal predictive power beyond mic-only observation variants.

All runs use **existing real WAVs only** and treat **chirp calibration as mandatory truth-reference**.

---

## Locked dataset inputs
- `data_root`: `/home/sbplab/jiawei/data`
- Speakers: `18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V`
- Truth reference (mandatory): `results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000`
- Windowing: `5.0 s`, centers `100..600` step `1`
- Speech-active filter: `RMS(MicL_clean) >= p50` per speaker (computed on clean MicL)

---

## Estimator (fixed)
All scoring/evaluation uses **MIC–MIC guided GCC-PHAT**:
- Analysis band: `500–2000 Hz`
- Guided center: `truth_reference.tau_ref_ms`
- Guided radius: `0.3 ms`
- Max lag: `±10 ms`

Band-policy setup (teacher/student):
- Bands: `B=64` linear bands over `500–2000 Hz`
- Horizon: `K=6`

---

## Stressor: common-mode coherent interference (mic-only)
We inject **the same** interference waveform into both microphones:

`MicL_mix(t) = MicL_mix(t) + i(t)`
`MicR_mix(t) = MicR_mix(t) + i(t)`

Where `i(t)` is:
1) **sourced from real silence windows** (existing WAVs only),
2) **band-limited** to the analysis band (500–2000 Hz),
3) scaled to a target **in-band SNR** relative to a clean reference signal.

### Noise source (existing WAVs only)
Per speaker:
- Define silence windows from the same center grid: bottom `1%` by `RMS(MicL_clean)` (minimum 3 windows; else fail-fast).
- For each speech window, choose a silence-window center deterministically using a fixed seed and record it in artifacts.

### Band-limiting
Deterministic FFT hard-mask bandpass:
- `bandpass_fft(x, fs, 500, 2000)` via `rfft -> mask -> irfft`.

### In-band SNR scaling
Let:
- `s_bp = bandpass_fft(signal_for_alpha)` (typically clean MicL)
- `n_bp = bandpass_fft(noise_window)` (silence noise)

Compute powers:
- `P_s = mean(s_bp^2)`
- `P_n = mean(n_bp^2)`

For target `snr_db` (in-band):
- `alpha = sqrt(P_s / (P_n * 10^(snr_db/10) + eps))`
- `i_bp = alpha * n_bp`

Achieved SNR diagnostic:
- `snr_achieved_db = 10*log10(P_s / (alpha^2*P_n + eps))`

Implementation:
- `scripts/mic_corruption.py::add_common_mode_interference()`

---

## Fixed “near-fail” base condition (kept constant)
To keep continuity with prior Claim-2 ablation runs, the sweep and ablation also keep:
- Independent mic-local in-band noise + gain→clip→de-gain corruption (`snr_db=0`, `preclip_gain=100`, `clip_limit=0.99`)
- Occlusion spectral shaping on one mic (MicR lowpass at 800 Hz, zero-phase magnitude shaping)
- LDV is never corrupted

This base setting is already encoded in `scripts/run_claim2_micobs_ablation_suite.py`.

---

## Sweep: choose a “coherence trap” severity
Script:
- `scripts/sweep_claim2_coherence_trap_cm_interference.py`

Grid (pre-registered):
- `cm_snr_db ∈ {+20, +10, +5, 0, -5, -10, -15}` (in-band 500–2000 Hz)

Evaluation subset:
- **Test windows only**: `center_sec > 450` (matches DTmin split)

### Selection criteria (locked)
Define:
- `theta_fail_deg = 4.0`
- `psr_good_db = 3.0`

Near-fail must satisfy (pooled over all speakers, test windows):
- `fail_rate(theta_error_ref_deg > 4.0°) >= 0.40`
- `frac(PSR > 3.0 dB) <= 0.10`

Coherence trap must satisfy:
- `median(mic_coh_band_median) >= 0.20`
  - where `mic_coh_band_median` is derived from `mic_coh_speech_band_summary.median` in teacher JSONL.

Selection rule:
- Among grid points satisfying both conditions, pick the one with **highest** `median(mic_coh_band_median)`.
- If none satisfy both, record a negative result and stop (do not silently change thresholds).

---

## Follow-up: mic-observation ablation at the chosen level
After the sweep chooses a `cm_snr_db`, run:
- `scripts/run_claim2_micobs_ablation_suite.py` with `--cm_interf_enable 1 --cm_interf_snr_db <chosen> --cm_interf_seed 1337`

Observation variants (DTmin students):
- `ldv_mic`
- `mic_only_control`
- `mic_only_coh_only`
- `mic_only_psd_only`

Teacher identity must hold:
- actions, noise centers, and forbidden masks identical across obs modes.

Primary interpretation:
- If `mic_only_coh_only` ≈ `mic_only_control`, then **coherence is close to sufficient** under this stressor.
- If `ldv_mic` materially beats `mic_only_control`, then LDV provides **marginal predictive power beyond strong mic-only cues** in this coherence-trap regime.

---

## Required artifacts
Every run writes artifacts under `results/<run_name>/`:

Sweep:
- `run.log`, `run_config.json`, `code_state.json`
- per grid point: `cm_snr_*/teacher_mic_only_control/...` + `summary.json`
- `sweep_summary.json`, `report.md`

Ablation:
- suite standard artifacts: teachers + students + `teacher_identity.json`, `summary_table.json`, `report.md`

Note: `results/` is gitignored; commits must use `git add -f`.

