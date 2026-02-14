# Spec: Claim 2 Verification — LDV Irreducibility Under Mic-Local Degradation (Existing WAVs Only)

**Purpose**
Demonstrate (or falsify) the conditional claim:

> Under **mic-local** degradations (noise + saturation/clipping affecting microphones but not LDV), an LDV+MIC-observation student policy outperforms an otherwise identical MIC-only student at reproducing a teacher’s band-selection decisions and stabilizing **MIC–MIC guided GCC-PHAT** on speech tails.

This is an **existing-WAVs-only** verification. No new recordings.

---

## 1) Locked dataset inputs
- `data_root`: `/home/sbplab/jiawei/data`
- Speakers: `18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V`
- Each speaker uses a 3-channel triplet:
  - `*LDV*.wav`, `*LEFT*.wav`, `*RIGHT*.wav`
- `fs` invariant: `48000` Hz, mono. Fail fast otherwise.

## 2) Truth reference (mandatory)
- `truth_ref_root`: `results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000`
- Guided center for MIC–MIC GCC: `truth_reference.tau_ref_ms`
- Primary evaluation truth for angle: `truth_reference.theta_ref_deg`

---

## 3) Windowing (fixed)
- Window length: `5.0 s`
- Candidate centers: `t=100..600 s` inclusive, step `1 s`
- Speech-active filter (computed on **clean MicL**):
  - keep windows where `RMS(MicL) >= p50` per speaker
- Silence set (computed on **clean MicL**):
  - bottom `1%` by `RMS(MicL)` over candidates, minimum `3` windows

---

## 4) Corruption model (mic-local, deterministic)
Corrupt **MicL** and **MicR** only. **LDV is never corrupted.**

### 4.1 Parameters (pre-registered grid)
- Analysis band (also corruption SNR band): `500–2000 Hz`
- `snr_db ∈ [20, 10, 5, 0, -5, -10]`
- `seed = 1337`
- `preclip_gain = 100`
- `clip_limit = 0.99`

### 4.2 Noise sourcing (existing data only)
For each speech window:
- Select independent noise centers `noise_center_sec_L`, `noise_center_sec_R` from the speaker’s silence-center pool (deterministic RNG).
- Extract `noise_window` from the same channel at that center (`MicL` noise from MicL track, MicR noise from MicR track).

### 4.3 Band-limiting (deterministic)
Define a deterministic frequency-domain hard-mask bandpass:

- `bandpass_fft(x, fs, lo, hi)`:
  1) `X = rfft(x)`
  2) keep bins with `lo <= f <= hi`, zero others
  3) `x_bp = irfft(X_masked)`

This is implemented in `scripts/mic_corruption.py`.

### 4.4 In-band SNR scaling (pre-clip)
Let:
- `s = bandpass_fft(mic_clean, 500, 2000)`
- `n = bandpass_fft(noise_window, 500, 2000)`
- `P_s = mean(s^2)`
- `P_n = mean(n^2)`

Target SNR is defined by:
- `SNR_db = 10 log10(P_s / P_n_scaled)`

We choose a scale `alpha` so that:
- `P_n_scaled = alpha^2 * P_n`
- `alpha = sqrt(P_s / (P_n * 10^(snr_db/10) + eps))`

Then:
- `mic_noisy = mic_clean + alpha * n`

### 4.5 Saturation proxy: gain→clip→de-gain
Apply:
- `mic_pre = preclip_gain * mic_noisy`
- `mic_clip = clip(mic_pre, -clip_limit, +clip_limit)`
- `mic_corrupt = mic_clip / preclip_gain`

Diagnostics per window (per channel):
- `snr_achieved_db_preclip`: computed from `P_s` and `alpha^2 * P_n`
- `clip_frac = mean(|mic_pre| >= clip_limit)`

---

## 5) Coupling mask policy (primary for Claim 2)
We compute the silence-derived coupling mask from **MicL–MicR coherence only** on **clean** silence windows:

- For each silence window, estimate Welch spectra (`nperseg=8192`, `noverlap=4096`, Hann):
  - `S_ll(f)`, `S_rr(f)`, `S_lr(f)`
- Coherence:
  - `gamma2_mic(f) = |S_lr|^2 / (S_ll * S_rr)`
- Band aggregation:
  - `cpl_band[b] = mean_{f in band b}(gamma2_mic(f))`

Forbidden bands:
- `forbidden_raw[b] = (cpl_band[b] >= 0.20)`
- If `forbidden_raw` would forbid **all** bands, we set `forbidden_effective[:] = False` and mark `forbid_rule_degenerated=true` in artifacts.

Rationale: keep the coupling metric as a penalty feature without collapsing the action space.

---

## 6) Teacher + student definitions (what is being learned)
### 6.1 Teacher task
For each corrupted speech window:
- Compute MIC–MIC PHAT cross-spectrum in `500–2000 Hz`
- Greedily select up to `K=6` bands (from `B=64`) maximizing:
  - `score(S) = -( |tau_S_ms - tau_ref_ms| / 0.30 ) + ( psr_S_db / 3.0 )`

Teacher actions do **not** depend on `obs_mode`.

### 6.2 Observation modes (ablation)
Let per-band features be computed on the **corrupted** window:
- `ldv_band[b] = mean_{f in band}( log PSD(LDV)(f) )`
- `mic_band[b] = mean_{f in band}( log PSD(0.5*(MicL+MicR))(f) )`
- `coh_band[b] = mean_{f in band}( coherence(MicL,MicR)(f) )`
- `cpl_band[b]` from Section 5 (clean silence windows)

Let `z(x)` be z-score across `b=0..B-1` within the window.

Then:
- `obs_mode=ldv_mic`:
  - `obs[b] = z(ldv_band)[b] + z(mic_band)[b] + z(coh_band)[b] - 2*z(cpl_band)[b]`
- `obs_mode=mic_only_control`:
  - `obs[b] = z(mic_band)[b] + z(coh_band)[b] - 2*z(cpl_band)[b]`

The MIC-only control uses the **same** coupling penalty and forbidden bands, isolating the marginal value of dynamic LDV features.

---

## 7) Artifact schemas (must be produced)
### 7.1 Teacher run (`scripts/teacher_band_omp_micmic.py`)
Under `results/.../per_speaker/<speaker>/windows.jsonl`, each record includes:
- `corruption`:
  - `enabled`
  - `noise_center_sec_L`, `noise_center_sec_R`
  - per-channel diag: `snr_achieved_db_preclip`, `clip_frac`, `alpha`, etc.

`teacher_trajectories.npz` includes:
- `observations`: `(N, K, B)` float32
- `actions`: `(N, K)` int32 (padding -1)
- `valid_len`: `(N,)` int32
- `speaker_id`: `(N,)` string
- `center_sec`: `(N,)` float64
- `forbidden_mask`: `(N, B)` bool
- `band_edges_hz`: `(B+1,)` float64
- Corruption arrays (length `N`):
  - `noise_center_sec_L`, `noise_center_sec_R`: float64
  - `snr_target_db`: float64
  - `snr_achieved_db_L`, `snr_achieved_db_R`: float64 (pre-clip)
  - `clip_frac_L`, `clip_frac_R`: float64
  - `corruption_config_json`: scalar string (JSON)

### 7.2 Student run (`scripts/train_dtmin_from_band_trajectories.py`)
`test_windows.jsonl` must include a `corruption` block mirroring the per-window corruption used in evaluation.

`summary.json` includes:
- `corruption.use_traj_corruption`
- `corruption.config` (if present)
- test summaries of achieved SNR and clip fraction.

---

## 8) Acceptance criteria (Claim 2)
Claim 2 is supported if at least one severity in `{0, -5, -10} dB` satisfies:

1) Near-fail baseline: baseline MIC–MIC `fail_rate_ref(theta_error_ref_deg > 4°) >= 0.40`
2) LDV feasibility: LDV+MIC student `fail_rate_ref(>4°) <= 0.10`
3) Irreducibility: LDV+MIC student beats MIC-only student:
   - `p95(theta_error_ref_deg)` at least 10% lower than MIC-only
   - `fail_rate_ref(>4°)` at least 20% lower (relative)

If (1) never happens, record a negative result (the proxy did not create sufficiently extreme windows).
