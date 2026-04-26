# Spec: Claim 2 Mic-Observation Ablation (Coherence vs PSD vs LDV) Under a Fixed Near-Fail Occlusion Setting

## Purpose
This experiment isolates **which observation signals** are sufficient to learn a band-selection policy that stabilizes MIC–MIC guided GCC-PHAT on speech tails.

It answers:
1) Is **MicL–MicR coherence** alone sufficient (so LDV is not uniquely needed)?
2) How much do **mic PSD** and **LDV PSD** add beyond coherence?

This is **not** a new Claim-2 “pass/fail” stressor sweep; it is an **ablation** run at a single, pre-chosen near-fail setting.

---

## Locked dataset inputs
- `data_root`: `/home/sbplab/jiawei/data`
- Speakers: `18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V`
- Truth reference: `results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000`
- Windowing: 5.0 s, centers `100..600` step `1.0`
- Speech-active filter: keep windows with `RMS(MicL_clean) >= p50` per speaker (computed on clean MicL)

---

## Locked estimator and band policy
- Estimator: MIC–MIC guided GCC-PHAT over 500–2000 Hz
- Guided center: `truth_reference.tau_ref_ms`
- Guided radius: 0.3 ms
- Max lag: ±10 ms
- Bands: `B=64` equal-width linear bands in 500–2000 Hz
- Horizon: `K=6`

---

## Locked coupling mask policy (fair across ablations)
- `coupling_mode = mic_only`
- Silence windows: bottom 1% by `RMS(MicL_clean)` on the same center grid (minimum 3 else raise)
- Compute silence coherence:
  - `gamma2_silence(f) = |S_lr(f)|^2 / (S_ll(f) S_rr(f))` using Welch (`nperseg=8192`, `noverlap=4096`, Hann)
- Band aggregation:
  - `cpl_band[b] = mean_f_in_band( gamma2_silence(f) )`
- Forbidden band:
  - `forbidden[b] = (cpl_band[b] >= 0.20)`

The same forbidden mask is applied to all observation modes.

---

## Fixed near-fail mic-local corruption setting
This experiment uses a single fixed corruption configuration designed to reproduce a baseline near-fail regime:

### Noise + clipping (mic-local; deterministic)
- Enable corruption: `corrupt_enable=1`
- In-band SNR target: `corrupt_snr_db = 0` (defined in 500–2000 Hz)
- Global seed: `corrupt_seed = 1337`
- Saturation proxy:
  - `preclip_gain = 100`
  - `clip_limit = 0.99`

Noise is sourced from clean silence windows (existing WAVs only) and replayed via stored `noise_center_sec_*` arrays.

### Occlusion (mic-local spectral shaping on one mic)
- Enable occlusion: `occlusion_enable=1`
- Target mic: `occlusion_target = micr`
- Kind: `occlusion_kind = lowpass`
- Lowpass cutoff: `occlusion_lowpass_hz = 800`

LDV is never corrupted.

---

## Observation vector definitions (per window)
All observation vectors are length `B` (one scalar per band).

Let:
- `logPSD_MicAvg_band[b]`: band-mean Welch log-PSD of `(MicL + MicR)/2` magnitude-squared (implemented as average of mic PSDs), computed from **corrupted** mic windows.
- `mic_coh_speech_band[b]`: band-mean coherence(MicL, MicR) computed from **corrupted** mic windows.
- `logPSD_LDV_band[b]`: band-mean Welch log-PSD of LDV, computed from clean LDV windows.
- `cpl_band[b]`: silence coupling band metric computed from **clean** silence windows (mic-only coupling).

`zscore_inband(x)[b] = (x[b] - mean_b x[b]) / std_b x[b]`, computed across `b=0..B-1` within the window. Raise if `std < 1e-12`.

Observation modes:
- `ldv_mic`:
  - `obs[b] = z(logPSD_LDV_band)[b] + z(logPSD_MicAvg_band)[b] + z(mic_coh_speech_band)[b] - 2*z(cpl_band)[b]`
- `mic_only_control`:
  - `obs[b] = z(logPSD_MicAvg_band)[b] + z(mic_coh_speech_band)[b] - 2*z(cpl_band)[b]`
- `mic_only_coh_only`:
  - `obs[b] = z(mic_coh_speech_band)[b] - 2*z(cpl_band)[b]`
- `mic_only_psd_only`:
  - `obs[b] = z(logPSD_MicAvg_band)[b] - 2*z(cpl_band)[b]`

Hard-lock: teacher actions are derived from MIC–MIC score only and must be identical across obs modes.

---

## Driver script contract
Script: `scripts/run_claim2_micobs_ablation_suite.py`

### Full run
```bash
python -u scripts/run_claim2_micobs_ablation_suite.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_micobs_ablation_<YYYYMMDD_HHMMSS> \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V
```

### Smoke run (real data, fast)
```bash
python -u scripts/run_claim2_micobs_ablation_suite.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_micobs_ablation_smoke_<YYYYMMDD_HHMMSS> \
  --smoke 1
```

---

## Required outputs
Under `results/<run_name>/`:
- `run.log`, `run_config.json`, `code_state.json`
- `teacher/<obs_mode>/teacher_trajectories.npz` and per-speaker JSONL summaries
- `student/<obs_mode>/test_windows.jsonl`
- `teacher_identity.json` (asserts identity across teacher runs)
- `summary_table.json` (all pooled + per-speaker metrics)
- `report.md`

---

## Guardrails (hard requirements)
The run is valid only if:
1) Teacher identity checks pass (actions + noise centers + forbidden mask identical across obs modes).
2) No NaNs; non-empty test set for each obs mode.

Near-fail precondition:
- Baseline pooled near-fail criteria:
  - `fail_rate_theta_gt4deg >= 0.40` AND
  - `frac_psr_gt3db <= 0.10`
  Note: with `guided_radius_ms=0.3` and sub-sample shift clamp, `theta_error_ref_deg > 5°` can be mathematically unreachable. The suite enforces a reachability guardrail.
If not met, record “near-fail not reproduced” and do not interpret ablation differences.

---

## Interpretation rules (pre-registered)
Not a pass/fail claim—these are labels for the report narrative:

- **Coherence nearly sufficient** if `mic_only_coh_only` is within:
  - `<= 5%` relative pooled p95 of `mic_only_control`, and
  - `<= 0.02` absolute pooled fail-rate(>4°) of `mic_only_control`.
- **PSD materially helps beyond coherence** if `mic_only_psd_only` closes `>= 50%` of the pooled p95 gap between `mic_only_coh_only` and `mic_only_control`.
- **LDV adds marginal info beyond strong mic-only** if `ldv_mic` improves vs `mic_only_control` by:
  - `>= 10%` pooled p95 relative AND `>= 20%` pooled fail-rate relative.
