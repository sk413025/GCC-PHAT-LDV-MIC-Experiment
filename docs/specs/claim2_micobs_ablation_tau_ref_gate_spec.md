# Spec: Fix Speaker-Dependent Negative Transfer via a Tau-Ref Support Gate (Claim-2 Ablation v3)

## Purpose
This spec defines a minimal, reproducible follow-up to the existing **Claim-2 mic-observation ablation** to address a specific regression observed in the v2 fix run:

> LDV+MIC can still be **worse for some speakers** (speaker-dependent negative transfer), even after switching to soft coupling + dynamic mic-coherence gating.

The core issue is that **band coherence is necessary but not sufficient**: under multipath / coupling, a band can be coherent but dominated by a **coherent-but-wrong delay** (peak far from `tau_ref`). We therefore add a truth-guided but physically consistent **tau-ref support gate** that forbids bands whose per-band GCC peak is not supported near the chirp-reference delay.

This spec is intended for an executed results commit (code + docs + artifacts under `results/`).

---

## Background evidence (what this is fixing)
The v2 run:
- `results/claim2_micobs_ablation_fix_20260214_144138/summary_table.json`
- `results/claim2_micobs_ablation_fix_20260214_144138/report.md`

shows negative transfer for speaker `20-0.1V` (LDV+MIC worse than mic-only(control) on test windows), while speaker `21-0.1V` was improved vs v1.

The most plausible causal story is:
- Disabling silence-coupling hard forbids reintroduces **low-frequency bands** that may be dominated by a coherent non-LOS / coupling path under the occlusion stressor.
- A per-window coherence floor (`coh_min`) does not eliminate “coherent-but-wrong” bands.

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
- Guided center: `truth_reference.tau_ref_ms` (chirp-reference truth)
- Guided radius: 0.3 ms
- Max lag: ±10 ms
- Bands: `B=64` equal-width linear bands in 500–2000 Hz
- Horizon: `K=6`

---

## Locked near-fail setting (unchanged)
Mic-local stress is identical to the v1/v2 ablations to isolate the new gate:

### Noise + saturation proxy (mic-local; deterministic)
- Enable corruption: `corrupt_enable=1`
- In-band SNR target: `corrupt_snr_db = 0` (defined in 500–2000 Hz)
- Seed: `corrupt_seed = 1337`
- Saturation proxy: `preclip_gain = 100`, `clip_limit = 0.99`
- Noise is sourced from clean silence windows and replayed via stored `noise_center_sec_*`.

### Occlusion (mic-local spectral shaping)
- Enable occlusion: `occlusion_enable=1`
- Target mic: `micr`
- Kind: `lowpass`
- Cutoff: `800 Hz`

LDV is never corrupted.

---

## Gate policy (v3)
This v3 run uses three gates/penalties:

### 1) Soft coupling only (unchanged from v2)
We keep silence coupling `cpl_band[b]` as a penalty feature in observations, but do **not** hard-forbid any band due to coupling:
- `coupling_hard_forbid_enable = 0`

### 2) Dynamic per-window mic-coherence floor (unchanged from v2)
After corruption+occlusion on the current speech window:
- Compute per-band MicL–MicR coherence `mic_coh_speech_band[b]` via Welch (`8192/4096`, Hann).
- Forbid if:
  - `dynamic_coh_gate_enable = 1`
  - `dynamic_coh_min = 0.05`
  - `forbidden_dyn_coh[b] = (mic_coh_speech_band[b] < 0.05)`

### 3) NEW: Tau-ref support gate (hard forbid)
For each window and band `b` (computed from the same corrupted mic signals):

1. Compute the band-limited PHAT correlation:
   - `R_phat(f) = (X(f) * conj(Y(f))) / (|X(f) * conj(Y(f))| + eps)`
   - `cc_b = irfft(R_phat * band_mask_b)` then extract ±maxlag window.

2. Define peaks:
   - `peak_global = max_{tau in ±maxlag} |cc_b(tau)|`
   - `peak_guided = max_{tau in tau_ref ± guided_radius} |cc_b(tau)|`

3. Guided support ratio:
   - `ratio_b = peak_guided / (peak_global + 1e-30)`

4. Forbid rule (locked for this experiment):
   - `tau_ref_gate_enable = 1`
   - `tau_ref_gate_ratio_min = 0.60`
   - `forbidden_tau_ref[b] = (ratio_b < 0.60)`

Effective forbidden mask per window:
- `forbidden_mask[b] = forbidden_dyn_coh[b] OR forbidden_tau_ref[b]`

Interpretation:
- A coherent-but-wrong band tends to have a strong global peak far from `tau_ref`, giving a small `ratio_b`.

---

## Sub-sample tau estimate stability (bugfix-level guardrail)
Both the teacher and student evaluation use quadratic interpolation around the guided-window argmax. When the argmax hits the guided-window boundary, quadratic interpolation can extrapolate far outside the guided window.

To keep “guided” behavior meaningful, clamp the interpolation shift:
- `shift = clip(shift, -0.5, +0.5)` samples

This ensures the estimate remains a local refinement and does not violate the guided window constraint by construction.

---

## Script contracts

### Teacher
Script: `scripts/teacher_band_omp_micmic.py`

New CLI args:
- `--tau_ref_gate_enable {0,1}` (default 0)
- `--tau_ref_gate_ratio_min <float>` (default 0.60)

Artifacts required per speaker:
- `per_speaker/<speaker>/coupling_mask.json` must include:
  - `tau_ref_gate_enable`, `tau_ref_gate_ratio_min`
- `per_speaker/<speaker>/windows.jsonl` must include per window:
  - `forbidden_tau_ref_bands`, `forbidden_tau_ref_count`
  - `tau_ref_gate.guided_ratio_band_summary` (when enabled)

Trajectory NPZ must include:
- `forbidden_mask`: `(N, B)` bool, per-window effective forbidden.

Hard requirement:
- Teacher actions/noise centers/forbidden_mask are **independent of obs_mode**.

### Suite driver
Script: `scripts/run_claim2_micobs_ablation_suite.py`

New CLI args:
- `--tau_ref_gate_enable {0,1}` default 0
- `--tau_ref_gate_ratio_min <float>` default 0.60

Report additions:
- “Policy gates (teacher forbidden mask)” must list the 3 gate flags and thresholds.

### Student train/eval
Script: `scripts/train_dtmin_from_band_trajectories.py`

Requirement:
- Must apply the same sub-sample shift clamp as the teacher.

---

## Reproduction commands

### Smoke run (real data, fast)
```bash
python -u scripts/run_claim2_micobs_ablation_suite.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_micobs_ablation_tau_ref_gate_smoke_<YYYYMMDD_HHMMSS> \
  --smoke 1 \
  --coupling_hard_forbid_enable 0 \
  --dynamic_coh_gate_enable 1 \
  --dynamic_coh_min 0.05 \
  --tau_ref_gate_enable 1 \
  --tau_ref_gate_ratio_min 0.60
```

### Full run
```bash
python -u scripts/run_claim2_micobs_ablation_suite.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_micobs_ablation_tau_ref_gate_<YYYYMMDD_HHMMSS> \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V \
  --coupling_hard_forbid_enable 0 \
  --dynamic_coh_gate_enable 1 \
  --dynamic_coh_min 0.05 \
  --tau_ref_gate_enable 1 \
  --tau_ref_gate_ratio_min 0.60
```

---

## Acceptance / verification

### Validity guardrails
1) `results/<run>/teacher_identity.json`: PASS (actions + noise centers + forbidden mask identical across obs modes).
2) Baseline near-fail precondition (pooled test windows): `fail_rate_ref(>5°) >= 0.40`.
3) No NaNs; non-empty test windows for every obs_mode.

### Fix acceptance (non-worse per speaker; locked)
In `results/<run>/report.md` “Negative transfer check (per speaker)”:
- No speaker should be flagged by:
  - `p95_ldv > 1.05 * p95_mic` OR
  - `fail_ldv > fail_mic + 0.02`

### Pooled guardrail (avoid “fix by deleting LDV benefit”)
On pooled test windows:
- `p95_improvement_frac(ldv_mic vs mic_only_control) >= 0.05`
- `fail_rate_improvement_frac(ldv_mic vs mic_only_control) >= 0.10`

If any condition fails, commit the negative result with causal analysis and artifacts.

