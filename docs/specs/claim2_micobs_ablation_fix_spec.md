# Spec: Fix Speaker-Dependent Negative Transfer via Soft Coupling + Dynamic Mic-Coherence Gate (Claim-2 Ablation v2)

## Purpose
This spec defines a minimal, reproducible modification to the existing **Claim-2 mic-observation ablation** that targets a specific failure mode:

> LDV+MIC improves pooled speech-tail metrics but can be **worse for some speakers** (speaker-dependent negative transfer).

The fix is physically motivated under the **fixed near-fail occlusion** stressor:

1) **Soft coupling only**: do **not** hard-forbid bands based on silence coupling; keep coupling as a penalty feature only.  
2) **Dynamic coherence gate**: hard-forbid bands whose **per-window MicL–MicR coherence** is too low to support stable MIC–MIC GCC-PHAT.

This is an executed experiment spec (code + artifacts under `results/`).

---

## Background evidence (what this is fixing)
In the baseline run:
- `results/claim2_micobs_ablation_20260214_112341/summary_table.json`

Speaker `21-0.1V` shows negative transfer for LDV+MIC vs mic-only(control) on test windows:
- LDV+MIC student p95 worse and fail-rate worse.

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

## Locked near-fail setting (unchanged)
Mic-local stress is kept identical to the v1 ablation so the fix is isolated:

### Noise + clipping (mic-local; deterministic)
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

## Fix policy definitions

### 1) Soft coupling only (disable coupling hard forbids)
We still compute silence coupling per band from **clean silence windows**:
- `cpl_band[b] = mean_f_in_band( coherence_silence(MicL, MicR) )`

But we do **not** hard-forbid any band due to coupling in this v2 run:
- `coupling_hard_forbid_enable = 0`
- `forbidden_static_effective[b] = False` for all `b`

Coupling still enters the observation vector as a penalty term:
- subtract `2*zscore_inband(cpl_band)[b]` (same as v1)

### 2) Dynamic per-window mic-coherence gate (hard forbid)
For each **speech window** after corruption+occlusion are applied:
- Compute MicL–MicR magnitude-squared coherence via Welch (`nperseg=8192`, `noverlap=4096`, Hann).
- Aggregate to bands:
  - `mic_coh_speech_band[b] = mean_f_in_band( gamma2_speech(f) )`

Dynamic forbidden rule (locked):
- `dynamic_coh_gate_enable = 1`
- `dynamic_coh_min = 0.05`
- `forbidden_dyn[b] = (mic_coh_speech_band[b] < 0.05)`

Effective forbidden mask per window:
- `forbidden_mask[b] = forbidden_dyn[b]` (since static coupling forbids are disabled)

Teacher and student must share the **same** forbidden mask per window (stored in the teacher trajectory NPZ).

---

## Script contracts

### Teacher changes
Script: `scripts/teacher_band_omp_micmic.py`

New CLI args:
- `--coupling_hard_forbid_enable {0,1}` (default 1)
- `--dynamic_coh_gate_enable {0,1}` (default 0)
- `--dynamic_coh_min <float>` (default 0.05)

Artifacts required per speaker:
- `per_speaker/<speaker>/coupling_mask.json` must include:
  - `coupling_hard_forbid_enable`, `dynamic_coh_gate_enable`, `dynamic_coh_min`
- `per_speaker/<speaker>/windows.jsonl` must include per window:
  - `mic_coh_speech_band_summary` = `{min, median, p90, max}`
  - `forbidden_dyn_count`, `forbidden_dyn_bands`

Trajectory NPZ must include:
- `forbidden_mask`: `(N, B)` bool, per-window effective forbidden.

Hard requirement:
- Teacher actions/noise centers/forbidden_mask are **independent of obs_mode**.

### Suite driver changes
Script: `scripts/run_claim2_micobs_ablation_suite.py`

New CLI args:
- `--coupling_hard_forbid_enable {0,1}` default 1
- `--dynamic_coh_gate_enable {0,1}` default 0
- `--dynamic_coh_min <float>` default 0.05

Report additions:
- A “Negative transfer check (per speaker)” section comparing `ldv_mic` vs `mic_only_control` on test windows.

Flag rule (locked):
- Flag if `p95_ldv > 1.05 * p95_mic` OR `fail_ldv > fail_mic + 0.02`

---

## Reproduction commands

### Smoke run (real data, fast)
```bash
python -u scripts/run_claim2_micobs_ablation_suite.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_micobs_ablation_fix_smoke_<YYYYMMDD_HHMMSS> \
  --smoke 1 \
  --coupling_hard_forbid_enable 0 \
  --dynamic_coh_gate_enable 1 \
  --dynamic_coh_min 0.05
```

### Full run
```bash
python -u scripts/run_claim2_micobs_ablation_suite.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_micobs_ablation_fix_<YYYYMMDD_HHMMSS> \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V \
  --coupling_hard_forbid_enable 0 \
  --dynamic_coh_gate_enable 1 \
  --dynamic_coh_min 0.05
```

---

## Acceptance / verification

### Validity guardrails
1) `results/<run>/teacher_identity.json`: PASS (actions + noise centers + forbidden mask identical across obs modes).
2) Baseline near-fail precondition (pooled test windows): `fail_rate_ref(>5°) >= 0.40`.
3) No NaNs; non-empty test windows for every obs_mode.

### Fix acceptance (non-worse per speaker; locked)
For **each speaker** on test windows, LDV+MIC student vs mic-only(control) must satisfy:
- `p95_ldv <= 1.05 * p95_mic`
- `fail_ldv <= fail_mic + 0.02`
Additionally, speaker `21-0.1V` must satisfy the same.

### Pooled guardrail (avoid “fix by deleting LDV benefit”)
On pooled test windows:
- `p95_improvement_frac(ldv_mic vs mic_only_control) >= 0.05`
- `fail_rate_improvement_frac(ldv_mic vs mic_only_control) >= 0.10`

If any condition fails, the result is still committed as a negative outcome with causal analysis.

