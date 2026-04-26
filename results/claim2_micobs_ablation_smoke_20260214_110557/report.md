# Claim-2 Mic-Observation Ablation Report

Generated: 2026-02-14T11:06:13.208376

Run dir: `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_micobs_ablation_smoke_20260214_110557`

## Fixed near-fail setting

- corruption: SNR=0.0 dB in-band (500–2000), seed=1337, preclip_gain=100.0, clip_limit=0.99
- occlusion: target=micr, kind=lowpass, lowpass_hz=800.0

## Teacher identity checks

- actions/noise centers/forbidden mask identical across obs modes: PASS

## Near-fail precondition (baseline MIC–MIC)

- baseline fail_rate_ref(>5°): 0.000 (>= 0.40 required) => FAIL

## Pooled test metrics (vs chirp reference)

| obs_mode | baseline p95 | baseline fail | student p95 | student fail |
| --- | ---: | ---: | ---: | ---: |
| ldv_mic | 4.364 | 0.000 | 4.601 | 0.000 |
| mic_only_control | 4.364 | 0.000 | 4.601 | 0.000 |
| mic_only_coh_only | 4.364 | 0.000 | 4.601 | 0.000 |
| mic_only_psd_only | 4.364 | 0.000 | 3.719 | 0.000 |

## Key deltas (student vs student)

| comparison | p95 improvement frac | fail-rate improvement frac |
| --- | ---: | ---: |
| ldv_mic_vs_mic_only_control | 0.000 | nan |
| mic_only_control_vs_mic_only_coh_only | 0.000 | nan |
| mic_only_control_vs_mic_only_psd_only | -0.237 | nan |

## Interpretation rules (pre-registered)

- Coherence nearly sufficient if coh_only is within <=5% p95 and <=0.02 abs fail-rate of mic_only_control.
- PSD materially helps beyond coherence if psd_only closes >=50% of the p95 gap between coh_only and control.
- LDV adds marginal info beyond strong mic-only if ldv_mic improves vs mic_only_control by >=10% p95 and >=20% fail-rate (relative).
