# Claim-2 Mic-Observation Ablation Report

Generated: 2026-02-14T19:01:57.219807

Run dir: `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_coherence_trap_ablation_smoke_20260214_190130`

## Fixed near-fail setting

- corruption: SNR=0.0 dB in-band (500–2000), seed=1337, preclip_gain=100.0, clip_limit=0.99
- occlusion: target=micr, kind=lowpass, lowpass_hz=800.0

- common-mode interference: enabled=True, snr_db=0.0, seed=1337

## Policy gates (teacher forbidden mask)

- coupling_hard_forbid_enable: False
- dynamic_coh_gate_enable: True (coh_min=0.050)
- tau_ref_gate_enable: False (ratio_min=0.60)

## Teacher identity checks

- actions/noise centers/forbidden mask identical across obs modes: PASS

## Metric reachability (guided window)

- guided_radius_ms: 0.300
- quadratic_shift_clamp_ms: 0.010417
- theta_fail_deg: 4.00

| speaker | max_theta_error_possible_deg |
| --- | ---: |
| 20-0.1V | 4.377 |

## Near-fail precondition (baseline MIC–MIC)

- Criteria: fail_rate_theta_gt4deg>=0.40 AND frac_psr_gt3db<=0.10 (pooled test windows)
- baseline fail_rate_theta_gt4deg: 0.000
- baseline frac_psr_gt3db: 1.000
- near-fail => FAIL

## Pooled test metrics (vs chirp reference)

### Theta error

| obs_mode | baseline p95 | baseline fail(>4°) | student p95 | student fail(>4°) |
| --- | ---: | ---: | ---: | ---: |
| ldv_mic | 3.142 | 0.000 | 3.180 | 0.000 |
| mic_only_control | 3.142 | 0.000 | 3.180 | 0.000 |
| mic_only_coh_only | 3.142 | 0.000 | 3.180 | 0.000 |
| mic_only_psd_only | 3.142 | 0.000 | 3.192 | 0.000 |

### Tau error (vs tau_ref)

| obs_mode | baseline p95 (ms) | baseline fail(>0.25ms) | student p95 (ms) | student fail(>0.25ms) |
| --- | ---: | ---: | ---: | ---: |
| ldv_mic | 0.224 | 0.000 | 0.226 | 0.000 |
| mic_only_control | 0.224 | 0.000 | 0.226 | 0.000 |
| mic_only_coh_only | 0.224 | 0.000 | 0.226 | 0.000 |
| mic_only_psd_only | 0.224 | 0.000 | 0.227 | 0.000 |

### PSR (guided peak)

| obs_mode | baseline median (dB) | baseline frac(>3dB) | student median (dB) | student frac(>3dB) |
| --- | ---: | ---: | ---: | ---: |
| ldv_mic | 13.066 | 1.000 | 5.515 | 1.000 |
| mic_only_control | 13.066 | 1.000 | 5.515 | 1.000 |
| mic_only_coh_only | 13.066 | 1.000 | 5.515 | 1.000 |
| mic_only_psd_only | 13.066 | 1.000 | 7.048 | 1.000 |

## Key deltas (student vs student)

| comparison | p95 improvement frac | fail-rate improvement frac |
| --- | ---: | ---: |
| ldv_mic_vs_mic_only_control | 0.000 | nan |
| mic_only_control_vs_mic_only_coh_only | 0.000 | nan |
| mic_only_control_vs_mic_only_psd_only | 0.003 | nan |

## Interpretation rules (pre-registered)

- Coherence nearly sufficient if coh_only is within <=5% p95 and <=0.02 abs fail-rate of mic_only_control.
- PSD materially helps beyond coherence if psd_only closes >=50% of the p95 gap between coh_only and control.
- LDV adds marginal info beyond strong mic-only if ldv_mic improves vs mic_only_control by >=10% p95 and >=20% fail-rate (relative).

## Negative transfer check (per speaker)

- Comparison: `ldv_mic` student vs `mic_only_control` student (test windows only)
- Flag if: p95_ldv > 1.05 * p95_mic OR fail_ldv > fail_mic + 0.02

| speaker | p95_ldv | p95_mic | Δp95 | fail_ldv(>4°) | fail_mic(>4°) | Δfail | flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20-0.1V | 3.180 | 3.180 | +0.000 | 0.000 | 0.000 | +0.000 |  |

- Flagged speakers: `[]`
