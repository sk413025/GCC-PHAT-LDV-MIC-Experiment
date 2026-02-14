# Claim-2 Mic-Observation Ablation Report

Generated: 2026-02-14T17:57:24.200866

Run dir: `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_micobs_ablation_metricfix_20260214_174101`

## Fixed near-fail setting

- corruption: SNR=0.0 dB in-band (500–2000), seed=1337, preclip_gain=100.0, clip_limit=0.99
- occlusion: target=micr, kind=lowpass, lowpass_hz=800.0

## Policy gates (teacher forbidden mask)

- coupling_hard_forbid_enable: False
- dynamic_coh_gate_enable: True (coh_min=0.050)
- tau_ref_gate_enable: True (ratio_min=0.60)

## Teacher identity checks

- actions/noise centers/forbidden mask identical across obs modes: PASS

## Metric reachability (guided window)

- guided_radius_ms: 0.300
- quadratic_shift_clamp_ms: 0.010417
- theta_fail_deg: 4.00

| speaker | max_theta_error_possible_deg |
| --- | ---: |
| 18-0.1V | 4.820 |
| 19-0.1V | 4.517 |
| 20-0.1V | 4.377 |
| 21-0.1V | 4.443 |
| 22-0.1V | 4.669 |

## Near-fail precondition (baseline MIC–MIC)

- Criteria: fail_rate_theta_gt4deg>=0.40 AND frac_psr_gt3db<=0.10 (pooled test windows)
- baseline fail_rate_theta_gt4deg: 0.714
- baseline frac_psr_gt3db: 0.063
- near-fail => PASS

## Pooled test metrics (vs chirp reference)

### Theta error

| obs_mode | baseline p95 | baseline fail(>4°) | student p95 | student fail(>4°) |
| --- | ---: | ---: | ---: | ---: |
| ldv_mic | 4.625 | 0.714 | 4.570 | 0.393 |
| mic_only_control | 4.625 | 0.714 | 4.483 | 0.399 |
| mic_only_coh_only | 4.625 | 0.714 | 4.551 | 0.371 |
| mic_only_psd_only | 4.625 | 0.714 | 4.472 | 0.387 |

### Tau error (vs tau_ref)

| obs_mode | baseline p95 (ms) | baseline fail(>0.25ms) | student p95 (ms) | student fail(>0.25ms) |
| --- | ---: | ---: | ---: | ---: |
| ldv_mic | 0.312 | 0.770 | 0.308 | 0.453 |
| mic_only_control | 0.312 | 0.770 | 0.308 | 0.450 |
| mic_only_coh_only | 0.312 | 0.770 | 0.308 | 0.418 |
| mic_only_psd_only | 0.312 | 0.770 | 0.312 | 0.437 |

### PSR (guided peak)

| obs_mode | baseline median (dB) | baseline frac(>3dB) | student median (dB) | student frac(>3dB) |
| --- | ---: | ---: | ---: | ---: |
| ldv_mic | -4.305 | 0.063 | -0.747 | 0.016 |
| mic_only_control | -4.305 | 0.063 | -0.777 | 0.016 |
| mic_only_coh_only | -4.305 | 0.063 | -0.667 | 0.019 |
| mic_only_psd_only | -4.305 | 0.063 | -0.366 | 0.025 |

## Key deltas (student vs student)

| comparison | p95 improvement frac | fail-rate improvement frac |
| --- | ---: | ---: |
| ldv_mic_vs_mic_only_control | -0.019 | 0.016 |
| mic_only_control_vs_mic_only_coh_only | 0.015 | -0.076 |
| mic_only_control_vs_mic_only_psd_only | -0.002 | -0.033 |

## Interpretation rules (pre-registered)

- Coherence nearly sufficient if coh_only is within <=5% p95 and <=0.02 abs fail-rate of mic_only_control.
- PSD materially helps beyond coherence if psd_only closes >=50% of the p95 gap between coh_only and control.
- LDV adds marginal info beyond strong mic-only if ldv_mic improves vs mic_only_control by >=10% p95 and >=20% fail-rate (relative).

## Negative transfer check (per speaker)

- Comparison: `ldv_mic` student vs `mic_only_control` student (test windows only)
- Flag if: p95_ldv > 1.05 * p95_mic OR fail_ldv > fail_mic + 0.02

| speaker | p95_ldv | p95_mic | Δp95 | fail_ldv(>4°) | fail_mic(>4°) | Δfail | flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 18-0.1V | 4.570 | 4.570 | +0.000 | 0.244 | 0.231 | +0.013 |  |
| 19-0.1V | 4.339 | 4.290 | +0.049 | 0.321 | 0.321 | +0.000 |  |
| 20-0.1V | 4.147 | 4.147 | +0.000 | 0.449 | 0.469 | -0.020 |  |
| 21-0.1V | 4.472 | 4.472 | +0.000 | 0.444 | 0.476 | -0.032 |  |
| 22-0.1V | 4.360 | 4.344 | +0.016 | 0.520 | 0.520 | +0.000 |  |

- Flagged speakers: `[]`
