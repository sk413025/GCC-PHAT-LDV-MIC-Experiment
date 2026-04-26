# Claim-2 Mic-Observation Ablation Report

Generated: 2026-02-14T14:58:05.969417

Run dir: `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_micobs_ablation_fix_20260214_144138`

## Fixed near-fail setting

- corruption: SNR=0.0 dB in-band (500–2000), seed=1337, preclip_gain=100.0, clip_limit=0.99
- occlusion: target=micr, kind=lowpass, lowpass_hz=800.0

## Teacher identity checks

- actions/noise centers/forbidden mask identical across obs modes: PASS

## Near-fail precondition (baseline MIC–MIC)

- baseline fail_rate_ref(>5°): 0.453 (>= 0.40 required) => PASS

## Pooled test metrics (vs chirp reference)

| obs_mode | baseline p95 | baseline fail | student p95 | student fail |
| --- | ---: | ---: | ---: | ---: |
| ldv_mic | 10.606 | 0.453 | 7.387 | 0.195 |
| mic_only_control | 10.606 | 0.453 | 6.753 | 0.214 |
| mic_only_coh_only | 10.606 | 0.453 | 6.691 | 0.192 |
| mic_only_psd_only | 10.606 | 0.453 | 7.111 | 0.233 |

## Key deltas (student vs student)

| comparison | p95 improvement frac | fail-rate improvement frac |
| --- | ---: | ---: |
| ldv_mic_vs_mic_only_control | -0.094 | 0.088 |
| mic_only_control_vs_mic_only_coh_only | -0.009 | -0.115 |
| mic_only_control_vs_mic_only_psd_only | 0.050 | 0.081 |

## Interpretation rules (pre-registered)

- Coherence nearly sufficient if coh_only is within <=5% p95 and <=0.02 abs fail-rate of mic_only_control.
- PSD materially helps beyond coherence if psd_only closes >=50% of the p95 gap between coh_only and control.
- LDV adds marginal info beyond strong mic-only if ldv_mic improves vs mic_only_control by >=10% p95 and >=20% fail-rate (relative).

## Negative transfer check (per speaker)

- Comparison: `ldv_mic` student vs `mic_only_control` student (test windows only)
- Flag if: p95_ldv > 1.05 * p95_mic OR fail_ldv > fail_mic + 0.02

| speaker | p95_ldv | p95_mic | Δp95 | fail_ldv | fail_mic | Δfail | flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 18-0.1V | 5.645 | 6.405 | -0.760 | 0.090 | 0.115 | -0.026 |  |
| 19-0.1V | 8.127 | 8.688 | -0.561 | 0.151 | 0.189 | -0.038 |  |
| 20-0.1V | 13.832 | 9.037 | +4.795 | 0.408 | 0.327 | +0.082 | YES |
| 21-0.1V | 5.795 | 5.804 | -0.010 | 0.222 | 0.238 | -0.016 |  |
| 22-0.1V | 5.655 | 5.655 | +0.000 | 0.173 | 0.240 | -0.067 |  |

- Flagged speakers: `['20-0.1V']`
