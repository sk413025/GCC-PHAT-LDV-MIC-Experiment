# Coherence-Trap Sweep: Common-Mode Interference (Teacher Baseline)

Generated: 2026-02-14T18:36:07.995529

Run dir: `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_coherence_trap_sweep_smoke_20260214_183545`

## Selection criteria

- Near-fail: fail_rate(theta_error_ref_deg > 4.0°) >= 0.40 AND frac(PSR>3.0dB) <= 0.10
- Coherence trap: median(mic_coh_band_median) >= 0.20
- Windows summarized: center_sec > -1

## Sweep results (test windows, pooled)

| cm_snr_db | fail_rate(>4°) | frac_psr(>3dB) | psr_median | coh_median | near_fail | coh_trap |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| +20 | 0.000 | 0.333 | -1.57 | 0.115 | FAIL | FAIL |
| +10 | 0.000 | 0.667 | 5.92 | 0.166 | FAIL | FAIL |
| +5 | 0.000 | 0.833 | 10.38 | 0.250 | FAIL | PASS |
| +0 | 0.000 | 1.000 | 13.98 | 0.451 | FAIL | PASS |
| -5 | 0.000 | 1.000 | 15.86 | 0.692 | FAIL | PASS |
| -10 | 0.000 | 1.000 | 17.06 | 0.858 | FAIL | PASS |
| -15 | 0.000 | 1.000 | 17.34 | 0.935 | FAIL | PASS |

## Chosen level

- None (no grid point met both near-fail and coherence-trap criteria).
