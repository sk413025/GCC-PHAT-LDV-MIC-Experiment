# Coherence-Trap Sweep: Common-Mode Interference (Teacher Baseline)

Generated: 2026-02-14T18:59:49.737772

Run dir: `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_coherence_trap_sweep_20260214_183626`

## Selection criteria

- Near-fail: fail_rate(theta_error_ref_deg > 4.0°) >= 0.40 AND frac(PSR>3.0dB) <= 0.10
- Coherence trap: median(mic_coh_band_median) >= 0.20
- Windows summarized: center_sec > 450

## Sweep results (test windows, pooled)

| cm_snr_db | fail_rate(>4°) | frac_psr(>3dB) | psr_median | coh_median | near_fail | coh_trap |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| +20 | 0.682 | 0.088 | -4.26 | 0.021 | PASS | FAIL |
| +10 | 0.560 | 0.154 | -5.35 | 0.036 | FAIL | FAIL |
| +5 | 0.541 | 0.252 | -6.24 | 0.095 | FAIL | FAIL |
| +0 | 0.484 | 0.352 | -8.06 | 0.273 | FAIL | PASS |
| -5 | 0.390 | 0.352 | -9.15 | 0.564 | FAIL | PASS |
| -10 | 0.346 | 0.352 | -9.72 | 0.806 | FAIL | PASS |
| -15 | 0.292 | 0.352 | -9.90 | 0.918 | FAIL | PASS |

## Chosen level

- None (no grid point met both near-fail and coherence-trap criteria).
