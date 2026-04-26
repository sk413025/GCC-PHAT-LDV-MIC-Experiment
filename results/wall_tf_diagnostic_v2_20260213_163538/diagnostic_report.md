# Wall Panel Transfer Function Diagnostic Report

**Generated**: 2026-02-13T16:38:37.217785
**Speakers**: 18, 19, 20, 21, 22

## Checkpoint Summary

| CK | Name | Status |
|----|------|--------|
| 0 | Data sanity (18) | PASS |
| 1 | MicL-MicR control (18) | 0/5 PASS |
| 2 | LDV quality (18) | 5/5 PASS |
| 0 | Data sanity (19) | PASS |
| 1 | MicL-MicR control (19) | 0/5 PASS |
| 2 | LDV quality (19) | 5/5 PASS |
| 0 | Data sanity (20) | PASS |
| 1 | MicL-MicR control (20) | 0/5 PASS |
| 2 | LDV quality (20) | 5/5 PASS |
| 0 | Data sanity (21) | PASS |
| 1 | MicL-MicR control (21) | 0/5 PASS |
| 2 | LDV quality (21) | 5/5 PASS |
| 0 | Data sanity (22) | PASS |
| 1 | MicL-MicR control (22) | 0/5 PASS |
| 2 | LDV quality (22) | 5/5 PASS |
| 3 | γ² freq profile | ratio=1.47 (does NOT support wall modal) |
| 4 | Phase behaviour | jumps=5, roughness_ratio=1.0, R²=0.971 |
| 5 | Speaker independence | γ² corr=0.946, jump overlap=0.482 |

## Per-Speaker Diagnostic Summary

| Speaker | Pair | γ²_mid | γ²_ratio | R² | Jumps | Roughness | frac_correct |
|---------|------|--------|----------|-----|-------|-----------|-------------|
| 18 | LDV_MicL | 0.2505 | 1.47 | 0.971 | 5 | 0.506 | 0.013 |
| 18 | LDV_MicR | 0.5446 | 0.43 | 0.950 | 5 | 0.488 | 0.020 |
| 18 | MicL_MicR | 0.2768 | 1.61 | 0.901 | 8 | 0.523 | 0.053 |
| 19 | LDV_MicL | 0.2574 | 1.75 | 0.965 | 7 | 0.491 | 0.013 |
| 19 | LDV_MicR | 0.5510 | 0.64 | 0.951 | 6 | 0.475 | 0.020 |
| 19 | MicL_MicR | 0.2899 | 1.71 | 0.928 | 7 | 0.514 | 0.059 |
| 20 | LDV_MicL | 0.2200 | 2.10 | 0.966 | 5 | 0.505 | 0.007 |
| 20 | LDV_MicR | 0.5151 | 0.71 | 0.947 | 4 | 0.448 | 0.026 |
| 20 | MicL_MicR | 0.3004 | 2.03 | 0.932 | 8 | 0.508 | 0.086 |
| 21 | LDV_MicL | 0.2388 | 1.94 | 0.964 | 5 | 0.492 | 0.013 |
| 21 | LDV_MicR | 0.5401 | 0.62 | 0.950 | 5 | 0.452 | 0.033 |
| 21 | MicL_MicR | 0.3071 | 1.82 | 0.911 | 8 | 0.533 | 0.059 |
| 22 | LDV_MicL | 0.2487 | 1.49 | 0.947 | 5 | 0.485 | 0.020 |
| 22 | LDV_MicR | 0.5592 | 0.64 | 0.911 | 7 | 0.481 | 0.020 |
| 22 | MicL_MicR | 0.3260 | 1.76 | 0.936 | 5 | 0.494 | 0.072 |

## Hypothesis Verdict

**Best match**: `nonlinear_ldv` (score: 50%)

| Hypothesis | Score | Criteria met |
|------------|-------|-------------|
| wall_modal | 1/7 | speaker_corr_gt0.5 |
| broadband_noise | 1/5 | frac_correct_approx0 |
| room_modal | 1/4 | jump_count_3_5 |
| clock_drift | 1/4 | R2_gt0.95 |
| nonlinear_ldv | 2/4 | jump_count_5_15, gamma2_mid_0.1_0.3 |

### Observed values vs. predictions

- Phase jumps [500-2000 Hz]: **5** (wall modal predicts 30-60)
- γ² median [500-2000 Hz]: **0.2487** (wall modal predicts 0.03-0.12)
- γ² ratio (low/mid): **1.75** (wall modal predicts >2)
- Phase R² [500-2000 Hz]: **0.965** (wall modal predicts <0.2)
- Narrowband correct τ₂ fraction: **1.3%** (wall modal predicts 10-30%)
- Phase roughness: **0.492** rad (wall modal predicts >1.5)
- Speaker γ² correlation: **0.946** (wall modal predicts >0.5)

### Wall-modal distinctive signatures found:
- Speaker γ² correlation > 0.5 (structural, not random)
