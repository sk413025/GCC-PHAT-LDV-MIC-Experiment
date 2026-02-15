# Claim-2 Root-Cause Audit: Confidently-Wrong vs Uncertainly-Wrong, Global vs Guided Peaks

Generated: 2026-02-15T02:53:59.913486

Run dir: `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_rootcause_audit_20260215_025310`

## Definitions (guided peak)

- Confidently-wrong (CW): PSR >= 3.0 dB AND theta_error_ref >= 4.0 deg
- Confidently-right (CR): PSR >= 3.0 dB AND theta_error_ref < 4.0 deg
- Uncertainly-wrong (UW): PSR < 3.0 dB AND theta_error_ref >= 4.0 deg

## Case summary (pooled)

| case | n | guided p95 err (deg) | P(CW) | P(UW) | guided PSR median | P(global near0 & PSR>3) | global tau median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metricfix_mic_only | 318 | 4.625 | 0.053 | 0.660 | -4.30 | 0.006 | -1.667 |
| cm_snr_m15 | 318 | 4.581 | 0.198 | 0.094 | -9.90 | 1.000 | 0.000 |

## Per-speaker CW rates (guided)

### metricfix_mic_only

| speaker | n | P(CW) | P(UW) | P(global near0 & PSR>3) |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 78 | 0.000 | 0.526 | 0.000 |
| 19-0.1V | 53 | 0.000 | 0.698 | 0.000 |
| 20-0.1V | 49 | 0.020 | 0.816 | 0.020 |
| 21-0.1V | 63 | 0.079 | 0.698 | 0.016 |
| 22-0.1V | 75 | 0.147 | 0.640 | 0.000 |

### cm_snr_m15

| speaker | n | P(CW) | P(UW) | P(global near0 & PSR>3) |
| --- | ---: | ---: | ---: | ---: |
| 18-0.1V | 78 | 0.000 | 0.000 | 1.000 |
| 19-0.1V | 53 | 0.000 | 0.000 | 1.000 |
| 20-0.1V | 49 | 0.000 | 0.000 | 1.000 |
| 21-0.1V | 63 | 1.000 | 0.000 | 1.000 |
| 22-0.1V | 75 | 0.000 | 0.400 | 1.000 |

