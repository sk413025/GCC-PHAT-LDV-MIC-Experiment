# Claim-2 Root-Cause Audit: Confidently-Wrong vs Uncertainly-Wrong, Global vs Guided Peaks

Generated: 2026-02-15T02:54:33.453217

Run dir: `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_rootcause_audit_smoke_20260215_025425`

## Definitions (guided peak)

- Confidently-wrong (CW): PSR >= 3.0 dB AND theta_error_ref >= 4.0 deg
- Confidently-right (CR): PSR >= 3.0 dB AND theta_error_ref < 4.0 deg
- Uncertainly-wrong (UW): PSR < 3.0 dB AND theta_error_ref >= 4.0 deg

## Case summary (pooled)

| case | n | guided p95 err (deg) | P(CW) | P(UW) | guided PSR median | P(global near0 & PSR>3) | global tau median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| metricfix_smoke | 4 | 3.142 | 0.000 | 0.000 | 13.07 | 1.000 | 0.000 |
| cm_snr_smoke | 6 | 3.130 | 0.000 | 0.000 | 17.34 | 1.000 | 0.000 |

## Per-speaker CW rates (guided)

### metricfix_smoke

| speaker | n | P(CW) | P(UW) | P(global near0 & PSR>3) |
| --- | ---: | ---: | ---: | ---: |
| 20-0.1V | 4 | 0.000 | 0.000 | 1.000 |

### cm_snr_smoke

| speaker | n | P(CW) | P(UW) | P(global near0 & PSR>3) |
| --- | ---: | ---: | ---: | ---: |
| 20-0.1V | 6 | 0.000 | 0.000 | 1.000 |

