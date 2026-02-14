# Claim 2 Verification Report: Occlusion + Noise/Clipping Sweep

Generated: 2026-02-14T04:13:14.305386

Run dir: `/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/claim2_irreducible_occlusion_20260214_005000`


## Grid

| tag | baseline fail_rate_ref(>5°) | LDV+MIC student fail_rate_ref(>5°) | LDV beats MIC-only | PASS |
| --- | ---: | ---: | ---: | ---: |
| snr_p0_occ_lp800_micr | 0.453 | 0.035 | False | False |
| snr_m5_occ_lp800_micr | 0.355 | 0.060 | False | False |
| snr_m10_occ_lp800_micr | 0.220 | 0.072 | False | False |
| snr_p0_occ_lp600_micr | 0.233 | 0.075 | False | False |
| snr_m5_occ_lp600_micr | 0.192 | 0.063 | False | False |
| snr_m10_occ_lp600_micr | 0.179 | 0.069 | False | False |
| snr_p0_occ_tilt2_micr | 0.248 | 0.116 | False | False |
| snr_m5_occ_tilt2_micr | 0.267 | 0.101 | False | False |
| snr_m10_occ_tilt2_micr | 0.198 | 0.066 | False | False |

## Final
- claim2_supported_any: **False**
- near_fail_observed_any: True
- note: FAIL: near-fail occurred but LDV+MIC did not meet feasibility and/or did not beat MIC-only.