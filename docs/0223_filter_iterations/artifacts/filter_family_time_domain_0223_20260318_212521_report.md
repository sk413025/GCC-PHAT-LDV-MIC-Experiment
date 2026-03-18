# 0223 Time-Domain Filter Family Sweep

- Generated: 2026-03-18T21:25:28
- Data root: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223`
- Slice sec: `5.0`

## Summary

| variant | valid_cases | physical_count | delta_tau_mae_ms | theta_v_mae_deg | max_delta_tau_err_ms | max_theta_v_err_deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ldv_diff2_bp500_2000 | 4 | 4 | 0.158 | 2.220 | 0.295 | 4.148 |
| baseline_ldv_diff_bp500_2000 | 4 | 4 | 0.199 | 2.806 | 0.358 | 5.043 |
| ldv_diff_rmsnorm_bp500_2000 | 4 | 4 | 0.205 | 2.879 | 0.379 | 5.335 |
| both_rmsnorm_ldv_diff_bp500_2000 | 4 | 4 | 0.205 | 2.879 | 0.379 | 5.335 |
| both_zscore_ldv_diff_bp500_2000 | 4 | 4 | 0.205 | 2.879 | 0.379 | 5.335 |
| ldv_preemph_diff_bp500_2000 | 4 | 4 | 0.345 | 4.855 | 0.577 | 8.118 |
| ldv_diff_signedsqrt_bp500_2000 | 4 | 4 | 0.382 | 5.368 | 0.764 | 10.752 |
| ldv_envelope_bp500_2000 | 4 | 4 | 0.475 | 6.684 | 0.775 | 10.897 |
| ldv_diff_envelope_bp500_2000 | 4 | 4 | 0.512 | 7.200 | 0.973 | 13.688 |