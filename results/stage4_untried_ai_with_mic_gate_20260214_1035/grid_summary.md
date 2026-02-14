# Untried Directions Grid Summary

| config | direction_tags | signal_pair | pass_mode | pass_count | failing_speakers | run_dir |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_fft_band200_4000 | baseline | ldv_micl | omp_vs_raw | 2 | 18-0.1V, 19-0.1V, 20-0.1V | results\stage4_untried_ai_with_mic_gate_20260214_1035\baseline_fft_band200_4000 |
| F_welch_phat_band200_4000 | F | ldv_micl | omp_vs_raw | 1 | 18-0.1V, 20-0.1V, 21-0.1V, 22-0.1V | results\stage4_untried_ai_with_mic_gate_20260214_1035\F_welch_phat_band200_4000 |
| A_welch_coherence_weighted_band200_4000 | A+F | ldv_micl | omp_vs_raw | 1 | 18-0.1V, 20-0.1V, 21-0.1V, 22-0.1V | results\stage4_untried_ai_with_mic_gate_20260214_1035\A_welch_coherence_weighted_band200_4000 |
| D_plus_F_welch_phat_ldv_comp | D+F | ldv_micl | omp_vs_raw | 0 | 18-0.1V, 19-0.1V, 20-0.1V, 21-0.1V, 22-0.1V | results\stage4_untried_ai_with_mic_gate_20260214_1035\D_plus_F_welch_phat_ldv_comp |
| D_plus_H_plus_F_ldv_comp_cohmask | D+H+F | ldv_micl | omp_vs_raw | 0 | 18-0.1V, 19-0.1V, 20-0.1V, 21-0.1V, 22-0.1V | results\stage4_untried_ai_with_mic_gate_20260214_1035\D_plus_H_plus_F_ldv_comp_cohmask |
| B_narrowband_fusion_welch_phat | B+F | ldv_micl | omp_vs_raw | 1 | 18-0.1V, 19-0.1V, 20-0.1V, 21-0.1V | results\stage4_untried_ai_with_mic_gate_20260214_1035\B_narrowband_fusion_welch_phat |
| I_dual_baseline_joint_weighted | I+A+F | ldv_micl | omp_vs_raw | 2 | 20-0.1V, 21-0.1V, 22-0.1V | results\stage4_untried_ai_with_mic_gate_20260214_1035\I_dual_baseline_joint_weighted |
| E_full_segment_micl_micr_welch_weighted | E+A+F | micl_micr | theta_only | 1 | 19-0.1V, 20-0.1V, 21-0.1V, 22-0.1V | results\stage4_untried_ai_with_mic_gate_20260214_1035\E_full_segment_micl_micr_welch_weighted |