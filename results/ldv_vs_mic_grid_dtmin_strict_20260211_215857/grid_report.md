# LDV-vs-Mic Grid Comparison Report (GCC-PHAT Only)

Generated: 2026-02-11 22:31:39

## Purpose
Compare LDV-MicL and MicL-MicR under chirp and geometry guidance.

## Data & Truth Sources
- Speech data root: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\dataset\GCC-PHAT-LDV-MIC-Experiment`
- Chirp truth reference file: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\exp-ldv-perfect-geometry-cloud\exp-validation\ldv-perfect-geometry\validation-results\stage4_doa_validation_speech_truthref_chirp_scan_guided_5s_local_20260209_170241\chirp_truthref_5s.json`

## Grid Summary

| truth_type | config | tau_err_max_ms | bandpass | pass_mode | alignment_mode | pass_count | failing_speakers | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chirp | ldv_micl_chirp_tau2_band0 | 2.0 | 0-0 | omp_vs_raw | dtmin | 4 | 22-0.1V | C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\exp-ldv-vs-mic-doa-comparison\results\ldv_vs_mic_grid_dtmin_strict_20260211_215857\ldv_micl_chirp_tau2_band0 |
| chirp | ldv_micl_chirp_tau2_band500_2000 | 2.0 | 500-2000 | omp_vs_raw | dtmin | 4 | 19-0.1V | C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\exp-ldv-vs-mic-doa-comparison\results\ldv_vs_mic_grid_dtmin_strict_20260211_215857\ldv_micl_chirp_tau2_band500_2000 |
| chirp | micl_micr_chirp_tau2_band500_2000 | 2.0 | 500-2000 | theta_only | dtmin | 5 |  | C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\exp-ldv-vs-mic-doa-comparison\results\ldv_vs_mic_grid_dtmin_strict_20260211_215857\micl_micr_chirp_tau2_band500_2000 |
| geometry | ldv_micl_geometry_tau2_band500_2000 | 2.0 | 500-2000 | omp_vs_raw | dtmin | 3 | 18-0.1V, 21-0.1V | C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\exp-ldv-vs-mic-doa-comparison\results\ldv_vs_mic_grid_dtmin_strict_20260211_215857\ldv_micl_geometry_tau2_band500_2000 |
| chirp | ldv_micl_chirp_tau0p3_band500_2000 | 0.3 | 500-2000 | omp_vs_raw | dtmin | 4 | 19-0.1V | C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\exp-ldv-vs-mic-doa-comparison\results\ldv_vs_mic_grid_dtmin_strict_20260211_215857\ldv_micl_chirp_tau0p3_band500_2000 |

Best pass count: 5/5 (config micl_micr_chirp_tau2_band500_2000).
