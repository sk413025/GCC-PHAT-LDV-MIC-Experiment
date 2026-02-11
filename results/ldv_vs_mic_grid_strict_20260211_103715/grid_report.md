# LDV-vs-Mic Grid Comparison Report (GCC-PHAT Only)

Generated: 2026-02-11 11:07:23

## Purpose
Compare LDV-MicL and MicL-MicR under chirp and geometry guidance.

## Data & Truth Sources
- Speech data root: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\dataset\GCC-PHAT-LDV-MIC-Experiment`
- Chirp truth reference file: `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\exp-ldv-perfect-geometry-cloud\exp-validation\ldv-perfect-geometry\validation-results\stage4_doa_validation_speech_truthref_chirp_scan_guided_5s_local_20260209_170241\chirp_truthref_5s.json`

## Chirp Truth Reference Provenance
The chirp truth reference JSON was produced locally in the `exp-ldv-perfect-geometry-cloud` worktree.

Source commit: `a53c778281208309c00b3873c0353c7efe26e047` (Results: Stage4C local speech truth-ref chirp scan guided 5s)

Method string (from JSON): `gcc_phat_full_analysis scan 5s hop 1s, max_lag=10ms, psr_exclude=50, no bandpass`

Truth-ref entries (speaker -> chirp mic pair):

| speaker | label | tau_ms | theta_deg | center_sec | psr_db | left_wav | right_wav |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 18-0.1V | chirp 29 mic truth-ref (5s) | 1.585930 | 22.864486 | 9.5 | 12.384265 | `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\dataset\chirp\+0.8\0128-LEFT-MIC-29-chirp(+0.8m).wav` | `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\dataset\chirp\+0.8\0128-RIGHT-MIC-29-chirp(+0.8m).wav` |
| 19-0.1V | chirp 28 mic truth-ref (5s) | 0.914682 | 12.949784 | 8.5 | 12.743691 | `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\dataset\chirp\+0.4\0128-LEFT-MIC-28-chirp(+0.4m).wav` | `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\dataset\chirp\+0.4\0128-RIGHT-MIC-28-chirp(+0.4m).wav` |
| 20-0.1V | chirp 25 mic truth-ref (5s) | 0.222393 | 3.123379 | 3.5 | 12.499607 | `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\dataset\chirp\+0.0\0128-LEFT-MIC-25-chirp(0.0m).wav` | `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\dataset\chirp\+0.0\0128-RIGHT-MIC-25-chirp(0.0m).wav` |
| 21-0.1V | chirp 24 mic truth-ref (5s) | -0.635458 | -8.956648 | 12.5 | 13.365166 | `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\exp-ldv-perfect-geometry-cloud\_old_reports\24-chirp(-0.4m)\0128-LEFT-MIC-24-chirp(-0.4m).wav` | `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\exp-ldv-perfect-geometry-cloud\_old_reports\24-chirp(-0.4m)\0128-RIGHT-MIC-24-chirp(-0.4m).wav` |
| 22-0.1V | chirp 23 mic truth-ref (5s) | -1.307046 | -18.676611 | 8.5 | 10.873455 | `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\exp-ldv-perfect-geometry-cloud\_old_reports\23-chirp(-0.8m)\0128-LEFT-MIC-23-chirp(-0.8m).wav` | `C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\exp-ldv-perfect-geometry-cloud\_old_reports\23-chirp(-0.8m)\0128-RIGHT-MIC-23-chirp(-0.8m).wav` |

## Grid Summary

| truth_type | config | tau_err_max_ms | bandpass | pass_mode | pass_count | failing_speakers | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chirp | ldv_micl_chirp_tau2_band0 | 2.0 | 0-0 | omp_vs_raw | 4 | 22-0.1V | results\ldv_vs_mic_grid_strict_20260211_103715\ldv_micl_chirp_tau2_band0 |
| chirp | ldv_micl_chirp_tau2_band500_2000 | 2.0 | 500-2000 | omp_vs_raw | 4 | 19-0.1V | results\ldv_vs_mic_grid_strict_20260211_103715\ldv_micl_chirp_tau2_band500_2000 |
| chirp | micl_micr_chirp_tau2_band500_2000 | 2.0 | 500-2000 | theta_only | 5 |  | results\ldv_vs_mic_grid_strict_20260211_103715\micl_micr_chirp_tau2_band500_2000 |
| geometry | ldv_micl_geometry_tau2_band500_2000 | 2.0 | 500-2000 | omp_vs_raw | 3 | 18-0.1V, 21-0.1V | results\ldv_vs_mic_grid_strict_20260211_103715\ldv_micl_geometry_tau2_band500_2000 |
| chirp | ldv_micl_chirp_tau0p3_band500_2000 | 0.3 | 500-2000 | omp_vs_raw | 4 | 19-0.1V | results\ldv_vs_mic_grid_strict_20260211_103715\ldv_micl_chirp_tau0p3_band500_2000 |

Best pass count: 5/5 (config micl_micr_chirp_tau2_band500_2000).
