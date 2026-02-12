# LDV Replacement Side-by-Side Comparison (DTmin, K=3, Strict)

Generated: 2026-02-12

## Scope
This report compares two LDV replacement directions under the same strict Stage4 protocol:
- `ldv_micl`: LDV replaces left-mic side pairing (reference run)
- `ldv_micr`: LDV replaces right-mic side pairing (newly completed run)

No new rerun is introduced in this report. It summarizes committed experiment artifacts.

## Source Artifacts
- L-side DTmin run root:
  - `results/ldv_vs_mic_grid_dtmin_strict_20260211_215857`
- R-side DTmin run root:
  - `results/ldv_vs_mic_grid_ldv_micr_dtmin_20260212_145737`
- L-side analysis:
  - `results/ldv_vs_mic_grid_dtmin_strict_20260211_215857/analysis/analysis_report.md`
- R-side analysis:
  - `results/ldv_vs_mic_grid_ldv_micr_dtmin_20260212_145737/analysis/analysis_report.md`
- Grid summaries:
  - `results/ldv_vs_mic_grid_dtmin_strict_20260211_215857/grid_summary.json`
  - `results/ldv_vs_mic_grid_ldv_micr_dtmin_20260212_145737/grid_summary.json`

## Shared Evaluation Conditions
- `alignment_mode=dtmin`
- `max_k=3`
- `pass_mode=omp_vs_raw` for LDV replacement configs
- Speakers: `18-0.1V, 19-0.1V, 20-0.1V, 21-0.1V, 22-0.1V`
- Chirp-truth path:
  - `C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/worktree/exp-ldv-perfect-geometry-cloud/exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation_speech_truthref_chirp_scan_guided_5s_local_20260209_170241/chirp_truthref_5s.json`

## L/R Pass Count Comparison

| matched condition | L (`ldv_micl`) | R (`ldv_micr`) | delta (R-L) |
| --- | ---: | ---: | ---: |
| chirp, tau=2.0, band=0-0 | 4/5 | 3/5 | -1 |
| chirp, tau=2.0, band=500-2000 | 4/5 | 4/5 | 0 |
| chirp, tau=0.3, band=500-2000 | 4/5 | 4/5 | 0 |
| geometry, tau=2.0, band=500-2000 | 3/5 | 4/5 | +1 |

## L/R Median Metric Comparison

| matched condition | L theta_err_median (deg) | R theta_err_median (deg) | delta R-L | L psr_median (dB) | R psr_median (dB) | delta R-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chirp, tau=2.0, band=0-0 | 2.929309 | 3.394569 | +0.465260 | -4.296013 | -7.474900 | -3.178887 |
| chirp, tau=2.0, band=500-2000 | 2.806957 | 3.033109 | +0.226152 | -28.049744 | -13.251107 | +14.798637 |
| chirp, tau=0.3, band=500-2000 | 2.806957 | 3.033109 | +0.226152 | -28.049744 | -13.251107 | +14.798637 |
| geometry, tau=2.0, band=500-2000 | 1.962231 | 2.509403 | +0.547172 | -1.378806 | -3.409515 | -2.030709 |

## Failure Speaker Distribution

| matched condition | L failing speakers | R failing speakers |
| --- | --- | --- |
| chirp, tau=2.0, band=0-0 | `22-0.1V` | `21-0.1V, 22-0.1V` |
| chirp, tau=2.0, band=500-2000 | `19-0.1V` | `20-0.1V` |
| chirp, tau=0.3, band=500-2000 | `19-0.1V` | `20-0.1V` |
| geometry, tau=2.0, band=500-2000 | `18-0.1V, 21-0.1V` | `22-0.1V` |

## Interpretation
- The two replacement directions are not strictly symmetric per condition.
- R-side (`ldv_micr`) is weaker on `chirp tau2 band0` (3/5 vs 4/5).
- R-side is stronger on `geometry tau2 band500_2000` (4/5 vs 3/5).
- Both sides tie on the two chirp band-limited settings (4/5 each), but fail on different speakers.
- Therefore, current evidence supports: both sides are workable under K=3 DTmin strict mode, but error/failure profiles are side-dependent and should both be reported.

## Per-Config Detail Paths
- L-side details:
  - `results/ldv_vs_mic_grid_dtmin_strict_20260211_215857/ldv_micl_chirp_tau2_band0/summary_table.md`
  - `results/ldv_vs_mic_grid_dtmin_strict_20260211_215857/ldv_micl_chirp_tau2_band500_2000/summary_table.md`
  - `results/ldv_vs_mic_grid_dtmin_strict_20260211_215857/ldv_micl_chirp_tau0p3_band500_2000/summary_table.md`
  - `results/ldv_vs_mic_grid_dtmin_strict_20260211_215857/ldv_micl_geometry_tau2_band500_2000/summary_table.md`
- R-side details:
  - `results/ldv_vs_mic_grid_ldv_micr_dtmin_20260212_145737/ldv_micr_chirp_tau2_band0/summary_table.md`
  - `results/ldv_vs_mic_grid_ldv_micr_dtmin_20260212_145737/ldv_micr_chirp_tau2_band500_2000/summary_table.md`
  - `results/ldv_vs_mic_grid_ldv_micr_dtmin_20260212_145737/ldv_micr_chirp_tau0p3_band500_2000/summary_table.md`
  - `results/ldv_vs_mic_grid_ldv_micr_dtmin_20260212_145737/ldv_micr_geometry_tau2_band500_2000/summary_table.md`

## Reproducibility Notes
- This document is derived from existing committed artifacts and does not alter raw run outputs.
- To regenerate this comparison, read the two `grid_summary.json` files and two `analysis_report.md` files listed above and recompute the paired table rows by matched `(truth_type, tau_err_max_ms, bandpass)`.
