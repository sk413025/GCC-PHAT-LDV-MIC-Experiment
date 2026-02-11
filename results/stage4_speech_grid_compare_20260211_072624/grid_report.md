# Stage4 Speech Grid Comparison Report (GCC-PHAT Only)

Generated: 2026-02-11 08:26:01

## Purpose
Run a systematic grid over geometry vs chirp truth guidance with 5 speakers and identify which configuration yields the most GCC-PHAT passes.

## Data & Truth Sources
- Speech data root: `dataset/GCC-PHAT-LDV-MIC-Experiment` (speakers 18-0.1V..22-0.1V)
- Chirp truth reference file (precomputed):
  - `worktree/exp-ldv-perfect-geometry-cloud/exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation_speech_truthref_chirp_scan_guided_5s_local_20260209_170241/chirp_truthref_5s.json`
  - Used as truth override for chirp-guided runs (tau_ref_ms/theta_ref_deg from this file)

## Grid Configuration
- segment_mode=scan, n_segments=5, analysis_slice_sec=5, eval_window_sec=5
- scan_start=100s, scan_end=600s, scan_hop=1s, scan_psr_min_db=-20, scan_sort_by=tau_err, scan_min_separation_sec=5
- gcc_guided_peak_radius_ms=0.3, ldv_prealign=gcc_phat

Configs:
1. tau_err_max_ms=2.0, bandpass=0-0
2. tau_err_max_ms=2.0, bandpass=500-2000
3. tau_err_max_ms=0.3, bandpass=0-0
4. tau_err_max_ms=0.3, bandpass=500-2000

Truth types:
- geometry (no override)
- chirp (override tau/theta from chirp_truthref_5s.json)

## Grid Summary (GCC-PHAT)

| truth_type | config | tau_err_max_ms | bandpass | pass_count | failing_speakers | run_dir |
| --- | --- | --- | --- | --- | --- | --- |
| geometry | tau2_band0 | 2.0 | 0-0 | 3 | 19-0.1V,21-0.1V | results\stage4_speech_geometry_tau2_band0_20260211_072624 |
| geometry | tau2_band500_2000 | 2.0 | 500-2000 | 3 | 18-0.1V,21-0.1V | results\stage4_speech_geometry_tau2_band500_2000_20260211_072624 |
| geometry | tau0p3_band0 | 0.3 | 0-0 | 2 | 18-0.1V,19-0.1V,21-0.1V | results\stage4_speech_geometry_tau0p3_band0_20260211_072624 |
| geometry | tau0p3_band500_2000 | 0.3 | 500-2000 | 3 | 18-0.1V,21-0.1V | results\stage4_speech_geometry_tau0p3_band500_2000_20260211_072624 |
| chirp | tau2_band0 | 2.0 | 0-0 | 4 | 22-0.1V | results\stage4_speech_chirp_tau2_band0_20260211_072624 |
| chirp | tau2_band500_2000 | 2.0 | 500-2000 | 4 | 19-0.1V | results\stage4_speech_chirp_tau2_band500_2000_20260211_072624 |
| chirp | tau0p3_band0 | 0.3 | 0-0 | 4 | 22-0.1V | results\stage4_speech_chirp_tau0p3_band0_20260211_072624 |
| chirp | tau0p3_band500_2000 | 0.3 | 500-2000 | 4 | 19-0.1V | results\stage4_speech_chirp_tau0p3_band500_2000_20260211_072624 |

Best pass count: 4/5 (all chirp truth configurations).
Geometry truth configurations maxed at 3/5.

## Per-Run Artifacts
Each run directory includes:
- `summary_table.md` (per-speaker GCC-PHAT summary)
- `{speaker}/summary.json` and `{speaker}/run.log`
- `subset_manifest.json`, `code_state.json`, `run_config.json`

Grid-level summary files:
- `results\stage4_speech_grid_compare_20260211_072624\grid_summary.md`
- `results\stage4_speech_grid_compare_20260211_072624\grid_summary.json`

## Smoke & Guardrail Tests
- Smoke (geometry): `results/stage4_speech_grid_compare_20260211_072624_smoke_geometry/`
- Smoke (chirp truth override): `results/stage4_speech_grid_compare_20260211_072624_smoke_chirp/`
- Guardrail (expected fail: zero segments): `results/stage4_speech_grid_compare_20260211_072624_guardrail/`

## Notes
- Chirp truth guidance consistently achieved 4/5 passes, failing either 19-0.1V or 22-0.1V depending on bandpass.
- Geometry guidance did not exceed 3/5; 21-0.1V was a persistent failure case across geometry settings.
