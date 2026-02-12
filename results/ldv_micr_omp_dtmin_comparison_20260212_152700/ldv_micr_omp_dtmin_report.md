# LDV-MicR Replacement Experiment (OMP vs DTmin, K=3)

Generated: 2026-02-12

## Background
- This run completes the missing `ldv_micr` branch (replace right microphone by LDV) under the same strict Stage4 protocol used for the existing `ldv_micl` branch.
- `max_k=3` is fixed for both OMP and DTmin, matching the pinned plan from commits `b63a4ab` and `3d54603`.

## Purpose
- Verify strict-pass behavior for `ldv_micr`.
- Compare OMP and DTmin with identical truth source, speakers, scan window, and pass criteria.

## Setup
- Branch: `exp/ldv-vs-mic-doa-comparison`
- Data root:
  - `C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/dataset/GCC-PHAT-LDV-MIC-Experiment`
- Chirp truth file:
  - `C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/worktree/exp-ldv-perfect-geometry-cloud/exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation_speech_truthref_chirp_scan_guided_5s_local_20260209_170241/chirp_truthref_5s.json`
- DTmin policy:
  - `C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/worktree/exp-ldv-vs-mic-doa-comparison/results/ldv_vs_mic_grid_dtmin_strict_20260211_215857/dtmin_model/model_dtmin_policy_k3.npz`
- Speakers: `18-0.1V, 19-0.1V, 20-0.1V, 21-0.1V, 22-0.1V`
- Scan/eval: `segment_mode=scan`, `scan_start_sec=100`, `scan_end_sec=600`, `scan_hop_sec=1`, `n_segments=5`, `analysis_slice_sec=5`, `eval_window_sec=5`
- Guided GCC and prealign: `gcc_guided_peak_radius_ms=0.3`, `ldv_prealign=gcc_phat`
- Pass mode: `omp_vs_raw`

## Data Lineage
- total_files: `15`
- fingerprint_sha256: `13cc941023969dc8cfc4bc4c1711cce02803228d8dcc1bd7fd5c126e1e84a3e5`

## Executed Commands
```bash
python -u scripts/run_ldv_vs_mic_grid.py \
  --data_root "C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/dataset/GCC-PHAT-LDV-MIC-Experiment" \
  --chirp_truth_file "C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/worktree/exp-ldv-perfect-geometry-cloud/exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation_speech_truthref_chirp_scan_guided_5s_local_20260209_170241/chirp_truthref_5s.json" \
  --alignment_mode omp \
  --max_k 3 \
  --signal_pairs ldv_micr \
  --output_base "results/ldv_vs_mic_grid_ldv_micr_omp_20260212_142946"

python -u scripts/run_ldv_vs_mic_grid.py \
  --data_root "C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/dataset/GCC-PHAT-LDV-MIC-Experiment" \
  --chirp_truth_file "C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/worktree/exp-ldv-perfect-geometry-cloud/exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation_speech_truthref_chirp_scan_guided_5s_local_20260209_170241/chirp_truthref_5s.json" \
  --alignment_mode dtmin \
  --dtmin_model_path "C:/Users/Jenner/Documents/SBP Lab/LDVReorientation/worktree/exp-ldv-vs-mic-doa-comparison/results/ldv_vs_mic_grid_dtmin_strict_20260211_215857/dtmin_model/model_dtmin_policy_k3.npz" \
  --max_k 3 \
  --signal_pairs ldv_micr \
  --output_base "results/ldv_vs_mic_grid_ldv_micr_dtmin_20260212_145737"
```

## Artifacts
- OMP run root:
  - `results/ldv_vs_mic_grid_ldv_micr_omp_20260212_142946`
- DTmin run root:
  - `results/ldv_vs_mic_grid_ldv_micr_dtmin_20260212_145737`
- Per-config artifacts (both roots):
  - `summary_table.md`, `run_config.json`, `subset_manifest.json`, `code_state.json`
  - per-speaker `summary.json` and `run.log`
- Analysis outputs:
  - `results/ldv_vs_mic_grid_ldv_micr_omp_20260212_142946/analysis/analysis_report.md`
  - `results/ldv_vs_mic_grid_ldv_micr_dtmin_20260212_145737/analysis/analysis_report.md`

## Strict Pass Results

| config | OMP pass_count | DTmin pass_count | OMP failing_speakers | DTmin failing_speakers |
| --- | --- | --- | --- | --- |
| ldv_micr_chirp_tau2_band0 | 3/5 | 3/5 | 21-0.1V, 22-0.1V | 21-0.1V, 22-0.1V |
| ldv_micr_chirp_tau2_band500_2000 | 4/5 | 4/5 | 20-0.1V | 20-0.1V |
| ldv_micr_geometry_tau2_band500_2000 | 4/5 | 4/5 | 22-0.1V | 22-0.1V |
| ldv_micr_chirp_tau0p3_band500_2000 | 4/5 | 4/5 | 20-0.1V | 20-0.1V |

## Key Metrics (Median Across 5 Speakers)

| config | OMP theta_err_median_deg | DTmin theta_err_median_deg | OMP psr_median_db | DTmin psr_median_db |
| --- | --- | --- | --- | --- |
| ldv_micr_chirp_tau2_band0 | 3.395821 | 3.394569 | -7.488270 | -7.474900 |
| ldv_micr_chirp_tau2_band500_2000 | 3.034012 | 3.033109 | -13.210392 | -13.251107 |
| ldv_micr_geometry_tau2_band500_2000 | 2.516718 | 2.509403 | -3.418664 | -3.409515 |
| ldv_micr_chirp_tau0p3_band500_2000 | 3.034012 | 3.033109 | -13.210392 | -13.251107 |

## Log Interpretation
- OMP and DTmin produce the same strict pass counts across all `ldv_micr` configs.
- DTmin is numerically close to OMP BECAUSE the policy is trained from OMP trajectories and executed with the same `max_k=3` horizon.
- `tau2_band0` is consistently weaker (3/5) DUE TO broader-band ambiguity relative to 500-2000 Hz constrained runs.
- Guardrail run fails by design (`segment_mode=scan selected 0 segments`) in both grids, confirming fail-fast enforcement.

## Analysis (Successes and Failures)
- Success: `4/5` is stable for band-limited `ldv_micr` configs BECAUSE guided GCC and scan gating reduce unstable windows.
- Failure: speaker `20-0.1V` remains failing in both chirp band-limited configs DUE TO segment-level mismatch where aligned result is not strictly better than raw under `omp_vs_raw`.
- Failure: speaker `22-0.1V` remains failing in `tau2_band0` and geometry band-limited config, indicating persistent hard-case behavior independent of OMP/DTmin choice.
- Method comparison: DTmin does not degrade the strict result envelope; therefore the `ldv_micr` path is now covered for both alignment backends.

## Next Steps
1. Keep `max_k=3` as the default for OMP and DTmin in strict comparisons.
2. Focus targeted diagnostics on `20-0.1V` and `22-0.1V` using per-segment records in each config folder.
3. If further gains are needed, run controlled ablations on guided radius and band limits while keeping strict comparability fixed.

