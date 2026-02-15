# Spec: Claim-2 Root-Cause Audit — Confidently-Wrong vs Uncertainly-Wrong, Global vs Guided Peaks

## Purpose
This analysis-only experiment audits why Claim-2 “LDV irreducible” effects are small or speaker-dependent by separating two fundamentally different failure modes for MIC–MIC guided GCC-PHAT:

1) **Uncertainly-wrong**: low confidence (PSR low) and high error.
2) **Confidently-wrong**: high confidence (PSR high) but high error (a “trap” regime).

It also diagnoses whether the **global GCC peak** (within ±maxlag) competes with or dominates the **guided peak** (around chirp-reference `tau_ref_ms`), which indicates multipath/coupling-driven peak selection.

This experiment:
- Uses **existing real WAVs only**.
- Replays the exact mic corruption described in existing teacher/student JSONL artifacts.
- Recomputes guided and global peaks on corrupted MicL/MicR windows.

No model, teacher, or student is modified.

---

## Inputs (locked)
- `data_root`: `/home/sbplab/jiawei/data`
- `truth_ref_root`: `results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000`
- Analysis band: `500–2000 Hz`
- Windowing: `5.0 s`
- GCC-PHAT: `max_lag = ±10 ms`, guided radius `±0.3 ms`
- PSR exclude samples: `50`
- Default analysis subset: `center_sec > 450` (aligns with DTmin test split)

The script accepts one or more cases:
- A **teacher dir** containing `per_speaker/*/windows.jsonl`, or
- A **student dir** containing `test_windows.jsonl`.

---

## Definitions (guided peak)
Let the guided peak be the peak found within `tau_ref_ms ± 0.3 ms` on the MIC–MIC GCC-PHAT correlation window.

Thresholds:
- `theta_fail_deg = 4.0`
- `psr_good_db = 3.0`

Classes:
- **CW (confidently-wrong)**: `PSR_guided >= 3 dB` AND `theta_error_ref_deg >= 4 deg`
- **CR (confidently-right)**: `PSR_guided >= 3 dB` AND `theta_error_ref_deg < 4 deg`
- **UW (uncertainly-wrong)**: `PSR_guided < 3 dB` AND `theta_error_ref_deg >= 4 deg`

Global-peak flag:
- **global near-0** if `|tau_global_ms| < 0.3 ms` AND `PSR_global >= 3 dB`

---

## Procedure (per window)
For each JSONL row (speaker_id, center_sec):
1) Load MicL/MicR WAVs from `data_root/<speaker>/*LEFT*.wav` and `*RIGHT*.wav`.
2) Extract clean 5s windows centered at `center_sec`.
3) Replay corruption from the JSONL `corruption` record:
   - occlusion magnitude shaping (if enabled),
   - common-mode coherent interference (if enabled),
   - independent in-band noise+clip corruption (if enabled).
4) Compute MIC–MIC GCC-PHAT over 500–2000 Hz and extract a ±10 ms correlation window.
5) Compute:
   - guided peak (tau_guided_ms, psr_guided_db),
   - global peak (tau_global_ms, psr_global_db),
   - theta errors vs chirp-reference truth.
6) Classify into CW/CR/UW.

---

## Outputs (locked)
Under `results/<run_name>/`:
- `run.log`, `run_config.json`, `code_state.json`
- `report.md` (pooled + per-speaker tables)
- `summary.json`
- `cases/<label>/per_window.jsonl` (recomputed peaks + flags)
- `cases/<label>/summary.json`
- `cases/<label>/manifest.json` (WAV + input JSONL sha256 list)

---

## Acceptance (validity)
A run is considered valid if:
- No NaNs in peak metrics.
- Non-empty analyzed window set for every case.
- The report clearly separates CW vs UW rates per speaker.

Interpretation goal (not pass/fail):
- If CW dominates, the next stressor should target “confidently-wrong” regimes (e.g., coherent interferer with wrong TDOA).
- If UW dominates, improvements should target SNR/coherence conditioning rather than “trap avoidance”.

