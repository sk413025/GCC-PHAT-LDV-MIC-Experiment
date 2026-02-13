# Agent Prompt: Verify "Extreme Window" Feasibility (Existing WAVs Only)

## Mission
Verify whether the already-trained **LDV-assisted band-policy student** remains robust on **truth-free defined extreme speech windows** where mic-only tends to degrade.

This does **not** create real wind/occlusion/saturation data. It defines extreme windows using waveform diagnostics (coherence, HF imbalance, clipping, low RMS).

## Inputs (locked)
- Data root: `/home/sbplab/jiawei/data`
- Student run: a completed Band-DTmin run containing `test_windows.jsonl`
  - Example: `results/band_dtmin_student_20260213_175234/`

## Quickstart
```bash
python -u scripts/eval_speech_extreme_subsets.py \
  --data_root /home/sbplab/jiawei/data \
  --student_run results/band_dtmin_student_20260213_175234 \
  --out_dir results/extreme_subset_eval_<YYYYMMDD_HHMMSS>
```

## What to read
- `results/extreme_subset_eval_<ts>/subset_report.md`
  - Check `OVERALL: PASS/FAIL`
  - Check pooled tables for each subset
- `results/extreme_subset_eval_<ts>/subset_metrics.json` for exact numbers

## Acceptance (Claim 1)
We consider the claim supported if:
1) At least **2/4** subsets satisfy:
   - `p95(theta_error_ref_deg)` improvement ≥ 15% AND
   - `fail_rate(theta_error_ref_deg > 5°)` improvement ≥ 20%
2) At least **1** subset is "near-fail" for baseline but not for student:
   - baseline fail-rate ≥ 0.40 AND student fail-rate ≤ 0.10

If this fails, record it as a negative result: this dataset may not contain sufficiently extreme windows under these proxies.

## Pitfalls (do NOT do this)
1) Do not define extreme subsets using truth (no filtering based on theta/tau errors).
2) Do not change the test split: use `test_windows.jsonl` from the student run.
3) Do not write artifacts to repo root; always under `results/<run_name>/`.

