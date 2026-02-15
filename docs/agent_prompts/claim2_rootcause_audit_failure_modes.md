# Agent Prompt: Run Claim-2 Root-Cause Audit (Failure Modes + Global vs Guided Peaks)

## Goal
Diagnose why Claim-2 “LDV irreducible” effects are small or speaker-dependent by separating:
- **confidently-wrong** vs **uncertainly-wrong** guided-peak failures, and
- **global peak** vs **guided peak** competition in MIC–MIC GCC-PHAT.

This is analysis-only and must use **existing real WAVs**.

Spec: `docs/specs/claim2_rootcause_audit_failure_modes_spec.md`

---

## Quickstart

### Run (two cases)
```bash
python -u scripts/analyze_claim2_failure_modes.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_rootcause_audit_<YYYYMMDD_HHMMSS> \
  --test_min_center_sec 450 \
  --case metricfix_mic_only=results/claim2_micobs_ablation_metricfix_20260214_174101/student/mic_only_control \
  --case cm_snr_m15=results/claim2_coherence_trap_sweep_20260214_183626/cm_snr_-15db/teacher_mic_only_control
```

---

## What to read
- Primary report: `results/<run>/report.md`
- Per-window recomputation: `results/<run>/cases/<label>/per_window.jsonl`
- Case summaries: `results/<run>/cases/<label>/summary.json`

---

## Pitfalls
1) Do not use synthetic data.
2) Do not change thresholds silently; record the exact run_config.
3) Ensure corruption replay uses the JSONL `corruption` record (noise centers, occlusion, common-mode interference).
4) Remember `results/` is gitignored: if committing, use `git add -f results/<run>/`.

