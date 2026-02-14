# Agent Prompt: Run Claim 2 “Coherence Trap” — Common-Mode Coherent Interference (Existing WAVs Only)

## Goal
Create a **coherence trap** where:
- MicL–MicR coherence is **high** (misleading),
- but MIC–MIC guided GCC-PHAT becomes **near-failing** vs chirp-reference truth,
then run the standard mic-observation ablation suite to quantify whether LDV adds marginal predictive power beyond mic-only observation variants.

The exact definitions and acceptance gates are in:
`docs/specs/claim2_coherence_trap_cm_interference_spec.md`.

---

## Quickstart

### 1) Smoke sweep (fast, real data)
```bash
python -u scripts/sweep_claim2_coherence_trap_cm_interference.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_coherence_trap_sweep_smoke_<YYYYMMDD_HHMMSS> \
  --smoke 1
```

Read:
- `results/<run>/report.md`
- `results/<run>/sweep_summary.json`

If the sweep chooses no level (exit code 2), stop and record a negative result.

### 2) Full sweep (choose cm_snr_db)
```bash
python -u scripts/sweep_claim2_coherence_trap_cm_interference.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_coherence_trap_sweep_<YYYYMMDD_HHMMSS> \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V
```

Read:
- `results/<run>/report.md` (chosen level section)

### 3) Ablation suite at the chosen level
Replace `<CHOSEN_CM_SNR_DB>` with the sweep’s chosen `cm_snr_db`:
```bash
python -u scripts/run_claim2_micobs_ablation_suite.py \
  --data_root /home/sbplab/jiawei/data \
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \
  --out_dir results/claim2_coherence_trap_ablation_<YYYYMMDD_HHMMSS> \
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V \
  --cm_interf_enable 1 \
  --cm_interf_snr_db <CHOSEN_CM_SNR_DB> \
  --cm_interf_seed 1337 \
  --coupling_hard_forbid_enable 0 \
  --dynamic_coh_gate_enable 1 \
  --dynamic_coh_min 0.05 \
  --tau_ref_gate_enable 0
```

Read:
- `results/<run>/teacher_identity.json` (must PASS)
- `results/<run>/report.md` and `results/<run>/summary_table.json`

---

## What to check (pass/fail gates)
1) **Sweep pass**: at least one grid point meets both:
   - Near-fail: high tail failure + low PSR-good fraction
   - Coherence trap: median mic coherence is high
2) **Ablation validity**:
   - Teacher identity checks pass (actions/noise/forbidden identical across obs modes)
   - Non-empty test windows for every variant
   - No NaNs in metrics

---

## Pitfalls (do not do this)
1) Do not optimize or filter windows using truth (no cherry-picking).
2) Do not change the window grid or RMS speech-active filter.
3) Do not “fix” the coherence trap by enabling `tau_ref_gate` in the sweep; it can trivially forbid the trap.
4) Ensure the interference is **common-mode** (same waveform added to both mics) and **in-band** (500–2000 Hz).
5) Remember `results/` is gitignored: commit artifacts with `git add -f results/<run>/`.

