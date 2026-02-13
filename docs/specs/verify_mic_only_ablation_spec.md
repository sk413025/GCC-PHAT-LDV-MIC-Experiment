# Spec: Mic-Only Ablation (LDV Irreducible Information Test)

## Scope
This spec defines a controlled ablation to quantify whether LDV provides predictive information for band selection that cannot be replicated by MIC-only features.

## Key idea
- Keep **teacher actions identical** (truth-guided MIC-MIC score).
- Change only the **student observation vector**.

## Observation modes (locked)
Let per-band features be z-scored across bands within the same window:
- `f_ldv[b]`: log-PSD of LDV (Welch) aggregated to bands
- `f_mic[b]`: log-PSD of mic average `0.5*(MicL+MicR)`
- `f_coh[b]`: MicL–MicR magnitude-squared coherence in the window
- `f_cpl_*[b]`: band coupling metric computed from silence windows

Modes:
- `ldv_mic` (default):
  - `obs[b] = z(f_ldv)[b] + z(f_mic)[b] + z(f_coh)[b] - 2*z(f_cpl_max)[b]`
- `mic_only_strict`:
  - `obs[b] = z(f_mic)[b] + z(f_coh)[b] - 2*z(f_cpl_mic_only)[b]`
- `ldv_only`:
  - `obs[b] = z(f_ldv)[b] - 2*z(f_cpl_max)[b]`

## Coupling / forbidden bands (locked)
- Forbidden bands always derive from `f_cpl_max[b]`, computed from silence windows using:
  - `max(γ²_mic, γ²_ldvL, γ²_ldvR)` per frequency, then band-mean.
- Degenerate-case handling remains: if raw forbid would forbid all bands, effective forbid is empty.

In `mic_only_strict`, `f_cpl_mic_only[b]` is computed only from MIC coherence during silence and is used as a penalty feature only.

## Acceptance (Claim 2)
Compare LDV+MIC student vs MIC-only student on pooled test windows (vs chirp reference):
- p95 improvement (LDV+MIC vs MIC-only) ≥ 10%
- fail-rate improvement ≥ 20%

Report file: `results/ablation_compare_<ts>/ablation_report.md`

