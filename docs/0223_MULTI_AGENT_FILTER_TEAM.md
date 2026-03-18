# 0223 Multi-Agent Filter Team Contract

This document defines the operating contract for continued 0223
LDV-MIC `delta_tau` improvement work in `exp/ldv-vs-mic-doa-comparison`.

## Goal

Improve or robustify the current best validated 0223 `delta_tau` recovery
pipeline while keeping the work auditable and safe to promote to git.

Current validated baseline:

- variant: `ldv_diff_bp500_2000`
- strategy: `amp_product_mean_tau_delta_quad`
- `lag_min_ms = 4.4`
- `delta_scale_ms = 0.6`
- `mean_tau_center_ms = 4.8`
- `mean_tau_scale_ms = 0.45`
- `delta_tau_mae_ms = 0.199`
- `theta_v_mae_deg = 2.806`
- `physical_count = 4 / 4`
- `max_delta_tau_abs_err_ms = 0.358`

## Team

### Supervisor

Owner:

- experiment plan
- task routing
- acceptance gate decisions
- final promotion recommendation

Responsibilities:

- assign disjoint write scopes
- reject candidates that fail mandatory tests
- require reruns when engineer outputs are incomplete
- maintain a compact summary of accepted and rejected attempts

Write scope:

- `docs/0223_MULTI_AGENT_FILTER_TEAM.md`
- `docs/0223_filter_iterations/`

### Engineer A: Time-Domain Filter Engineer

Focus:

- LDV-side time-domain transforms
- derivative families
- pre-emphasis families
- envelope or rectified variants
- simple temporal normalization

Write scope:

- `scripts/filter_family_time_domain_0223.py`
- `results/filter_family_time_domain_0223_*/`

### Engineer B: Spectral Filter Engineer

Focus:

- band-pass sweeps
- spectral flattening or whitening
- sub-band voting
- harmonic suppression or notch-style ideas

Write scope:

- `scripts/filter_family_spectral_0223.py`
- `results/filter_family_spectral_0223_*/`

### Engineer C: Pairing And Scoring Engineer

Focus:

- candidate extraction
- positive-lag gating
- sub-band pairing fusion
- PSR-aware ranking
- adaptive pair scoring

Write scope:

- `scripts/peak_pair_family_0223.py`
- `results/peak_pair_family_0223_*/`

### Validator

Owner:

- reproduce engineer outputs
- compare candidates against the baseline
- issue pass or reject recommendation

Write scope:

- `docs/0223_filter_iterations/validator_*.md`

## Mandatory Test Ladder

Every engineer candidate must pass all of the following before the Supervisor
can accept it for comparison:

1. Script compiles

```powershell
python -m py_compile <candidate_script.py>
```

2. Script runs end-to-end on the 4-case 0223 subset

3. Output contains:

- machine-readable summary
- human-readable report
- per-case error values

4. Candidate keeps:

- `valid_cases = 4`
- `physical_count = 4`

If any of the above fails, the Supervisor rejects the candidate and sends it
back to the owning engineer.

## Promotion Gates

### Minimum Acceptable Candidate

A candidate can be recorded as technically valid only if:

- `delta_tau_mae_ms <= 0.22`
- `theta_v_mae_deg <= 3.20`
- `max_delta_tau_abs_err_ms <= 0.40`
- `physical_count = 4`

### Baseline Replacement Gate

A candidate can replace the current baseline only if one of the following is
true:

- `delta_tau_mae_ms < 0.199`
- or `delta_tau_mae_ms <= 0.205` and `theta_v_mae_deg < 2.806`
- or `delta_tau_mae_ms <= 0.205` with a materially simpler or more robust
  configuration, as explicitly approved by the Supervisor

## Rejection Policy

The Supervisor must reject and return work when:

- the script does not run end-to-end
- results are missing per-case details
- the candidate is worse than the minimum gate
- the engineer modifies files outside the assigned write scope

## Git Promotion Rule

Only the following are allowed into the final commit:

- accepted scripts
- validation summaries
- iteration notes
- final tracked artifact bundle under `docs/`

Raw `results/` directories remain local unless intentionally copied into a
tracked `docs/` artifact bundle.
