# 2026-03-18 Round 2 Oracle Validator Note

## Validation Scope

Validated artifact:

- `results/oracle_guided_filter_search_0223_20260318_215951/summary.json`

Tracked copy to be promoted under `docs/` after review.

Reference accepted baseline before this round:

- preprocessing: `ldv_diff`
- band: `700-1800 Hz`
- pairing: accepted `mean_tau` rule
- `delta_tau_mae_ms = 0.152`
- `theta_v_mae_deg = 2.147`

## Validation Question

Can an oracle-guided filter search reveal a filter family that is both:

1. strongly aligned with the correct lag peaks
2. still better after oracle information is removed

## Validation Result

### Teacher-Side Winner

- `diff2_bp700_1800`
- `oracle_rank_sum_mean = 6.000`
- `oracle_tau_abs_err_sum_mean_ms = 0.165`
- `oracle_amp_ratio_mean = 0.630`

Validator judgment:

- `TEACHER-ONLY WIN`
- not accepted for promotion

Reason:

- non-oracle result degrades to `delta_tau_mae_ms = 0.350`
- this indicates the filter improves correct-peak visibility but does not make
  the correct pair reliably selectable by the current blind rule

### Best Transferable Candidate

- `diff2_bp500_2000`
- `oracle_rank_sum_mean = 6.750`
- `oracle_tau_abs_err_sum_mean_ms = 0.099`
- `oracle_amp_ratio_mean = 0.641`
- `non_oracle_delta_tau_mae_ms = 0.158`
- `non_oracle_theta_v_mae_deg = 2.220`

Validator judgment:

- `TRANSFERABLE BUT NOT BETTER`

Reason:

- it transfers much better from oracle to non-oracle than `diff2_bp700_1800`
- but it still does not beat the currently accepted baseline `0.152 / 2.147`

### Baseline Stability Check

- `baseline_diff_bp700_1800`
- `oracle_rank_sum_mean = 9.500`
- `non_oracle_delta_tau_mae_ms = 0.152`
- `non_oracle_theta_v_mae_deg = 2.147`

Validator judgment:

- `KEEP CURRENT BASELINE`

## Key Lesson

The answer-conditioned search exposed a teacher-student gap:

- some filters improve the true peak rank
- but they do not preserve enough ranking stability for blind selection

Therefore, this round does not justify replacing the current accepted baseline.

## Generalizability Judgment

The most generalizable signal from this round is not a new promoted filter.
It is a design rule:

- evaluate filter families by both oracle recoverability and blind
  recoverability
- reject filters that only win under oracle selection

This means future rounds should prioritize proxy objectives that better match
the blind selection rule, for example:

- correct-vs-wrong peak margin
- PSR near the correct lag
- score gap between the correct pair and the best wrong pair

## Validator Recommendation

Commit:

- the oracle-guided search script
- supervisor and validator notes for this round
- tracked summary artifact bundle

Do not promote a new baseline from this round.
