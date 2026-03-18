# 2026-03-18 Round 2 Oracle Supervisor Memo

## Objective

This round intentionally used oracle knowledge of the correct `tau_VL` and
`tau_VR` values to answer a narrower question:

- which filter families make the correct lag peaks easiest to recover
- and which of those families remain useful after oracle information is removed

## Oracle Design Principle

The Supervisor approved an answer-conditioned search because the current
accepted baseline already has a stable non-oracle pairing rule.

Therefore the round objective was:

1. improve visibility of the correct lag peaks
2. inspect whether improved visibility transfers into the existing non-oracle
   pairing rule

The search was not allowed to change the promotion rule:

- accepted baseline replacement still depends on non-oracle metrics
- oracle-only wins are not enough

## Oracle Metrics

Each candidate was compared using:

- `oracle_hit_cases`
- `oracle_rank_sum_mean`
- `oracle_tau_abs_err_sum_mean_ms`
- `oracle_amp_ratio_mean`

Interpretation:

- lower `oracle_rank_sum_mean` is better
- lower oracle tau error is better
- higher amplitude ratio means the correct peak is more visible relative to the
  strongest peak

## Generalization Gate

A candidate was considered transferable only if it also improved the current
non-oracle pairing result:

- `non_oracle_delta_tau_mae_ms`
- `non_oracle_theta_v_mae_deg`
- `non_oracle_physical_count`

## Round 2 Findings

### Oracle-Optimal Family

Best oracle visibility:

- `diff2_bp700_1800`
- `oracle_rank_sum_mean = 6.000`
- `oracle_amp_ratio_mean = 0.630`

Decision:

- not accepted as a new baseline
- rejected as non-transferable

Reason:

- once oracle access is removed, non-oracle metrics collapse to
  `delta_tau_mae_ms = 0.350`
- the filter makes the correct peak visible, but not naturally top-ranked under
  the fixed non-oracle pairing rule

### Most Transferable Oracle Candidate

Best transfer from oracle to non-oracle:

- `diff2_bp500_2000`
- `oracle_rank_sum_mean = 6.750`
- `non_oracle_delta_tau_mae_ms = 0.158`
- `non_oracle_theta_v_mae_deg = 2.220`

Decision:

- accepted as a strong backup
- still worse than the current accepted baseline

### Current Winner Still Stands

Current accepted baseline remains:

- `ldv_diff_bp700_1800`
- `non_oracle_delta_tau_mae_ms = 0.152`
- `non_oracle_theta_v_mae_deg = 2.147`

Reason:

- it is not the most oracle-friendly filter
- but it is the most stable when oracle information is removed

## Supervisor Interpretation

The most important outcome of this round is:

- oracle-optimal is not the same as deployment-optimal

The answer-conditioned search suggests that second-derivative filters can
surface the correct peaks more strongly, but they also make the ranking more
fragile under blind selection.

The accepted `ldv_diff_bp700_1800` filter is therefore best interpreted as the
most transferable compromise:

- not the strongest teacher-side filter
- but the strongest student-side filter under the existing pairing rule

## Next-Step Instruction

If Round 3 is opened later, the next task should not blindly chase the best
oracle score. It should instead focus on:

- why `diff2_bp700_1800` exposes the correct peaks but fails non-oracle
- whether proxy scores such as PSR or local margin can bridge that teacher-
  student gap
