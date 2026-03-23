# 2026-03-23 Round 5 Lane 1 Validator Note

## Validation Scope

Validated artifact bundle:

- `artifacts/round5_pair_formation_0223_20260323_225242_report.md`
- `artifacts/round5_pair_formation_0223_20260323_225242_summary.json`

Validation rule for this lane:

- pair formation must improve strict rescue-rank first
- then that improvement must survive the fixed blind scorer

## Validation Result

Lane 1 passed the structural test but failed the deployment test.

Observed:

- `cond_prune_medium` improved strict rescue pair rank to `3 / 3`
- rescue hit@5 also improved to `5 / 4`
- but central hard-case selections were still wrong
- hard-case `delta_tau_mae_ms` worsened from `0.249` to `0.769`

## Validator Judgment

Verdict:

- `REJECT FOR PROMOTION`
- `KEEP AS STRUCTURAL POSITIVE EVIDENCE`

Reason:

- rescue pairs were no longer buried
- but the final blind scorer did not convert that structural gain into correct
  top-1 selection

This means the lane succeeded only at the first half of the validator contract.

## Important Finding

The key information from lane 1 is not the worse MAE by itself.

It is this separation:

- pair formation can expose the rescue pair
- blind ranking still prefers the wrong family after exposure

That sharply narrows the next move.

The validator therefore treats round 5 lane 1 as:

- not deployable
- but strongly informative

## Validator Conclusion

Do not promote any lane 1 pair-formation-only family.

Use the better rescue structure as the input for the next scorer lane.
