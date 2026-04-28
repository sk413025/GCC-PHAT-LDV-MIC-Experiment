# strict_v14 reproduction note

Local reproduction succeeded for commit `726d4a5247e2716b2fabc89cc2823eb7084a2ad2` in `worktree/repro-726d4a52`. This reproduces the `strict_v14` independent PI-GS audit result for the speech evaluation path.

Before running, `scripts/independent_pigs_audit.py` needed one minimal structural refactor: a too-deep nested loop was replaced with `itertools.product(...)` to fix `SyntaxError: too many statically nested blocks`. The change was structural only and preserved `strict_v14` semantics.

Command used:

```bash
python scripts/independent_pigs_audit.py --profile strict_v14 --offset_model affine --out_dir results\independent_pigs_audit_strict_v14_full_20260428_2303
```

Output directory:

```text
results/independent_pigs_audit_strict_v14_full_20260428_2303
```

Headline reproduced metrics from the successful run:

- Combined speech MAE: `1.2378519331656777`
- Combined speech max: `4.564810209091938`
- LORO MAE: `1.2378519331656777`
- Chirp MAE: `11.857073296336484`

These match the rounded `strict-v14` values documented in `docs/INDEPENDENT_PIGS_AUDIT_SUMMARY.md` (`1.24` combined/LORO MAE and `4.56` max), while the chirp result remains far from a full paper-table reproduction.
