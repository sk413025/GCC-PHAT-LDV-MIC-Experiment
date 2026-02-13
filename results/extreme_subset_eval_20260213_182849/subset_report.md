# Extreme Subset Evaluation Report
Generated: 2026-02-13T18:28:56.195493
Run dir: /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/extreme_subset_eval_20260213_182849
## Subset definitions (truth-free)
- LOW_COH: bottom 10% by median MSC(MicL,MicR) in 500–2000 Hz
- HIGH_HF_IMB: top 10% by |HF imbalance| in 2–8 kHz
- CLIPPED: clip_frac_max >= 0.001
- LOW_RMS: bottom 10% by RMS(MicL) dB

## Acceptance (Claim 1)
- subsets meeting (p95>=15% AND fail>=20%): 3
- near-fail subset met (baseline fail>=0.40 and student<=0.10): False
- OVERALL: FAIL

## Pooled metrics (vs chirp reference)

### LOW_COH (n=32)
| Method | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2.195 | 5.861 | 5.899 | 0.344 |
| student | 0.324 | 0.848 | 2.014 | 0.031 |
- p95 improvement frac: 0.659
- fail-rate improvement frac: 0.909

### HIGH_HF_IMB (n=32)
| Method | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.161 | 4.697 | 5.335 | 0.094 |
| student | 0.298 | 0.709 | 2.564 | 0.000 |
- p95 improvement frac: 0.519
- fail-rate improvement frac: 1.000

### CLIPPED (n=0)
| Method | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: |
| baseline | nan | nan | nan | nan |
| student | nan | nan | nan | nan |
- p95 improvement frac: nan
- fail-rate improvement frac: nan

### LOW_RMS (n=32)
| Method | median | p90 | p95 | fail_rate(>5°) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.578 | 5.108 | 5.144 | 0.188 |
| student | 0.212 | 1.470 | 3.197 | 0.031 |
- p95 improvement frac: 0.379
- fail-rate improvement frac: 0.833
