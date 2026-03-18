# 2026-03-18 Round 3 Oracle Roadmap

## Status Before Round 3

Current accepted blind baseline:

- preprocessing: `ldv_diff`
- band: `700-1800 Hz`
- pairing: accepted `mean_tau` pairing rule
- `delta_tau_mae_ms = 0.152`
- `theta_v_mae_deg = 2.147`
- `physical_count = 4 / 4`

Round 2 established a teacher-student gap:

- `diff2_bp700_1800` is oracle-optimal for correct-peak visibility
- but it degrades badly once oracle access is removed
- `diff2_bp500_2000` transfers better, but still does not beat the blind
  baseline

Round 3 therefore does not optimize for oracle rank alone.
It optimizes for transfer:

- make the correct pair easier to select
- then prove that a blind rule can recover most of that gain

## Supervisor Objective

The supervisor is explicitly allowed to inspect the correct
`tau_VL` / `tau_VR` values during candidate generation.

However, promotion is still based on blind deployment metrics.

Round 3 question:

- which preprocessing or filter families increase the correct-pair margin
- which observable proxy features explain that gain
- which blind scoring rule can recover the teacher signal without using the
  answers

## Teacher Metrics

Oracle-side evaluation must report more than final `delta_tau` error.

Each candidate should report:

- `correct_vl_rank`
- `correct_vr_rank`
- `correct_pair_rank`
- `correct_pair_margin`
- `correct_peak_psr_vl`
- `correct_peak_psr_vr`
- `wrong_pair_suppression`
- `teacher_delta_tau_abs_err_ms`

Interpretation:

- low correct-pair rank is necessary but not sufficient
- a family that lifts the correct peak without widening the margin is fragile
- a family that wins only by answer-conditioned ranking cannot be promoted

## Blind Proxy Distillation

Round 3 should convert oracle observations into blind, observable features.

The first proxy table should contain:

- `mean_tau`
- `abs(delta_tau)`
- `pair_amp_product`
- `pair_balance`
- `psr_vl`
- `psr_vr`
- `pair_margin`
- `cross_band_agreement`
- `window_stability`

The first blind score family should start from:

`pair_amp_product * mean_tau_prior * delta_tau_gate * psr_weight * balance_weight`

The design goal is not to guess the answer directly.
The goal is to reproduce the structure that oracle scoring rewards:

- stable mean lag
- visible and isolated peaks
- suppression of strong but inconsistent wrong pairs

## Lane Assignment

### Lane A: Oracle Landscape Mining

Owner:

- engineer lane for teacher-side diagnostics

Deliverables:

- case-by-case `rank`, `margin`, `psr`, and `band sensitivity` tables
- failure breakdown for the hardest cases
- evidence for whether oracle wins come from true margin gain or only local
  rank reshuffling

Reject if:

- no per-case rank table
- no correct-vs-wrong margin analysis
- no explanation for why `diff2_bp700_1800` fails blindly

### Lane B: Blind Proxy Scoring

Owner:

- engineer lane for student-side scoring

Deliverables:

- blind pair-scoring families that do not use reference lags
- ablation of `mean_tau`, `psr`, `balance`, and `pair_margin`
- top-k candidate ranking traces for every case

Reject if:

- any reference-derived feature is used
- scoring still requires per-case tuning
- no candidate ranking trace is produced

### Lane C: Filter Family Transfer

Owner:

- engineer lane for local family exploration

Priority families:

- `diff1 + local energy normalization`
- `diff1/diff2 hybrid`
- `diff1 + coherence-weighted band mask`

Reject if:

- search expands to broad unfocused sweeps
- no candidate beats or tightly matches the baseline under blind scoring
- gains appear only on a single case

### Lane D: Sub-Band Robustness

Owner:

- engineer lane for band-structure analysis

Deliverables:

- per-band or grouped-band support for the correct lag
- cross-band agreement scores
- band rejection or band voting candidates that stay blind

Reject if:

- band selection depends on oracle-only labels at deployment
- improvements appear only in one fragile band configuration

### Lane E: Hard-Case Failure Analysis

Owner:

- engineer lane for worst-case diagnostics

Deliverables:

- why the worst cases fail: low margin, wrong peak hijack, window fragility, or
  cross-band inconsistency
- proposal for one targeted fix that remains globally deployable

Reject if:

- no case-level mechanism is identified
- the proposed fix only works for a single case

## Priority Search Space

Do not start with another large brute-force sweep.
Start from the current stable blind winner and explore local, explainable
changes.

### Priority 1: Diff1 + Local Energy Normalization

Families:

- `ldv_diff1 + running_rms_norm`
- `ldv_diff1 + local_zscore`
- `ldv_diff1 + clip_then_norm`

Initial ranges:

- bandpass: `600-1700`, `700-1800`, `700-2000`, `800-1800`
- normalization window: `20`, `40`, `60`, `80 ms`
- clipping percentile: `95`, `97`, `99`

Hypothesis:

- wrong peaks are often helped by short high-energy bursts or reverberant tails
- local normalization should widen the correct-vs-wrong margin without the
  blind fragility seen in pure `diff2`

### Priority 2: Diff1/Diff2 Hybrid

Families:

- waveform fusion: `a * diff1 + b * diff2`
- score fusion: `alpha * gcc(diff1) + (1 - alpha) * gcc(diff2)`
- candidate intersection between `diff1` and `diff2`

Initial ranges:

- `a:b = 1:0.25`, `1:0.5`, `1:1`
- `alpha = 0.6`, `0.7`, `0.8`, `0.9`
- merge window: `+-0.15`, `+-0.25`, `+-0.35 ms`

Hypothesis:

- `diff1` supplies ranking stability
- `diff2` supplies peak sharpening
- the useful transfer may come from mixing them, not replacing one with the
  other

### Priority 3: Diff1 + Coherence-Weighted Band Mask

Families:

- per-band weighting
- top-k band mask
- cross-band consistency voting

Initial ranges:

- STFT window: `512`, `1024`
- band bin width: `125 Hz`, `250 Hz`, `500 Hz`
- top-k bands: `2`, `4`, `6`, `8`

Hypothesis:

- the full band contains harmful regions that inflate wrong peaks
- blind band agreement may preserve transfer without answer leakage

### Priority 4: Diff1 + Envelope-Assisted Gating

Families:

- Hilbert envelope gating
- rectified-smoothed gating
- transient-only segment selection

Initial ranges:

- envelope LPF cutoff: `20`, `40`, `80 Hz`
- gate percentile: top `30%`, `40%`, `50%`, `60%`
- segment length: `100`, `200`, `300 ms`

Hypothesis:

- select windows where useful cross-modal activity is present
- reduce static resonance and tail-dominated windows

### Lower Priority

Only test these after the first four tracks produce a clear signal:

- mic-side symmetric cleanup
- adaptive notch or de-ringing
- aggressive stationary-band rejection

These are higher-risk for overfit and harder to interpret.

## Promotion Gates

Any candidate proposed for promotion must satisfy:

- `valid_cases = 4`
- `physical_count = 4`
- blind `delta_tau_mae_ms <= 0.18`
- blind `max_delta_tau_abs_err_ms <= 0.35`
- no non-physical selections
- no success caused only by left-right error cancellation

Promotion over the current accepted baseline requires:

- blind median `abs_err_delta_ms` improvement of at least `30%` over raw
  baseline
- blind worst-case `abs_err_delta_ms` improvement of at least `15%` over raw
  baseline
- at least `3 / 4` cases improved individually
- teacher-to-student median gap `<= 0.25 ms`
- teacher-to-student worst-case gap `<= 0.50 ms`
- window-stability standard deviation `< 0.35 ms`

Hard rejection conditions:

- oracle win with no blind improvement
- any reference-derived feature in the blind scorer
- per-case manual tuning
- improvement carried by only one case
- non-physical lag selection
- fragile success that disappears under minor window shifts

## Round 3 Deliverables

The supervisor should not accept another round without the following tracked
artifacts:

- supervisor memo for the round
- validator memo for the round
- one summary artifact per lane
- a comparison table against the current accepted baseline
- per-case top-k candidate traces
- one short transfer conclusion: why the winning method should generalize

## Immediate Next Step

Round 3 should begin with a narrow, auditable experiment matrix:

1. `diff1 + running_rms_norm + bp`
2. `diff1/diff2 score fusion + bp`
3. `diff1 + top-k band mask`

Do not open a larger search until at least one of these three families shows:

- a larger correct-pair margin under oracle metrics
- and a measurable blind gain over `ldv_diff_bp700_1800`

## Supervisor Conclusion

The next meaningful step is not to chase a stronger oracle-only filter.

The next meaningful step is to distill oracle structure into a blind proxy that
keeps the current baseline's stability while recovering some of `diff2`'s
teacher-side sharpness.
