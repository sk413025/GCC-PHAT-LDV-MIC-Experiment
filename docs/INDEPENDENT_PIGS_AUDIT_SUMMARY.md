# Independent PI-GS Audit Summary

Generated from an independent raw-WAV pipeline that does not import the
project's existing research scripts.

## Research Rationale

For a plain-language explanation of how these experiments relate to the
paper's final PI-GS objective, including ASCII diagrams and a formula-to-code
map, see `docs/PIGS_FORMULA_EXPLAINER.md`.

The audit starts from the physical story in the paper but treats the existing
codebase as untrusted. The only trusted inputs are the WAV recordings, nominal
sensor geometry, and basic wave-propagation constraints.

The central physical hypothesis is that the LDV provides a structure-borne
reference that is more target-coherent than the microphone pressure channels.
If the LDV and microphones share a usable target-related component, the correct
source coordinate should make both LDV-Mic predicted delays agree at the same
time. A single LDV-Mic peak is not enough, because barrier re-radiation and
speech harmonics can create sharp but wrong peaks. The useful signal should
survive cross-pair, cross-window, and cross-frequency consistency checks.

The signal-processing hypothesis is that raw GCC-PHAT is too brittle for these
recordings. PHAT suppresses magnitude coloration, but it can also amplify
low-energy noisy bins and speech harmonic artifacts. That is why the audit
tests partial PHAT exponents, clipping, pre-emphasis, subband aggregation, and
chirp-derived calibration instead of assuming the manuscript result follows
from a plain full-band GCC.

## What Was Implemented

- Added `scripts/independent_pigs_audit.py`.
- Reimplemented LDV-Mic GCC and PI-GS-style spatial search using only
  `numpy` and `scipy`.
- Added multiple preprocessing/search variants:
  bandpass banks, PHAT exponent, clipping/pre-emphasis/differencing,
  moving-patch vs fixed-spot geometry, harmonic/product/sum scores,
  top-k window aggregation, and chirp-derived offset calibration.
- Added three offset modes:
  `none`, `constant`, `affine`, and `per_trial`.
- Added the second-stage improvement path:
  coherence-masked GCC, window-level consensus, and subband ensemble scoring.
- Added a strict-v2 validation path:
  canonical-vs-holdout trial sets, score-level subband aggregation, and
  incremental window diagnostics.
- Added the strict-v3/strict-v4 adaptive path:
  stable-prefix window stopping, basin validation, and
  leave-one-recording-out reporting across the combined speech set.
- Added the strict-v5 diagnostic-adaptive path:
  confidence-prefix selection, pair-overlap basins, chirp-only subband
  weighting, and prefix-level confidence diagnostics.
- Added the strict-v8/strict-v9/strict-v10 physics-probe path:
  wall-wave delay templates, common-delay marginalization, and chirp-derived
  spatial coordinate calibration.
- Added the strict-v11/strict-v12/strict-v13/strict-v14 correction path:
  signed-GCC polarity checks, edge-dilation compensation, selected-K-gated
  edge dilation, and a center deadband symmetry prior.

## Analysis Process

1. Rebuilt a naive PI-GS objective from first principles:
   compute LDV-Mic GCC curves, evaluate them at geometry-predicted delays, and
   search over source lateral coordinate.
2. Tested plain wideband and midband GCC-PHAT variants. These stayed far from
   the paper claim, which suggested that the original result likely depended on
   preprocessing, calibration, or window selection.
3. Added chirp-only calibration modes. Global/affine chirp calibration helped,
   while per-position chirp calibration did not transfer well to speech. This
   argues against a simple "same-position chirp reference solves speech" story.
4. Diagnosed window-level behavior. High PSR-like reliability sometimes picked
   confidently wrong windows, confirming that single-pair peak sharpness is not
   a reliable correctness signal in this barrier setup.
5. Added subband ensemble, coherence-masked GCC, and window consensus. The best
   improvement came from subband ensemble with clipped audio and product
   scoring, not from the first coherence mask or consensus estimator.
6. Added holdout block repeats to test whether canonical improvements
   generalize. This exposed that some canonical gains were fragile and that
   holdout performance is the better guardrail for future work.
7. Tested score-level subband aggregation. Penalizing positions with high
   cross-subband disagreement improved combined canonical+holdout performance,
   which supports the hypothesis that remaining errors are driven by
   frequency-local false peaks.
8. Tested adaptive stopping with `stable_prefix`. This improved both average
   and worst-case combined speech error, showing that a fixed window count was
   mixing different recording regimes.
9. Tested basin validation. This treats per-window/per-subband candidates as
   votes for spatial basins, then either gates or softly reweights the final
   score. It helped one holdout repeat, but did not fix the largest stable
   wrong basins.
10. Tested confidence-prefix, pair-overlap, and chirp-weighted variants. These
    did not improve performance, showing that naive internal confidence can be
    fooled by stable wrong basins and that chirp frequency stability does not
    automatically transfer to speech.
11. Tested three less code-anchored physics ideas. A fixed-speed wall-wave
    template did not explain the data, common LDV-Mic time-shift
    marginalization admitted too many false alignments, and chirp-based
    spatial de-warping failed because chirp raw coordinates were not a stable
    monotonic ruler.
12. Tested GCC polarity. Positive-only and negative-only correlations both
    failed, so the current absolute GCC is not merely hiding an easy polarity
    fix.
13. Tested whether the remaining bias is a lateral compression effect. A
    constrained edge-dilation correction improved the hard extreme-position
    rows, and gating that correction by selected-K avoided damaging early
    rollback estimates.
14. Tested a center deadband symmetry prior. Combining selected-K-gated edge
    dilation with center snapping reduced combined speech MAE below 2 deg, but
    this is a strong prior and must be treated as a hypothesis needing external
    validation.

The current conclusion is split by metric. For speech, strict-v13 and
strict-v14 reach paper-level MAE on the current canonical+holdout set, with
strict-v14 numerically exceeding the manuscript-level speech target. For chirp,
the same strict-v14 configuration still performs poorly, so it should not be
read as reproducing the full paper table. The best interpretation is that the
speech result is now explainable by an explicit physical post-correction
hypothesis, while the broader paper claim still needs external validation and
clearer chirp accounting.

## Method Details

- Geometry models:
  `moving_patch` assumes the source primarily excites a nearby barrier patch
  that then radiates to each microphone. `fixed_spot` assumes the LDV spot is a
  fixed reference point and compares source-to-LDV vs source-to-microphone
  travel times.
- Offset calibration:
  offsets are estimated from chirp only and frozen for speech. `constant`
  estimates one VL/VR offset pair, `affine` lets the offset vary linearly with
  lateral coordinate, and `per_trial` estimates a per-recording offset.
- Preprocessing:
  bandpass, partial PHAT, clipping, pre-emphasis, and differencing are treated
  as hypotheses about suppressing low-SNR bins, harmonic artifacts, and
  barrier-induced coloration.
- Advanced scoring:
  subband ensemble splits the wideband score into narrower bands before
  aggregation, while product/harmonic scores require both LDV-Mic pairs to be
  strong. The diagnostic output records per-window/per-subband candidates so
  remaining false peaks can be traced.
- Strict-v2 scoring:
  the original subband path averages normalized GCC curves before scoring.
  The new score-level path scores each subband separately and then aggregates
  the score curves. `subband_score_exp_cv` downweights coordinates that are
  strong in one frequency band but inconsistent across bands.
- Strict-v3 adaptive stopping:
  `stable_prefix` adds reliability-ranked speech windows until the cumulative
  coordinate estimate stops moving by more than a small lateral threshold. This
  is intended to avoid both under-using recordings that need more evidence and
  over-using recordings where later windows introduce false peaks.
- Strict-v4 basin validation:
  per-window/per-subband candidates are converted into a smooth spatial prior.
  `basin_gate` keeps only score regions supported by that candidate basin,
  while `basin_mul` softly multiplies the final score by the prior. The prior
  uses no speech labels; it asks whether independent windows and frequency
  bands point to the same physical coordinate.
- Strict-v5 adaptive diagnostics:
  `confidence_prefix` evaluates every prefix by margin, basin support,
  pair-overlap support, candidate spread, and subband spread. Pair-overlap
  builds separate VL-only and VR-only basins before combining them. Chirp
  subband weights are learned only from chirp and frozen for speech.
- Strict-v8 wall-wave templates:
  `wall_wave_sub` and `wall_wave_add` test whether the LDV observes a
  structural wall wave that first propagates laterally to the laser spot before
  being compared with microphone radiation. Structural speeds of 80, 160, and
  320 m/s are tested as fixed hypotheses.
- Strict-v9 common-delay marginalization:
  each candidate coordinate can shift both LDV-Mic predicted delays together
  by a small amount. This preserves the left-right differential geometry while
  allowing an unknown common wall/instrument delay.
- Strict-v10 chirp spatial calibration:
  chirp estimates are treated as a calibration target for a raw-coordinate to
  true-coordinate de-warp, then the learned mapping is frozen for speech. This
  is a legitimate instrument-calibration hypothesis only if the chirp spatial
  ordering is itself stable.
- Strict-v11 GCC polarity:
  the previous pipeline used absolute GCC. Strict-v11 compares absolute,
  positive-only, and negative-only correlations to test whether false basins
  are mainly coming from anti-correlated peaks.
- Strict-v12/strict-v13 edge dilation:
  the post-estimate coordinate is expanded away from center only after it
  exceeds a threshold. Strict-v13 additionally requires enough selected speech
  windows before applying the correction, so early rollback estimates are not
  over-corrected.
- Strict-v14 center deadband:
  if the final estimate stays within a small broadside band after enough
  selected windows, it is snapped to center. This encodes a symmetry prior:
  small signed offsets near broadside may be wall/room bias rather than a
  reliable lateral displacement.

## Commands Run

```bash
python scripts/independent_pigs_audit.py \
  --profile quick \
  --out_dir results/independent_pigs_audit_quick

python scripts/independent_pigs_audit.py \
  --profile targeted \
  --offset_model affine \
  --out_dir results/independent_pigs_audit_targeted_affine

python scripts/independent_pigs_audit.py \
  --profile targeted \
  --offset_model per_trial \
  --out_dir results/independent_pigs_audit_targeted_per_trial

python scripts/independent_pigs_audit.py \
  --profile targeted \
  --no-calibrate_offsets \
  --out_dir results/independent_pigs_audit_targeted_nocal

python scripts/independent_pigs_audit.py \
  --profile advanced \
  --offset_model affine \
  --out_dir results/independent_pigs_audit_advanced_affine

python scripts/independent_pigs_audit.py \
  --profile strict_v2 \
  --offset_model affine \
  --out_dir results/independent_pigs_audit_strict_v2

python scripts/independent_pigs_audit.py \
  --profile strict_v3 \
  --offset_model affine \
  --out_dir results/independent_pigs_audit_strict_v3

python scripts/independent_pigs_audit.py \
  --profile strict_v4 \
  --offset_model affine \
  --out_dir results/independent_pigs_audit_strict_v4

python scripts/independent_pigs_audit.py \
  --profile strict_v5 \
  --offset_model affine \
  --out_dir results/independent_pigs_audit_strict_v5

python scripts/independent_pigs_audit.py \
  --profile strict_v6 \
  --offset_model affine \
  --out_dir results/independent_pigs_audit_strict_v6_hysteresis_v2

python scripts/independent_pigs_audit.py \
  --profile strict_v7 \
  --offset_model affine \
  --out_dir results/independent_pigs_audit_strict_v7

python scripts/independent_pigs_audit.py \
  --profile strict_v8 \
  --offset_model affine \
  --out_dir results/independent_pigs_audit_strict_v8

python scripts/independent_pigs_audit.py \
  --profile strict_v9 \
  --offset_model affine \
  --out_dir results/independent_pigs_audit_strict_v9

python scripts/independent_pigs_audit.py \
  --profile strict_v10 \
  --offset_model affine \
  --out_dir results/independent_pigs_audit_strict_v10

python scripts/independent_pigs_audit.py \
  --profile strict_v11 \
  --offset_model affine \
  --out_dir results/independent_pigs_audit_strict_v11

python scripts/independent_pigs_audit.py \
  --profile strict_v12 \
  --offset_model affine \
  --out_dir results/independent_pigs_audit_strict_v12

python scripts/independent_pigs_audit.py \
  --profile strict_v13 \
  --offset_model affine \
  --out_dir results/independent_pigs_audit_strict_v13

python scripts/independent_pigs_audit.py \
  --profile strict_v14 \
  --offset_model affine \
  --out_dir results/independent_pigs_audit_strict_v14
```

## Best Results So Far

| Run | Best Speech MAE | Speech Max | Best Config |
|---|---:|---:|---|
| quick + constant chirp offset | 4.55 deg | 9.66 deg | `1000-4000_b0.5_raw_moving_patch_y0.5_harmonic` |
| targeted + affine chirp offset | 4.18 deg | 12.02 deg | `80-8000_b0.3_raw_moving_patch_y0.5_harmonic_k4_r2` |
| targeted + per-trial chirp offset | 6.71 deg | 17.84 deg | `80-8000_b0.3_raw_moving_patch_y0.5_sum_k12_r1` |
| targeted + no calibration | 6.80 deg | 21.68 deg | `1000-4000_b0.5_clip_moving_patch_y0.6_product_k4_r1` |
| advanced + affine chirp offset | 3.65 deg | 7.76 deg | `80-8000_b0.3_clip_moving_patch_y0.5_product_k8_r2_plain0_score_sub` |
| strict-v2 baseline, canonical | 3.65 deg | 7.76 deg | `80-8000_b0.3_clip_moving_patch_y0.5_product_k8_r2_plain0_score_sub` |
| strict-v2 baseline, holdout | 6.94 deg | 13.14 deg | same config, evaluated on complete block repeats |
| strict-v2 score-level, canonical | 3.18 deg | 8.02 deg | `80-8000_b0.3_clip_moving_patch_y0.5_product_k9_r2_plain0_score_sub_subband_score_exp_cv0.2` |
| strict-v2 score-level, holdout | 5.88 deg | 9.90 deg | same config, evaluated on complete block repeats |
| strict-v2 score-level, combined | 4.53 deg | 9.90 deg | same config, canonical + holdout speech |
| strict-v3 stable-prefix, canonical | 3.12 deg | 8.02 deg | `80-8000_b0.3_clip_moving_patch_y0.5_product_k9_r2_plain0_score_sub_subband_score_exp_cv0.2_stable_prefix_m8_s0.03_n2_fb9` |
| strict-v3 stable-prefix, holdout | 5.23 deg | 7.80 deg | same config, adaptive selected-K speech windows |
| strict-v3 stable-prefix, combined | 4.18 deg | 8.02 deg | same config, canonical + holdout speech |
| strict-v4 basin gate, canonical | 3.12 deg | 8.02 deg | `80-8000_b0.3_clip_moving_patch_y0.5_product_k9_r2_plain0_score_sub_subband_score_exp_cv0.2_stable_prefix_m8_s0.03_n2_fb9_basin_gate_sig0.08_g0.1_p0.25` |
| strict-v4 basin gate, holdout | 5.02 deg | 7.80 deg | same config, adaptive selected-K plus basin gate |
| strict-v4 basin gate, combined | 4.07 deg | 8.02 deg | same config, canonical + holdout speech |
| strict-v4 LORO basin selection | 4.07 deg | 8.02 deg | leave-one-recording-out selected `basin_gate` for every held-out row |
| strict-v5 stable-prefix baseline, combined | 4.07 deg | 8.02 deg | same as strict-v4 best, included as the strict-v5 guardrail |
| strict-v5 confidence-prefix, combined | 6.36 deg | 22.44 deg | `confidence_prefix` with normal basin gate |
| strict-v5 pair-overlap confidence, combined | 6.87 deg | 22.44 deg | `confidence_prefix` with pair-overlap gate |
| strict-v5 chirp-weighted pair-overlap, combined | 10.22 deg | 37.19 deg | pair-overlap plus chirp-only subband weights |
| strict-v6 LR-prior variants, combined | 7.57-7.84 deg | 22.71 deg | mic-mic soft prior added to stable-prefix variants |
| strict-v6 hysteresis-prefix, canonical | 2.59 deg | 8.02 deg | `hysteresis_prefix` with basin gate and no LR prior |
| strict-v6 hysteresis-prefix, holdout | 4.00 deg | 6.73 deg | same config, rollback/hysteresis selected-K |
| strict-v6 hysteresis-prefix, combined | 3.30 deg | 8.02 deg | same config, canonical + holdout speech |
| strict-v6 LORO hysteresis selection | 3.30 deg | 8.02 deg | leave-one-recording-out selected hysteresis for every held-out row |
| strict-v7 jackknife, combined | 3.35 deg | 8.02 deg | `subband_jackknife0.5` with hysteresis |
| strict-v7 cluster-max, combined | 5.53 deg | 22.44 deg | winner-take-subband-cluster with hysteresis |
| strict-v8 wall-wave best non-baseline, combined | 12.18 deg | 22.44 deg | `wall_wave_add` at 80 m/s |
| strict-v9 common-shift best non-baseline, combined | 11.60 deg | 32.31 deg | 0.5 ms common LDV-Mic delay marginalization |
| strict-v10 chirp spatial calibration best non-baseline, combined | 10.11 deg | 17.76 deg | affine raw-x to true-x calibration learned on chirp |
| strict-v11 signed-GCC best non-baseline, combined | 16.66 deg | 30.07 deg | positive-only GCC polarity |
| strict-v12 edge dilation, combined | 2.55 deg | 6.73 deg | edge dilation at 0.3 m, gain 1.75 |
| strict-v13 selected-K edge dilation, combined | 1.97 deg | 6.73 deg | edge dilation at 0.3 m, gain 1.75, min K 4 |
| strict-v13 selected-K edge dilation, LORO | 2.16 deg | 7.40 deg | leave-one-recording-out over edge-dilation variants |
| strict-v14 edge + center deadband, combined | 1.24 deg | 4.56 deg | edge dilation plus 0.25 m center deadband |
| strict-v14 edge + center deadband, LORO | 1.24 deg | 4.56 deg | leave-one-recording-out selected the same correction for every held-out row |

## Current Interpretation

- Independent raw-WAV reconstruction can improve substantially over naive
  PI-GS. On the current speech set, strict-v13 and strict-v14 now reach or
  exceed the manuscript-level speech MAE, but they do so with explicit
  post-estimate physical priors and are not yet external validation.
- Chirp-derived calibration helps, but not enough by itself.
- Per-position chirp calibration did not improve speech, which argues against
  a simple "same-position chirp reference fixes everything" explanation.
- Window-level diagnosis shows that high PSR/reliability windows can be
  confidently wrong; structural/harmonic false peaks are likely dominating.
- The best current result is still driven by failure at center and +0.4 m
  positions, while extreme positions can be accurate under some configs.
- The advanced run improved MAE from 4.18 deg to 3.65 deg. The winning change
  was subband ensemble with clipped audio and product score; coherence masking
  and the first consensus estimator did not outperform the simpler score
  aggregation.
- The new best result reduces center/+0.4 m errors but shifts the largest
  residual error to +0.8 m, suggesting the remaining problem is subband
  disagreement / structural false peaks rather than a single global offset.
- Strict-v2 shows that canonical-only MAE is not a sufficient success metric.
  The best holdout-aware result uses `K=9` plus `subband_score_exp_cv0.2`,
  improving combined MAE from 5.30 deg to 4.53 deg and reducing combined max
  error from 13.14 deg to 9.90 deg.
- The holdout set remains harder than canonical. The largest strict-v2
  residual is `+0.4m #15`, which is estimated near center, so future work
  should focus on why that recording's barrier/microphone correlation pulls
  inward.
- Strict-v3 improves on strict-v2 by replacing fixed `K=9` speech aggregation
  with a stable-prefix selector. Combined MAE improves from 4.53 deg to
  4.18 deg, while combined max error improves from 9.90 deg to 8.02 deg.
- Strict-v4 adds basin validation. The best variant, `basin_gate`, improves
  combined MAE from 4.18 deg to 4.07 deg and holdout MAE from 5.23 deg to
  5.02 deg, but does not improve the worst-case 8.02 deg error.
- The strict-v4 improvement is narrow: it mainly corrects `+0.4m #13` from
  2.60 deg to 1.57 deg. The largest errors, `+0.8m #17` and `+0.0m #22`,
  remain unchanged, which implies those recordings have stable but wrong
  candidate basins rather than merely isolated false peaks.
- Strict-v5 tested a more adaptive confidence-prefix selector, pair-overlap
  basin validation, and chirp-only subband weights. None beat the strict-v4
  baseline. This negative result is useful: the naive confidence score tends
  to prefer late, over-concentrated wrong basins, and chirp-derived high-band
  weights did not transfer safely to speech.
- Strict-v6 separates two ideas. The mic-mic LR prior is not useful as a
  direct score multiplier because its chirp peaks collapse near zero lag rather
  than following source position. The rollback/hysteresis selector is useful:
  it improves combined MAE from 4.07 deg to 3.30 deg while keeping max error at
  8.02 deg.
- Strict-v7 tested subband jackknife and winner-take-cluster aggregation. The
  best jackknife variant is close to strict-v6 but does not beat it, while
  cluster-max badly hurts holdout. This suggests the remaining hard cases are
  not solved by simply selecting a subband cluster or penalizing leave-one-band
  instability.
- Strict-v8 through strict-v10 are negative but clarifying. A simple
  fixed-speed wall-wave model pulls many estimates toward center or the wrong
  side. Allowing a common LDV-Mic delay shift makes false alignments too easy,
  especially on canonical trials. Chirp spatial calibration fails because the
  chirp raw spatial estimates are not monotonic enough to serve as a reliable
  coordinate ruler.
- Strict-v11 shows that absolute GCC is still the safer choice; fixed
  positive-only or negative-only polarity breaks badly.
- Strict-v12 and strict-v13 are the first post-strict-v6 changes to beat the
  3.30 deg guardrail. The improvement supports the idea that the remaining
  speech estimates are laterally compressed toward center, likely by the
  effective wall/patch aperture or by a compressed geometric mapping from
  wall vibration to microphone TDOA.
- Strict-v14 reaches 1.24 deg combined MAE and 4.56 deg max error by adding a
  center deadband to the selected-K edge dilation. This is close to or better
  than the manuscript-level target, but it should be treated as a physically
  motivated correction hypothesis rather than a fully validated reproduction.

## Paper Comparison Interpretation

The most important distinction is speech versus chirp. The paper-facing
numbers discussed in the surrounding analysis are approximately 2.23 deg MAE
for blocked speech and 1.94 deg MAE for blocked chirp. Strict-v14 reaches
1.24 deg combined speech MAE and 4.56 deg max speech error on the current
canonical+holdout speech set, so the speech number is no longer the main gap.
However, the same strict-v14 run has 11.86 deg chirp MAE, so it does not
reproduce the full paper table or the chirp claim.

In plain terms, strict-v14 appears to explain how paper-level speech
performance can be achieved: the raw PI-GS estimate finds a mostly correct
basin, hysteresis prevents late-window contamination, edge dilation compensates
for lateral compression toward center, and the center deadband avoids
over-interpreting small broadside offsets. This is a coherent physical story,
but it is a post-correction story rather than "plain GCC-PHAT PI-GS works out
of the box."

The recommended reading is therefore:

- `strict-v13` is the more conservative algorithmic result. It uses selected-K
  gated edge dilation, reaches 1.97 deg combined speech MAE, and is easier to
  defend as a continuous lateral-compression correction.
- `strict-v14` is the strongest numerical result. It adds a broadside center
  deadband and reaches 1.24 deg combined speech MAE, but it is more prior-like
  and should be validated with leave-position-out or new recordings before it
  is treated as a general method.
- The remaining paper gap is not "can speech reach ~2 deg?" on this dataset;
  it is "can the correction be justified without dataset-specific priors, and
  can the chirp claim be reproduced or honestly separated?"

## Strict-v2 Interpretation

Strict-v2 was added because canonical-only evaluation was starting to look too
optimistic. The original five canonical speech trials can be improved by
choosing the right preprocessing and window count, but that does not prove the
method generalizes. The extra complete block repeats act as a small holdout set:
they use the same physical setup and labels, but different recordings. If a
change improves canonical and holdout together, it is more likely to reflect a
real signal-processing improvement rather than a lucky fit to five files.

The strict-v2 baseline keeps the previous best physical hypothesis fixed:
clipped audio, partial PHAT, `moving_patch` geometry, affine chirp offset, and
five subbands. Its combined canonical+holdout speech MAE is 5.30 deg. The best
strict-v2 variant changes only the aggregation logic: each subband first forms
its own spatial score, then `subband_score_exp_cv0.2` penalizes coordinates
whose support is strong in one band but inconsistent across bands. This lowers
combined MAE to 4.53 deg and combined max error to 9.90 deg.

This result is physically plausible. True LDV-Mic target correlation should not
usually appear in only one narrow frequency band; the same geometry should
receive at least partial support across several bands. By contrast, wall
resonances, speech harmonics, and reflected paths can create sharp but
frequency-local false peaks. Penalizing cross-subband disagreement therefore
directly targets the most likely failure mechanism observed in diagnostics.

The `K=9` window count should be read as an empirical stability point, not a
physical constant. With too few windows, a small number of sharp false peaks can
dominate. With too many windows, lower-quality speech windows add structural
and harmonic artifacts. The fact that `K=10` can improve canonical MAE but
hurt holdout is exactly why strict-v2 ranks by combined/holdout-aware metrics
instead of canonical alone.

The remaining error pattern is also informative. The best strict-v2 result is
good on most canonical trials, but holdout `+0.4m #15` is pulled close to the
center. That suggests the next bottleneck is not the basic TDOA geometry; it is
recording-specific false-peak rejection. The next experiments should inspect
incremental window diagnostics for `+0.4m #15` and turn the fixed `K=9` choice
into an automatic stopping/selection rule based on spatial stability and
subband agreement.

## Strict-v3 Interpretation

Strict-v3 implements that automatic stopping idea. It keeps the strict-v2 best
physical and spectral hypothesis fixed, but changes speech aggregation from a
fixed `K=9` prefix to `stable_prefix`. The selector still sorts windows by the
same reliability measure, but it evaluates the cumulative spatial estimate
after each added window. Once the estimate stops moving by more than 0.03 m for
two consecutive updates after at least eight windows, it stops. If no stable
point is found, it falls back to `K=9`.

This is a direct response to the strict-v2 diagnostics. Some recordings are
hurt by adding too many windows; others need more than the first few windows
before the estimate stabilizes. Fixed `K=9` is therefore a compromise, not a
physical rule. Stable-prefix treats each recording independently without using
the speech label: the only evidence it uses is whether the estimated source
coordinate becomes self-consistent as more high-reliability windows are added.

The strict-v3 result improves both average and worst-case error. Canonical MAE
is 3.12 deg, holdout MAE is 5.23 deg, combined MAE is 4.18 deg, and combined
max error is 8.02 deg. The selected-K distribution is also informative:
`K=8` for five trials, `K=9` for one trial, `K=10` for one trial, `K=11` for two
trials, and `K=17` for one trial. This confirms that the method is no longer
secretly just using one global window count.

The largest remaining errors are now `+0.8m #17` and `+0.0m #22`. The next
likely bottleneck is therefore not just when to stop adding windows, but how to
reject or downweight spatial basins that remain stable while still being
biased by barrier/room false paths.

The causal chain is now clearer. Strict-v2 showed that cross-subband agreement
helps reject frequency-local false peaks, but its fixed `K=9` window count
still mixed good and bad speech windows. Incremental diagnostics showed two
opposite failure modes: some trials were already good before `K=9` and then
got pulled away by later windows, while other trials needed more windows before
the cumulative estimate settled. Strict-v3 therefore changes the question from
"how many windows should every recording use?" to "when has this recording's
spatial estimate stopped moving?" That is why the improvement is
interpretably tied to the observed failure, not just another parameter sweep.

The remaining failures should be interpreted differently from the strict-v2
failures. Stable-prefix can prevent unstable or late-arriving bad windows from
continuing to perturb the estimate, but it cannot detect a wrong basin that is
already stable. A stable wrong basin is physically plausible in this setup:
barrier vibration modes, room reflections, and speech harmonics can create a
repeatable LDV-Mic delay pattern that points to the wrong lateral coordinate.
The next algorithmic target should therefore be basin validation, not just
window stopping.

## Strict-v4 Interpretation

Strict-v4 implements the basin-validation idea directly. After stable-prefix
chooses how many speech windows to use, each selected window and subband
produces its own candidate coordinate. These candidates are turned into a
smooth spatial prior. The assumption is physical rather than statistical
label-fitting: a real source coordinate should be supported repeatedly across
time windows and frequency bands, while speech harmonics, wall resonances, and
reflections are more likely to appear as isolated or frequency-local peaks.

Two basin mechanisms were tested. `basin_gate` keeps only the parts of the
final PI-GS score that sit inside a candidate-supported basin. `basin_mul`
keeps the whole score but softly boosts regions with basin support. Both use
the same chirp-frozen calibration and do not look at speech labels inside a
recording. The strict-v4 grid is intentionally small: baseline strict-v3,
narrow hard gate, and wider soft multiplier.

The result is useful but sobering. `basin_gate` improves combined speech MAE
from 4.18 deg to 4.07 deg and holdout MAE from 5.23 deg to 5.02 deg. However,
it only changes one best-row estimate: `+0.4m #13` moves from 0.50 m to
0.46 m, lowering that row's error from 2.60 deg to 1.57 deg. The large
failures `+0.8m #17`, `+0.0m #22`, and `+0.4m #15` are not corrected.

The leave-one-recording-out check selects `basin_gate` for every held-out row
and reports the same 4.07 deg MAE / 8.02 deg max error. This is a small
positive sign: within this three-config grid, the basin-gate choice is not
driven by one single held-out recording. But it is not a formal reproduction
claim, because the basin parameters themselves were motivated by prior
diagnostics on the same dataset.

The strict-v4 causal lesson is that some errors are now "stable wrong." Basin
validation can reject a peak when the final score chooses a coordinate that
other windows/subbands do not support. It cannot reject a wrong coordinate
when many windows and bands agree on the same wrong path. That failure mode is
physically plausible for a barrier system: a wall mode or reflection can
produce a repeatable LDV-Mic delay relation that is internally consistent but
geometrically biased.

## Strict-v5 Interpretation

Strict-v5 was a deliberately small adaptive experiment. It kept the strict-v4
best configuration as a guardrail, then added three stricter hypotheses:
`confidence_prefix`, `confidence_prefix + pair_overlap_gate`, and
`confidence_prefix + pair_overlap_gate + chirp_stability` subband weights.
All three were label-free at speech time. The selector used only internal
score margin, candidate-basin support, pair-overlap support, candidate spread,
and subband spread. The chirp weights were learned from chirp recordings only
and then frozen for speech.

The result is a clear negative. The strict-v4 baseline remains best at 4.07 deg
combined MAE / 8.02 deg max error. Plain `confidence_prefix` falls to 6.36 deg
combined MAE and 22.44 deg max error. Adding pair-overlap makes it slightly
worse at 6.87 deg combined MAE. Chirp-derived subband weights fail badly at
10.22 deg combined MAE and 37.19 deg max error.

The failure mode is informative. `+0.4m #15` is nearly correct at early
prefixes (`K=2..4`), but the confidence score later prefers a wrong basin near
center because that wrong basin has high basin support and high pair-overlap
support. In other words, internal consistency is not the same as correctness:
a reflection or wall mode can be very self-consistent. The pair-overlap prior
did not solve this because both LDV-left and LDV-right can still agree on the
same biased structural path.

The chirp-only subband weights are also a cautionary result. They assign high
weight to `4000-8000 Hz` and low weight to lower bands. That can look stable on
chirp, but speech in that band is more vulnerable to harmonics, fricatives,
noise, and barrier coloration. The transfer from chirp to speech is therefore
not guaranteed, even if the weighting rule uses no speech labels.

The practical lesson is that the next adaptive method should not maximize
confidence over all prefixes. It needs a guardrail for "late confidence
inflation" and a way to penalize basins that become more confident only after
the estimate jumps far from an earlier stable coordinate. A promising next
direction is a rollback/hysteresis selector: keep early accurate prefixes when
later windows increase confidence but require a large spatial jump, unless the
new basin is supported by an independent physical test stronger than simple
pair overlap.

## Strict-v6 Interpretation

Strict-v6 tested two physical intuitions. The first was to add a third acoustic
constraint: left-mic to right-mic TDOA. In principle this should be independent
of the LDV-wall path, so it might reject wall-mode false basins. In practice it
failed. Chirp diagnostics show the LR peak stays near 0 ms across source
positions instead of tracking the expected mic-mic geometry. That means the
mic-mic GCC is dominated by common-mode/direct electronics/room components or
by a signal path that is not the simple source-to-left/right acoustic delay.
Using it as a soft prior therefore hurts holdout badly.

The second intuition worked. `hysteresis_prefix` explicitly protects against
late confidence inflation. It starts from the confidence-prefix idea, but adds
two guardrails: rollback is allowed only before a large spatial jump when the
pre-jump basin has enough pair support, and the stable-prefix guardrail is
computed with at least eight windows so the method does not over-trust a very
early isolated peak. This matches the observed failure mechanism from
strict-v5: wrong basins can become very confident after additional speech
windows, even when an earlier prefix was physically more plausible.

The best strict-v6 result uses `hysteresis_prefix` with basin gate and no LR
prior. It improves canonical MAE from 3.12 deg to 2.59 deg, holdout MAE from
5.02 deg to 4.00 deg, and combined MAE from 4.07 deg to 3.30 deg. LORO selects
the same hysteresis config for every held-out row, so within this small
strict-v6 grid the improvement is not driven by one recording alone.

The main recovered case is `+0.4m #15`: strict-v4 selected `K=17` and estimated
0.19 m, while strict-v6 rolls back to `K=3` and estimates 0.38 m, reducing the
error from 5.58 deg to 0.53 deg. It also improves `-0.8m #20`, `+0.0m #18`,
and `+0.4m #16`. The remaining hard cases are still `+0.8m #17` and
`+0.0m #22`; hysteresis prevents them from getting worse but does not fully
identify the true source basin. This suggests the remaining problem is a
stable wrong LDV-wall basin rather than just late-window contamination.

## Strict-v7 Interpretation

Strict-v7 asked whether the remaining stable wrong basins are caused by one
frequency band dominating the average. Two ideas were tested. `subband_jackknife`
recomputes the score while leaving out each band, then penalizes positions that
are unstable under that leave-one-band test. `subband_cluster_max` does the
opposite: it lets a local cluster of agreeing subbands win instead of forcing
all bands to average together.

The result is mostly negative. `subband_jackknife0.5` is close to strict-v6
at 3.35 deg combined MAE / 8.02 deg max error, but does not improve the best
3.30 deg result. `subband_cluster_max` lowers a few canonical rows but damages
holdout, especially by pulling `-0.8m #21` toward the wrong side. This means
the false basins are not merely single-band outliers; they can be supported by
a plausible cluster of bands, just not the right physical path.

A separate geometry probe tested alternative `moving_patch` effective depths.
`ldv_y=0.35` improved some hard rows such as `+0.8m #17` and `+0.0m #22`, but
destroyed many other rows. This is an important physical clue: the effective
wall/patch path is probably not constant across recordings or frequency
content, but letting geometry adapt freely would overfit to alternate false
paths. For now geometry adaptation should be diagnostic-only unless there is a
label-free way to decide when an alternate path is physically valid.

## Strict-v8 Interpretation

Strict-v8 deliberately stepped away from the current best code path and tested
a different physical picture. Instead of assuming the LDV instantly observes
the same wall patch that re-radiates to the microphones, it asked whether the
LDV signal might include lateral structural propagation along the wall. Two
sign conventions were tested: one where wall propagation subtracts from the
mic path and one where it adds as extra delay. Speeds of 80, 160, and 320 m/s
were used as coarse wall-wave hypotheses.

The result is strongly negative. The existing moving-patch guardrail remains
best at 3.30 deg combined MAE / 8.02 deg max error. The best non-baseline
wall-wave config is 12.18 deg combined MAE / 22.44 deg max error, and most
wall-wave variants pull estimates toward center or to the wrong side.

The interpretation is not that wall vibration is absent. It is narrower than
that: a fixed-speed, fixed-spot structural-wave correction is not the missing
model. If wall dynamics matter, they are probably modal, frequency-dependent,
or recording-dependent rather than a single lateral propagation speed that can
be inserted into the TDOA template.

## Strict-v9 Interpretation

Strict-v9 tested a more forgiving version of the same intuition. If the LDV
contains an unknown common wall/instrument delay, then both LDV-Mic curves
should be allowed to slide together by a small amount. This preserves the
left-right differential geometry but no longer demands that the absolute VL
and VR delays be exactly right.

This also fails. A 0.5 ms common-shift radius is the best non-baseline variant,
but it reaches only 11.60 deg combined MAE and produces a 32.31 deg max error.
The reason is physically understandable: speech and wall responses contain
many sharp peaks, so allowing a free common shift gives the algorithm too many
ways to make a wrong pair of peaks look jointly strong. The differential
geometry alone is not selective enough under these noisy/harmonic recordings.

The useful lesson is that absolute LDV-Mic timing, even if imperfect, is still
acting as an important guardrail. Removing too much of that constraint turns
the search into a false-alignment machine.

## Strict-v10 Interpretation

Strict-v10 tested a calibration-lab idea: use chirp as a known-position source
to learn a mapping from raw PI-GS coordinate to true coordinate, then freeze
that mapping for speech. This is not speech-label leakage, because the speech
labels are not used to fit the mapping. But it is still risky because it
assumes chirp and speech share the same spatial distortion.

The result is negative and revealing. Affine chirp spatial calibration worsens
combined MAE to 10.11 deg, and piecewise-linear calibration worsens it to
14.57 deg. The affine fit collapses toward a very small slope, while the
piecewise mapping becomes non-monotonic in practice: the chirp raw estimates
do not preserve the left-to-right order of the true source positions.

That means chirp is useful for timing offset calibration, but not currently a
reliable spatial ruler. This matters because it blocks an otherwise tempting
path: we cannot safely "fix" speech coordinates by learning a raw-x de-warp
from chirp unless we first make chirp spatial estimates monotonic and
physically stable.

## Strict-v11 Interpretation

Strict-v11 tested whether the use of absolute GCC was creating false peaks by
making positive and negative correlations equally acceptable. This is a
reasonable suspicion because an LDV velocity signal and microphone pressure
signal can invert polarity depending on wall motion, reflection, and sensor
chain details.

The result is negative. The absolute-GCC baseline remains 3.30 deg combined
MAE / 8.02 deg max error. Positive-only GCC worsens to 16.66 deg combined MAE,
and negative-only GCC worsens to 19.46 deg combined MAE. The practical lesson
is that polarity is not stable enough across recordings, windows, or bands to
be used as a hard constraint. Absolute GCC may admit false peaks, but it also
keeps real peaks that flip sign.

## Strict-v12 Interpretation

Strict-v12 tested a more geometric idea. The hard residuals after hysteresis
look compressed toward center: `+0.8m #17` sits around 0.48 m, `+0.8m #21`
around 0.61 m, and `-0.8m #21` around -0.58 m. That pattern looks less like
random false detection and more like an effective lateral-coordinate
compression. Physically, this could come from wall aperture effects,
patch-spreading, or a mismatch between the nominal free-space geometry and the
actual structure-borne reference.

The tested correction, `edge_dilation`, leaves center estimates alone and
expands only coordinates beyond a threshold. The best strict-v12 variant uses
a 0.3 m threshold and 1.75 gain. It improves combined MAE from 3.30 deg to
2.55 deg and reduces max error from 8.02 deg to 6.73 deg. It fixes the
extreme holdout rows especially well: `-0.8m #21` moves from -0.58 m to
-0.79 m, and `+0.8m #21` moves from 0.61 m to 0.84 m.

The failure mode is also clear. Applying dilation unconditionally damages
early rollback cases with only two selected windows, especially `+0.4m #13`.
That means the correction is plausible only when enough windows support a
compressed basin; it should not be blindly applied to every estimate.

## Strict-v13 Interpretation

Strict-v13 adds that guardrail. Edge dilation is applied only when the selected
speech prefix contains enough windows. The best setting uses threshold 0.3 m,
gain 1.75, and minimum selected K of 4. This keeps the early rollback rows
unchanged while still expanding the stable compressed extreme rows.

The result improves combined MAE to 1.97 deg with max error 6.73 deg. LORO is
also encouraging at 2.16 deg MAE / 7.40 deg max error. Most rows now have
small errors, and the remaining max error is `+0.0m #22`, which is still
estimated as -0.25 m. In plain language: the edge correction solves the
"extremes pulled toward center" problem, but not the "near-center small bias"
problem.

This is the first result in the independent audit that reaches the approximate
paper-level average error without using speech labels inside a recording.
However, the correction parameters were derived from current diagnostics, so
it is still an audit hypothesis, not an external validation result.

## Strict-v14 Interpretation

Strict-v14 adds a center deadband on top of selected-K edge dilation. If the
post-dilation estimate remains within a small center band after enough selected
windows, it snaps to 0 m. The physical idea is broadside symmetry: near the
center, a small signed lateral estimate can be caused by wall asymmetry or
room bias rather than a real source displacement. This is similar to saying
"do not over-interpret a tiny left/right bias near the symmetric point."

The best strict-v14 result uses edge threshold 0.3 m, edge gain 1.75, minimum
edge K 4, and a 0.25 m center deadband with minimum K 4. It reports 1.24 deg
combined MAE and 4.56 deg max error. LORO selects the same correction for
every held-out row and reports the same 1.24 deg MAE / 4.56 deg max error.

The improvement mechanism is transparent. Edge dilation corrects compressed
extreme estimates while leaving low-K rollback cases alone. The center
deadband fixes `+0.0m #22` by snapping -0.25 m to 0 m, and it also makes the
canonical center row exactly centered. The remaining largest error is
`+0.8m #17`, which improves from 8.02 deg to 4.56 deg but still remains
somewhat compressed.

This is the strongest numerical result so far, but also the most prior-driven.
It may represent the missing physical postprocessing behind the manuscript
claim, or it may be partially exploiting the fact that the current experiment
uses a small set of known discrete source positions including exactly 0 m. The
next validation step should therefore be a leave-position-out or new-recording
test before treating strict-v14 as a reproduced algorithm.

## Chirp Physics Diagnostics Round

After strict-v14, the remaining uncomfortable fact is that speech is strong
but chirp is still poor in the main audit. To attack that directly, a separate
raw-WAV diagnostic script was added at `scripts/chirp_physics_diagnostics.py`.
It deliberately does not reuse the main PI-GS estimator. Instead, it tests
basic physical interpretations of the chirp:

- mic-mic TDOA with ordinary cross-correlation/GCC;
- mic-mic TDOA after synthetic chirp matched filtering;
- LDV-mic geometric scoring after synthetic chirp matched filtering;
- fixed-window, all-trial replay, and explicitly marked sliding-window oracle
  reports.

The best mic-mic-only chirp condition reaches only 5.66 deg canonical MAE. It
is monotonic, but it is strongly compressed toward the center except at the
edges. This matches the physical suspicion that the microphones see a strong
common chirp/reflection field, so left-right timing alone is not enough.

The best canonical LDV-mic matched-filter condition is much more revealing:
1500-4000 Hz, differentiated signals, a 500-7000 Hz / 1.5 s synthetic chirp,
LDV effective y of 0.5 m, and a +2 ms common shift produce 2.25 deg canonical
chirp MAE. That is close to the paper-facing chirp claim around 1.94 deg. The
sequence is also monotonic:

```text
true x:  -0.8   -0.4    0.0   +0.4   +0.8
est x:   -1.063 -0.409 +0.087 +0.334 +0.765
```

But the same condition does not generalize cleanly to the repeated holdout
recordings. On canonical+holdout it is 7.22 deg MAE, and the direct all-trial
condition ranking does not find a better non-oracle fixed-window rule. This
means the 2.25 deg result is a real clue, but not yet a robust reproduction.

The most important diagnostic is the sliding-window oracle. If each trial is
allowed to choose its 0.5 s chirp window using the known source label, the best
LDV-mic matched-filter family reaches 1.05 deg MAE / 3.19 deg max error across
all 10 canonical+holdout recordings:

```text
est x: -0.786 -0.421 -0.084 +0.341 +0.795 | -0.798 +0.118 +0.396 +0.347 +0.840
err:    0.34   0.55   2.27   1.56   0.12 |  0.05   3.19   0.11   1.40   0.96 deg
```

This should not be interpreted as a valid blind algorithm, because it uses the
answer to pick the window. Its value is diagnostic: the chirp recordings do
contain paper-level spatial information, but the correct physical event is not
always the strongest matched-filter peak. Different repeats place the useful
window in different parts of the first 2 seconds. In plain language, the data
has the needle; our current automatic selector is still often grabbing a
shinier piece of hay.

Several attempted automatic selectors were negative. Choosing the largest
matched-filter/geometric score fails because false reflections often normalize
to score 1.0. Requiring agreement with mic-mic TDOA also fails because the
mic-mic estimate itself tends to collapse toward broadside and disagrees with
the true edge cases. A useful next selector probably has to inspect the
LDV-mic score-shape over time, adjacent-window stability, and whether VL/VR
peaks form a physically plausible pair, rather than using raw peak height.

The next iteration expanded the LDV-mic physical grid. The original diagnostic
mostly assumed an effective LDV y of 0.5 m and a small set of common shifts.
The expanded run varies effective LDV y from 0.0 m to 0.8 m, common shift from
0.5 ms to 3.0 ms, and includes denser half-second chirp windows. The best
non-oracle all-trial chirp result improves from 7.22 deg MAE to 6.01 deg MAE:

```text
condition: 80-800 Hz, diff, window 0.75-1.25 s,
           ldv_y=0.8 m, common_shift=1.25 ms, product/sum
est x:     -0.963 -1.025 -0.005 +0.260 +0.271 | -0.734 +0.403 +0.403 +0.381 +0.381
MAE:       6.01 deg all, 7.31 deg canonical, 4.72 deg holdout
```

This is a real non-oracle improvement, but it changes the error profile rather
than solving chirp. The earlier 1500-4000 Hz condition is excellent on the
canonical five rows but fails on holdout. The new 80-800 Hz condition is less
accurate on the canonical rows but more stable across repeats. That is
physically plausible: low-frequency wall/structure response is less sensitive
to fine chirp timing and high-frequency modal nulls, but it also has worse
spatial resolution and pulls the right-edge positions toward the center.

Two other ideas were tested and rejected. A window-aware subchirp template is
more physically literal than using the full 1.5 s chirp template inside each
0.5 s window, but it worsens the fixed-window result and raises the oracle
upper bound from 1.05 deg to about 2.09 deg. That suggests the full-template
matched filter is acting like a useful empirical chirp/reflection fingerprint,
not just a clean local sweep detector. Multi-geometry or multi-template
agreement also fails as an automatic selector because stable reflections can
be consistently wrong.

A further chirp iteration tested fusion rather than trying to pick one perfect
condition. The physical motivation is simple: low-frequency structure response
is repeat-stable but spatially blunt, while mid-frequency chirp response has
sharper spatial information but is more vulnerable to repeat-specific
reflections. Averaging several views can cancel some path-specific errors.

The fixed, row-label-free three-regime fusion combines:

- high-resolution early chirp: 1500-4000 Hz, 0.25-0.75 s, `ldv_y=0.3`,
  `common_shift=1.5 ms`;
- low-frequency stable chirp: 80-800 Hz, 0.75-1.25 s, `ldv_y=0.8`,
  `common_shift=1.25 ms`;
- mid-frequency repeat-stable chirp: 1500-4000 Hz, 0.375-0.875 s,
  `ldv_y=0.1`, `common_shift=1.0 ms`.

Equal averaging of these three regimes improves the non-oracle all-trial chirp
result from 6.01 deg MAE to 4.28 deg MAE:

```text
equal-fusion x:
-1.037 -0.669 +0.260 +0.336 +0.755 | -0.315 -0.062 +0.291 +0.346 +0.705
MAE: 4.28 deg all, 4.43 deg canonical, 4.14 deg holdout
```

Adding a simple center/edge prior to the same fixed fusion gives 3.38 deg MAE:

```text
postprocess: center deadband 0.275 m, edge threshold 0.2 m, edge gain 1.1
x: -1.121 -0.716 +0.000 +0.350 +0.811 | -0.327 +0.000 +0.300 +0.360 +0.755
MAE: 3.38 deg all, 3.40 deg canonical, 3.37 deg holdout
```

This is the strongest chirp result so far that is based on a fixed physical
fusion recipe, but its postprocess thresholds were still chosen after seeing
the current dataset. Treat it as a plausible missing postprocessing hypothesis,
not external proof.

The most aggressive in-dataset diagnostic ranks the expanded LDV grid on the
same current recordings, ensembles the top 10 conditions by inverse-MAE
weight, and applies a mild center/edge prior. It reaches 2.36 deg MAE:

```text
top-K diagnostic: K=10, weighted mean, center deadband 0.15 m,
                  edge threshold 0.2 m, edge gain 1.35
x: -1.067 -0.499 +0.000 +0.398 +0.797 | -0.430 +0.000 +0.431 +0.275 +0.749
MAE: 2.36 deg all, 1.78 deg canonical, 2.94 deg holdout
```

This average is now close to the paper-facing chirp claim around 1.94 deg, but
it is explicitly not a clean validation result. A leave-one-recording-out
stress test of condition ranking collapses badly, because the best-ranked
conditions change depending on which recording is left out. In plain language:
ensembling can recover much of the paper-level chirp behavior inside this
dataset, but we still do not have a reliable blind rule for choosing the
ensemble from new data.

The next iteration reframed chirp as a calibration signal rather than a blind
continuous-localization signal. This matters because the experiment itself has
known discrete calibration positions: -0.8, -0.4, 0.0, +0.4, and +0.8 m. Under
that interpretation, it is physically reasonable to use the canonical chirp
rows to learn a monotone raw-x-to-true-x map, then apply that calibrated axis
to repeat recordings.

The new `canonical_grid_calibration` report does exactly that. It takes a
fused chirp estimate, learns a piecewise monotone map from the canonical five
rows, and optionally snaps the calibrated value to the known discrete grid.
With the dataset-ranked top-K chirp ensemble as input, piecewise grid snapping
gives the strongest chirp number so far:

```text
calibration: canonical grid, piecewise map + discrete grid snap
source:      top-K chirp ensemble, K=10, weighted mean, center/edge prior
x:           -0.8 -0.4 +0.0 +0.4 +0.8 | -0.4 +0.0 +0.4 +0.4 +0.8
MAE:         1.01 deg all, 0.00 deg canonical, 2.02 deg holdout
max error:   10.11 deg
```

This result is now better than the paper-facing average chirp claim, but it
has a very specific meaning. It says the current chirp data can support
paper-level performance if chirp is treated as a discrete calibration task.
It does not prove a general continuous blind localizer, and the max error
shows the weakness clearly: the `-0.8m #21` repeat is snapped to -0.4 m. In
plain language, the method mostly recovers the calibration grid, but one
left-edge repeat is still too ambiguous.

This calibration-grid result is important because it reconciles otherwise
contradictory observations. The sliding-window oracle showed that good chirp
information exists. Fixed blind conditions were too brittle. Fusion recovered
much of the information but remained biased. Canonical grid calibration then
converts the biased fused coordinate into the experiment's known discrete
coordinate system. That sequence is a plausible explanation for how a paper
pipeline could report strong chirp numbers after substantial preprocessing and
calibration.

## Important Caveats

- Results under `results/` are intentionally not committed; they are generated
  artifacts and are ignored by `.gitignore`.
- The audit does not prove the manuscript numbers. It documents which
  physically motivated assumptions improve performance and which do not.
- Any future result that uses speech labels, hand-picked speech windows, or
  per-speech-position tuning must be marked as oracle/label leakage rather than
  a valid reproduction.
- The advanced result has poor chirp MAE but better speech MAE, so it should be
  interpreted as a speech-specific processing hypothesis, not as a universal
  solved calibration.
- The strict-v2 report ranks a fixed hypothesis grid against holdout speech.
  Treat it as an audit guardrail, not as proof that holdout labels may be used
  freely for tuning.
- Strict-v3 improves holdout-aware metrics, but its selector thresholds were
  chosen from the current diagnostic set. The next formal validation step
  should use leave-one-recording-out selection before treating the numbers as a
  reproduction claim.
- Strict-v4 adds leave-one-recording-out model selection over three configs,
  but the basin parameters were still chosen from current diagnostics. Treat
  the LORO number as a stronger sanity check than combined ranking, not as an
  independent external validation set.
- Strict-v5 includes negative results. Do not reuse `confidence_prefix`,
  `pair_overlap_gate`, or `chirp_stability` as-is as if they were improvements;
  their value is diagnostic, not performance.
- Strict-v6's hysteresis thresholds were derived from this dataset's diagnostic
  behavior. LORO is encouraging, but external validation or additional repeats
  are still needed before treating 3.30 deg as a reproduction-level claim.
- Strict-v7 confirms that subband clustering and alternate patch geometry can
  expose hidden candidate basins, but should not be used directly as selectors
  without stronger physical validity checks.
- Strict-v8, strict-v9, and strict-v10 are implemented as negative controls as
  much as improvement attempts. Do not reuse wall-wave templates, common-shift
  marginalization, or chirp spatial de-warping as improvements unless a future
  diagnostic fixes the specific failure modes documented above.
- Strict-v12 through strict-v14 add post-estimate priors. They are physically
  motivated and label-free at speech-evaluation time, but their thresholds were
  chosen after inspecting this dataset. Treat the 1.24 deg strict-v14 result
  as a strong hypothesis about the missing manuscript postprocessing, not as
  independent proof of generalization.
- The center deadband in strict-v14 assumes broadside symmetry and a meaningful
  center position. It may not be valid for arbitrary continuous source
  locations or for a deployment where the set of candidate positions is not
  known in advance.

## Next Most Likely Improvements

- Validate strict-v14 with a leave-position-out protocol, not only
  leave-recording-out, because the center deadband and edge dilation are
  position-prior-like corrections.
- Search for more repeats or a truly external recording set. The strict-v14
  numbers are good enough that validation quality matters more than further
  in-sample tuning.
- Derive edge dilation from a physical or chirp diagnostic if possible, rather
  than choosing its threshold/gain from speech outcomes.
- Replace the hard center deadband with a softer confidence-based broadside
  prior that can degrade gracefully for continuous source locations.
- Replace the failed mic-mic prior with residual-shape diagnostics on the
  LDV-Mic curves themselves. A true source should have a plausible local score
  shape around both VL and VR delays, not merely a high peak.
- Try a conservative oracle-free max-error guardrail: if hysteresis selects a
  prefix whose score is much less stable under subband jackknife than the
  stable-prefix estimate, fall back to stable-prefix.
- Add diagnostic-only multi-geometry reports for `ldv_y` alternatives, but do
  not rank by them until a chirp- or physics-derived validity test exists.
- Investigate the residual hard rows as multi-path ambiguity: report the top
  two spatial basins and their subband/window support instead of forcing a
  single estimate too early.
- Improve the consensus estimator using subband agreement rather than the
  current margin/PSR-like weight.
- If chirp-derived weights are revisited, regularize them much more strongly
  and test whether high-frequency chirp stability actually predicts speech
  stability before applying them to speech.
- For chirp specifically, replace fixed-window scoring with a sliding-window
  selector that looks for physically plausible LDV-Mic peak pairs across
  adjacent windows. Do not select by raw matched-filter peak height alone.
- Use the sliding-window oracle only as an upper-bound diagnostic. If a future
  adaptive selector approaches the 1.05 deg oracle without using labels, then
  chirp reproduction becomes much more credible.
- Keep both chirp regimes visible: 1500-4000 Hz currently gives the best
  canonical chirp result, while 80-800 Hz gives the best non-oracle
  canonical+holdout result. A future adaptive method may need to switch
  between these regimes based on detected repeat/stability rather than choose
  one global band.
- Treat chirp fusion as the most promising path now. Fixed multi-regime
  fusion reaches 4.28 deg without row-level label selection, and a simple
  center/edge prior reaches 3.38 deg. The 2.36 deg top-K result is useful as
  an in-dataset upper bound, but needs a non-leaky condition-selection rule.
- If chirp is intended as a calibration signal, canonical-grid calibration is
  now the best explanation of paper-level chirp performance: it reaches
  1.01 deg all-trial MAE / 2.02 deg holdout MAE when calibrated on the
  canonical grid. Keep this separate from blind continuous-localization claims.
- Before any future chirp-based spatial calibration, first require chirp raw
  estimates to be monotonic with known source position under leave-one-position
  checks.
- If wall physics is revisited, model it as frequency-dependent/modal behavior
  rather than a single lateral speed added to every band equally.
- Explicitly audit whether the paper text used a different subset of repeated
  recordings or hand-selected windows.
