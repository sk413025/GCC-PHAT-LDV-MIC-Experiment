# Independent PI-GS Audit Summary

Generated from an independent raw-WAV pipeline that does not import the
project's existing research scripts.

## Research Rationale

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

The current conclusion is conservative: the independent pipeline can move
toward the manuscript claim, but it has not reproduced the claimed ~2 deg
speech MAE without further assumptions or better false-peak rejection.

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

## Current Interpretation

- Independent raw-WAV reconstruction can improve substantially over naive
  PI-GS, but did not reproduce the manuscript's claimed ~2 deg speech MAE.
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

## Next Most Likely Improvements

- Extend stable-prefix and score-level subband consistency to a
  leave-one-recording-out protocol so selector thresholds are chosen without
  looking at the same speech rows used for reporting.
- Inspect `results/independent_pigs_audit_strict_v3/best_incremental_window_diagnostics.json`
  to identify why `+0.8m #17` and `+0.0m #22` remain biased after adaptive
  window selection.
- Improve the consensus estimator using subband agreement rather than the
  current margin/PSR-like weight.
- Inspect `results/independent_pigs_audit_strict_v3/best_window_diagnostics.json`
  to identify which subbands create the remaining holdout false peaks.
- Try per-subband reliability learning from chirp only: weight subbands by
  chirp stability, then freeze weights for speech.
- Explicitly audit whether the paper text used a different subset of repeated
  recordings or hand-selected windows.
