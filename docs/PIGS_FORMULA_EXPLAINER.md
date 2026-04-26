# PI-GS Formula Explainer

This note explains, in plain language, how the paper's final PI-GS objective
relates to the recent reproduction experiments.

The short version:

```text
The paper formula is the skeleton.
The recent experiments are the muscles, nerves, and calibration fixtures needed
to make that skeleton stand up on noisy real recordings.
```

## The Paper's Last Formula

The final objective in the paper is:

```text
S(p) = |R_VL(tau_VL(p))| + |R_VR(tau_VR(p))|

p_hat = argmax_{p in Omega} S(p)
```

Plain-language translation:

```text
Try one possible source position p.
        |
        v
Use geometry to predict two delays:

  tau_VL(p): expected delay between LDV and left microphone
  tau_VR(p): expected delay between LDV and right microphone
        |
        v
Look up how strong the measured correlation curves are at those two delays.
        |
        v
Add the two strengths.
        |
        v
Repeat this for every possible position.
        |
        v
Pick the position with the highest joint score.
```

In one picture:

```text
                         candidate source p
                                *
                               /|\
                              / | \
                             /  |  \
                            /   |   \
                         LDV   MicL  MicR

For this p:

  predicted LDV-MicL delay = tau_VL(p)
  predicted LDV-MicR delay = tau_VR(p)

Then:

  score from left pair  = |R_VL(tau_VL(p))|
  score from right pair = |R_VR(tau_VR(p))|

  S(p) = left score + right score
```

The method is called physics-informed because the search is not merely asking
"where is the largest correlation peak?" It asks a more constrained question:

```text
Which physical source position would make BOTH LDV-microphone pairs happy at
the same time?
```

## Why The Formula Is Useful

With microphones only, a wall or barrier can create a coherence trap:

```text
source sound
    |
    v
barrier vibrates / reflects / re-radiates
    |
    v
many microphone peaks appear
```

A microphone-only correlation can therefore lock onto a strong but wrong peak:

```text
MicL-MicR correlation:

score
  ^
  |              false structural peak
  |                    /\
  |                   /  \
  |     true peak    /    \
  |       /\        /      \
  |______/  \______/        \________> delay
```

The LDV gives an extra anchor. Instead of trusting one microphone-microphone
delay, PI-GS asks for two LDV-microphone delays that agree with the same source
position:

```text
LDV-MicL says: "positions along this delay ridge are possible"
LDV-MicR says: "positions along this other delay ridge are possible"

The likely source is where the ridges cross.
```

ASCII version:

```text
Search area Omega

   y
   ^
   |
   |     LDV-MicL ridge
   |        \      true source
   |         \        *
   |          \      /
   |           \    /  LDV-MicR ridge
   |            \  /
   |-------------\/------------> x
```

That is the central physical idea behind the paper objective.

## The Hidden Assumptions

The formula is elegant, but it only works well if several things are true:

```text
Assumption A:
  The useful LDV-related peak is visible in R_VL and R_VR.

Assumption B:
  The predicted geometry delays tau_VL(p), tau_VR(p) are close to reality.

Assumption C:
  False multipath peaks do not accidentally agree across both pairs.

Assumption D:
  The chosen time window and frequency band actually contain source-coherent
  information, not mostly noise, harmonics, or wall modes.
```

The recent experiments show that the raw recordings often violate these
assumptions. That is why plain GCC-PHAT plus the final formula is not enough to
reproduce the paper-level chirp result from raw WAV files.

## How The Recent Implementation Maps To The Formula

The implementation in `scripts/chirp_physics_diagnostics.py` keeps the same
core objective, but makes the correlation curves and delay templates more
realistic.

The core implementation pattern is:

```text
1. Build tau templates:

     tau_l_base, tau_r_base = tau_ldv_mic_templates(xs_grid, ldv_y)

2. Apply effective delay corrections:

     tau_l = delay_sign * tau_l_base + common_shift
     tau_r = delay_sign * tau_r_base + common_shift

3. Build cleaner LDV-Mic score curves:

     preprocess signal
     matched-filter with chirp template
     compute LDV-Mic correlation envelopes

4. Evaluate the paper score over candidate x positions:

     sampled_l = score_l(tau_l[x])
     sampled_r = score_r(tau_r[x])

     score[x] = sampled_l * sampled_r
              or min(sampled_l, sampled_r)
              or sampled_l + sampled_r

5. Estimate:

     x_hat = argmax_x score[x]
```

The formula says:

```text
p_hat = argmax_p S(p)
```

The diagnostic code says:

```text
x_hat = argmax_x score[x]
```

So the code is the paper formula specialized to the project's 1D source grid,
with additional preprocessing and calibration around it.

## What Each Experimental Layer Adds

### 1. Matched Filtering

What it changes:

```text
It improves R_VL(tau) and R_VR(tau).
```

Why:

Chirp is a known signal. If we know the emitted sweep, we can correlate the
recordings against a synthetic chirp template before doing LDV-microphone delay
scoring.

Plain-language analogy:

```text
Raw audio:
  "I hear many things. Which part is the chirp?"

Matched filter:
  "Only light up the parts that look like the chirp pattern."
```

This helps because the chirp recordings contain noise, harmonics, reflections,
and wall vibration modes. The matched filter is not a new localization formula;
it is a way to make the formula's correlation curves less misleading.

### 2. Frequency Bands And Time Windows

What it changes:

```text
It chooses which physical evidence goes into R_VL and R_VR.
```

Why:

Different frequency bands see different versions of the wall:

```text
low frequency:
  more stable, often less spatially sharp

mid frequency:
  more spatial detail, but more modal/null risk

high frequency:
  sharper timing, but easier to lose under attenuation or reflections
```

Different chirp windows also behave differently:

```text
early window:
  may capture direct/early response

later window:
  may include stronger wall/reflection energy

too late:
  may be dominated by ringing or unrelated structure
```

So these experiments ask:

```text
Which parts of the recording best satisfy the paper formula's assumptions?
```

### 3. Effective LDV Geometry And Common Delay Shift

What it changes:

```text
It modifies tau_VL(p) and tau_VR(p).
```

Why:

The paper formula uses geometric delay templates. In a perfect free-field
system, those templates would match the real measured delay. In this project,
the LDV is not a normal microphone in air. It measures barrier motion, which can
include:

```text
source -> barrier coupling
barrier lateral propagation
LDV instrument delay
re-radiation from a distributed wall patch
unknown effective measurement point
```

So the true effective template may look like:

```text
paper ideal:

  tau_VM(p) = air_path_to_mic(p) - air_path_to_LDV_point(p)

real recording:

  tau_VM(p) ~= paper_template(p)
              + common electronics/wall delay
              + effective LDV position bias
              + frequency-dependent wall behavior
```

The expanded geometry and common-shift sweeps are therefore tests of whether
the paper's delay curves need practical calibration before the `argmax` step.

### 4. Product / Min / Sum Score Modes

What it changes:

```text
It changes how the two LDV-Mic pair scores become S(p).
```

Paper formula:

```text
S(p) = left + right
```

Product score:

```text
S(p) = left * right
```

The product version is stricter:

```text
left high, right low  -> product stays low
left low, right high  -> product stays low
left high, right high -> product becomes high
```

This matches the physical intuition of joint consistency. A real source should
make both LDV-Mic pairs agree, not only one.

### 5. Multi-Regime Fusion

What it changes:

```text
It averages several PI-GS score/estimate regimes instead of trusting one.
```

Why:

The wall is not a simple object. One band/window may be fooled by a structural
mode; another may be closer to the direct source-related response. Fusion asks
for agreement across multiple imperfect views:

```text
regime A: high-resolution early chirp
regime B: low-frequency stable chirp
regime C: mid-frequency repeat-stable chirp
        |
        v
average / combine estimates
```

This is not a replacement for the paper formula. It is more like running the
paper formula through three different physical lenses, then keeping the
position that remains stable.

### 6. Center / Edge Priors

What it changes:

```text
It post-processes p_hat after the formula has produced a raw estimate.
```

Why:

The experiments show two recurring bias patterns:

```text
near center:
  tiny nonzero estimates may be room/wall bias rather than true lateral motion

near edges:
  estimates may be compressed toward center because the usable delay evidence
  loses spatial leverage
```

So the center/edge prior is a physically motivated correction:

```text
if estimate is very close to center:
  snap or soften toward center

if estimate is confidently away from center:
  expand slightly away from center
```

This is not in the paper's final formula. It is a practical hypothesis about
the bias pattern in this testbed.

### 7. Canonical-Grid Calibration

What it changes:

```text
It treats chirp as a calibration signal, not as blind continuous localization.
```

Why:

The chirp data was recorded at known discrete positions. If the purpose of
chirp is calibration, then it is reasonable to learn a mapping:

```text
raw PI-GS/fusion estimate  --->  known canonical source grid
```

ASCII version:

```text
raw estimates:

  -0.71    -0.31    -0.03     0.28     0.76

known grid:

  -0.80    -0.40     0.00     0.40     0.80

calibration learns:

  raw x ---------------> corrected x
```

This can produce paper-level chirp numbers, but it has a specific meaning:

```text
It supports "chirp can calibrate this known experimental grid."

It does NOT by itself prove "blind continuous localization works this well."
```

That distinction matters for honest interpretation.

## Results Interpreted Through The Formula

The recent chirp diagnostic progression can be read as progressively repairing
the formula's assumptions:

```text
single fixed condition:
  about 6.01 deg MAE
  -> formula works somewhat, but one band/window/geometry is not enough

fixed three-regime fusion + center/edge prior:
  about 3.38 deg MAE
  -> multiple physical views reduce false peaks and compression bias

dataset-ranked top-K ensemble + prior:
  about 2.36 deg MAE
  -> data contains near-paper-level information, but this is an in-dataset
     diagnostic because condition ranking uses evaluated data

canonical-grid calibration:
  about 1.01 deg all-trial MAE / 2.02 deg holdout MAE
  -> paper-level chirp behavior is plausible if chirp is treated as a known-grid
     calibration signal

sliding-window oracle upper bound:
  about 1.05 deg MAE
  -> the recordings contain good windows, but an oracle-free selector is still
     needed before this can be claimed as a fair blind pipeline
```

The practical reading:

```text
The paper formula is real and useful.
But the raw data does not naturally satisfy the clean conditions assumed by the
formula.

To approach the paper numbers, the pipeline needs preprocessing, geometry
calibration, regime fusion, and sometimes grid calibration.
```

## What Is Formula-Core Versus Extra Processing

Formula-core:

```text
compute tau_VL(p), tau_VR(p)
sample R_VL and R_VR at those predicted delays
combine the pair scores
argmax over candidate positions
```

Preprocessing that supports the formula:

```text
matched filtering
bandpass selection
differencing / signal transforms
window selection
robust score normalization
```

Calibration that supports the formula:

```text
effective LDV geometry
delay sign
common delay shift
chirp-derived coordinate correction
```

Post-processing after the formula:

```text
center deadband
edge dilation
known-grid snapping
multi-regime estimate fusion
```

Important interpretation:

```text
If we report a result using only formula-core, it is a stricter blind
localization claim.

If we report a result with chirp grid calibration, it is a calibration-assisted
claim and should be labeled that way.
```

## Bottom Line

The paper's last formula says:

```text
Find the source position whose geometry-predicted LDV-Mic delays are jointly
supported by the measured correlation curves.
```

The recent experiments show:

```text
That idea is the right backbone, but the measured correlation curves and delay
templates are messy in real chirp/speech recordings.
```

Therefore, the current best explanation is:

```text
Paper-level performance probably required substantial practical handling around
the final formula:

  cleaner chirp/speech evidence
  calibrated effective delay templates
  multiple physical regimes
  and possibly known-grid chirp calibration
```

So the relationship is not:

```text
recent experiments replaced the formula
```

It is:

```text
recent experiments made explicit the hidden work needed for the formula to
behave like the paper says it behaves.
```

