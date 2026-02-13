# Spec: Band-OMP Teacher → Band-DTmin Student (LDV+MIC Helps MIC-MIC on Speech Tails)

## Scope
This spec defines a decision-complete workflow to:
- Generate **band-selection** teacher trajectories (OMP-like greedy selection).
- Train a lightweight DTmin student to imitate these trajectories.
- Evaluate student vs baseline MIC-MIC on speech tail metrics (vs chirp reference truth).

This spec intentionally does **not** attempt to recover LDV↔MIC `τ₂/τ₃` as geometry constraints.

## Inputs

### Data root (WAVs)
Directory structure:
```
<DATA_ROOT>/
  18-0.1V/
    *LDV*.wav
    *LEFT*.wav
    *RIGHT*.wav
  ...
```

### Chirp truth-reference root
Per speaker:
`<TRUTH_REF_ROOT>/<speaker>/summary.json` containing:
```
truth_reference.tau_ref_ms: float
truth_reference.theta_ref_deg: float
```

## Windowing (locked)
- `fs = 48000` (fail-fast)
- window length: `5.0 s`
- centers: `100..600` seconds inclusive, step `1.0 s` (501 candidates)
- speech-active windows: keep windows where `RMS(MicL)` ≥ the 50th percentile over candidate windows for that speaker
- silence windows: bottom `1%` RMS(MicL) (minimum 3 windows; else fail)

## Bands (locked)
- analysis band: `500–2000 Hz`
- number of bands: `B = 64`
- band edges: linear spacing over `[500, 2000]` Hz

## Guardrail G1: Coupling-forbidden bands (locked)

### Compute silence-window coherence
For each silence window, compute magnitude-squared coherence for:
- MicL–MicR
- LDV–MicL
- LDV–MicR

Using Welch:
- `nperseg=8192`, `noverlap=4096`, Hann

Compute:
`γ²_xy(f) = |S_xy(f)|² / (S_xx(f) S_yy(f))`

Average γ² over silence windows **per pair**, then take:
`γ²_max_silence(f) = max(γ²_mic, γ²_ldvL, γ²_ldvR)`

Convert to band metric:
`cpl_band[b] = mean_{f in band b}( γ²_max_silence(f) )`

Forbidden rule:
- band `b` forbidden if `cpl_band[b] >= 0.20`

Artifact:
- `per_speaker/<speaker>/coupling_mask.json` containing `cpl_band`, `forbidden_bands`, and `band_edges_hz`.

### Degenerate case (must handle)
If the raw forbidden rule would mark **all** bands as forbidden (common under strong coupling),
disable the hard-forbid and use `cpl_band` only as a penalty feature in the observation vector.
Record both:
- `forbidden_bands_raw`
- `forbidden_bands_effective`
- `forbid_rule_degenerated = true`

## Teacher (Band-OMP) (locked)

### Baseline estimator
MIC-MIC GCC-PHAT on full analysis band (500–2000 Hz), guided peak search:
- guided center: `tau_ref_ms`
- guided radius: `0.3 ms`
- max lag window: `±10 ms`
- PSR exclude: `50 samples`

### Teacher action space
- action is a band index `b ∈ {0..B-1}`
- cannot select a forbidden band
- cannot select a band already selected
- horizon/sparsity budget `K=6`

### FFT-domain PHAT spectrum
For a window (MicL=x, MicR=y) and FFT length `n_fft = len(x)+len(y)`:
```
R(f)      = X(f) * conj(Y(f))
R_phat(f) = R(f) / (|R(f)| + eps)
```
`eps = 1e-12`

### Band masks on FFT grid
For each band b, define a binary mask `M_b(f)` over FFT bins in that band.

### Correlation for a selected set S
```
R_S(f) = mean_{b in S}( M_b(f) * R_phat(f) )
cc_S   = irfft(R_S)
cc_win = concat( cc_S[-max_shift:], cc_S[:max_shift+1] )
```

Compute guided peak (argmax of `|cc_win|` in the guided window), and PSR:
```
PSR = 20 log10( peak / (max_sidelobe + 1e-10) )
```
Exclude ±50 samples around the peak when finding sidelobe max.

### Teacher score (per step)
Let `tau_S_ms` be the selected delay in milliseconds, and `psr_S_db` the PSR.
```
score(S) = -(|tau_S_ms - tau_ref_ms| / 0.30) + (psr_S_db / 3.0)
```

### Greedy selection rule
Initialize `S = ∅`, `score(S) = -∞`.
For step `k=1..K`:
1) Evaluate all allowed candidates `b ∉ S`:
   - compute `score(S ∪ {b})`
2) Choose `b* = argmax score(S ∪ {b})`
3) Stop early if `score(S ∪ {b*}) <= score(S) + 0.01`
4) Else set `S := S ∪ {b*}` and continue

### Teacher observation vector (truth-free)
For each window compute band-level features:
- `logPSD_LDV_band[b]`
- `logPSD_MicAvg_band[b]`
- `mic_coh_speech_band[b]` (MicL–MicR coherence within the window)
- `cpl_band[b]` from silence windows (speaker-level)

Then:
```
obs[b] = zscore(logPSD_LDV_band)[b]
       + zscore(logPSD_MicAvg_band)[b]
       + zscore(mic_coh_speech_band)[b]
       - 2.0 * zscore(cpl_band)[b]
```
`zscore` is computed across bands within a window (mean/std over b=0..B-1), and must fail if std≈0.

### Teacher trajectories NPZ schema
File: `teacher_trajectories.npz`
- `observations`: float32, shape `(N, K, B)` (repeat obs across steps)
- `actions`: int32, shape `(N, K)` (band indices, -1 padding)
- `valid_len`: int32, shape `(N,)`
- `speaker_id`: string, shape `(N,)`
- `center_sec`: float64, shape `(N,)`
- `forbidden_mask`: bool, shape `(N, B)`
- `band_edges_hz`: float64, shape `(B+1,)`

## Student (Band-DTmin) (locked)

### Model
Nearest-centroid per step:
- Centroid for (k,b) is the mean observation vector for samples whose teacher action at step k is b.
- Predict action by minimum L2 distance to centroids, restricted to actions with at least one training example.

### Invalid action rule (locked)
If the predicted action at a step is:
- forbidden, or
- already selected,
then **stop early** (do not search fallback).

## Evaluation (locked)
Test set: all samples with `center_sec > 450 s`.
Compute baseline vs student on these windows and report:
- pooled and per-speaker:
  - median/p90/p95 of `theta_error_ref_deg`
  - fail rate `theta_error_ref_deg > 5°`

Win condition: see `docs/agent_prompts/band_omp_teacher_student.md`.
