# Spec: Extreme Subset Evaluation (Existing WAVs Only)

## Scope
This spec defines how to evaluate baseline vs student performance on **truth-free** extreme subsets of speech windows.

## Inputs
- Student run directory containing `test_windows.jsonl`
  - Each record includes `speaker_id`, `center_sec`, and:
    - `baseline.theta_error_ref_deg`
    - `student.theta_error_ref_deg`
- Raw WAVs in `--data_root`:
  - `<speaker>/*LEFT*.wav`
  - `<speaker>/*RIGHT*.wav`

## Locked constants
- `fs = 48000` (fail-fast)
- window length: 5.0 s
- Welch parameters: `nperseg=8192`, `noverlap=4096`, Hann
- Analysis band: 500–2000 Hz
- HF band: 2–8 kHz
- Extreme percentile: 10%
- Clipping thresholds:
  - `clip_frac = mean(|x| >= 0.99)`
  - extreme clipping: `clip_frac_max >= 1e-3`

## Truth-free diagnostics per test window
1) `mic_coh_median_500_2000`
   - Compute Welch coherence:
     - `γ²(f) = |Sxy(f)|² / (Sxx(f) Syy(f))`
   - Then `median(γ²(f))` over `f ∈ [500, 2000]`
2) `hf_imbalance_db`
   - Compute Welch PSD integrals in 2–8 kHz:
     - `E_L = ∑ Pxx_L(f)` over `f ∈ [2000, 8000]`
     - `E_R = ∑ Pxx_R(f)` over `f ∈ [2000, 8000]`
   - `hf_imbalance_db = 10 log10((E_L+eps)/(E_R+eps))`
3) `clip_frac_max`
   - `max(clip_frac_L, clip_frac_R)`
4) `speech_rms_db`
   - `20 log10(RMS(MicL)+eps)`

## Extreme subsets (computed on test windows only)
- `LOW_COH`: bottom 10% by `mic_coh_median_500_2000`
- `HIGH_HF_IMB`: top 10% by `|hf_imbalance_db|`
- `CLIPPED`: `clip_frac_max >= 1e-3`
- `LOW_RMS`: bottom 10% by `speech_rms_db`

## Metrics per subset (vs chirp reference)
Compute for both baseline and student:
- count, median, p90, p95 of `theta_error_ref_deg`
- fail rate: `mean(theta_error_ref_deg > 5°)`

## Acceptance (Claim 1)
PASS if:
1) At least 2/4 subsets satisfy:
   - p95 improvement frac ≥ 0.15
   - fail-rate improvement frac ≥ 0.20
2) At least 1 subset satisfies:
   - baseline fail-rate ≥ 0.40 AND student fail-rate ≤ 0.10

If FAIL, record as negative evidence: dataset likely lacks sufficiently extreme windows under these proxies.

