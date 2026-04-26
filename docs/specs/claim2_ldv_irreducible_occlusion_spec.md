# Spec: Claim 2 Verification — LDV Irreducibility Under Mic Occlusion (Existing WAVs Only)

## Purpose
This experiment is a follow-up Claim-2 verification designed to intentionally create a **mic-local near-fail regime** using an occlusion-style frequency-response distortion applied to **one mic only**, while keeping LDV unchanged.

Claim under test (conditional):

> Under mic-local degradations (occlusion spectral distortion + in-band noise + saturation/clipping), an **LDV+MIC** student policy provides a measurable benefit over an otherwise identical **MIC-only** student for stabilizing MIC–MIC guided GCC-PHAT on speech tails.

---

## Locked dataset inputs
- `data_root`: `/home/sbplab/jiawei/data`
- Speakers: `18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V`
- Truth reference (unchanged): `results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000`
- Windowing: 5.0 s, centers `100..600` step `1`
- Speech-active filter: `RMS(MicL) >= p50` per speaker (computed on clean MicL)
- MIC–MIC analysis band: `500–2000 Hz`
- Guided radius: `0.3 ms`, max lag: `±10 ms`
- Band policy: `B=64` linear bands (500–2000), horizon `K=6`
- Coupling mask policy: **mic-only coupling** from clean silence windows (MicL–MicR coherence), forbid threshold `gamma^2 >= 0.20` with explicit “forbid-all degeneracy” handling.

---

## Corruption model (mic-local, deterministic)
We corrupt MicL/MicR using:
1) **In-band uncorrelated noise** sourced from silence windows (existing WAVs only), and
2) **Occlusion-style spectral shaping** applied to **one mic only** (default MicR), and
3) **Gain→clip→de-gain** saturation proxy.

LDV is never corrupted.

### Noise + clipping (same as prior Claim-2 sweep)
- SNR is defined in `500–2000 Hz` and enforced by scaling band-limited noise.
- `preclip_gain = 100`, `clip_limit = 0.99`.

### Occlusion (zero-phase spectral shaping)
Occlusion is applied as **magnitude-only shaping** on the target mic window:
- This preserves phase to avoid introducing a new “delay truth” inconsistent with the chirp reference.
- The goal is to reduce effective in-band SNR and coherence, pushing baseline MIC–MIC into a near-fail regime.

Implementation: `scripts/mic_corruption.py::apply_occlusion_fft()`

Supported kinds:
- `lowpass`: hard spectral mask `f <= lowpass_hz`
- `tilt`: power-law attenuation above a pivot: `gain(f) = (max(f,pivot)/pivot)^(-k)` for `f >= pivot`

### Mixing rule (important)
Noise scaling `alpha` is computed using the **clean** mic as the reference (`signal_for_alpha = mic_clean`), but the mixed signal may be occluded:
- `signal_for_mix = apply_occlusion_fft(mic_clean)` if occluded; else `mic_clean`.
- `mic_noisy = signal_for_mix + alpha * noise_bp`

This makes the occluded mic effectively have **lower achieved SNR** than the nominal grid, which is the intended stress.

---

## Swept grid (pre-registered)
We sweep:
- `snr_db ∈ {0, -5, -10}` (in-band 500–2000)
- `occlusion ∈ {lp800_micr, lp600_micr, tilt2_micr}`

Where:
- `lp800_micr`: lowpass cutoff 800 Hz on MicR
- `lp600_micr`: lowpass cutoff 600 Hz on MicR
- `tilt2_micr`: tilt exponent k=2 above pivot 800 Hz on MicR

Global seed: `1337`.

---

## Acceptance criteria (Claim 2)
We accept Claim 2 if **any** grid point satisfies:
1) Baseline near-fail: `fail_rate_ref(theta_error_ref_deg > 4°) >= 0.40`
2) LDV+MIC feasibility: LDV+MIC student `fail_rate_ref(>4°) <= 0.10`
3) Irreducibility (ablation): LDV+MIC student beats MIC-only student:
   - p95 improvement ≥ 10% and fail-rate improvement ≥ 20% (relative)

If (1) never happens across the grid, record a negative result: the proxy did not create sufficiently extreme windows.
