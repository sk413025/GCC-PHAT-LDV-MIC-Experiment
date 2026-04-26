# Commit Strategy: Integrating PI-GS Empirical PGFPlots Data

## 1. Commit Message Structure

**Commit Title:**
`feat(paper): inject empirical pgfplots for Coherence Trap and Jammer Immunity`

**Commit Body Format:**

```text
feat(paper): inject empirical pgfplots for Coherence Trap and Jammer Immunity

**Background & Motivation:**
The `doc/interspeech-2026-rewrite` branch successfully pivoted the narrative from PI-DNN to a zero-shot Physics-Informed Geometric Search (PI-GS). While the theoretical arguments for LDV anchor isolation (Sec 2.2) and the Coherence Trap (Sec 3.1) are established, the paper lacked raw empirical visualizations to substantiate these physical claims against reviewer scrutiny. Specifically, the "Jammer Immunity" claim needed quantifiable proof showing how Microphone arrays collapse while the LDV-Mic fusion remains anchored.

**Objective:**
1. Visualize the "Coherence Trap" (Rayleigh Blur) using raw experimental cross-spectral data to prove why naive blind search fails.
2. Provide a "Confidently Wrong" Jammer Resilience Curve to empirically validate the impedance mismatch hypothesis (LDV immunity).

**Expected Results:**
- Mic-Mic cross-spectrum should show massive spatial smearing (multipath) across all lags.
- LDV-Mic cross-spectrum should show a clean, isolated negative-lag geometry.
- As Signal-to-Jammer Ratio (SJR) decreases (jammer gets louder), the Mic-Mic DoA error should rapidly saturate toward the jammer's location, while PI-GS error should remain flat and immune.

**Actual Results & Interpretation:**
- **Scalograms:** Mic-Mic shows severe chaotic multipath in both positive/negative domains. LDV-Mic cleanly isolates the negative-lag direct path. This visually proves the necessity of the PI-GS joint optimization to unwrap the Mic-Mic blur.
- **Jammer Curve:** Extracted using the official `multi_sensor_fusion_doa.py` (`E_msnf_3` method) from the `master` branch. As SJR approaches -40dB, Mic-Mic error saturates at ~3.2°. PI-GS completely ignores the jammer, locking its error at a highly stable ~1.6°. This physically proves the structural low-pass filtering effect of the barrier.

**Reproduction Steps:**
1. **Scalograms (Fig. 2):**
   - Run: `python scripts/generate_freq_x_scalogram.py`
   - Inputs: `0223-LDV-40-boy(+0.4m)-13-block.wav`, `0223-MIC-LEFT...`, `0223-MIC-RIGHT...`
   - Outputs: `results/freq_x_scalogram_MicMic.dat`, `results/freq_x_scalogram_PIGS.dat`
2. **Jammer Curve (Fig. 4):**
   - Run: `python scripts/generate_jammer_curve_msnf.py`
   - Core Logic: Sweeps SJR by mixing `0223-unblock-7(high)` jammer into the target mics. Feeds mixed signals into `multi_sensor_fusion_doa.py`'s exact `E_msnf_3` WLS formulation.
   - Outputs: `results/jammer_resilience_curve_msnf.dat`

**Data Lineage:**
The LaTeX source (`paper/main.tex`) has been explicitly commented to map each `\addplot` directly to the Python script and output `.dat` file responsible for its generation, ensuring full experimental provenance.
```

## 2. Next Actions
- I will modify `main.tex` to inject explicit LaTeX comments above each `pgfplots` figure detailing the *Data Lineage* (which python script generates the `.dat` file, what the source `.wav` files are, and where the `.dat` file resides).
- I will then execute the `git add`, `git commit` using the detailed message planned above, and finally `git push`.
