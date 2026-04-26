# Spec: Offsets + Sparse Multipath Verification

## Scope
Define a reproducible diagnostic to test whether observed GCC-PHAT peaks are dominated by stable offsets / coupling paths and whether the measurement is consistent with a sparse multipath model.

## Inputs
- Data root with WAV triplets:
  - `<speaker>/*LDV*.wav`, `*LEFT*.wav`, `*RIGHT*.wav`
- Chirp truth-ref root:
  - `<truth_ref_root>/<speaker>/summary.json` containing `truth_reference.tau_ref_ms`

## Windowing (locked)
- `fs=48000` (fail-fast)
- 5.0 s windows
- centers: 100..600 s step 1 s
- speech-active: RMS(MicL) >= p50 over candidates
- silence: bottom 1% RMS(MicL), minimum 3 windows

## GCC-PHAT (locked)
- analysis band: 500–2000 Hz (FFT mask)
- max lag: ±10 ms
- global peaks: top-K=5 in |cc| with exclusion ±50 samples between picks
- guided MIC-MIC peak: around `tau_ref_ms ± 0.3 ms`

## Silence-derived offsets
For each speaker and pair (MicL–MicR, LDV–MicL, LDV–MicR):
- compute global top-1 peak tau for each silence window
- bin taus at 0.1 ms and take the mode bin center as `tau_offset_silence_ms[pair]`

## Outputs (locked)
- `per_speaker/<speaker>/windows.jsonl` containing per-window peak lists
- `per_speaker/<speaker>/summary.json` containing `tau_offset_silence_ms`
- `pair_offset_estimates.json` + `offsets_report.md`

## Acceptance (Claim 3) (locked; may fail)
PASS if:
1) For LDV–MicL and LDV–MicR, across-speaker std of `tau_offset_silence_ms` <= 0.3 ms
2) MIC-MIC offset subtraction reduces pooled median |tau_guided - tau_ref| by >= 20%
3) Sparse proxy: any LDV pair has silence top-1 bin mass >= 0.6 (approx)

If FAIL, record the negative result and do not tune thresholds silently.

