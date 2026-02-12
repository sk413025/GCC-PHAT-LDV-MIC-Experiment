# GCC-PHAT Experimental Dataset

**Archive Version**: 1.0
**Created**: 2026-02-12
**Source Repository**: https://github.com/anthropics/nmf-sound-localizer
**Archive Repository**: https://github.com/sk413025/GCC-PHAT-LDV-MIC-Experiment

---

## Overview

This dataset consolidates chirp calibration signals and speech recordings used in the GCC-PHAT Laser Vibrometer (LDV) vs. Microphone Direction-of-Arrival (DoA) comparison experiment. It combines data from two independent collection sessions:

1. **Chirp Data** (5 speakers at 5 spatial positions): Used for truth-reference DoA calibration via GCC-PHAT
2. **Speech Data** (5 speakers at 5 spatial positions): Evaluation targets for DoA localization algorithms

---

## Directory Structure

```
GCC-PHAT-dataset/
├── README.md                    ← You are here
├── PROVENANCE.md               ← Detailed source tracking & collection metadata
├── chirp/                       ← Chirp calibration signals (5 speakers, 3 sensors each)
│   ├── 23-chirp(-0.8m)/        ← Speaker 23 at -0.8m (18.8s)
│   ├── 24-chirp(-0.4m)/        ← Speaker 24 at -0.4m (18.8s)
│   ├── 25-chirp(+0.0m)/        ← Speaker 25 at +0.0m (12.1s)
│   ├── 28-chirp(+0.4m)/        ← Speaker 28 at +0.4m (12.1s)
│   └── 29-chirp(+0.8m)/        ← Speaker 29 at +0.8m (12.1s)
│
└── speech/                      ← Speech recordings (5 speakers, 9 files each = 3 sensors × 3 utterances)
    ├── 18-0.1V/                ← Speaker 18 at +0.8m
    ├── 19-0.1V/                ← Speaker 19 at +0.4m
    ├── 20-0.1V/                ← Speaker 20 at +0.0m (center)
    ├── 21-0.1V/                ← Speaker 21 at -0.4m
    └── 22-0.1V/                ← Speaker 22 at -0.8m
```

**File Count**: 75 total (30 chirps + 45 speech)

---

## Chirp Signals

### Purpose
Ground-truth calibration for Direction-of-Arrival (DoA) via GCC-PHAT analysis.

### File Format
- **Format**: WAV (48kHz, 24-bit, mono PCM)
- **Duration**: 12.1-18.8 seconds (see PROVENANCE.md for details)
- **Files per speaker**: 3 (LDV, LEFT-MIC, RIGHT-MIC)
- **Total chirp files**: 15 (5 speakers × 3 sensors)

### Speaker-Position Mapping

| Speaker | Position | Duration | Filename Pattern |
|---------|----------|----------|------------------|
| 29 | +0.8m | 12.1s | `0128-{SENSOR}-29-chirp(+0.8m).wav` |
| 28 | +0.4m | 12.1s | `0128-{SENSOR}-28-chirp(+0.4m).wav` |
| 25 | +0.0m | 12.1s | `0128-{SENSOR}-25-chirp(0.0m).wav` |
| 24 | -0.4m | 18.8s | `0128-{SENSOR}-24-chirp(-0.4m).wav` |
| 23 | -0.8m | 18.8s | `0128-{SENSOR}-23-chirp(-0.8m).wav` |

*Note: Sensor = LDV, LEFT-MIC, or RIGHT-MIC*

### Typical Usage
```python
# Load GCC-PHAT truth-reference tau and theta from chirp signals
chirp_truthref = json.load(open('chirp_truthref_5s.json'))
tau_ms = chirp_truthref['22-0.1V']['tau_ms']        # -1.307 ms
theta_deg = chirp_truthref['22-0.1V']['theta_deg']  # -18.677 degrees
```

---

## Speech Signals

### Purpose
Evaluation targets for DoA localization algorithms, guided by chirp-derived truth-references.

### File Format
- **Format**: WAV (48kHz, 24-bit, mono PCM)
- **Duration**: ~3-5 seconds per utterance
- **Files per speaker**: 9 (3 sensors × 3 utterances)
- **Total speech files**: 45 (5 speakers × 9 files)

### Speaker-Position Mapping

| Speaker ID | Voltage | Position | Corresponds to Chirp |
|-----------|---------|----------|---------------------|
| 18-0.1V | 0.1V | +0.8m | Chirp 29 |
| 19-0.1V | 0.1V | +0.4m | Chirp 28 |
| 20-0.1V | 0.1V | +0.0m | Chirp 25 |
| 21-0.1V | 0.1V | -0.4m | Chirp 24 |
| 22-0.1V | 0.1V | -0.8m | Chirp 23 |

*Voltage indicates speaker amplification level during recording.*

### File Organization

Each speaker directory contains:
```
18-0.1V/
├── 0128-LDV-18-0.1V.wav         ← Utterance 1
├── 0128-LEFT-MIC-18-0.1V.wav
├── 0128-RIGHT-MIC-18-0.1V.wav
├── 0128-LDV-18-0.1V-v2.wav      ← Utterance 2
├── 0128-LEFT-MIC-18-0.1V-v2.wav
├── 0128-RIGHT-MIC-18-0.1V-v2.wav
├── 0128-LDV-18-0.1V-v3.wav      ← Utterance 3
├── 0128-LEFT-MIC-18-0.1V-v3.wav
└── 0128-RIGHT-MIC-18-0.1V-v3.wav
```

### Typical Usage
```python
# Extract speech features and estimate DoA using chirp truth-references
from gcc_phat import gcc_phat_analysis

left_wav = load_audio('speech/18-0.1V/0128-LEFT-MIC-18-0.1V.wav')
right_wav = load_audio('speech/18-0.1V/0128-RIGHT-MIC-18-0.1V.wav')

# GCC-PHAT analysis guided by chirp truth (tau=-1.307ms from chirp 23)
tau_estimated = gcc_phat_analysis(left_wav, right_wav,
                                   truth_tau_ms=chirp_truthref['22-0.1V']['tau_ms'])
doa_error_ms = tau_estimated - chirp_truthref['22-0.1V']['tau_ms']
```

---

## Experimental Design

### Core Question
*Can LDV-aligned-to-Mic reference, when guided by chirp truth calibration, achieve comparable DoA performance to baseline Mic-Mic reference systems?*

### Signal Pairs Evaluated
1. **LDV-MicL** (Primary): Laser Vibrometer aligned to Left Microphone
2. **MicL-MicR** (Baseline): Left Microphone vs Right Microphone pair
3. **LDV-MicL + Geometry** (Negative Control): Using geometry-based truth instead of chirp

### Expected Results
- **LDV-MicL + Chirp**: ≥4/5 speakers pass (should match MicL-MicR baseline)
- **MicL-MicR + Chirp**: 4/5 speakers pass (established baseline)
- **LDV-MicL + Geometry**: ≤3/5 speakers pass (negative control - geometry insufficient alone)

---

## Technical Specifications

### Audio Specifications
- **Sample Rate**: 48,000 Hz (48 kHz)
- **Bit Depth**: 24-bit
- **Encoding**: Linear PCM (uncompressed)
- **Channels**: Mono (each WAV is single-channel)
- **Byte Order**: Little-endian (standard WAV)

### Equipment Used
- **LDV**: Laser Vibrometer (single-axis measurement)
- **Microphones**: Standard omnidirectional microphones (LEFT-MIC, RIGHT-MIC)
- **Microphone Array Geometry**: 0.128m baseline (speaker IDs derived from session code 0128)

### Calibration
- All signals are assumed to have undergone pre-collection alignment verification
- LDV→Mic OMP-based alignment performed during Stage 1-2 of experiment
- No post-processing or filtering applied to raw WAV files

---

## Data Lineage

### Chirps 25, 28, 29 (Positive Positions)
```
Original Location: dataset/chirp/{+0.0, +0.4, +0.8}/
Status: ✅ Complete metadata available
Stored in: Main repository tracked files
```

### Chirps 23, 24 (Negative Positions)
```
Original Location: worktree/exp-ldv-perfect-geometry-cloud/_old_reports/
Source Commit: 754a7e8 (2026-02-01, author: yu_chen)
Cloud Branch: exp-ldv-perfect-geometry-clean
Status: ⚠️ Collection metadata incomplete (see PROVENANCE.md)
```

### Speech 18-22
```
Original Location: dataset/GCC-PHAT-LDV-MIC-Experiment/
Status: ✅ Complete metadata available
Stored in: Main repository tracked files
```

**Full Details**: See `PROVENANCE.md` for complete lineage including commit hashes, collection dates, and data integrity notes.

---

## Storage & Access

### Git LFS
All WAV files are stored using **Git Large File Storage (LFS)** to reduce repository size:
- LFS pointer files are committed to git (lightweight)
- Actual WAV binary data is stored on LFS server
- Download: Automatic on `git clone` (requires `git lfs`)

### Remote Locations
- **Primary Archive**: https://github.com/sk413025/GCC-PHAT-LDV-MIC-Experiment
- **Branch**: `dataset/canonical-v1`
- **Installation**: `git clone https://github.com/sk413025/GCC-PHAT-LDV-MIC-Experiment.git && git lfs pull`

### Local Access (Development)
```bash
# If you have local copy
ls dataset/GCC-PHAT-dataset/chirp/29-chirp*/
ls dataset/GCC-PHAT-dataset/speech/18-0.1V/
```

---

## Verification Checklist

Before using this dataset, verify:
- ✅ All 75 WAV files are present (30 chirp + 45 speech)
- ✅ Chirps: 5 directories, 3 files each (15 total)
- ✅ Speech: 5 directories, 9 files each (45 total)
- ✅ No corrupted files: Open a few random WAVs in audio editor
- ✅ Format consistency: All should be 48kHz, 24-bit, mono PCM
- ✅ Directory structure matches layout above

---

## Citation

If using this dataset in research, cite:

```bibtex
@dataset{gcc_phat_ldv_mic_2026,
  author = {Jenner and Yu Chen},
  title = {GCC-PHAT Laser Vibrometer vs. Microphone DoA Comparison Dataset},
  year = {2026},
  url = {https://github.com/sk413025/GCC-PHAT-LDV-MIC-Experiment},
  note = {Branch: dataset/canonical-v1}
}
```

---

## Questions & Support

For questions about this dataset:
1. Check `PROVENANCE.md` for detailed source tracking
2. Refer to experiment plan: `EXP_LDV_VS_MIC_DOA_COMPARISON.md`
3. Contact: Original contributors (see commit history)

---

**Dataset Version**: 1.0
**Last Updated**: 2026-02-12
**Storage**: Git LFS
**License**: (As specified in parent repository)
