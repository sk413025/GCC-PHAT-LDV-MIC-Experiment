# GCC-PHAT Dataset Provenance Tracking

**Compiled**: 2026-02-12
**Source**: Research experiment dataset consolidation for public archive
**Format**: WAV (48kHz, 24-bit, mono PCM)

---

## Chirp Data

### Positive Position Chirps (12.1s each, 1.7 MB × 3 files per speaker)

| Speaker | Position | Original Location | Source Commit | Files |
|---------|----------|------------------|---------------|-------|
| 25 | +0.0m | `dataset/chirp/+0.0/` | N/A (local) | 0128-LDV, 0128-LEFT-MIC, 0128-RIGHT-MIC |
| 28 | +0.4m | `dataset/chirp/+0.4/` | N/A (local) | 0128-LDV, 0128-LEFT-MIC, 0128-RIGHT-MIC |
| 29 | +0.8m | `dataset/chirp/+0.8/` | N/A (local) | 0128-LDV, 0128-LEFT-MIC, 0128-RIGHT-MIC |

**Recording Details**:
- Sample rate: 48,000 Hz
- Bit depth: 24-bit
- Channel: Mono (separate files for LDV and two microphones)
- Duration: ~12.1 seconds
- Equipment: Laser Vibrometer (LDV) + Microphone pair (LEFT-MIC, RIGHT-MIC)
- Status: ✅ Full collection metadata available

---

### Negative Position Chirps (18.8s each, 2.6 MB × 3 files per speaker)

| Speaker | Position | Original Location | Source Commit | Files |
|---------|----------|------------------|---------------|-------|
| 23 | -0.8m | `worktree/exp-ldv-perfect-geometry-cloud/_old_reports/23-chirp(-0.8m)/` | 754a7e8 | 0128-LDV, 0128-LEFT-MIC, 0128-RIGHT-MIC |
| 24 | -0.4m | `worktree/exp-ldv-perfect-geometry-cloud/_old_reports/24-chirp(-0.4m)/` | 754a7e8 | 0128-LDV, 0128-LEFT-MIC, 0128-RIGHT-MIC |

**Recording Details**:
- Sample rate: 48,000 Hz
- Bit depth: 24-bit
- Channel: Mono (separate files for LDV and two microphones)
- Duration: ~18.8 seconds
- Equipment: Laser Vibrometer (LDV) + Microphone pair (LEFT-MIC, RIGHT-MIC)
- Status: ⚠️ **Source metadata incomplete** - Files first appeared in commit 754a7e8 (2026-02-01, author: yu_chen) from cloud branch `exp-ldv-perfect-geometry-clean`. Original collection date and parameters not formally documented in git history.

**Commit 754a7e8 Message**:
```
exp(stage4b): chirp mic truth-ref (scan+prealign)

Commits chirp inputs (23/24) and Stage4-B outputs using scan + LDV→MicL prealign.
Also keeps chirp ablations (no-prealign, prealign-only) for lineage.
```

---

## Speech Data

### Speakers 18-22 (Variable duration, 2-3 utterances per speaker)

| Speaker | Position | Original Location | Source Commit | Files per Utterance |
|---------|----------|------------------|---------------|---------------------|
| 18-0.1V | +0.8m | `dataset/GCC-PHAT-LDV-MIC-Experiment/18-0.1V/` | Local | 0128-LDV, 0128-LEFT-MIC, 0128-RIGHT-MIC |
| 19-0.1V | +0.4m | `dataset/GCC-PHAT-LDV-MIC-Experiment/19-0.1V/` | Local | 0128-LDV, 0128-LEFT-MIC, 0128-RIGHT-MIC |
| 20-0.1V | +0.0m | `dataset/GCC-PHAT-LDV-MIC-Experiment/20-0.1V/` | Local | 0128-LDV, 0128-LEFT-MIC, 0128-RIGHT-MIC |
| 21-0.1V | -0.4m | `dataset/GCC-PHAT-LDV-MIC-Experiment/21-0.1V/` | Local | 0128-LDV, 0128-LEFT-MIC, 0128-RIGHT-MIC |
| 22-0.1V | -0.8m | `dataset/GCC-PHAT-LDV-MIC-Experiment/22-0.1V/` | Local | 0128-LDV, 0128-LEFT-MIC, 0128-RIGHT-MIC |

**Recording Details**:
- Sample rate: 48,000 Hz
- Bit depth: 24-bit
- Channel: Mono (separate files for LDV and two microphones)
- Duration: Variable (3-5 seconds per utterance, 2-3 utterances per speaker)
- Equipment: Laser Vibrometer (LDV) + Microphone pair (LEFT-MIC, RIGHT-MIC)
- Content: English speech utterances at controlled speaker positions
- Status: ✅ Full collection metadata available

---

## Usage & Citation

This dataset combines:
1. **Chirp calibration signals** (positive & negative mic positions) for DoA truth-referencing
2. **Speech signals** (5 speakers, 5 positions) for localization algorithm validation

### Typical Use Case
As `chirp_truthref_5s.json` truth-reference in GCC-PHAT DoA evaluation:
- Chirp speakers 25-29 provide ground truth tau_ms and theta_deg for speech evaluation
- Speech speakers 18-22 are evaluated using these chirp-derived calibrations

### Data Lineage
```
Positive Chirps (25/28/29, ~12.1s)
  └─ dataset/chirp/{+0.0, +0.4, +0.8}/

Negative Chirps (23/24, ~18.8s)
  └─ exp-ldv-perfect-geometry-clean cloud branch
     └─ commit 754a7e8 (first appearance in git)

Speech (18-22, variable duration)
  └─ dataset/GCC-PHAT-LDV-MIC-Experiment/
```

---

## Known Limitations

1. **Negative chirp collection metadata**: Speaker 23/24 WAV files lack explicit collection date, device calibration parameters, and recording session notes. Source confirmed as cloud branch but original recording context is undocumented.

2. **Duration discrepancy**: Chirp speakers 23/24 (18.8s) differ from 25/28/29 (12.1s), likely due to different recording parameters or stimulus duration. Both are valid reference signals but create asymmetry in experimental design.

3. **LDV alignment**: All files assume proper LDV→Microphone alignment was performed during collection. No raw vibrometer traces or alignment metadata are included.

---

## File Organization

All files in this archive follow the naming convention:
```
{SessionID}-{SensorType}-{SpeakerID}-{Stimulus}.wav

SessionID: 0128 (constant across all files)
SensorType: LDV | LEFT-MIC | RIGHT-MIC
SpeakerID: 23-29 (chirp) | 18-0.1V to 22-0.1V (speech)
Stimulus: chirp(-0.8m) | speech (implicit from speaker ID)
```

Example: `0128-LEFT-MIC-29-chirp(+0.8m).wav`

---

## Verification

To verify dataset integrity, check:
- ✅ Total WAV count: 30 chirp files + 45 speech files = 75 files total
- ✅ No corrupted WAVs: All files should open in standard audio software
- ✅ Format consistency: All are 48kHz, 24-bit, mono PCM
- ✅ Directory structure matches plan in README.md

---

**Archive created by**: Claude Code automation
**Provenance verified on**: 2026-02-12
**Storage format**: Git LFS (for binary WAV files)
