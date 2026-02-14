# LDV vs Mic Detailed Record (A-I Sweep with Mic-Superiority Gate)

Date: 2026-02-14  
Run root: `results/stage4_untried_ai_with_mic_gate_20260214_1035`

## 1. LDV-Better-Than-Mic Summary (Prioritized)

Primary criterion tracked in this run (for `ldv_micl` configs):

- `ldv_better_than_mic := theta_error_median_deg(LDV path) < theta_error_median_deg(MicL-MicR baseline)`

Observed counts:

| Config | LDV better than Mic | Pass count |
| --- | --- | --- |
| `B_narrowband_fusion_welch_phat` | **4/5** | 1/5 |
| `baseline_fft_band200_4000` | **2/5** | 2/5 |
| `I_dual_baseline_joint_weighted` | **2/5** | 2/5 |
| `A_welch_coherence_weighted_band200_4000` | 1/5 | 1/5 |
| `F_welch_phat_band200_4000` | 1/5 | 1/5 |
| `D_plus_H_plus_F_ldv_comp_cohmask` | 1/5 | 0/5 |
| `D_plus_F_welch_phat_ldv_comp` | 0/5 | 0/5 |

Key point:

- LDV can outperform Mic in several configs (especially `B`, 4/5), **but pass still stays low** when the run also enforces `omp_better_than_raw`.

## 2. Pass Rule Used in This Sweep

For all `ldv_micl` configs in this run:

- `pass = omp_better_than_raw AND omp_error_small(theta<5deg) AND psr_min_ok AND ldv_better_than_mic`

Therefore, a case can satisfy `ldv_better_than_mic=true` and still fail if `omp_better_than_raw=false`.

## 3. Detailed Per-Config Diagnosis

### 3.1 baseline_fft_band200_4000

- pass: 2/5
- ldv_better_than_mic: 2/5
- omp_better_than_raw: 4/5
- both (ldv_better + omp_better + theta_small): 2/5
- failing speakers: `18, 19, 20`
  - `18`: failed because `ldv_better_than_mic=false` (`ldv_err=1.650`, `mic_err=0.643`)
  - `19`: failed because `ldv_better_than_mic=false` and `omp_better_than_raw=false`
  - `20`: failed because `ldv_better_than_mic=false` (`ldv_err=0.026`, `mic_err=0.001`)

### 3.2 F_welch_phat_band200_4000

- pass: 1/5
- ldv_better_than_mic: 1/5
- omp_better_than_raw: 1/5
- both: 1/5
- failing speakers: `18, 20, 21, 22`
  - all four failed due `ldv_better_than_mic=false` and `omp_better_than_raw=false`

### 3.3 A_welch_coherence_weighted_band200_4000

- pass: 1/5
- ldv_better_than_mic: 1/5
- omp_better_than_raw: 3/5
- both: 1/5
- failing speakers: `18, 20, 21, 22`
  - `18`: `ldv_better=false`, `omp_better=false`
  - `20`: `ldv_better=false`, `omp_better=false`, and `theta_small=false`
  - `21`, `22`: `ldv_better=false` (even though `omp_better=true`)

### 3.4 D_plus_F_welch_phat_ldv_comp

- pass: 0/5
- ldv_better_than_mic: 0/5
- omp_better_than_raw: 3/5
- both: 0/5
- failing speakers: `18, 19, 20, 21, 22`
  - all failed primarily because `ldv_better_than_mic=false`

### 3.5 D_plus_H_plus_F_ldv_comp_cohmask

- pass: 0/5
- ldv_better_than_mic: 1/5
- omp_better_than_raw: 2/5
- both: 0/5
- failing speakers: `18, 19, 20, 21, 22`
  - `19` had `ldv_better=true` but failed due `omp_better=false`
  - others mainly failed on `ldv_better=false`

### 3.6 B_narrowband_fusion_welch_phat

- pass: 1/5
- ldv_better_than_mic: **4/5** (best)
- omp_better_than_raw: 1/5
- both: 1/5
- failing speakers: `18, 19, 20, 21`
  - `18`, `20`, `21`: failed mainly because `omp_better_than_raw=false` despite `ldv_better=true`
  - `19`: failed both (`ldv_better=false`, `omp_better=false`)

### 3.7 I_dual_baseline_joint_weighted

- pass: 2/5
- ldv_better_than_mic: 2/5
- omp_better_than_raw: **5/5**
- both: 2/5
- failing speakers: `20, 21, 22`
  - all three failed because `ldv_better_than_mic=false`
  - note: `omp_better_than_raw=true` for all speakers in this config

### 3.8 E_full_segment_micl_micr_welch_weighted (Mic baseline reference)

- signal pair: `micl_micr`
- pass: 1/5
- this config is not in LDV-vs-Mic replacement mode; it is a Mic baseline run under `theta_only` pass mode.

## 4. Reproduction Commands

Smoke test used for new I + mic-gate path:

```bash
python -u scripts/stage4_doa_ldv_vs_mic_comparison.py \
  --data_root dataset/GCC-PHAT-dataset/speech \
  --speaker 20-0.1V \
  --output_dir results/stage4_dual_baseline_smoke_20260214_0201 \
  --signal_pair ldv_micl \
  --segment_mode fixed --n_segments 1 \
  --analysis_slice_sec 5 --eval_window_sec 5 \
  --gcc_method welch_coherence_weighted \
  --gcc_bandpass_low 200 --gcc_bandpass_high 4000 \
  --dual_baseline_fusion \
  --ldv_prealign gcc_phat \
  --alignment_mode omp --max_k 3 \
  --use_geometry_truth --pass_mode omp_vs_raw \
  --pass_require_ldv_better_than_mic
```

Full A-I sweep with mic-gate:

```bash
python -u scripts/run_ldv_vs_mic_untried_directions.py \
  --data_root dataset/GCC-PHAT-dataset/speech \
  --output_base results/stage4_untried_ai_with_mic_gate_20260214_1035
```

## 5. Data Lineage

- Dataset root: `dataset/GCC-PHAT-dataset/speech`
- Speakers: `18-0.1V, 19-0.1V, 20-0.1V, 21-0.1V, 22-0.1V`
- Total files used: 15
- Fingerprint SHA256: `5bad7debd72ebc97f8cef94fd39049ad171c599a2ad8ad65c4f9d83574999f93`

## 6. Main Takeaway

- Under strict pass definition, LDV replacement remains difficult.
- However, LDV is not uniformly inferior to Mic: the `B` direction shows LDV can beat Mic on most speakers, but still misses pass due the additional `omp_vs_raw` constraint.
- Direction `I` (dual-baseline) improves stability of `omp_vs_raw` (5/5 true) but still only 2/5 beats Mic on median theta error.
