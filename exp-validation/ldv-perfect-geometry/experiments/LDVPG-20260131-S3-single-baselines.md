# LDVPG-20260131-S3-single-baselines

## 背景 / 動機

Stage 3（Cross-Mic TDoA）曾出現「結果全部固定在 `-1.6875 ms`」的怪現象（等價於 `-81 samples @ 48kHz`）。
初步判斷不是 OMP 本身壞掉，而是 **baseline τ 的定義**在短窗、低 PSR 的情況下容易鎖到穩定假峰，導致「看起來每次都一樣」。

本實驗把 Stage 3 的 baseline 明確拆成兩種做法（`report` vs `windowed(PSR>=10)`），確認 baseline 對通過率的影響。

## 實驗目的

1. 驗證 Stage 3 對 baseline 定義的敏感度
2. 用更穩健的 baseline（windowed + PSR 篩選）減少假峰主導

## 假設 / 預期

- `baseline_method=windowed` 且 `baseline_psr_min_db>=10` 時，baseline τ 會更接近「穩定的物理解」，Stage 3 通過率應該優於 `baseline_method=report`

## 輸入資料

- Speech 18–22（未 commit；校驗見 `exp-validation/ldv-perfect-geometry/datasets/LDVPG_speech_18-22_root.sha256`）

## 參數（關鍵）

- `fs=48000`
- OMP STFT 頻段：`freq_min=500`, `freq_max=2000`
- GCC-PHAT：`gcc_bandpass_low=500`, `gcc_bandpass_high=2000`, `gcc_segment_sec=1.0`
- baseline 區間（對齊 `full_analysis.py`）：`baseline_start_sec=100`, `baseline_end_sec=600`
- `analysis_slice_sec=5.0`（避免整段 STFT 造成記憶體負擔）

## 實驗指令（可復現）

> 下面假設資料夾在 repo root（`18-0.1V/`…），因此用 `--data_root .`。

### 1) Single-Segment baseline=report

```bash
for spk in 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V; do
  python exp-validation/ldv-perfect-geometry/scripts/stage3_tdoa_evaluation.py \
    --data_root . \
    --speaker "$spk" \
    --output_dir exp-validation/ldv-perfect-geometry/validation-results/stage3_tdoa_evaluation \
    --center_sec 300 \
    --analysis_slice_sec 5 --gcc_segment_sec 1 \
    --freq_min 500 --freq_max 2000 \
    --baseline_method report --baseline_start_sec 100 --baseline_end_sec 600
done
```

### 2) Single-Segment baseline=windowed(PSR>=10)

```bash
for spk in 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V; do
  python exp-validation/ldv-perfect-geometry/scripts/stage3_tdoa_evaluation.py \
    --data_root . \
    --speaker "$spk" \
    --output_dir exp-validation/ldv-perfect-geometry/validation-results/stage3_tdoa_evaluation_windowed_psr10 \
    --center_sec 300 \
    --analysis_slice_sec 5 --gcc_segment_sec 1 \
    --freq_min 500 --freq_max 2000 \
    --baseline_method windowed --baseline_start_sec 100 --baseline_end_sec 600 \
    --baseline_window_sec 5 --baseline_hop_sec 5 --baseline_psr_min_db 10
done
```

## 實際結果摘要

| baseline | 通過率（5 speakers） | 備註 |
|---|---:|---|
| report | 2/5 | baseline PSR 仍可能偏低，某些 speaker baseline τ 不穩 |
| windowed (PSR>=10) | 3/5 | baseline 更接近穩定值（多窗 + 篩選） |

> 詳細數值請見 commit 內 artifacts（summary.json）。

## 解讀

- Stage 3 的「成功/失敗」主要受 baseline 定義影響；在短窗且低 PSR 時，GCC-PHAT 很容易鎖到穩定假峰（造成 τ 看似固定）
- `windowed + PSR` 能提升 baseline 穩定度，因此通過率上升，但仍有 speaker 在語音條件下受混響/多源影響

## Artifacts（本 commit 內）

- `exp-validation/ldv-perfect-geometry/scripts/stage3_tdoa_evaluation.py`
- `exp-validation/ldv-perfect-geometry/validation-results/stage3_tdoa_evaluation/**/summary.json`
- `exp-validation/ldv-perfect-geometry/validation-results/stage3_tdoa_evaluation_windowed_psr10/**/summary.json`

## Lineage

- 依據：`exp-validation/ldv-perfect-geometry/STAGE_VALIDATION_PLAN.md` 的 Stage 3 設計
- 相關前置：Stage 1/2 outputs（commit `5ef5f81`）與後續 Stage 4 truth-ref 設計

