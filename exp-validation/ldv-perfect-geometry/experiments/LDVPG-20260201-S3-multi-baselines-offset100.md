# LDVPG-20260201-S3-multi-baselines-offset100

## 背景 / 動機

延伸 `LDVPG-20260131-S3-single-baselines`：Stage 3 在 multi-segment 情境下更容易遇到「部分窗 PSR 很低 / 多峰」的狀況，因此 baseline 需要更明確且更穩健。

本實驗做三件事：
1. 以 `baseline_method=segment` 重新產生 legacy multi-segment 結果，觀察短窗 baseline 在不同 segment 間的變異與 outlier（包含偶發的 `~ -1.69 ms` 假峰）
2. 用 `baseline_method=report` 且 `segment_offset=100s` 對齊 `full_analysis.py` 的分析區間（100–600s）
3. 用 `baseline_method=windowed` 且 `baseline_psr_min_db=10` 建立更穩健 baseline，對照通過率差異

## 實驗目的

- 建立可復現的 multi-segment Stage 3 跑法（避免整段 STFT、避免 baseline 定義不清）
- 比較：legacy segment baseline vs report baseline vs windowed baseline（PSR 篩選）

## 輸入資料

- Speech 18–22（未 commit；校驗見 `exp-validation/ldv-perfect-geometry/datasets/LDVPG_speech_18-22_root.sha256`）

## 參數（關鍵）

- `fs=48000`
- OMP STFT 頻段：`freq_min=500`, `freq_max=2000`
- GCC-PHAT：`gcc_bandpass_low=500`, `gcc_bandpass_high=2000`, `gcc_segment_sec=1.0`
- baseline 區間（對齊 `full_analysis.py`）：`baseline_start_sec=100`, `baseline_end_sec=600`
- multi-seg：`n_segments=10`, `segment_spacing=50s`, `segment_offset=100s`（對齊報告）

## 實驗指令（可復現）

> 下面假設資料夾在 repo root（`18-0.1V/`…），因此用 `--data_root .`。

### 0) Legacy：baseline_method=segment（multi-seg）

```bash
for spk in 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V; do
  python exp-validation/ldv-perfect-geometry/scripts/stage3_multi_segment.py \
    --data_root . \
    --speaker "$spk" \
    --output_dir exp-validation/ldv-perfect-geometry/validation-results/stage3_multi_segment \
    --n_segments 10 --segment_spacing 50 --segment_offset 0 \
    --analysis_slice_sec 5 --gcc_segment_sec 1 \
    --freq_min 500 --freq_max 2000 \
    --baseline_method segment
done
```

### 1) Report baseline + offset=100（對齊 full_analysis.py）

```bash
for spk in 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V; do
  python exp-validation/ldv-perfect-geometry/scripts/stage3_multi_segment.py \
    --data_root . \
    --speaker "$spk" \
    --output_dir exp-validation/ldv-perfect-geometry/validation-results/stage3_multi_segment_report_offset100 \
    --n_segments 10 --segment_spacing 50 --segment_offset 100 \
    --analysis_slice_sec 5 --gcc_segment_sec 1 \
    --freq_min 500 --freq_max 2000 \
    --baseline_method report --baseline_start_sec 100 --baseline_end_sec 600
done
```

### 2) Windowed baseline (PSR>=10) + offset=100

```bash
for spk in 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V; do
  python exp-validation/ldv-perfect-geometry/scripts/stage3_multi_segment.py \
    --data_root . \
    --speaker "$spk" \
    --output_dir exp-validation/ldv-perfect-geometry/validation-results/stage3_multi_segment_windowed_psr10_offset100 \
    --n_segments 10 --segment_spacing 50 --segment_offset 100 \
    --analysis_slice_sec 5 --gcc_segment_sec 1 \
    --freq_min 500 --freq_max 2000 \
    --baseline_method windowed --baseline_start_sec 100 --baseline_end_sec 600 \
    --baseline_window_sec 5 --baseline_hop_sec 5 --baseline_psr_min_db 10
done
```

## 解讀（重點）

- multi-segment 下，短窗 baseline 的分布可能有 outlier（偶發鎖到穩定假峰），造成「某些 segment 很怪」但 median 看起來正常
- `segment_offset=100s` 讓 multi-seg 對齊報告分析區間，便於跟 Stage 3 single / full_analysis.py 做一致比較
- `windowed + PSR` 的 baseline 比 `report` 更穩健，但仍可能受語音內容/混響影響

## Artifacts（本 commit 內）

- `exp-validation/ldv-perfect-geometry/scripts/stage3_multi_segment.py`
- `exp-validation/ldv-perfect-geometry/validation-results/stage3_multi_segment/**/summary.json`
- `exp-validation/ldv-perfect-geometry/validation-results/stage3_multi_segment_report_offset100/**/summary.json`
- `exp-validation/ldv-perfect-geometry/validation-results/stage3_multi_segment_windowed_psr10_offset100/**/summary.json`

## Lineage

- Parent Exp: `LDVPG-20260131-S3-single-baselines`

