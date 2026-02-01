# LDVPG-20260201-S4B-chirp-truthref

## 背景 / 動機

Stage 4-A 顯示：在語音（短窗、混響/多源）下，`MicL–MicR` 的 GCC-PHAT 容易 peak 錯鎖，導致「幾何真值驗證」不可靠。

因此改用 `_old_reports` 的 chirp（23/24）建立 **Mic truth-ref**：
- chirp 在 mic-mic 上通常更容易得到穩定的 `τ`（相對語音）
- 先把 chirp 的 `MicL–MicR τ` 當作 reference，再看 `OMP_LDV` 是否能回到相同的 τ/θ

## 實驗目的

建立可復現且穩定的 Stage 4-B（Chirp truth-ref）流程，作為後續 Stage 4-C 的真值來源。

## 輸入資料

- `_old_reports/23-chirp(-0.8m)/`（commit 內）
- `_old_reports/24-chirp(-0.4m)/`（commit 內）

## 參數（關鍵）

- `segment_mode=scan`（掃窗挑出 mic-mic 最穩的片段）
- `eval_window_sec=1.0`
- `scan_hop_sec=0.1`
- `scan_sort_by=psr`
- `ldv_prealign=gcc_phat`（先做 LDV→MicL 的 fractional-delay 對齊）
- GCC-PHAT：不帶通（`gcc_bandpass_low/high <= 0`）

## 實驗指令（可復現）

> chirp 資料夾名稱包含 `(-0.8m)` / `(-0.4m)`，因此用 `--speaker_key` 指向幾何 speaker key（22 對應 -0.8m、21 對應 -0.4m）。

```bash
python exp-validation/ldv-perfect-geometry/scripts/stage4_doa_validation.py \
  --data_root _old_reports \
  --speaker '23-chirp(-0.8m)' --speaker_key 22 \
  --output_dir exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation_chirp_prealign_scan_psr \
  --segment_mode scan --eval_window_sec 1 --analysis_slice_sec 5 \
  --gcc_bandpass_low 0 --gcc_bandpass_high 0 \
  --ldv_prealign gcc_phat \
  --scan_hop_sec 0.1 --scan_sort_by psr \
  --scan_psr_min_db 5 --scan_ldv_micl_psr_min_db 4 --scan_tau_err_max_ms 0.3 \
  --scan_allow_fallback

python exp-validation/ldv-perfect-geometry/scripts/stage4_doa_validation.py \
  --data_root _old_reports \
  --speaker '24-chirp(-0.4m)' --speaker_key 21 \
  --output_dir exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation_chirp_prealign_scan_psr \
  --segment_mode scan --eval_window_sec 1 --analysis_slice_sec 5 \
  --gcc_bandpass_low 0 --gcc_bandpass_high 0 \
  --ldv_prealign gcc_phat \
  --scan_hop_sec 0.1 --scan_sort_by psr \
  --scan_psr_min_db 5 --scan_ldv_micl_psr_min_db 4 --scan_tau_err_max_ms 0.3 \
  --scan_allow_fallback
```

## 實際結果摘要

- `23-chirp(-0.8m)`：`|Δθ| < 1°`
- `24-chirp(-0.4m)`：`|Δθ| ~ 0°`

> 詳細數值請見 `summary.json`。

## 解讀

在 chirp 條件下，`OMP_LDV`（GCC-PHAT）可以把 DoA 拉回到非常接近 mic truth-ref；
因此 Stage 4-B 可作為後續 Stage 4-C 的 reference 來源（用 chirp 的 mic-mic τ 當 truth）。

## Artifacts（本 commit 內）

- Chirp inputs（可直接復現）：
  - `_old_reports/23-chirp(-0.8m)/*.wav`
  - `_old_reports/24-chirp(-0.4m)/*.wav`
- Stage 4-B outputs：
  - `exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation_chirp_prealign_scan_psr/**/summary.json`
- 另外保留的 ablation outputs（同為 chirp；非必要但便於回溯）：
  - `exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation_chirp/**/summary.json`
  - `exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation_chirp_prealign/**/summary.json`

## Lineage

- Parent Exp: `LDVPG-20260201-S4A-speech-geometry-truth`

