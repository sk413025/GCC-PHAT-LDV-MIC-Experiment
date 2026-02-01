# LDVPG-20260201-S4A-speech-geometry-truth

## 背景 / 動機

Stage 4 的原始設計是用幾何位置計算 DoA 真值（θ_true），再用多種方法（GCC-PHAT/CC/NCC/MUSIC）去看：
`OMP_LDV` 是否能讓 DoA 更接近幾何真值。

但在語音資料上，`MicL–MicR` 的 TDoA 可能因為短窗、多源/混響造成 **peak 錯鎖**（例如鎖到穩定但非物理的 sidelobe），使得「幾何真值」驗證在實務上不成立。

## 實驗目的

以語音長檔（18–22）跑 Stage 4-A：
- truth：幾何計算的 `θ_true`
- 檢查 Stage 4 是否能穩定量到合理的 `MicL–MicR τ`（否則 DoA 無法作為驗證）

## 假設 / 預期

若 `MicL–MicR` 的 baseline 可靠，則：
- `OMP_LDV` 應該能讓 DoA 誤差下降（至少在 GCC-PHAT 上）

## 輸入資料

- Speech 18–22（未 commit；校驗見 `exp-validation/ldv-perfect-geometry/datasets/LDVPG_speech_18-22_root.sha256`）

## 參數（關鍵）

- `eval_window_sec=1.0`
- `gcc_bandpass_low/high = 500–2000 Hz`
- `segment_mode=fixed`，`segment_offset_sec=100`，`segment_spacing_sec=50`，共 `n_segments=5`

## 實驗指令（可復現）

```bash
for spk in 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V; do
  python exp-validation/ldv-perfect-geometry/scripts/stage4_doa_validation.py \
    --data_root . \
    --speaker "$spk" \
    --output_dir exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation \
    --segment_mode fixed \
    --segment_offset_sec 100 --segment_spacing_sec 50 --n_segments 5 \
    --analysis_slice_sec 5 --eval_window_sec 1 \
    --gcc_bandpass_low 500 --gcc_bandpass_high 2000
done
```

## 實際結果摘要（定性）

- 多數 speaker 的 `MicL–MicR` 在短窗語音下並不穩定，導致 DoA 與幾何真值偏差很大
- GCC-PHAT 僅在 speaker 20（θ_true≈0°）呈現「看似正確」的現象；其餘位置容易被 peak 錯鎖主導

> 完整 per-method/per-segment 數值請見 commit 內 `summary.json`。

## 解讀

Stage 4-A 在語音（1 秒窗、500–2000 Hz）下**不適合作為幾何真值驗證**；
後續 Stage 4 的真值來源應改成「可量測且穩定」的 reference（例如 chirp 的 mic-mic τ），或先掃窗挑出 mic-mic τ/PSR 合理的片段再驗證。

## Artifacts（本 commit 內）

- `exp-validation/ldv-perfect-geometry/scripts/stage4_doa_validation.py`
- `exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation/**/summary.json`

## Lineage

- Parent：Stage 3 完成後的 Stage 4-A 嘗試（參考 `STAGE_VALIDATION_PLAN.md`）
- Next：Stage 4-B/4-C 改用 chirp truth-ref（mic-mic τ）驗證

