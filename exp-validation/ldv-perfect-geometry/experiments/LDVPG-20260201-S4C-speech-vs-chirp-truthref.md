# LDVPG-20260201-S4C-speech-vs-chirp-truthref

## 背景 / 動機

Stage 4-A 在語音上不適合用幾何真值驗證；因此 Stage 4-C 改成：
- 用 Stage 4-B 的 chirp `MicL–MicR` 當 truth-ref
- 假設「相同擺位」的 speech（21/22）與 chirp（24/23）在理論上應有相同的 TDoA/DoA

對應關係：
- `21-0.1V`（x=-0.4）↔ `24-chirp(-0.4m)`
- `22-0.1V`（x=-0.8）↔ `23-chirp(-0.8m)`

## 實驗目的

在 speech 上，用 chirp truth-ref 驗證 `OMP_LDV` 是否能把 DoA 拉回到 reference。

## 輸入資料

- Speech 21/22（未 commit；校驗見 `exp-validation/ldv-perfect-geometry/datasets/LDVPG_speech_18-22_root.sha256`）
- Chirp truth-ref 來源（已 commit）：`_old_reports/23-chirp(-0.8m)`、`_old_reports/24-chirp(-0.4m)`

## 設計（兩個版本都要做）

### Version A：1 秒窗（不 guided）

- `eval_window_sec=1.0`
- `segment_mode=scan`（100–600s、hop=1.0s）挑出最接近 truth-ref 的窗
- GCC-PHAT 不帶通（`gcc_bandpass_low/high <= 0`）

Artifacts：`validation-results/stage4_doa_validation_speech_truthref_chirp_scan/`

### Version B：5 秒窗 + guided peak

- `eval_window_sec=5.0`（scan 與最終估計都用 5 秒）
- truth-ref 也用 5 秒窗重算（避免 1 秒 vs 5 秒不公平）
- `gcc_guided_peak_radius_ms=0.3`：GCC-PHAT 只在 `τ_ref ± 0.3ms` 內找峰，避免全域假峰
- 注意：guided peak 的 PSR 可能 < 0 dB（代表 guided peak 不是全域最大峰），scan 的 `scan_psr_min_db` 需放寬

Artifacts：`validation-results/stage4_doa_validation_speech_truthref_chirp_scan_guided_5s/`

## 實際結果摘要（重點）

- `21-0.1V`：在 5 秒窗下 `OMP_LDV` 幾乎貼住 truth-ref（誤差 ~0.06°）
- `22-0.1V`：即使用 guided + 5 秒窗，`OMP_LDV` 仍落在約 `-22°`；反而 raw 在該窗更接近 truth-ref（~1.2°）

## 解讀

- `21` 顯示：在更長窗 + 參考引導下，speech 的 `OMP_LDV` 可以維持與 chirp truth-ref 一致
- `22` 顯示：speech 的峰型/干擾仍會讓 `OMP_LDV` 鎖到另一個穩定峰；需要針對該 speaker 的窗選/頻帶/對齊策略再設計（例如更嚴格的掃窗條件、或把 guided 半徑/代價函數改成 joint objective）

## Artifacts（本 commit 內）

- `exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation_speech_truthref_chirp_scan/**/summary.json`
- `exp-validation/ldv-perfect-geometry/validation-results/stage4_doa_validation_speech_truthref_chirp_scan_guided_5s/**/summary.json`

## Lineage

- Parent Exp: `LDVPG-20260201-S4B-chirp-truthref`

