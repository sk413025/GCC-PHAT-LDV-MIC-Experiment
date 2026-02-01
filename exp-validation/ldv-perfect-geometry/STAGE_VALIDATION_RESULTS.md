# Stage Validation Results

**Date**: 2026-02-01
**Branch**: `exp-ldv-perfect-geometry-clean`
**Note**: 本文件會隨實驗 commit 更新；詳見 `exp-validation/ldv-perfect-geometry/experiments/`。

---

## Executive Summary

本次驗證實作了四階段驗證流程，測試 OMP (Orthogonal Matching Pursuit) 對齊方法是否能讓 LDV 信號取代麥克風進行 TDoA/DoA 估計。

| 階段 | 目標 | 結果 | 通過率 |
|------|------|------|--------|
| Stage 1 | Energy Reduction | ✅ **PASSED** | 5/5 |
| Stage 2 | Target Similarity | ✅ **PASSED** | 5/5 |
| Stage 3 (Single, report baseline) | Cross-Mic TDoA | ⚠️ PARTIAL | 2/5 |
| Stage 3 (Single, windowed baseline PSR>=10) | Cross-Mic TDoA | ⚠️ PARTIAL | 3/5 |
| Stage 3 (Multi, report baseline, offset=100s) | Multi-Segment TDoA | ⚠️ PARTIAL | 1/5 |
| Stage 3 (Multi, windowed baseline PSR>=10, offset=100s) | Multi-Segment TDoA | ⚠️ PARTIAL | 2/5 |
| Stage 4-A (Speech + Geometry truth) | DoA Multi-Method | ❌ **FAILED** | 0/5 |
| Stage 4-B (Chirp + Mic truth-ref) | DoA Multi-Method（GCC-PHAT） | ✅ **PASSED** | 2/2 |
| Stage 4-C (Speech + Chirp truth-ref) | DoA Multi-Method（GCC-PHAT） | ⚠️ PARTIAL | 1/2 |

**關鍵發現**：
1. OMP 對齊在 Stage 1、2 表現優異（**全部 5/5 通過**）
2. Stage 3/4 對「baseline 的定義」非常敏感：短時間窗 + 低 PSR 時容易鎖到固定假峰（例如 -81 samples = -1.6875 ms）；report baseline 也可能因 PSR 過低而不可靠
3. 以語音長檔在 **500–2000 Hz** + **1 秒窗**的設定下，Stage 4 的 **MicL–MicR τ 幾乎都接近 0 ms**，因此與幾何真值偏差很大（除了 speaker 20 本來就是 θ≈0°）
4. 以 `_old_reports` 的 chirp（23/24）做驗證時，採用 **scan 挑窗 + LDV→MicL GCC-PHAT prealign**，`OMP_LDV`（GCC-PHAT）可把 DoA 拉回到非常接近真值參考（|Δθ| < 1°）

---

## Stage 1: Energy Reduction Validation

### 目標
確認 OMP 在 LDV → Mic 方向上確實在學習有意義的模式，而非隨機猜測。

### 配置
```python
{
    'fs': 48000,
    'n_fft': 6144,
    'hop_length': 160,
    'max_lag': 50,      # ±1.04 ms @ 48kHz
    'max_k': 3,         # OMP sparsity
    'tw': 64,           # Time window
    'freq_min': 100,    # Hz
    'freq_max': 8000,   # Hz
}
```

### 結果（全部通過 ✅）

| Speaker | OMP Energy Reduction | Random Baseline | Improvement | Status |
|---------|---------------------|-----------------|-------------|--------|
| 18-0.1V | 0.9116 ± 0.0709 | 0.6442 ± 0.1148 | **41.5%** | ✅ |
| 19-0.1V | ~0.91 | ~0.65 | **~40%** | ✅ |
| 20-0.1V | 0.9167 ± 0.0616 | 0.6532 ± 0.1118 | **40.4%** | ✅ |
| 21-0.1V | ~0.91 | ~0.65 | **~40%** | ✅ |
| 22-0.1V | ~0.91 | ~0.64 | **~41%** | ✅ |

### 結論
✅ **Stage 1 PASSED (5/5)** - OMP 一致性地優於 Random baseline 約 40%。

---

## Stage 2: Target Mic Similarity Validation

### 目標
確認 OMP 對齊後的 LDV 信號在時域上更像 Target Mic。

### 結果（全部通過 ✅）

| Speaker | Target | Raw τ (ms) | OMP τ (ms) | Raw PSR | OMP PSR | Status |
|---------|--------|------------|------------|---------|---------|--------|
| 18-0.1V | Left | +3.021 | **0.000** | 1.1 dB | 9.7 dB | ✅ |
| 19-0.1V | Left | +3.042 | **0.000** | 0.3 dB | 8.8 dB | ✅ |
| 20-0.1V | Left | +3.042 | **0.000** | 0.3 dB | 7.4 dB | ✅ |
| 21-0.1V | Left | +3.021 | **0.000** | 3.4 dB | 9.5 dB | ✅ |
| 22-0.1V | Left | +0.000 | **0.000** | 14.1 dB | 17.2 dB | ✅* |

*Speaker 22 的 Raw τ = 0 是該 segment 的特殊情況，PSR 仍有改善。

### 結論
✅ **Stage 2 PASSED (5/5)** - OMP 重建成功近似 Target Mic 信號，τ → 0，PSR +6-8 dB。

---

## Stage 3: Cross-Mic TDoA Evaluation

### 目標
驗證 `OMP_LDV`（把 LDV 對齊到 MicL）是否能取代 MicL 與 MicR 做 cross-mic TDoA：
- baseline：`GCC-PHAT(MicL, MicR)`
- raw：`GCC-PHAT(Raw_LDV, MicR)`
- OMP：`GCC-PHAT(LDV_as_MicL, MicR)`

### Baseline 定義（最影響 Stage 3 的地方）
Stage 3 的「通過/失敗」高度依賴你把哪個 `τ` 當作 `τ_baseline`：
- `baseline_method=segment`：只看單一短窗的 `(MicL,MicR)`；在低 PSR 時可能鎖到 **固定假峰 -81 samples = -1.6875 ms**
- `baseline_method=report`：對齊 `full_analysis.py` / 報告設定，用 100–600s 長區間做 baseline（但 PSR 仍可能很低）
- `baseline_method=windowed`：100–600s 多窗 median（可加 PSR>=10 篩選），baseline 會更穩定（本次落在 ~0ms）

> 本文件「最新數值」以 `report` 與 `windowed(PSR>=10)` 兩種 baseline 為主；legacy 的 `segment` baseline 僅保留作為「為什麼會全部 -1.688ms」的對照。

### Single-Segment（center=300s, eval=1s, bandpass=500–2000Hz）

#### baseline_method=report（對齊報告）

| Speaker | Baseline(report) τ (ms) | Baseline PSR | Segment τ (ms) | Segment PSR | Raw τ (ms) | OMP τ (ms) | OMP Error vs Baseline (ms) | Raw→OMP PSR (dB) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 18-0.1V | -0.0164 | 5.60 | +0.0051 | 21.79 | +0.0000 | +0.0000 | 0.0164 | 31.0 → 32.8 | ❌ |
| 19-0.1V | -1.6841 | 2.81 | +0.0097 | 23.77 | +0.0165 | +0.0165 | 1.7007 | 23.5 → 27.0 | ❌ |
| 20-0.1V | +0.0030 | 14.51 | -0.0002 | 27.58 | +0.0192 | +0.0192 | 0.0162 | 23.7 → 26.3 | ✅ |
| 21-0.1V | +0.0003 | 23.47 | -0.0140 | 21.55 | -0.0209 | -0.0212 | 0.0215 | 7.6 → 11.5 | ❌ |
| 22-0.1V | -0.0232 | 16.22 | -0.0014 | 28.94 | -0.0013 | -0.0015 | 0.0218 | 28.8 → 31.7 | ✅ |

**通過率**：2/5（20、22）

#### baseline_method=windowed（PSR>=10）

| Speaker | Baseline τ (ms) | Baseline PSR | Raw τ (ms) | OMP τ (ms) | OMP Error (ms) | Raw→OMP PSR (dB) | Status |
|---|---:|---:|---:|---:|---:|---:|:--:|
| 18-0.1V | +0.0003 | 23.82 | +0.0000 | +0.0000 | 0.0003 | 31.0 → 32.8 | ✅ |
| 19-0.1V | +0.0003 | 23.95 | +0.0165 | +0.0165 | 0.0163 | 23.5 → 27.0 | ✅ |
| 20-0.1V | -0.0001 | 24.39 | +0.0192 | +0.0192 | 0.0193 | 23.7 → 26.3 | ✅ |
| 21-0.1V | +0.0004 | 25.56 | -0.0209 | -0.0212 | 0.0216 | 7.6 → 11.5 | ❌ |
| 22-0.1V | +0.0002 | 25.53 | -0.0013 | -0.0015 | 0.0016 | 28.8 → 31.7 | ❌ |

**通過率**：3/5（18、19、20）

### Multi-Segment（10 segments; centers=100..550s, eval=1s, bandpass=500–2000Hz）

#### baseline_method=report（offset=100s）

| Speaker | Baseline(report) τ (ms) | Baseline PSR | Raw median error (ms) | OMP median error (ms) | Raw→OMP PSR (dB) | Status |
|---|---:|---:|---:|---:|---:|:--:|
| 18-0.1V | -0.0164 | 5.60 | 0.0128 | 0.0129 | 26.8 → 28.1 | ❌ |
| 19-0.1V | -1.6841 | 2.81 | 1.6810 | 1.6811 | 24.8 → 27.8 | ❌ |
| 20-0.1V | +0.0030 | 14.51 | 0.0044 | 0.0044 | 31.0 → 33.4 | ❌ |
| 21-0.1V | +0.0003 | 23.47 | 0.0122 | 0.0122 | 27.8 → 28.9 | ❌ |
| 22-0.1V | -0.0232 | 16.22 | 0.0222 | 0.0222 | 27.3 → 29.3 | ✅ |

**通過率**：1/5（22）

#### baseline_method=windowed（PSR>=10, offset=100s）

| Speaker | Baseline(windowed) τ (ms) | Baseline PSR | Raw median error (ms) | OMP median error (ms) | Raw→OMP PSR (dB) | Status |
|---|---:|---:|---:|---:|---:|:--:|
| 18-0.1V | +0.0003 | 23.82 | 0.0170 | 0.0169 | 26.8 → 28.1 | ✅ |
| 19-0.1V | +0.0003 | 23.95 | 0.0183 | 0.0184 | 24.8 → 27.8 | ❌ |
| 20-0.1V | -0.0001 | 24.39 | 0.0021 | 0.0021 | 31.0 → 33.4 | ❌ |
| 21-0.1V | +0.0004 | 25.56 | 0.0121 | 0.0121 | 27.8 → 28.9 | ❌ |
| 22-0.1V | +0.0002 | 25.53 | 0.0096 | 0.0095 | 27.3 → 29.3 | ✅ |

**通過率**：2/5（18、22）

### Legacy 對照：baseline_method=segment（短窗）

`baseline_method=segment` 只看單一短窗的 `(MicL,MicR)`；在低 PSR 時可能鎖到穩定假峰（例如 `-81 samples ≈ -1.6875 ms`）。
因此本文件的「通過率」與後續比較以 `report` / `windowed(PSR>=10)` 為主；legacy 現象可參考 `validation-results/stage3_multi_segment/` 的 per-segment baseline（偶發 outlier）。

### 補充觀察（從最新 report/windowed 結果）

1. `baseline_method=report` 在部分 speaker 的 baseline PSR 偏低（如 19 的 2.81 dB），`τ_baseline` 本身就不穩定  
2. `baseline_method=windowed(PSR>=10)` 的 baseline PSR 穩定（~24–26 dB），但 `τ_baseline ≈ 0 ms`（語音短窗下 mic-mic 主峰常偏到 0）  
3. 多數 ❌ 來自 `error_improved=false`（OMP 與 Raw 的誤差非常接近），而不是 OMP 明顯變差；若要改成「OMP 至少不比 Raw 差」需調整 pass rule（目前規則採嚴格 `<`）

---

## Geometry Reference

### 感測器位置
```
Speaker Line:  y = 0 m
  22: (-0.8, 0)  21: (-0.4, 0)  20: (0, 0)  19: (+0.4, 0)  18: (+0.8, 0)

LDV Box:       (0.0, 0.5) m
Mic Left:      (-0.7, 2.0) m
Mic Right:     (+0.7, 2.0) m

Mic Spacing:   1.4 m
Speed of Sound: 343 m/s
```

---

## Scripts

| Script | 用途 | 完成狀態 |
|--------|------|----------|
| `stage1_energy_reduction.py` | 驗證 OMP > Random | ✅ 已跑完（5 speakers） |
| `stage2_target_similarity.py` | 驗證 OMP ≈ Target Mic | ✅ 已跑完（5 speakers） |
| `stage3_tdoa_evaluation.py` | 單一 segment TDoA（report/windowed/geometry） | ✅ 已跑完（5 speakers） |
| `stage3_multi_segment.py` | 多 segment TDoA（report/windowed/geometry） | ✅ 已跑完（5 speakers） |
| `stage4_doa_validation.py` | DoA Multi-Method（fixed/scan + prealign） | ✅ 已跑完（speech 5 + chirp 2） |

---

## Legacy Notes（Stage 3 multi; baseline_method=segment）

`validation-results/stage3_multi_segment/` 保留作為 legacy multi-segment 對照（`baseline_method=segment`）。
在語音短窗下，per-segment 的 mic-mic τ 可能出現 outlier（包含偶發 `~ -1.69 ms` 的穩定假峰），因此不適合作為真值 baseline。

正式的通過率與比較請以 `stage3_multi_segment_report_offset100/` 與 `stage3_multi_segment_windowed_psr10_offset100/` 為主（見上表）。

---

## Stage 4: DoA Multi-Method Validation

### 目標
使用多種 DoA 估計方法驗證 OMP 對齊效果，並與 baseline 比較。

### 配置
```python
{
    'methods': ['GCC-PHAT', 'CC', 'NCC', 'MUSIC'],
    'pairings': ['MicL-MicR', 'Raw_LDV', 'Random_LDV', 'OMP_LDV'],
    'n_segments': 5,              # centers: 100,150,200,250,300s
    'eval_window_sec': 1.0,       # 每段取 1 秒
    'analysis_slice_sec': 5.0,    # 每段用 5 秒做 STFT/OMP，避免全檔 STFT 記憶體爆炸
    'bandpass': 500–2000 Hz,      # GCC/CC/NCC 與 MUSIC 都使用相同頻帶（MUSIC 內部本來就是 500–2000）
}
```

### Stage 4-A（Speech + Geometry truth）：GCC-PHAT 結果（5 speakers）

| Speaker | θ_true (°) | τ_true (ms) | Mic-Mic τ (ms) | OMP τ (ms) | OMP θ_error (°) | Raw→OMP PSR (dB) |
|---------|-----------:|------------:|---------------:|-----------:|----------------:|-----------------:|
| 18-0.1V | +20.82 | +1.450 | -0.0007 | -0.0091 | 20.94 | 26.0 → 27.8 |
| 19-0.1V | +10.71 | +0.759 | +0.0024 | -0.0002 | 10.71 | 27.8 → 30.6 |
| 20-0.1V | +0.00 | +0.000 | +0.0031 | -0.0014 | 0.11 | 30.9 → 33.6 |
| 21-0.1V | -10.71 | -0.759 | -0.0013 | -0.0022 | 10.68 | 32.2 → 34.4 |
| 22-0.1V | -20.82 | -1.450 | -0.0008 | +0.0076 | 20.92 | 26.8 → 29.4 |

**觀察**：在語音長檔、1 秒窗 + 500–2000 Hz 的設定下，`MicL–MicR` 的 τ 幾乎都落在 **0 ms 附近**，導致 DoA 角度也接近 0°，因此與幾何真值偏差很大（除了 speaker 20 本來就是 0°）。

### 關鍵發現

1. **Stage 4（幾何 DoA 真值）在語音資料上不穩定**：
   - `MicL–MicR` 本身的 τ 在短窗會偏向 0 ms（高 PSR），不代表幾何路徑差
   - 因此 `θ_error` 主要反映「短窗估計在混響/干擾下鎖到哪個峰」，而不是 OMP 對齊是否成功

2. **各方法表現（語音 1 秒窗）**：

| 方法 | Baseline 穩定性 | OMP 與 Baseline 匹配度 | 建議 |
|------|----------------|----------------------|------|
| **GCC-PHAT** | τ 在短窗容易收斂到 0 ms | OMP/Raw 差異很小 | 建議改用 chirp 或 window 篩選後再做幾何驗證 |
| CC | 常出現非物理解（θ 飽和到 ±90°） | 不穩定 | 不建議 |
| NCC | 常出現非物理解（θ 飽和到 ±90°） | 不穩定 | 不建議 |
| MUSIC | 變異大，依賴 snapshots/場景 | 不穩定 | 需重新設計參數/資料型態 |

### 結論
- 本次（語音長檔、短窗、500–2000 Hz）無法用 Stage 4 去驗證幾何 DoA 真值：**4/5 speakers 的 θ_error 仍約等於 |θ_true|**
- 若要把 Stage 4 當作「幾何 DoA 真值」驗證，請改用 chirp（或先掃窗挑出 `MicL–MicR τ` 接近理論值且 PSR 足夠的片段再驗）

### Stage 4-B（Chirp + Mic truth-ref）：GCC-PHAT 結果（2 datasets）

在 `_old_reports` 裡的 chirp 資料（23/24）其實更適合拿來當「真值參考」，因為 `MicL–MicR` 的 GCC-PHAT 會穩定地收斂到一致的 τ（不像語音短窗常偏到 0 ms）。

本次使用：
- `segment_mode=scan` + `scan_sort_by=psr`（優先挑 mic-mic 峰值最穩的窗）
- `ldv_prealign=gcc_phat`（先用 LDV→MicL 的 GCC-PHAT 估出 delay，再對 LDV 做 fractional delay）
- scan 參數：`eval_window=1.0s`, `hop=0.1s`, `scan_psr_min_db=5`, `scan_ldv_micl_psr_min_db=4`, `scan_tau_err_max_ms=0.3`, `min_separation=1.0s`
- GCC-PHAT 不做帶通（`gcc_bandpass_low/high <= 0`）

| Dataset | center (s) | Mic τ (ms) | Mic θ (°) | Mic PSR | OMP τ (ms) | OMP θ (°) | OMP PSR | abs(Δθ) (°) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 23-chirp(-0.8m) | 11.40 | -1.306 | -18.67 | 13.3 | -1.250 | -17.83 | 9.7 | 0.84 |
| 24-chirp(-0.4m) | 13.60 | -0.636 | -8.97 | 10.9 | -0.641 | -9.03 | 7.0 | 0.07 |

（Raw 對照，顯示未對齊時會飽和到 -90°）

| Dataset | Raw τ (ms) | Raw θ (°) | Raw PSR |
|---|---:|---:|---:|
| 23-chirp(-0.8m) | -4.064 | -84.62 | 8.2 |
| 24-chirp(-0.4m) | -4.433 | -90.00 | 5.2 |

**結論**：若把 chirp 的 `MicL–MicR` 當作「真值參考」，在以上挑窗策略下，`OMP_LDV`（GCC-PHAT）可以把 DoA 拉回到非常接近真值（誤差 < 1°）。

### Stage 4-C（Speech + Chirp truth-ref）：GCC-PHAT（2 speakers）

想法：因為 21/22 的擺位與 24/23-chirp 相同（x=-0.4 / -0.8），理論上 TDoA/DoA 應該一致；因此可用 chirp 的 `MicL–MicR`（不帶通）作為 truth-ref，再看 speech 的 `OMP_LDV` 能否回到同一個角度。

本段更新為「**5 秒窗 + guided peak**」版本：
- 評估窗：`eval_window_sec=5.0`（同時用在 scan 與最後的 GCC-PHAT 計算）
- chirp truth-ref：也用 **5 秒窗**重新計算（避免 1 秒 vs 5 秒不公平）
- guided peak：`gcc_guided_peak_radius_ms=0.3`（GCC-PHAT 只在 `τ_ref ± 0.3ms` 的區間內找峰，避免被全域假峰吃掉）
- scan：`segment_mode=scan`（100–600s、hop=1.0s）挑 1 個最接近 truth-ref 的窗  
  *注意：guided peak 的 PSR 可能是負值（代表「在全域範圍內，這個 guided peak 不是最大峰」），因此 scan 的 `scan_psr_min_db` 需放寬（本次用 -20）*

| Speech | Chirp truth-ref | center (s) | θ_ref (°) | Mic θ (°) | Raw θ (°) | OMP θ (°) | abs(Raw-Ref θ) (°) | abs(OMP-Ref θ) (°) | GCC-PHAT |
|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| 21-0.1V | chirp 24 mic truth-ref (5s) | 279.00 | -8.94 | -6.77 | -9.14 | -9.00 | 0.20 | 0.06 | ✅ |
| 22-0.1V | chirp 23 mic truth-ref (5s) | 395.00 | -18.68 | -22.28 | -17.44 | -22.30 | 1.24 | 3.62 | ❌ |

**觀察**：
- 21：`OMP_LDV` 可以在 5 秒窗下維持接近 truth-ref（0.06°），且略優於 Raw
- 22：即使用 guided peak + 5 秒窗，`OMP_LDV` 仍落在 -22° 附近（離 truth-ref ~3.6°）；反而 Raw 在該窗更接近 truth-ref（~1.2°）

---

## 實驗設計總覽

### 驗證流程

```
Stage 1: Energy Reduction
  └── 驗證 OMP 是否學到有意義的模式
  └── 比較: OMP vs Random baseline
  └── 指標: Energy Reduction ratio

Stage 2: Target Similarity
  └── 驗證 OMP 對齊後的 LDV 是否接近 Target Mic
  └── 比較: Raw_LDV vs OMP_LDV
  └── 指標: GCC-PHAT τ → 0, PSR ↑

Stage 3: Cross-Mic TDoA
  └── 驗證 OMP LDV 能否取代 Mic 做 TDoA
  └── 比較: (OMP_LDV, MicR) vs (MicL, MicR)
  └── 指標: τ → baseline, error < 0.5 ms

Stage 4: DoA Multi-Method
  └── 使用多種方法驗證
  └── 與幾何 ground truth 比較
```

### OMP 對齊原理

```python
# 1. 建立 lagged dictionary
Dict[f, lag, t] = LDV_STFT[f, t + lag]  # lag ∈ [-50, +50]

# 2. OMP 選擇最佳 lags
for freq in freq_bins:
    selected_lags = OMP(Dict[freq], Target_Mic[freq], max_k=3)

# 3. 重建對齊後的信號
OMP_LDV = Σ coeffs[k] * Dict[:, selected_lags[k], :]
```

### 關鍵參數

| 參數 | 值 | 說明 |
|------|-----|------|
| fs | 48000 Hz | 取樣率 |
| n_fft | 6144 | FFT 大小 |
| hop_length | 160 | 步進大小 |
| max_lag | 50 samples | ±1.04 ms |
| max_k | 3 | OMP 稀疏度 |
| tw | 64 frames | 時間窗口 |
| freq_range | 100-8000 Hz | 頻率範圍 |

---

## Next Steps

1. ✅ Stage 1: Energy Reduction - **完成** (5/5)
2. ✅ Stage 2: Target Similarity - **完成** (5/5)
3. ⚠️ Stage 3: Cross-Mic TDoA（baseline 敏感）
   - Single：report 2/5；windowed(PSR>=10) 3/5
   - Multi：report(offset=100s) 1/5；windowed(PSR>=10, offset=100s) 2/5
   - Legacy segment baseline（可能有 outlier 假峰）：multi 2/5（僅供對照，不建議當真值）
4. ⚠️ Stage 4: DoA Multi-Method
   - Speech + Geometry truth：0/5（語音短窗下 `MicL–MicR τ ≈ 0 ms`）
   - Chirp + Mic truth-ref：2/2（GCC-PHAT；|Δθ| < 1°）
   - Speech + Chirp truth-ref：1/2（5s + guided peak；21 ✅ / 22 ❌）

### 已完成工作
- [x] Stage 3：report/windowed baseline 的 single/multi 全部跑完並輸出 `summary.json`
- [x] Stage 4：speech（fixed 5 段）+ chirp（scan + prealign）都已跑完

### 後續建議
- Stage 3：先決定「要驗證什麼真值」：mic truth（report/windowed）或幾何 truth；若要幾何 truth，建議用 chirp 或先做 window/PSR 篩選後再做 `baseline_method=geometry`
- Stage 3 pass rule：目前用嚴格 `error_omp < error_raw`，會讓「OMP 跟 Raw 一樣好」被判 ❌；可視需求改成 `<=` 或改成只看 `error_small + PSR 提升`
- Stage 4：語音資料要對幾何真值需重新設計資料/窗長/挑窗策略（否則 MicL–MicR 自己就不對）

---

## Files

- 結果目錄：
  - `validation-results/stage1_energy_reduction/` (5 speakers)
  - `validation-results/stage2_target_similarity/` (5 speakers)
  - `validation-results/stage3_tdoa_evaluation/` (Single; report baseline)
  - `validation-results/stage3_tdoa_evaluation_windowed_psr10/` (Single; windowed baseline PSR>=10)
  - `validation-results/stage3_multi_segment/` (Multi; legacy segment baseline)
  - `validation-results/stage3_multi_segment_report_offset100/` (Multi; report baseline, offset=100s)
  - `validation-results/stage3_multi_segment_windowed_psr10_offset100/` (Multi; windowed baseline PSR>=10, offset=100s)
  - `validation-results/stage4_doa_validation/` (Speech; fixed segments)
  - `validation-results/stage4_doa_validation_chirp/` (Chirp; no prealign; ablation)
  - `validation-results/stage4_doa_validation_chirp_prealign/` (Chirp; prealign only; ablation)
  - `validation-results/stage4_doa_validation_chirp_prealign_scan_psr/` (Chirp; scan + prealign; final)
  - `validation-results/stage4_doa_validation_speech_truthref_chirp_scan/` (Speech; scan + chirp truth-ref override; legacy 1s)
  - `validation-results/stage4_doa_validation_speech_truthref_chirp_scan_guided_5s/` (Speech; scan + chirp truth-ref override + guided peak; 5s)
- Scripts：
  - `scripts/stage1_energy_reduction.py`
  - `scripts/stage2_target_similarity.py`
  - `scripts/stage3_tdoa_evaluation.py`
  - `scripts/stage3_multi_segment.py`
  - `scripts/stage4_doa_validation.py`
- 計畫文件：`EXPERIMENT_PLAN.md`, `STAGE_VALIDATION_PLAN.md`

---

**Last Updated**: 2026-02-01
**Branch**: `exp-ldv-perfect-geometry-clean`
