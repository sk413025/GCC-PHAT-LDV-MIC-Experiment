# Staged Validation Plan: LDV-to-Mic OMP Alignment

**Date**: 2026-01-30
**Branch**: `exp-ldv-perfect-geometry`
**Goal**: 階段性驗證 LDV→Mic OMP 對齊的有效性，再進行 TDoA 評估

---

## 問題背景

在 `exp-interspeech-GRU2` 分支成功驗證了 **Mic→LDV** 方向的 OMP 對齊（Energy Reduction 指標）。
現在要在 `exp-ldv-perfect-geometry` 實現 **LDV→Mic** 方向的對齊並用於 TDoA。

之前直接跳到 TDoA 評估，結果 OMP 無法改進 GCC-PHAT PSR。
需要**階段性驗證**，確認每一步都正確，再進行最終評估。

---

## 驗證階段總覽

| 階段 | 目標 | 指標 | 通過條件 |
|------|------|------|----------|
| 1 | OMP 對齊本身有效 | Energy Reduction | OMP > Random |
| 2 | 對齊後信號與 Target Mic 相似 | GCC-PHAT(LDV_OMP, Target_Mic) | τ→0, PSR↑ |
| 3 | 對齊後可取代 Mic 做 TDoA | GCC-PHAT(LDV_as_MicL, MicR) | τ→baseline |

---

## 階段 1: Energy Reduction 驗證

### 目的
確認 OMP 在 LDV→Mic 方向能有效學習對齊（不只是亂猜）

### 信號角色
```
Dictionary (X): LDV 的 lagged versions (多個延遲版本)
Target (Y):     Target Mic (左或右)
```

### 標準化方式
**逐頻 max-abs normalization**（每個頻段獨立標準化到 [-1, 1]）

原因：
- 讓每個頻段有相同的「起跑點」
- 避免高能量頻段主導 OMP 選擇
- 概念上類似 GCC-PHAT 的 PHAT weighting

實作：
```python
def normalize_per_freq(X_stft):
    """
    X_stft: shape (n_freq, n_time) or (n_freq, n_lags, n_time)
    """
    max_abs = np.abs(X_stft).max(axis=-1, keepdims=True)
    max_abs[max_abs == 0] = 1  # avoid division by zero
    return X_stft / max_abs
```

### 指標計算
```python
def energy_reduction(Y_target, Y_reconstructed):
    """
    Y_target: 原始 target 信號 (頻域)
    Y_reconstructed: OMP 重建的信號
    """
    residual = Y_target - Y_reconstructed
    E_target = np.sum(np.abs(Y_target)**2)
    E_residual = np.sum(np.abs(residual)**2)
    return (E_target - E_residual) / E_target
```

### 比較條件
| 方法 | 說明 |
|------|------|
| Random | 隨機選擇 lag，作為 baseline |
| OMP | Orthogonal Matching Pursuit 選擇最佳 lag |

### 預期結果
```
Energy Reduction:
  Random: 0.35 - 0.40 (隨機選擇的 baseline)
  OMP:    0.50 - 0.60 (應該明顯高於 Random)
  
差異: OMP - Random > 0.10 (至少 10% 改進)
```

### 通過條件
- [ ] OMP Energy Reduction > Random Energy Reduction
- [ ] 改進幅度 > 10%

### 失敗時的 Debug 方向
1. 檢查標準化是否正確（逐頻 vs 全頻）
2. 檢查 lag 範圍是否足夠（max_lag 參數）
3. 檢查頻段選擇（是否需要 bandpass）
4. 檢查 LDV 和 Mic 的取樣率是否一致

---

## 階段 2: Target Mic 相似度驗證

### 目的
確認對齊後的 `LDV_OMP` 在時域上確實更像 `Target_Mic`

### 信號角色
```
Signal A: LDV_OMP (OMP 對齊後，ISTFT 回時域)
Signal B: Target_Mic (原始麥克風信號)
```

### 指標
| 指標 | 說明 | 預期 |
|------|------|------|
| CC peak | Cross-correlation 最大值 | OMP > Raw |
| NCC peak | Normalized CC 最大值 | OMP > Raw |
| GCC-PHAT τ | 延遲估計 | OMP: τ→0 |
| GCC-PHAT PSR | Peak-to-sidelobe ratio | OMP > Raw |

### 比較條件
| 方法 | 說明 |
|------|------|
| Raw_LDV | 原始 LDV，未對齊 |
| Random_LDV | 隨機 lag 對齊 |
| OMP_LDV | OMP 對齊後 |

### 預期結果
```
GCC-PHAT(LDV_aligned, Target_Mic):
  Raw:    τ ≠ 0 (因為 LDV 和 Mic 位置不同)
  OMP:    τ → 0 (因為已對齊到同一位置)
  
GCC-PHAT PSR:
  Raw:    低 (因為未對齊)
  OMP:    高 (因為已對齊，相干性提高)
```

### 通過條件
- [ ] OMP 的 |τ| < Raw 的 |τ|
- [ ] OMP 的 PSR > Raw 的 PSR
- [ ] τ 接近 0（誤差 < 0.5 ms）

### 失敗時的 Debug 方向
1. 檢查 ISTFT 是否正確（相位重建）
2. 檢查 OMP 選擇的 lag 分布是否合理
3. 嘗試頻率相關的增益補償（不只是延遲）

---

## 階段 3: 跨麥克風 TDoA 評估

### 目的
確認對齊到 MicL 位置的 LDV 可以取代 MicL 與 MicR 做 TDoA

### 信號配對
| 配對 | 說明 | 預期 τ |
|------|------|--------|
| (MicL, MicR) | Baseline | τ_baseline (真實 TDoA) |
| (Raw_LDV, MicR) | LDV 在原位 | τ_raw (不同於 baseline) |
| (LDV_as_MicL, MicR) | LDV 對齊到 MicL 後 | τ → τ_baseline |

### 指標
| 指標 | 說明 |
|------|------|
| τ_median | 多 segment 的 τ 中位數 |
| τ_std | τ 的標準差（穩定性） |
| PSR_median | Peak-to-sidelobe ratio 中位數 |
| τ_error | |τ_OMP - τ_baseline| |

### 預期結果
```
(MicL, MicR):        τ = +1.69 ms, PSR = 44 dB (baseline)
(Raw_LDV, MicR):     τ = -1.33 ms, PSR = 8 dB
(LDV_as_MicL, MicR): τ → +1.69 ms, PSR > 8 dB

τ_error: < 0.3 ms (目標)
```

### 通過條件
- [ ] |τ_OMP - τ_baseline| < |τ_Raw - τ_baseline|
- [ ] PSR_OMP > PSR_Raw
- [ ] τ_error < 0.5 ms

### 失敗時的 Debug 方向
1. 回到階段 2 檢查對齊品質
2. 考慮傳感器模態差異（LDV vs Mic 的頻率響應）
3. 嘗試 frequency-dependent transfer function 補償

---

## 統一參數設定

| 參數 | 值 | 備註 |
|------|-----|------|
| fs | 48000 Hz | 維持原生取樣率 |
| n_fft | 6144 | 頻率解析度 ~7.8 Hz |
| hop_length | 160 | 時間解析度 ~3.3 ms |
| max_lag | 50 | ±50 samples @ 48kHz = ±1.04 ms |
| max_k | 16 | OMP 稀疏度 |
| tw | 32 | OMP 時間窗口 |
| normalization | per-freq max-abs | 逐頻標準化 |
| freq_range | full-band | 先不加 bandpass |
| segment_length | 50 s | 分段長度 |
| segment_overlap | 1 s | 分段重疊 |

---

## 實作檔案結構

```
exp-ldv-perfect-geometry/
├── scripts/
│   ├── stage1_energy_reduction.py    # 階段 1: Energy Reduction
│   ├── stage2_target_similarity.py   # 階段 2: Target 相似度
│   └── stage3_tdoa_evaluation.py     # 階段 3: TDoA 評估
├── results/
│   ├── stage1_energy_reduction/
│   ├── stage2_target_similarity/
│   └── stage3_tdoa_evaluation/
└── STAGED_VALIDATION_PLAN.md         # 本文件
```

---

## 執行順序

### Step 1: 執行階段 1
```bash
python scripts/stage1_energy_reduction.py \
    --data_root /path/to/data \
    --output_dir results/stage1_energy_reduction \
    --max_segments 5
```

檢查結果：
- `results/stage1_energy_reduction/summary.json`
- 確認 OMP > Random

### Step 2: 執行階段 2（僅當階段 1 通過）
```bash
python scripts/stage2_target_similarity.py \
    --data_root /path/to/data \
    --output_dir results/stage2_target_similarity \
    --max_segments 5
```

檢查結果：
- `results/stage2_target_similarity/summary.json`
- 確認 τ→0, PSR↑

### Step 3: 執行階段 3（僅當階段 2 通過）
```bash
python scripts/stage3_tdoa_evaluation.py \
    --data_root /path/to/data \
    --output_dir results/stage3_tdoa_evaluation \
    --max_segments 5
```

檢查結果：
- `results/stage3_tdoa_evaluation/summary.json`
- 確認 τ→baseline

---

## 預期時間表

| 階段 | 預計時間 | 備註 |
|------|----------|------|
| 階段 1 | 1-2 小時 | 單一 segment 可先快速驗證 |
| 階段 2 | 1-2 小時 | 依賴階段 1 結果 |
| 階段 3 | 2-4 小時 | 完整評估需要更多 segments |

---

## 詳細驗證流程與診斷指南

### 階段 1 詳細流程

**目標**：回答「OMP 真的在學東西還是亂猜？」

**執行步驟**：
1. 選擇單一 segment（建議 `20-0.1V` 中間 50 秒，speaker 在中央位置）
2. 載入 LDV 和 Target Mic 的 STFT
3. 建構 lagged dictionary（LDV 的多個延遲版本）
4. 執行 OMP：選擇最佳 lag 組合最小化重建誤差
5. 執行 Random baseline：隨機選擇相同數量的 lag
6. 計算兩者的 Energy Reduction 並比較

**診斷檢查點**：
```
□ lag 範圍檢查：max_lag=50 @ 48kHz = ±1.04 ms
  - 幾何預測的 LDV-to-MicL 延遲約 4.4 ms
  - 但我們不是要補償絕對延遲，而是學習頻率相關的相位校正
  - ±1.04 ms 應該足夠捕捉頻率 dispersion

□ 標準化檢查：確認是逐頻 max-abs
  - 錯誤：全頻標準化會讓高能量頻段主導
  - 正確：每個頻段獨立標準化到 [-1, 1]

□ 數值穩定性：避免除以零或極小值
  - max_abs[max_abs < 1e-10] = 1
```

**預期數值範圍**：
```
Energy Reduction (正常情況):
  Random: 0.30 - 0.45 (取決於 lag 數量和信號特性)
  OMP:    0.45 - 0.65 (應明顯高於 Random)

Energy Reduction (異常情況):
  兩者都 < 0.1  → 可能 lag 範圍不夠或信號問題
  兩者都 > 0.9  → 可能過擬合或數據洩漏
  OMP ≈ Random → OMP 沒學到有意義的模式
```

---

### 階段 2 詳細流程

**目標**：回答「對齊後的 LDV 確實變得像 Target Mic 嗎？」

**關鍵洞察**：
- Energy Reduction 高不代表「對齊正確」
- 需要直接驗證：對齊後的 LDV 與 Target Mic 的時間對齊是否改善

**執行步驟**：
1. 從階段 1 取得 OMP 選擇的 lag 和係數
2. 用這些 lag 重建 LDV_OMP（頻域）
3. ISTFT 轉回時域
4. 計算 GCC-PHAT(LDV_OMP, Target_Mic)
5. 比較 Raw_LDV 和 OMP_LDV 的 τ 和 PSR

**診斷檢查點**：
```
□ ISTFT 重建檢查：
  - 確認使用正確的 window 和 hop_length
  - 確認相位資訊保留（不是只用 magnitude）

□ OMP lag 分布檢查：
  - 視覺化 OMP 選擇的 lag 在各頻率的分布
  - 應該看到某種結構（不是完全隨機）
  - 如果所有頻率都選同一個 lag → 可能太簡單
  - 如果完全隨機 → OMP 可能沒學到物理意義

□ 頻段分析：
  - 低頻（< 500 Hz）：通常相干性較高
  - 中頻（500-2000 Hz）：語音主要能量
  - 高頻（> 2000 Hz）：可能有更多 dispersion
```

**預期數值範圍**：
```
GCC-PHAT τ (LDV vs Target_Mic):
  Raw_LDV:  τ ≈ -4.4 ms (幾何預測的 LDV-MicL 延遲)
  OMP_LDV:  τ → 0 ms (如果對齊成功)

GCC-PHAT PSR:
  Raw_LDV:  PSR ≈ 5-15 dB (未對齊)
  OMP_LDV:  PSR > Raw_LDV (相干性應提高)
```

---

### 階段 3 詳細流程

**目標**：回答「對齊後的 LDV 能用於實際 TDoA 任務嗎？」

**這是最終驗證**：如果階段 1、2 都通過但階段 3 失敗，表示有系統性問題。

**執行步驟**：
1. 計算 baseline：GCC-PHAT(MicL, MicR) → τ_baseline
2. 計算 raw：GCC-PHAT(Raw_LDV, MicR) → τ_raw
3. 計算 OMP：GCC-PHAT(LDV_as_MicL, MicR) → τ_omp
4. 比較 |τ_omp - τ_baseline| vs |τ_raw - τ_baseline|

**預期數值（以 speaker 20 為例）**：
```
幾何預測：
  MicL 到 speaker: 2.06 m
  MicR 到 speaker: 2.06 m
  τ_baseline ≈ 0 ms (對稱位置)

實際測量（來自之前的報告）：
  (MicL, MicR): τ ≈ +0.02 ms, PSR ≈ 44 dB
  (Raw_LDV, MicR): τ ≈ -1.33 ms, PSR ≈ 8 dB

目標：
  (LDV_as_MicL, MicR): τ → 0 ms, PSR > 8 dB
```

---

## 潛在風險與對策

### 風險 1：OMP 學到的不是物理延遲

**描述**：OMP 最小化重建誤差，但選擇的 lag 可能沒有物理意義。

**檢測方式**：
- 視覺化 OMP lag 分布
- 檢查是否與幾何預測的延遲相關

**對策**：
- 如果 lag 分布合理但 TDoA 失敗 → 可能需要增益補償
- 如果 lag 分布不合理 → 重新檢查 OMP 設定

### 風險 2：頻率相關的增益差異

**描述**：LDV 和 Mic 的頻率響應不同，純延遲校正可能不夠。

**檢測方式**：
- 比較 LDV 和 Mic 的頻譜包絡
- 檢查階段 2 的 PSR 是否有改善

**對策**：
- 考慮在 OMP 中加入 gain 參數
- 或在後處理中做頻譜正規化

### 風險 3：ISTFT 相位失真

**描述**：頻域操作後 ISTFT 可能產生 artifacts。

**檢測方式**：
- 聽對齊後的音訊
- 檢查重建信號的能量是否合理

**對策**：
- 使用 overlap-add 確保完美重建
- 確認 window 和 hop_length 設定正確

---

## Checklist

### 階段 1 Checklist
- [ ] 實作 `stage1_energy_reduction.py`
- [ ] 實作逐頻 max-abs normalization
- [ ] 執行 Random baseline
- [ ] 執行 OMP
- [ ] 比較 Energy Reduction
- [ ] 記錄結果到 `summary.json`

### 階段 2 Checklist
- [ ] 實作 `stage2_target_similarity.py`
- [ ] 計算 CC/NCC/GCC-PHAT
- [ ] 比較 Raw vs OMP
- [ ] 記錄 τ 和 PSR

### 階段 3 Checklist
- [ ] 實作 `stage3_tdoa_evaluation.py`
- [ ] 設定正確的配對 (LDV_as_MicL, MicR)
- [ ] 計算 τ_error
- [ ] 與 baseline 比較

---

## 附錄：關鍵程式碼片段

### 逐頻 Max-Abs Normalization
```python
def normalize_per_freq_maxabs(X_stft):
    """
    對 STFT 做逐頻率的 max-abs normalization
    
    Args:
        X_stft: complex array, shape (n_freq, n_time)
    
    Returns:
        X_norm: normalized array, same shape
        scale: normalization scale per frequency, shape (n_freq,)
    """
    # 計算每個頻率的最大絕對值
    max_abs = np.abs(X_stft).max(axis=-1)  # shape (n_freq,)
    max_abs[max_abs == 0] = 1  # 避免除以零
    
    # 標準化
    X_norm = X_stft / max_abs[:, np.newaxis]
    
    return X_norm, max_abs
```

### Energy Reduction 計算
```python
def compute_energy_reduction(Y_target, Y_reconstructed):
    """
    計算 OMP 的 Energy Reduction
    
    Args:
        Y_target: target signal (freq domain), shape (n_freq, n_time)
        Y_reconstructed: OMP reconstructed signal, same shape
    
    Returns:
        reduction: scalar, energy reduction ratio
    """
    residual = Y_target - Y_reconstructed
    E_target = np.sum(np.abs(Y_target)**2)
    E_residual = np.sum(np.abs(residual)**2)
    
    if E_target == 0:
        return 0.0
    
    return (E_target - E_residual) / E_target
```

### OMP 對齊（單一頻率）
```python
def omp_align_single_freq(X_dict, Y_target, max_k):
    """
    對單一頻率做 OMP 對齊
    
    Args:
        X_dict: dictionary matrix, shape (n_lags, n_time)
        Y_target: target vector, shape (n_time,)
        max_k: maximum sparsity
    
    Returns:
        selected_lags: list of selected lag indices
        coefficients: corresponding coefficients
        reconstructed: reconstructed signal
    """
    from sklearn.linear_model import OrthogonalMatchingPursuit
    
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=max_k)
    omp.fit(X_dict.T, Y_target)
    
    selected_lags = np.where(omp.coef_ != 0)[0]
    coefficients = omp.coef_[selected_lags]
    reconstructed = omp.predict(X_dict.T)
    
    return selected_lags, coefficients, reconstructed
```

---

## 階段 4: DoA 多方法驗證（新增）

### 目的
使用幾何真值（ground truth）驗證多種 DoA 估計方法，全面評估 OMP 對齊的效果。

### 信號配對（4 組）

| 配對 | 說明 | 用途 |
|------|------|------|
| **(MicL, MicR)** | 真實麥克風對 | Baseline |
| **(Raw_LDV, MicR)** | 原始 LDV | 未對齊參考 |
| **(Random_LDV, MicR)** | 隨機對齊 LDV | Random baseline |
| **(OMP_LDV, MicR)** | OMP 對齊 LDV | 測試目標 |

### DoA 估計方法（4 種）

| 方法 | 說明 | 特點 |
|------|------|------|
| **GCC-PHAT** | Phase Transform 加權 | 對混響魯棒 |
| **CC** | 標準互相關 | 對能量敏感 |
| **NCC** | 標準化互相關 | 對增益變化魯棒 |
| **MUSIC** | 子空間方法 | 高解析度，需要多快照 |

### 幾何真值計算

```python
# 感測器位置
mic_left = (-0.7, 2.0)   # m
mic_right = (0.7, 2.0)   # m
mic_spacing = 1.4        # m
c = 343                  # m/s

# Speaker 位置 (y=0)
speakers = {
    '18': (0.8, 0.0),   # 右側
    '19': (0.4, 0.0),
    '20': (0.0, 0.0),   # 中央
    '21': (-0.4, 0.0),
    '22': (-0.8, 0.0),  # 左側
}

# 真實 DoA 角度計算
def compute_true_doa(speaker_pos, mic_left, mic_right):
    """
    計算真實 DoA 角度（相對於麥克風陣列法線）
    """
    d_left = distance(speaker_pos, mic_left)
    d_right = distance(speaker_pos, mic_right)
    tau_true = (d_left - d_right) / c

    # DoA 角度：sin(θ) = τ * c / d
    sin_theta = tau_true * c / mic_spacing
    theta = np.arcsin(np.clip(sin_theta, -1, 1))
    return np.degrees(theta), tau_true
```

### 預期真實角度

| Speaker | Position | τ_true (ms) | θ_true (°) |
|---------|----------|-------------|------------|
| 18 | (+0.8, 0) | -0.69 | -9.9 |
| 19 | (+0.4, 0) | -0.38 | -5.5 |
| 20 | (0.0, 0) | 0.00 | 0.0 |
| 21 | (-0.4, 0) | +0.38 | +5.5 |
| 22 | (-0.8, 0) | +0.69 | +9.9 |

### 評估指標

| 指標 | 說明 | 單位 |
|------|------|------|
| **τ** | TDoA 估計 | ms |
| **θ** | DoA 角度估計 | 度 |
| **θ_error** | 角度誤差 `|θ_est - θ_true|` | 度 |
| **PSR** | Peak-to-Sidelobe Ratio | dB |

### 輸出格式

```
Speaker: 20-0.1V | θ_true = 0.0°

| Method   | Pairing      | τ (ms) | θ (°) | θ_error | PSR (dB) |
|----------|--------------|--------|-------|---------|----------|
| GCC-PHAT | MicL-MicR    | +0.02  | +0.3  |   0.3   |   44.2   |
| GCC-PHAT | Raw_LDV      | -1.33  | -19.1 |  19.1   |    8.3   |
| GCC-PHAT | Random_LDV   | -0.85  | -12.2 |  12.2   |    5.1   |
| GCC-PHAT | OMP_LDV      | +0.02  | +0.3  |   0.3   |   12.5   |
| CC       | MicL-MicR    | ...    | ...   |   ...   |   ...    |
| ...      | ...          | ...    | ...   |   ...   |   ...    |
```

### 通過條件

- [ ] OMP 的 θ_error < Raw 的 θ_error（所有方法）
- [ ] OMP 的 θ_error < Random 的 θ_error（所有方法）
- [ ] OMP 的 θ_error < 5°（實用門檻）
- [ ] OMP 的 PSR > Raw 的 PSR

### 實作檔案

```
scripts/
└── stage4_doa_validation.py    # 階段 4: 多方法 DoA 驗證
```

### 執行命令

```bash
python scripts/stage4_doa_validation.py \
    --data_root /path/to/data \
    --speaker 20-0.1V \
    --output_dir results/stage4_doa_validation \
    --n_segments 10
```

---

## 階段 3 結果總結（2026-01-30 更新）

### Multi-Segment TDoA 結果

| Speaker | Raw Error | OMP Error | 改善 | Status |
|---------|-----------|-----------|------|--------|
| 18-0.1V | 3.219 ms | 0.010 ms | 99.7% | ✅ |
| 19-0.1V | 3.229 ms | 0.000 ms | 100% | ✅ |
| 20-0.1V | 3.031 ms | 0.865 ms | 71.5% | ❌ |
| 21-0.1V | 3.240 ms | 0.000 ms | 100% | ✅ |
| 22-0.1V | 3.125 ms | 0.000 ms | 100% | ✅ |

### 關鍵發現

1. **4/5 通過**：只有 Speaker 20（中央位置）有殘差
2. **Speaker 20 異常**：需要進一步分析原因
3. **整體有效**：OMP 對齊方法在大多數位置有效

---

**Document Version**: 1.1
**Created**: 2026-01-30
**Updated**: 2026-01-31
**Author**: Staged validation plan for LDV-to-Mic alignment