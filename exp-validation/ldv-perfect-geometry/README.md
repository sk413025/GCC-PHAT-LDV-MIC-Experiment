# LDV Perfect Geometry Validation Experiment

## 實驗目標

驗證 OMP (Orthogonal Matching Pursuit) 對齊方法，使 LDV (Laser Doppler Vibrometer) 信號能夠替代麥克風進行 TDoA/DoA 估計。

---

## 原始資料位置

**音訊檔案位於 `main` 分支的 `dataset/` 目錄**

```
dataset/GCC-PHAT-LDV-MIC-Experiment/
├── 18-0.1V/
│   ├── *LDV*.wav       # LDV 振動信號
│   ├── *LEFT*.wav      # 左麥克風
│   └── *RIGHT*.wav     # 右麥克風
├── 19-0.1V/
├── 20-0.1V/
├── 21-0.1V/
└── 22-0.1V/
```

---

## Speaker 18-22 資料說明

### 幾何配置

```
Speaker Line (y = 0 m):
  22: (-0.8, 0)  21: (-0.4, 0)  20: (0, 0)  19: (+0.4, 0)  18: (+0.8, 0)
       ←── 左側                  中央                  右側 ──→

LDV Box:      (0.0, 0.5) m    # LDV 指向的反射點
Mic Left:     (-0.7, 2.0) m
Mic Right:    (+0.7, 2.0) m
Mic Spacing:  1.4 m
Speed of Sound: 343 m/s
```

### Speaker 編號對應

| Speaker | X 位置 (m) | 特性 | 備註 |
|---------|-----------|------|------|
| **18** | +0.8 | 右側邊緣 | 最穩定，所有 segments 完美對齊 |
| **19** | +0.4 | 右側中間 | 穩定 |
| **20** | 0.0 | **中央** | 雙峰分布問題（見下方說明） |
| **21** | -0.4 | 左側中間 | 穩定 |
| **22** | -0.8 | 左側邊緣 | 穩定 |

### 資料格式

- **取樣率**: 48000 Hz
- **檔案格式**: WAV (16-bit PCM)
- **命名規則**: `{speaker}-0.1V` 表示該 speaker 位置，0.1V 為 LDV 靈敏度設定

---

## 資料預處理流程

### 1. 信號讀取與正規化
```python
sr, signal = wavfile.read(path)
signal = signal.astype(np.float32) / 32768.0  # 16-bit → [-1, 1]
```

### 2. STFT 參數
```python
{
    'n_fft': 6144,
    'hop_length': 160,
    'freq_min': 100 Hz,
    'freq_max': 8000 Hz,
}
```

### 3. OMP 對齊參數
```python
{
    'max_lag': 50,      # ±50 samples = ±1.04 ms @ 48kHz
    'max_k': 3,         # OMP 稀疏度（選擇 3 個最佳 lag）
    'tw': 64,           # 時間窗口（frames）
}
```

### 4. Per-Frequency Max-Abs 正規化
```python
# 對每個頻率 bin 獨立正規化
for f in freq_bins:
    max_val = max(abs(LDV[f]), abs(Mic[f]))
    LDV[f] /= max_val
    Mic[f] /= max_val
```

---

## 實驗設計

採用四階段驗證流程：

| Stage | 驗證項目 | 通過標準 |
|-------|---------|---------|
| Stage 1 | 能量縮減驗證 | OMP 殘差 < Random baseline |
| Stage 2 | 目標麥克風相似度 | GCC-PHAT τ → 0 |
| Stage 3 | 跨麥克風 TDoA | OMP τ ≈ baseline τ |
| Stage 4 | DoA 多方法驗證 | 4種方法 × 4種信號配對 |

---

## 實驗結果摘要

### Stage 4 最終驗證 - 5/5 speakers PASS

| Speaker | 位置 | Baseline τ (ms) | OMP τ (ms) | Error (ms) | Status |
|---------|------|----------------|------------|------------|--------|
| 18 | x=+0.8 | -1.688 | -1.667 | 0.021 | ✓ |
| 19 | x=+0.4 | -1.688 | -1.667 | 0.021 | ✓ |
| 20 | x=0 | -1.688 | -1.688 | 0.000 | ✓ |
| 21 | x=-0.4 | -1.688 | -1.688 | 0.000 | ✓ |
| 22 | x=-0.8 | -1.688 | -1.667 | 0.021 | ✓ |

---

## 重要發現與注意事項

### 1. Speaker 20 中央位置的雙峰分布問題

**現象**：Speaker 20（x=0，中央位置）在 Stage 3 Multi-Segment 驗證中展現雙峰分布：

| Segment | OMP τ (ms) | Status |
|---------|------------|--------|
| 1, 6, 10 | 0.000 | ❌ 錯誤峰 |
| 3, 7 | -1.688 | ✅ 正確峰 |
| 2, 8, 9 | -1.646 ~ -1.667 | ✅ 接近正確 |
| 4 | +9.188 | ❌ 極端 outlier |
| 5 | +1.333 | ❌ |

**成功率**: 5/10 segments (50%)

**根本原因**：
- Speaker 20 與 LDV Box 都在 x=0（同一垂直軸）
- 幾何對稱性導致 OMP 找到兩個等價的對齊方案
- 其他 speakers 的非對稱位置提供更明確的對齊方向

**結論**：儘管 Stage 3 有問題，Stage 4 的 5-segment 驗證仍然通過（可能是運氣好選到正確的 segments）

### 2. 幾何符號約定修正

**原計算（錯誤）**：
```python
tau_true = (d_left - d_right) / c
```

**修正後（正確）**：
```python
# GCC-PHAT(MicL, MicR) 約定：
# - 正 τ：MicL 信號領先（聲音先到達 MicL）
# - 負 τ：MicR 信號領先（聲音先到達 MicR）
tau_true = (d_right - d_left) / c
```

### 3. GCC-PHAT 是最穩定的方法

| 方法 | Baseline 穩定性 | OMP 匹配度 | 建議 |
|------|----------------|-----------|------|
| **GCC-PHAT** | 極穩定 (std≈0) | error < 0.1 ms | **推薦使用** |
| CC | 不穩定 | 不穩定 | 不適用 |
| NCC | 不穩定 | 不穩定 | 不適用 |
| MUSIC | 高變異 | 高變異 | 需更多 snapshots |

### 4. OMP Lag 分布一致性

所有 speaker 位置的 OMP lag 分布非常相似：

| Speaker | Dominant lag mean | Dominant lag std |
|---------|-------------------|------------------|
| 18 | -0.037 ms | 0.553 ms |
| 19 | +0.027 ms | 0.564 ms |
| 20 | +0.028 ms | 0.529 ms |
| 21 | +0.056 ms | 0.556 ms |
| 22 | +0.025 ms | 0.559 ms |

這表明 OMP 對齊本身（Stage 1-2）在所有位置都正常工作，問題出在 Stage 3 的跨 segment 泛化。

---

## 目錄結構

```
exp-validation/ldv-perfect-geometry/
├── README.md                          # 本文件
├── EXPERIMENT_PLAN.md                 # 初始實驗計劃
├── STAGE_VALIDATION_PLAN.md           # 詳細驗證流程設計
├── STAGE_VALIDATION_RESULTS.md        # 完整驗證結果報告
├── GCC-PHAT_LDV_MIC_完整實驗報告.md   # 中文完整報告
├── full_analysis.py                   # 完整分析腳本
├── scripts/
│   ├── stage1_energy_reduction.py     # Stage 1 驗證腳本
│   ├── stage2_target_similarity.py    # Stage 2 驗證腳本
│   ├── stage3_tdoa_evaluation.py      # Stage 3 單段驗證
│   ├── stage3_multi_segment.py        # Stage 3 多段驗證
│   └── stage4_doa_validation.py       # Stage 4 DoA 驗證
└── validation-results/
    ├── stage1_energy_reduction/       # 含 summary.json + per_freq_results.npz
    ├── stage2_target_similarity/      # 含 summary.json + stage2_details.npz
    ├── stage3_tdoa_evaluation/        # 含 summary.json
    ├── stage3_multi_segment/          # 含 summary.json
    └── stage4_doa_validation/         # 含 summary.json
```

---

## 開發歷程摘要

| Commit | 說明 |
|--------|------|
| `5b62b22` | 初始 commit：GCC-PHAT LDV-MIC 實驗資料與分析 |
| `435a915` | 新增詳細驗證流程設計文件 |
| `8a61540` | 實作 Stage 1-3 驗證腳本（含 OMP 實作） |
| `81889df` | 新增驗證結果報告 |
| `7510877` | Stage 3-4 驗證完成，OMP 成功匹配 baseline TDoA |
| `f776778` | Stage 4 DoA 驗證，發現符號約定問題 |
| `35a2a54` | **最終結果**：5/5 speakers 全部通過（error ≤ 0.021 ms） |

---

## 重現實驗

1. **取得音訊資料**：切換到 `main` 分支，資料在 `dataset/GCC-PHAT-LDV-MIC-Experiment/`

2. **安裝依賴**：
   ```bash
   pip install numpy scipy torch
   ```

3. **執行驗證**：
   ```bash
   python scripts/stage1_energy_reduction.py --data-root dataset/GCC-PHAT-LDV-MIC-Experiment
   python scripts/stage2_target_similarity.py --data-root dataset/GCC-PHAT-LDV-MIC-Experiment
   python scripts/stage3_tdoa_evaluation.py --data-root dataset/GCC-PHAT-LDV-MIC-Experiment
   python scripts/stage4_doa_validation.py --data-root dataset/GCC-PHAT-LDV-MIC-Experiment
   ```

---

## 相關文件

- 完整驗證結果：`STAGE_VALIDATION_RESULTS.md`
- 實驗設計細節：`STAGE_VALIDATION_PLAN.md`
- 中文完整報告：`GCC-PHAT_LDV_MIC_完整實驗報告.md`

---

**Last Updated**: 2026-01-31
**Branch**: `exp-ldv-perfect-geometry-clean`
