# 最新Git Commit分析報告

## Commit 資訊
- **Commit Hash**: 62a5161
- **作者**: yu_chen <brandonhsu329@gmail.com>
- **日期**: 2026-02-01 04:57:02 UTC
- **標題**: docs(results): refresh stage validation report

## Commit 訊息
```
docs(results): refresh stage validation report

- Update exp-validation/ldv-perfect-geometry/STAGE_VALIDATION_RESULTS.md to match current experiment commits
- Materialize Stage1/2 summary.json artifacts (remove old placeholder pointers)
```

---

## 一、主要修正內容

### 1. 新增完整的實驗驗證文檔系統
本次 commit 新增了完整的 LDV-Perfect-Geometry 實驗驗證框架，包含：

- **實驗計劃文檔** (EXPERIMENT_PLAN.md, STAGE_VALIDATION_PLAN.md)
- **實驗結果報告** (STAGE_VALIDATION_RESULTS.md, README.md)
- **完整實驗報告** (GCC-PHAT_LDV_MIC_完整實驗報告.md)
- **實驗腳本** (5個 Python 驗證腳本)
- **驗證結果數據** (90個 JSON/NPZ 結果檔案)

### 2. 實作四階段驗證流程
建立了系統化的驗證方法，測試 OMP (Orthogonal Matching Pursuit) 對齊技術：

#### Stage 1: Energy Reduction (能量降低驗證)
- **目標**: 確認 OMP 在 LDV → Mic 方向上學習有意義的模式
- **結果**: ✅ **全部通過** (5/5)
- **成效**: OMP 一致性地優於 Random baseline 約 40%

#### Stage 2: Target Similarity (目標相似度驗證)
- **目標**: 確認 OMP 對齊後的 LDV 信號在時域上更像 Target Mic
- **結果**: ✅ **全部通過** (5/5)
- **成效**: τ → 0 ms, PSR 提升 +6~8 dB

#### Stage 3: Cross-Mic TDoA (跨麥克風時差評估)
- **目標**: 驗證 OMP_LDV 是否能取代 MicL 與 MicR 做 TDoA
- **結果**: ⚠️ **部分通過** (2-3/5，視 baseline 方法而定)
- **發現**: 對 baseline 定義極度敏感

#### Stage 4: DoA Validation (到達方向驗證)
- **目標**: 使用多種方法驗證 DoA 估計準確度
- **結果**: 
  - 4-A (語音 + 幾何真值): ❌ **失敗** (0/5)
  - 4-B (Chirp + 麥克風真值): ✅ **通過** (2/2)
  - 4-C (語音 + Chirp 真值): ⚠️ **部分通過** (1/2)

### 3. 新增 .gitignore 管理大型數據檔案
建立了適當的 gitignore 規則：
- 排除原始語音數據集 (18-22-0.1V 資料夾)
- 保留 chirp 測試數據 (23-chirp, 24-chirp)
- 排除 Python 編譯檔案

### 4. 保存歷史實驗數據
將舊的實驗報告移至 `_old_reports/` 資料夾，包含：
- 23-chirp(-0.8m) 測試數據
- 24-chirp(-0.4m) 測試數據

---

## 二、解決的問題

### 問題 1: 缺乏系統化的驗證流程
**以前的問題**: 
- 實驗結果散亂，沒有統一的驗證標準
- 難以追蹤每個階段的成功/失敗狀態

**本次解決**:
- 建立四階段驗證框架
- 每個階段都有明確的通過標準
- 自動化腳本產生可重現的結果

### 問題 2: Baseline 定義不明確導致結果不穩定
**以前的問題**:
- TDoA 基準值定義不一致
- 短時間窗 + 低 PSR 容易鎖到假峰 (例如: -81 samples = -1.6875 ms)

**本次解決**:
- 引入三種 baseline 方法：
  1. `segment`: 單一短窗
  2. `report`: 100-600s 長區間
  3. `windowed (PSR>=10)`: 多窗中位數 + PSR 篩選
- 詳細記錄每種方法的結果差異
- 發現 `windowed (PSR>=10)` 最穩定 (baseline PSR ~24-26 dB)

### 問題 3: OMP 方法效能未被量化
**以前的問題**:
- 不確定 OMP 是否真的在學習有意義的模式
- 缺乏與隨機基準的比較

**本次解決**:
- Stage 1 確認 OMP 比隨機猜測好 40%
- Stage 2 確認時域對齊品質 (PSR 提升 6-8 dB)
- 建立可量化的效能指標

### 問題 4: 語音與 Chirp 信號行為差異
**以前的問題**:
- 不清楚為什麼語音信號在某些情況下效果不佳
- 缺乏對不同信號類型的系統性比較

**本次解決**:
- 發現語音長檔在 500-2000 Hz + 1秒窗下，MicL-MicR τ ≈ 0 ms
  - 這與幾何真值偏差很大（除了 speaker 20 在 θ≈0°）
- Chirp 信號配合 scan 挑窗 + GCC-PHAT prealign 可達到 |Δθ| < 1° 的精度
- 明確記錄了語音與 chirp 的適用場景差異

### 問題 5: 實驗結果難以追溯和重現
**以前的問題**:
- 實驗參數散落各處
- 沒有標準的結果格式

**本次解決**:
- 所有實驗參數都記錄在 JSON 檔案中
- 建立標準的 summary.json 格式
- 每個 stage 都有對應的驗證腳本
- 可以完整重現任何實驗結果

---

## 三、關鍵發現

### 1. OMP 對齊技術的有效性
✅ **Stage 1-2 表現優異** (全部 5/5 通過)
- 證明 OMP 確實在學習有意義的模式
- 成功將 LDV 信號對齊到目標麥克風

⚠️ **Stage 3-4 對參數敏感**
- baseline 定義、時間窗長度、頻率範圍都會影響結果
- 需要仔細調整參數才能獲得穩定結果

### 2. Baseline 方法的重要性
不同 baseline 方法對通過率的影響：
- `segment`: 容易受假峰影響
- `report`: PSR 可能過低 (如 speaker 19 只有 2.81 dB)
- `windowed (PSR>=10)`: 最穩定，baseline PSR ~24-26 dB

### 3. 信號類型的適用性
- **語音信號**: 在短窗下 MicL-MicR τ 容易偏向 0 ms
  - 適合長時間平均
  - 需要更寬鬆的通過標準
  
- **Chirp 信號**: 配合適當的 prealign 可達到極高精度
  - |Δθ| < 1° 的 DoA 估計
  - 適合作為真值參考

### 4. 幾何配置
系統記錄了完整的感測器位置：
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

## 四、檔案統計

本次 commit 新增：
- **90 個檔案** (19,745 行新增)
- **文檔**: 6 個 Markdown 檔案
- **程式碼**: 5 個 Python 腳本
- **數據**: 12 個 NPZ 檔案, 73 個 JSON 檔案
- **音訊**: 6 個 WAV 檔案 (chirp 測試數據)
- **配置**: 2 個 .gitignore, 1 個 SHA256 清單

---

## 五、對未來工作的影響

### 已建立的基礎設施
1. ✅ 完整的四階段驗證流程
2. ✅ 可重現的實驗腳本
3. ✅ 標準化的結果格式
4. ✅ 清晰的通過/失敗標準

### 待改進的方向
1. ⚠️ Stage 3-4 的通過率仍需提升
2. ⚠️ 語音信號的 DoA 估計需要新方法
3. ⚠️ 需要更魯棒的 baseline 定義策略

### 建議後續研究
1. 探索不同的窗口長度和頻率範圍
2. 開發自適應的 baseline 選擇方法
3. 研究語音特定的 DoA 估計技術
4. 擴展到更多的說話者位置和距離

---

## 結論

這次 commit 是一個**重要的里程碑**，它：

1. ✅ **系統化**: 建立了完整的驗證框架
2. ✅ **可追溯**: 所有結果都有詳細記錄
3. ✅ **可重現**: 提供了自動化腳本
4. ✅ **問題識別**: 清楚指出現有方法的限制
5. ✅ **證明價值**: Stage 1-2 證明 OMP 方法的有效性
6. ⚠️ **指明方向**: Stage 3-4 的結果指出需要改進的地方

透過這次更新，團隊現在有了：
- 明確的效能基準
- 可靠的驗證方法
- 清晰的問題陳述
- 具體的改進方向

這為後續的研究和開發奠定了堅實的基礎。
