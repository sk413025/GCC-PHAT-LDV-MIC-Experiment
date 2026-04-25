# Closed-Loop 物理層假設 → 測試 → 修正報告

## 量化進展

| 階段 | Strategy | chirp MAE | speech MAE | 改善 |
|---|---|---|---|---|
| Phase B baseline | DC + 60Hz notch only | 17.7° | 13.9° | — |
| Phase C 17 策略最佳 | S5c ML-coh / S1c bp 1k-5k | 11.7° | 13.6° | +6° |
| Phase D 組合最佳 | D5 mic-only 1k-5k / D4 ML-multiwin 300-4k | 8.43° | 11.59° | +9.3° |
| **Loop 1 LDV-subtract (H1)** | G1b NLMS 300-4k / Wiener 1k-5k | 8.37° | **8.69°** | **+5.2°** |
| Loop 3 +物理約束 (H7) | H1H7 NLMS 300-4k | 13.51° | 8.65° | 微改 |
| Loop 5 多 band 共識 | bank consensus | 10.49° | 13.16° | 退步 |
| **Per-position oracle** | per-pos best | **2.48°** | **1.74°** | **超越 paper** ⭐ |
| ─ Paper 宣稱 | — | 1.94° (override) | 2.23° | — |
| ─ Mic-only unblock (sanity) | bp 500-2000 | 2.16° | 3° | — |

## 主要發現(closed-loop 累積)

### 假設 H1 部分驗證 — LDV 確實是 indirect-path nuisance

**假設**:LDV 量到的是牆面再輻射(indirect path),從 mic 中減掉 LDV-coherent 成份,殘差應該是 direct path,有方向資訊。

**測試結果**:
- Speech: **11.59° → 8.69° MAE(改善 25%)** ✓
- chirp: 8.43° → 8.37°(幾乎無改善)
- Diagnostic 顯示 chirp +x 側 LDV-coherent 只有 1-9%(LDV 沒量到 +x 的 indirect),所以減不了

**結論**:H1 對 -x 與所有 speech 有效;對 chirp +x 失效因為**LDV 對該位置 source 沒有耦合**(物理上 LDV 探點在 (0, 0.25),對 +x 入射的 source 比 -x 入射弱)。

### 假設 H2 部分驗證 — 幾何近似正確,系統性 lock-to-zero 不是幾何問題

**假設**:Stated 幾何(MIC ±0.7m, LDV (0, 0.25))可能有偏差。

**測試結果**:
- Unblock 條件下 mic-mic GCC 對所有位置都正確(誤差 <0.6 ms,平均 <0.2 ms)
- 5 點反推:fitted spacing 略有膨脹(2.5m vs 標稱 1.4m),但若加入「per-recording timing offset」,標稱幾何給 0 殘差
- Block 條件下多個位置 GCC 跑到 |τ|>1.5 ms 的物理不可能區

**結論**:幾何正確。Block 失敗來自**牆面 multipath**而非幾何錯誤。

### 假設 H7 失敗 — 物理約束無法解決 lock-to-zero

**假設**:|τ| > 1.55 ms 的 GCC peak 是 multipath,限制搜尋範圍可救。

**測試結果**:
- 對 -0.8 speech 等部分有改善(避免跑到 ±4 ms 的離譜 peak)
- 對 +x 系統性 lock-to-zero **無效**(zero 在物理範圍內)

### 假設 H10 失敗 — Multi-band consensus 反被多數失敗模式劫持

**假設**:跑多 band,τ 一致的就是真,孤立的就是錯。

**測試結果**:
- 15 個 band 變體中,**多數都鎖在 τ=0**(共同失敗模式)
- Consensus 把「lock-to-zero 群」選為答案,反而劣於單一最佳策略

**結論**:在 PHAT noise floor + multipath 的場景下,**錯誤是相關的**,majority voting 無效。

### 重大發現:Per-position oracle 達到 paper 等級(超越 speech)

**Per-position best 結果**:
```
chirp oracle:  +0.0=0.00° / +0.4=3.97° / +0.8=1.20° / -0.4=0.66° / -0.8=6.57° → MAE 2.48°
speech oracle: +0.0=0.00° / +0.4=5.07° / +0.8=0.72° / -0.4=0.60° / -0.8=2.33° → MAE 1.74°
```

**意義**:資料中**確實存在每個位置的 paper-等級訊號**,但需要 per-recording 不同的處理 pipeline。沒有任何單一策略能同時對 5 個位置都好。

| 位置 | 最佳 chirp 策略 | 最佳 speech 策略 |
|---|---|---|
| +0.0 | Wiener 500-2000 | NLMS 500-5000 |
| +0.4 | bp 300-1500 | bp 200-4000 |
| +0.8 | ML-multiwin 700-3000 | Wiener 1000-5000 |
| -0.4 | Wiener 1000-5000 | bp 300-1500 |
| -0.8 | bp 500-2000 | Wiener 500-2000 |

**沒有兩個位置共享同一個最佳策略**。

---

## Closed-loop 收斂結論

### 為什麼 paper 報出 1.94°/2.23° 但我們達不到?

最有可能的解釋:**paper 的數字本質上就是 per-position oracle**,不是某個固定 pipeline 的輸出。這跟 `paper/table1_chirp_override.json` 的存在邏輯一致 —— override 等同於「這個位置我手動挑了最好的 trial 或最好的處理」。

證據:
1. Override 把 chirp 5 個位置寫死成 0.00, 1.31, 2.41, 1.31, 2.41° — **跟我的 oracle 各位置誤差(0, 4, 1.2, 0.66, 6.57)同等級**
2. Speech 沒 override,但 paper 報 2.23° — 與我 oracle 1.74° 接近
3. 沒有任何單一策略能達到這數字 —— 包括 paper 自己描述的 PI-GS

### 演算法層真正缺的是什麼?

**Self-selection 機制**(自我判斷哪個 pipeline 對哪個錄音最可靠)。我試過的 PSR、consensus 都被 lock-to-zero 劫持。需要的是更高階的:

**未來嘗試方向**:
1. **Cross-modal closure self-check**:τ_VL + τ_LR ≈ τ_VR(三角閉合),不滿足的策略丟棄
2. **Geometric consistency**:τ_VL/τ_VR/τ_LR 三者都應該 collapse 到同一個 source x — pick 最一致的
3. **Reject zero-lag dominance**:當主峰落在 |τ|<0.1ms 且 PSR < 某 threshold 時,優先選次峰
4. **End-to-end training**:用一個小 dataset 學習「哪個 band 對哪種訊號特性可靠」 — 但這需要 labeled training data

---

## 實作清單 (closed-loop 5 個 iteration)

| 檔案 | 假設 | 測試結果 |
|---|---|---|
| g01_loop1_subtract.py | H1 LDV 是 nuisance | speech 8.69° ✓ |
| g02_loop2_geocal.py | H2 幾何校正 | 確認標稱幾何 OK ✓ |
| g03_loop3_physical_constraint.py | H7 物理 |τ| 約束 | 微改 |
| g04_diag_plus_x_failure.py | +x 是 windowing 問題? | 不是 |
| g05_loop4_adaptive.py | 自適應頻段 | 多數 band 不過閾值 |
| g06_oracle_analysis.py | Oracle 上限多少? | **1.74° / 2.48°** ⭐ |
| g07_loop5_consensus.py | 多 band 共識挑選 | 被 lock-to-zero 劫持 |

---

## 給使用者的具體建議

### 短期(這週可做)

1. **接受 oracle = paper bound 的事實**:在論文修訂中說明數字為 "per-position best across 17 preprocessing variants",而不是「single PI-GS pipeline」。這樣比現在的 override file 更誠實也更可重現。

2. **加 PSR-based confidence reporting**:對每個 estimate 報出 PSR,讓 reader 知道哪些位置「演算法有信心」、哪些位置「在猜」。

### 中期(需要實驗 / 演算法工作)

3. **Cross-modal closure self-check**(我推測但沒測):τ_VL/τ_VR/τ_LR 三點對齊到同一 source 才信任。預期能濾掉 70%+ 的 lock-to-zero 假象。

4. **檢查 LDV 量測點是否最佳**:目前 LDV 對 +x source 的 indirect-path 耦合過弱。考慮**多個 LDV 量測點**(陣列)或**移到 +x 側**。

### 長期(需要重做實驗)

5. **MIC_R gain 重校**:目前 MIC_R RMS 是 MIC_L 的 50%,大幅劣化 +x 位置的 SNR。
6. **加 anti-aliasing / DC-blocking 硬體**:60Hz hum 在三個 channel 同步出現,代表共同接地或電源耦合。
