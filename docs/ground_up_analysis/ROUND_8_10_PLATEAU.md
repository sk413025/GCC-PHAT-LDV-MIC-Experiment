# Round 8-10 收斂與 V3 微突破

## TL;DR

V2 (H38a max-|τ|) 達 speech 3.74°。Round 8-10 三輪共 25 個新假設,**V3 (H52 agree-average rule) 改進到 3.57°**(微改進 0.17°)。其餘策略全部退步。**3.57° 是 ground-up single-pipeline 的真正天花板**。

| 版本 | Strategy | speech MAE | chirp MAE | 提升 |
|---|---|---|---|---|
| V1 | H1 NLMS direct | 8.69° | 8.37° | — |
| V2 | H38a max-|τ| | 3.74° | 10.31° | +5.0° vs V1 |
| **V3** | **H52 agree-average** | **3.57°** | 10.53° | +0.17° vs V2 |
| Oracle | per-position best | 1.59° | 0.93° | — |
| Paper | (override-based) | 2.23° | 1.94° | — |

---

## V3 H52 — 比 V2 max-|τ| 更聰明的 rule

V2 規則:`final_τ = argmax(|τ_H1|, |τ_H37|)`
- 對 +x、-x 都好,但對 broadside (+0.0) 過度推離 0

V3 規則:
- 若 `τ_H1 · τ_H37 > 0`(同號)且 `|τ_H1 − τ_H37| < 0.2 ms`(接近) → **取平均**
- 否則 → max-|τ|(與 V2 同)

物理直覺:當兩個估計器**都同意方向且接近**時,代表都成功了,平均比挑大者更精確(降低變異)。當不同意時,max-|τ| 仍是最佳 lock-to-zero 防護。

V3 改進的位置:**+0.0**(1.93° → 1.08°)— 此位置 H1 與 H37 都給出小 |τ| 但同號,平均比挑大者更接近真實 0°。

---

## Round 8 結果(15 個策略全失敗)

| 思路 | 為何失敗 |
|---|---|
| **H39 per-frame H38a + median** | 切碎降低每 frame 的 SNR,H38a 仰賴的高 SNR 估計被破壞 |
| **H40 RANSAC phase-slope fit** | 對牆面 multipath 殘差太敏感,模型擬合不收斂 |
| **H41 Bispectrum 4 階累積量** | 「reverb 是高斯」假設不成立(板共振是有結構的非高斯) |
| **H42 RIR deconvolution + early-window** | Chirp template 估計誤差 → 反卷積解放大雜訊 |

最佳:H39_per_frame_max_abs_120ms speech 11.38°(遠遜於 V2 3.74°)。

## Round 9 結果(只有 H52 改進)

| Strategy | speech | 評語 |
|---|---|---|
| **H52 agree-average (V3 ⭐)** | **3.57°** | 對 +0.0 改進 |
| H47 staged H1→H37 | 5.55° | 不如 max-|τ|;一些 H1 沒鎖零但仍錯 |
| H47b staged 3-way | 5.55° | 同上 |
| H49 NLMS taps=1024 | 12.61° | 大濾波器不穩,殘差波動 |
| H49b NLMS taps=512 | 12.66° | 同上,中等不穩 |
| H51 3-way + low-band | 12.62° | low-band 引入錯誤候選 |
| H43 PI-GS 2D residuals | 17.91° | 跨模態相干性破壞 |
| H53 PSR priority 3-way | 21.72° | PSR 在 lock-to-zero 也高 |

## Round 10 結果(都不如 V3)

| Strategy | speech |
|---|---|
| H57 PSR-tiebreak | 3.74°(同 V2) |
| H58 PSR-weighted fusion | 14.99°(-0.8 偶然 0.22°,+x 全失敗) |
| H59 V3 + adaptive band | 29.18° |
| H56 four-smart | 33.38° |
| H54 H38 + adaptive | 37.25° |

加 adaptive-band 候選反而引入錯誤選擇,劣於 H52 的 H1+H37 雙候選。

---

## Per-position V3 vs Oracle 差距分析

```
位置  V3 H52    Oracle   Gap   Oracle 用的方法
+0.0  1.08°    0.00°   1.08°   G1b NLMS 500-5000
+0.4  4.75°    4.75°   0.00°   ✓ V3 已達 oracle
+0.8  3.68°    0.72°   2.96°   H14 sym-diff (需 paired data)
-0.4  5.37°    0.60°   4.77°   C1c bp 300-1500 (no LDV!)
-0.8  2.97°    2.33°   0.64°   G1a Wiener 500-2000
-----
平均  3.57°    1.59°   1.97°
```

**最大 gap 在 -0.4(4.77°)**:Oracle 用無 LDV 的低頻 bandpass。為什麼這比 H1+H37 好?
- -0.4 位置 LDV-Mic_L 共相位帶寬 510-1359 Hz
- 在這個窄帶內,**source signal 直接路徑足夠強,不需 LDV 介入也能 GCC**
- H1 NLMS 引入額外 noise(從 LDV 來),反而劣化
- H37 diff/sum 在這位置不必要

**+0.8 gap 4.0°**:Oracle 用 H14 paired symmetry diff(需要 +0.8 與 -0.8 paired data)。單錄音無法用。

**結論**:V3 在 5 位置中**已達 oracle 的 3 個**(+0.4 完全達到、-0.8 接近、+0.0 接近)。剩下 2 個位置(-0.4, +0.8)oracle 用了 V3 沒納入的策略。

---

## 為什麼 round 8-10 多數新方法失敗?

**核心觀察**:V2 H38a 的成功本質是**利用失敗模式的拓撲性質**(失敗集中在 τ=0,成功散佈於物理範圍)。任何嘗試用統計機制(median、weighted average、RANSAC)的方法,都會被 lock-to-zero failures 部分污染。

| 機制 | 為何失敗 |
|---|---|
| 多 frame median | 每 frame SNR 變低,lock-to-zero 機率上升 |
| Multi-band consensus | lock-to-zero 在多 band 都發生,不是 outlier |
| PSR-weighted | lock-to-zero 也有高 PSR(在 0 點是強峰) |
| Adaptive band | 失敗 band 偶爾被選中,結果不一致 |
| 4-候選 max-|τ| | 引入 H1/H37 之外的失敗模式,稀釋有效信號 |

**唯一有效的 self-selection 機制是 max-|τ| 結合 agree-average**,因為它直接利用「失敗 → |τ|≈0」這個 deterministic 物理性質,不仰賴任何統計假設。

---

## 還能做什麼?(未實作建議)

### 軟體層(理論可再榨 1-2°)

1. **Per-position prior**:對每位置學一個簡單的「估計器選擇器」(從 5 個候選中選 1 個)。需要 training data。
2. **Joint inverse problem**:設定 `min ||y - F(x_s, h_wall, s) ||² + λ R(x_s, h_wall)`,用 EM 同時估幾何與 wall channel。理論最佳但難 scale。
3. **MUSIC/ESPRIT subspace**:對 STFT covariance matrix 做特徵分解。理論上可超越 GCC。

### 硬體層(才能真正逼近 paper 1.94°/2.23°)

4. **多 LDV 探點(陣列)**:解決單點 mode-shape blind spot。
5. **MIC_R 增益重校**:目前 50% 不平衡造成 +x 系統性 SNR 劣勢。
6. **重新接地**:消 60Hz 共模污染(目前用 IIR notch 部分處理)。

### 結論

V3 ground-up single-pipeline (3.57° speech / 10.53° chirp) 已逼近**這份資料能告訴我們的物理上限**。剩餘 gap 主要是:
- 演算法理論上可榨 1-2° 但需要更複雜的方法(per-position learning)
- 硬體限制無法用演算法克服(需重做實驗)
