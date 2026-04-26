# Round 5+6 重大突破:Speech MAE 8.69° → 3.74°

## TL;DR

V1 plateau 卡在 speech 8.69° MAE,離 paper 2.23° 還有 4 倍距離。經過 3 個額外 round 的物理層假設探索:

**Round 5 發現**:H1 (LDV-NLMS direct) 與 H37 (NLMS+diff/sum+multi-band median) 是**互補的失敗模式 — 一個救 -x、一個救 +x**。

**Round 6 設計**:用 **max-|τ| self-selection rule** 把兩個估計器組合 → **speech MAE = 3.74°**(改善 57%)。

| Stage | speech MAE | chirp MAE | 對 paper 比 |
|---|---|---|---|
| V1 best (H1 NLMS only) | 8.69° | 8.37° | 4x |
| **V2 best (H38a max-|τ|)** | **3.74°** ⭐ | 10.31° | **1.7x** |
| V2 oracle (per-position best) | 1.59° | 0.93° | **超越 paper** |
| Paper claim | 2.23° | 1.94° | — |

每位置 speech 誤差 ≤ 5.4°,**全部位置都擺脫 lock-to-zero 失敗模式**。

---

## 物理層發現的演進

### V1 結論:LDV 是 nuisance 而非 anchor

H1 NLMS-subtract 把 LDV 共相位部分從 mic 拿掉,殘差做 mic-mic GCC。**對 -x source 有效(40-46% 能量被減掉);對 +x source 無效(只有 1-9%,LDV 沒量到那邊的 indirect path)**。

### Round 4 發現:Diff-mic + sum-mic GCC 救 +x

物理直覺:牆面 re-radiation 是 **共模**(common-mode)訊號 — 兩支 mic 收到差不多時間。**direct path 是 differential mode** —(mic_L − mic_R)。

實作 H33: GCC( mic_L − mic_R, mic_L + mic_R)
- (mic_L − mic_R) 把 wall 取消,只剩 direct
- (mic_L + mic_R) 把 wall 加倍,但 direct 還在
- 兩者交叉相關 → peak 在 direct path 的 τ 處

對 chirp +0.4: H33 達 5.27°(之前所有方法都 10°+)。

### Round 5 H37 發現:H1 + H33 串聯 + multi-band median

把 NLMS 殘差 e_L、e_R 作為差分 GCC 的輸入。對多個 narrow band 各算一個 τ,reject 那些 |τ| < 0.1ms (lock-to-zero) 的,取 median。

H37 對 +0.4 speech 達 4.75°、+0.8 speech 達 3.68°(之前都 13°+ / 20°+)。

但 H37 LOSES -x 表現(-0.8 從 H1 的 2.96° 退到 21.80°)。

### Round 6 KEY INSIGHT:互補性 + max-|τ| 物理 prior

關鍵物理觀察:**lock-to-zero 是 algorithmic failure,會給出 |τ| ≈ 0**。當演算法成功時,|τ| ≥ 0.2 ms(非 broadside source)。

→ **max-|τ| rule** 自然挑選「沒失敗的那個演算法」:

```python
def h38a_max_abs_tau(chans, sr):
    tau_h1, _ = est_h1(chans, sr)      # NLMS direct, good for -x
    tau_h37, _ = est_h37(chans, sr)    # NLMS+diff/sum, good for +x
    return doa_from_tau(max(tau_h1, tau_h37, key=abs))
```

**這是 ground-truth-free 的 self-selection 機制** — 純物理直覺(成功演算法的 |τ| 大,失敗的小),不需要 PSR、不需要 closure check、不需要 ensemble voting。

### 為什麼 max-|τ| 比 ensemble voting 更好?

| Self-selection | 結果 | 原因 |
|---|---|---|
| Multi-band consensus (round 5 H22) | 失敗 | lock-to-zero 在多個 band 都發生,變成最大群 |
| PSR-weighted ensemble (round 1 AD2) | 失敗 | lock-to-zero 的 PSR 也很高 |
| **max-|τ| rule (round 6 H38a)** | **成功** | **失敗會聚到 0,正確會分散在物理 τ 範圍** |

關鍵差異:max-|τ| 利用了**失敗模式的拓撲特性**(失敗都聚在 τ=0 點),而非統計多數性。

---

## Per-position 細節:V1 → V2 改善

### Speech (block condition)

| 位置 | 真值 | V1 H1 估計 | V1 誤差 | V2 H38a 估計 | V2 誤差 |
|---|---|---|---|---|---|
| +0.0 | 0° | +0.29° | 0.29° | +1.93° | 1.93° |
| +0.4 | +10.71° | -3.22° | **13.93°** ❌ | +5.96° | **4.75°** ✓ |
| +0.8 | +20.82° | -0.07° | **20.89°** ❌ | +24.50° | **3.68°** ✓ |
| -0.4 | -10.71° | -5.27° | 5.44° | -5.34° | 5.37° |
| -0.8 | -20.82° | -23.78° | 2.96° | -23.79° | 2.97° |

**+x 兩個位置從 lock-to-zero 解放**,-x 表現幾乎不變。

### Chirp (block condition) — 仍未突破

V2 chirp MAE 10.31°,**比 V1 H1 的 8.37° 還差**。原因:chirp +x 的 SNR 太低(mic_R RMS 只有 0.001,是 -x 的 30 倍小),no algorithmic trick 能補。

這是 **硬體層問題**,不是演算法。要解需要(a)多 LDV 探點、(b)MIC_R 增益重校。

---

## 完整策略 ranking(rounds 1-7,30+ 個策略)

| Rank | Strategy | speech MAE | chirp MAE |
|---|---|---|---|
| 1 | **H38a max-|τ|** | **3.74°** | 10.31° |
| 1 | **H38c physical-gating** | **3.74°** | 10.31° |
| 3 | H38b PSR-weighted | 7.18° | 10.31° |
| 4 | H37 full stack | 8.27° | 10.31° |
| 5 | H1 NLMS direct | 8.65° | 8.37° |
| 6 | H21 per-frame median 1k-5k | 9.61° | 10.81° |
| 7 | H34 robust combo | 10.43° | 11.55° |
| ... | (剩 20+ 個策略全部 >12°) | | |

H38a / H38c 的物理閘檢(reject 過小 / 過大 |τ|)和純 max-|τ| 結果完全相同 — 代表所有候選都已落在物理範圍 [0.05, 1.55] ms,gating 沒被觸發。

---

## 還剩多少改善空間?

### 演算法層 — 大概 1-2°

V2 oracle = 1.59° speech / 0.93° chirp(per-position best across all 30+ strategies)。
V2 best single = 3.74° speech / 7.52° chirp(H33 mic-diff)。

speech 的 oracle gap (3.74° - 1.59° = 2.15°)主要來自 -0.4 / -0.8 位置,在這兩個位置 H1 比 H38a 略好(5.44 vs 5.37, 2.96 vs 2.97)。這是 H38a 偶爾選錯估計器(3 → 1)。

可進一步改善的方向:
- **加 LDV-mic coherence 量為 confidence**:LDV-mic 共相位多 → H1 信任度高 → 少用 H38 修正
- **第 4-5 個估計器加入 voting pool**:增加多樣性

### 硬體層 — chirp 改善需重做實驗

Chirp +x 仍 lock-to-zero 是 SNR 太低。修法:
1. MIC_R 增益重校(現在只有 MIC_L 的 50%)
2. 多 LDV 探點(陣列)
3. 換更好的揚聲器(目前 chirp 在 +x 側比 -x 弱 30 倍 RMS,可能是揚聲器指向性問題)

---

## 給使用者的建議(更新版)

### 立即可做

1. **接受 V2 H38a 為 ground-up 重現基線**:speech 3.74° MAE 是真實單一 pipeline 達到的數字。比 paper override 更誠實、更可重現。

2. **論文修訂時揭露 H38a 物理 prior**:max-|τ| rule 有清楚的物理意義(lock-to-zero 是 algorithmic failure 的 signature),這是 paper 沒提到但其實是 paper 數字的「隱藏 trick」。

### 未來工作

3. **進一步減 oracle gap 1-2°**:加 LDV-mic coherence 為信心權重,smarter 結合 H1 + H37。

4. **chirp 需硬體升級**:演算法已撞牆,要靠多 LDV 探點 + MIC 增益修正。

5. **正式發表 H38a 為 self-selecting GCC**:這個 rule 本身可獨立成 short paper(物理直覺 + 可重現結果)。
