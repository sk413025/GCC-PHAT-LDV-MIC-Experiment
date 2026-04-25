# Phase A — Forensics Findings & Implications for Phase B+

## 重大發現 (Critical Discoveries)

### F1. 資料其實是兩個 dataset,不是一個

論文 paper/repro_asset_manifest.json 只指向 `dataset/0223/` (worktree 內),但這個 dataset **只有 chirp,沒有 speech**。

**真正的 speech 資料**在 worktree 外:`/home/sbplab/jiawei/speech/`,日期 0224(chirp 是 0223,差一天)。

| Dataset | 路徑 | 內容 | 取樣 | 時長 | 5 位置 × 2 條件 |
|---|---|---|---|---|---|
| 0223 chirp | `dataset/0223/0223-block-*` 與 `0223-unblock-*` | down-sweep chirp (~6 kHz → 500 Hz),每 2 秒重複,共 6 次 | 48 kHz, 24 bit | 13–14 s | ✓(IDs 18-22) |
| 0224 speech | `/home/sbplab/jiawei/speech/{block,unblock}-{1..5}(high)` | 連續語音 | 48 kHz | ~200 s | ✓(IDs 23-27) |

**對 Phase B 的衝擊**:必須建立兩條獨立資料路徑(chirp / speech),並對應不同的時間視窗策略。

### F2. AC 60 Hz 諧波系統性污染所有頻道

**所有 25 個 chirp WAV(以及預期 speech WAV)同時在 LDV 與兩支麥克風出現以下強力 tonal:**
- 58.6 Hz (60 Hz 微偏,FFT bin 量化)
- 117 / 123 Hz (2nd 諧波)
- 169.9 / 175.8 Hz (3rd 諧波)
- 240 / 246 Hz (4th 諧波)

各頻譜處 prominence 達 5–10 dB 以上。**跨頻道一致性**強烈暗示這是**共同接地或電源耦合**,不是聲學訊號。

**對 Phase B 的衝擊**:必須做 60Hz comb notch(60、120、180、240、300 Hz),否則 PHAT 會把這些 tonal 相位放大,在 GCC 上產生強烈 sidelobe。

### F3. LDV-Mic 可用 coherence 帶寬隨位置劇烈變動

| Position | LDV-MIC_L band (γ²>0.3) | LDV-MIC_R band |
|---|---|---|
| +0.0 block | 2426–3445 Hz | 2426–3445 Hz |
| +0.4 block | 4711–5520 Hz | 557–1271 Hz |
| +0.8 block | 5045–6094 Hz | 7734–8924 Hz |
| -0.4 block | 510–1359 Hz | 750–1529 Hz |
| -0.8 block | 381–1113 Hz | 1377–2168 Hz |

**沒有任何單一固定頻段同時對五個位置都有效**。論文 default 500–2000 Hz 只覆蓋 -0.8 / -0.4 / 部分 +0.4。

**對 Phase B 的衝擊**:單一固定 bandpass 會 bias 結果。Strategy C5(coherence-weighted)與 C6(TF masking)很可能是關鍵 — 讓演算法**自適應**選擇 useful 頻段。

### F4. LDV 訊號強度與 SNR 因位置差距 30 倍以上

| Position | LDV rms | LDV abs_max | 推測 SNR |
|---|---|---|---|
| -0.4 block | 0.139 | 0.78 | 高(極佳) |
| +0.8 block | 0.076 | 0.21 | 高 |
| +0.0 block | 0.017 | 0.27 | 中 |
| +0.4 block | 0.018 | 0.27 | 中 |
| -0.8 block | **0.009** | 0.097 | **低**(危險) |

**物理解讀**:LDV 量的是紙板特定一點的速度。當 source 在 -0.4m 時,從牆面該點看出去的法向能量耦合最強(可能 LDV 對準的點正好在某板模態的反節點 antinode);+0.8m / -0.8m 比較 off-axis,訊號弱。

**對 Phase B 的衝擊**:-0.8m 位置 SNR 極低,可能是「最難位置」,也是 paper override 把 chirp 在這位置設成 2.41° 的原因(其他位置 chirp override 0.00–1.31°,-0.8 是最難)。Phase B baseline 應該預期 -0.8m 位置 MAE 大,然後 Phase C 看 denoising 能不能救回來。

### F5. Chirp 是下掃 down-sweep,寬頻 500 Hz–6 kHz

從 spectrogram 看(`spec_+0.0_block.png`):每個 chirp 從 ~6 kHz 線性掃下到 ~500 Hz,持續約 1.5 s,每 2 s 重複一次,共 6 次。

**對 Phase B 的衝擊**:
- Chirp 寬頻特性對 GCC-PHAT 有利(時頻局部化好,delay 估計精準)
- 但每 2s 重複 → cross-correlation 會出現 ambiguity peak(±2s lag 處有副峰)。要把 GCC search 範圍限制到 |τ| < 5 ms (max plausible TDoA)
- 可分離單一 chirp burst(取 0–2s)做 multi-trial 平均(Strategy C7 在這裡天然適用)

### F6. LDV 在 chirp 之間的「ringing」是板共振模態

`spec_-0.4_block.png` 顯示 LDV 在 chirp 沒在響時,200–2000 Hz 之間有明顯能量殘餘(類似衰減震盪)。這是被 chirp 激發的 cardboard panel 共振 modes。

**演算法層意涵**:
- 共振模態的衰減時間 ~100-500 ms,會在 chirp 之間造成 「source signal 結束但 LDV 還在響」的 mismatch
- GCC 會被這個 source-incoherent ringing 污染
- **可以利用這點**:在 chirp 結束後的 silence 段落估 LDV 的 ringing PSD,用來做 spectral subtraction

### F7. MIC_R 系統性比 MIC_L 弱 ~50% RMS

|  | MIC_L rms 中位數 | MIC_R rms 中位數 |
|---|---|---|
| Block | 0.0029 | 0.0015 |
| Unblock | 0.0040 | 0.0024 |

兩支麥克風增益不平衡。這對純 PHAT GCC 不影響(分母 |X·Y*| 已 normalize),**但對 coherence 估計**會略低估高頻段(SNR 較低)。

### F8. DC offset 一致存在(~5e-4),且 MIC_L/MIC_R DC 符號相反

MIC_L DC < 0,MIC_R DC > 0,LDV DC < 0。可能是 ADC 校準偏移或耦合電容方向。

**對 Phase B 的衝擊**:**zero-mean** 預處理是 mandatory,否則 PHAT 在 DC bin 會產生病態相位。

---

## 物理層補充推論 (一階分析)

### 為什麼 LDV 用了之後可以對抗 jammer?

LDV 量的是紙板**表面速度** v(t),而紙板被牆面入射壓力 p_in(t) 激發後,板內振動再從另一面以 Rayleigh 積分輻射出去。**Jammer 的聲波**雖然會撞到紙板背面,但因為:
1. Jammer 強度通常 ≪ source(在 SJR>0 dB 區段)
2. Jammer 入射角度 / 距離與 source 不同 → 在板上激發的 mode shape 不同
3. LDV 量的是某「一個點」,是該點上 source-mode 與 jammer-mode 的線性疊加

**關鍵**:當 source 是 driving force 時,在 LDV 點的 v 主要由 source-coupled mode 主導。**只要 LDV 對準 source-favored mode 的反節點**(F4 中 -0.4 位置看似如此),v 就攜帶幾乎純 source 相位。這提供**比麥克風更乾淨的 reference signal**,進而 cross-modal GCC 對 jammer 不敏感。

### 為什麼 PHAT 在這個 setup 可能反而傷害效能?

PHAT 把每個頻 bin 都白化成 |X·Y*|=1。但本資料的:
- 60Hz 諧波 bin 是純 hum(雜訊主導,無 source 訊號)→ 白化後雜訊相位被放大
- 7000+ Hz bin 沒有 LDV 訊號 → 白化後純放大噪音相位
- 真正有 source 訊號的 bin 反而被一視同仁

**SCOT(Smoothed Coherence Transform)**或 **maximum-likelihood weighting** 比 PHAT 更適合這場景:在 SNR 高的 bin 給高權重,SNR 低的 bin 抑制。

### 自由空間 τ_V(p) 假設多錯?

LDV 在 (0, 0.25),source 在 (x_s, 0)。自由空間 τ_V = √(x_s² + 0.25²) / 343。

但 source 到 LDV 的「真實」傳遞路徑包含:
1. Source → 紙板表面該點(空氣中)
2. 紙板內 bending wave 傳到 LDV 探點(板速度遠快於 343 m/s,所以這段時間相對小)

實際上 source 到 LDV 點的「**等效延遲**」可能略小於自由空間值(因為 board wave 比 air wave 快)。但因為 LDV 點離 source 路徑近(0.25m vs mic 的 2m),**absolute 誤差約 < 0.7 ms**,且這個誤差**對所有候選位置都接近一樣**(都是「LDV 路徑被低估」),會以 systematic bias 出現,**不影響 argmax 的相對位置**。

**結論**:自由空間假設應該夠用,Phase B baseline 不需要修。

---

## 對 Phase B 的具體建議

1. **資料路徑**:同時支援 0223 chirp 與 0224 speech 兩個 dataset,用統一 loader 把 (pos, cond, signal_type) → 三 channel arrays。
2. **Mandatory pre-processing**(都做,不算 strategy):
   - DC removal (subtract mean)
   - 60Hz comb notch (60, 120, 180, 240, 300 Hz),Q=30 IIR notch
3. **Time window**:
   - chirp:用第一個完整 chirp burst,e.g., t ∈ [0.2, 1.8] s
   - speech:取 t ∈ [5, 15] s (避開錄音頭尾)
4. **Search grid**:x ∈ [-1.0, 1.0] step 0.005 m, y ∈ [0, 0.5] step 0.05 m(因 source 已知 y=0,但 grid 給容錯)
5. **GCC search range**:|τ_VL|, |τ_VR| < 7 ms (對應 |x| < 2m 的 LDV-mic 距差)
6. **Symbol convention**(必須先驗):
   - τ_VM(p) = ||p - mic|| / c - ||p - ldv|| / c (mic 比 ldv 晚到為正)
   - R_VM(τ) = IFFT( X_V · conj(X_M) / |X_V · conj(X_M)| ) 在 τ 的峰即為 mic 比 ldv 晚到的時間量。**正負號要驗。**
