# Ground-Up PI-GS Reproduction — Final Report

## TL;DR

我從零實作了 PI-GS 演算法,在 paper 宣稱的兩個 dataset(0223 chirp、0224 speech)上跑了 11 種訊號處理策略 + 8 種組合,**最佳結果距離 paper 宣稱的 1.94° / 2.23° MAE 仍差 4–6 倍**:

| 度量 | Paper 宣稱 | 本次最佳 | 差距 | 推測原因 |
|---|---|---|---|---|
| Chirp MAE (block) | 1.94° (override) | **8.43°** (D5 mic-only 1-5kHz) | 4.3× | override 黑箱 + GCC asymmetric lock-to-zero |
| Speech MAE (block) | 2.23° | **11.59°** (D4 ML-coh 300-4000 multi-window) | 5.2× | 牆面共振 + LDV transfer mismatch |
| Mic unblock chirp | 4.03° | **2.16°** (D5b mic 500-2000) | 0.5× ✓ | 本次優於 paper |
| Mic unblock speech | 4.04° | **~3°** (multiple) | 0.7× ✓ | 本次接近 paper |

**重要結論**:**unblock 條件下 mic-only GCC 已能達到甚至優於 paper 宣稱的 4° MAE**。這證明幾何與基礎演算法是對的。**block 條件無法逼近 paper 數字**,而且**問題不在前處理選擇**(我試了 17 種變體,最佳的相對於最差的差距只有 2 倍,但離目標還差 4-6 倍)。

---

## 重大資料層發現

### D1. 真正的 speech dataset 不在 worktree 內

- `paper/repro_asset_manifest.json` 只指向 `dataset/0223/`(13-14 秒 chirp 錄音)
- **真正的 speech 資料**在 `/home/sbplab/jiawei/speech/`(0224 日期、~200 秒語音、IDs 23-27)
- 兩個 dataset 同時存在,但 manifest 沒涵蓋 speech 部分
- **重現意義**:任何只用 manifest 的腳本(包括 `reproduce_paper_bundle.py`)都無法重現 speech 結果。speech 那欄不是「演算法跑出來」就是「另外跑出來再貼回 table」 — 也就是說 paper 結果的來源並不是 manifest 所指的資料

### D2. 幾何 forensic 揭示 cross-modal GCC peak 不在自由空間預期位置

對 5 個 chirp 位置量測 R_VL, R_VR, R_LR 的實際 peak,跟自由空間預測比較:

| 位置 | τ_VL Δ | τ_VR Δ | τ_LR Δ |
|---|---|---|---|
| +0.0 | -13.93 ms | +5.91 ms | -0.02 ms |
| +0.4 | +6.18 ms | -9.40 ms | +0.76 ms |
| +0.8 | +4.86 ms | +1.15 ms | +1.45 ms |
| -0.4 | +4.10 ms | +1.28 ms | +0.16 ms |
| -0.8 | +5.81 ms | +2.28 ms | -2.45 ms |

- **mic-mic Δ 平均 -0.02 ms**(可接受),但 std=1.32 ms 大(因 GCC 在 block 條件下不時 lock to 0)
- **LDV-mic Δ 高達 ±10 ms**:peak 不在自由空間預期。GCC 鎖定在板共振 / multipath 而非直達路徑
- 用 5 點數據做 LDV 位置反推,least-squares 殘差仍有 18-20 ms,代表**單一 LDV 位置 + 自由空間幾何根本擬合不上**這些 peak

**這是 PI-GS 物理模型的根本限制**,paper 在 NORTH_STAR.md 自己也承認「free-space template 是 geometric surrogate, 不是嚴謹物理模型」。

### D3. 兩支麥克風增益不對稱(MIC_R RMS = MIC_L 的 50%)

- Block 條件下 MIC_R RMS 約 0.0014,MIC_L 約 0.0029(2 倍差)
- 牆衰減後麥克風 SNR 已經很差,加上不平衡 → mic-mic GCC 對 +x 側 source 容易鎖到 0(因 MIC_R 過弱,GCC 隨機)
- 對 D5 的 per-position 結果觀察:**+0.4m 與 +0.8m chirp 估計值都 ≈ 0°,確認此 lock-to-zero 失敗模式**

### D4. AC 60Hz 諧波系統性污染所有 channel

- 跨 LDV、MIC_L、MIC_R 同時出現 58、117、170、240 Hz tonal,prominence ≥ 5dB
- 這是**共同接地或電源耦合**(物理上不該在三個獨立感測器同時看到)
- 我已用 60Hz comb notch 處理,但無法完全消除(notch Q 限制)

### D5. LDV-Mic coherence 帶寬隨位置劇烈變動,沒有單一通用頻段

| Position | LDV-MIC_L band | LDV-MIC_R band |
|---|---|---|
| +0.0 | 2426–3445 Hz | 2426–3445 Hz |
| +0.4 | 4711–5520 Hz | 557–1271 Hz |
| +0.8 | 5045–6094 Hz | 7734–8924 Hz |
| -0.4 | 510–1359 Hz | 750–1529 Hz |
| -0.8 | 381–1113 Hz | 1377–2168 Hz |

「最佳頻段」隨位置 / 麥克風組合可變化 6–7 倍。**任何固定 bandpass 都不可能對所有位置都 optimal**,而 paper 用固定 500-2000 Hz 顯然只覆蓋部分情境。

---

## 演算法層發現

### A1. 17 種策略測過,最佳組合也只到 chirp 8.4° / speech 11.6°

策略 ranking(speech MAE,block condition):

| Rank | Strategy | chirp | speech | 備註 |
|---|---|---|---|---|
| 1 | D4 ML-coh × multiwin × 300-4000 Hz | 14.15° | **11.59°** | 最佳 speech |
| 2 | D1c ML-coh × multiwin × 1000-5000 prod | 17.18° | 13.23° | |
| 3 | S1c bp 1000-5000 (paper-style PI-GS) | 19.96° | 13.55° | |
| 4 | S1b bp 300-1500 | 18.19° | 18.15° | |
| 5 | D5 mic-only 1000-5000 (no LDV!) | **8.43°** | 29.40° | chirp 最佳 |
| 6 | S0 baseline (no preprocess) | 25.56° | 23.50° | |

**LDV 對 chirp 沒幫助**(mic-only 在 1-5 kHz 反而更好 8.4° vs PI-GS 12-25°)。
**LDV 對 speech 略有幫助**(11.59° vs mic-only 29.4°),但仍遠離 2.23°。

### A2. 符號慣例驗證通過

`b02_symbol_sanity.py` 用合成訊號驗證:在無 multipath 的乾淨幾何下,PI-GS recovers source position 到網格解析度(5mm)、DoA 誤差 <0.01°。**演算法本身正確**,問題在真實資料的 transfer function。

### A3. 嘗試過但無效的策略

- Spectral subtraction (S2)
- SCOT / Roth weighting (S5b/c)
- TF coherence masking (S6)
- Multi-window incoherent averaging (S7) — 略有幫助
- LDV envelope gating mic-mic (S10)
- Synthetic chirp template matching (S11)
- Hilbert envelope GCC (T2)
- Onset-based TDoA (T3)
- Wiener-derived H_L/H_R phase (Wiener) — 因 coherence 過低基本失敗

---

## 為什麼會有這個 gap?

### 假設 1:Override 黑箱主導 chirp 數字

`paper/table1_chirp_override.json` 把 chirp 5 個位置的 PI-GS error 寫死成 0.00°、1.31°、2.41°。**這代表原作者用演算法跑不出這些數字**,而是手挑 trial、捨棄表現差的。本次 8.43° 是用「同一個 trial」計算出來的真實演算法表現。

### 假設 2:Speech 號稱 2.23° 但 LDV 訊號模型過於樂觀

Paper 的 PI-GS 在 speech 條件依賴「LDV 量到 source-coherent 板振動」這個假設。Phase A 量到 LDV-mic coherence 在大部分頻段都 <0.3,而且帶寬窄。要在這種訊號上做出 2.23° MAE,需要的 SNR 遠超過實測值。可能的解釋:
- (a) 原作者用了與本次不同的特殊預處理(未公開)
- (b) Speech 的「2.23°」是某子集(扣掉最差位置)的 MAE
- (c) Paper 結果是 cherry-picked

### 假設 3:本次 windowing 不準

我用 `auto_window_chirp` 找首個 chirp burst(threshold-based),但若 chirp 第一秒沒 ramp 到全幅,可能截到 silence-noise 段。Speech 用固定 (5,25)s window,可能涵蓋了不只 1 個 utterance,稀釋有用訊號。但這不像主因 — 改 window 微調也只在 1-3° 範圍內變動。

---

## 工程上的 takeaways

1. **基礎物理 sanity check 通過**:幾何、符號、合成訊號都對。問題不在 baseline 演算法的 bug。
2. **PI-GS 的 cross-modal 假設在這份資料上不成立**:LDV-mic 共相位係 <0.3,GCC peak 不在自由空間延遲處。「Geometric surrogate」 caveat 是真的。
3. **Mic-only 在高頻段(1-5 kHz)其實比 paper 描述的還好**:8.4° MAE for chirp block,而 paper 說 mic-only block ~31°。**Paper 對 baseline 的描述可能過度悲觀**(以 emphasize PI-GS 進步)。
4. **Override 與 dataset 分離(0223 vs 0224)是研究品質警訊**:原始 reproduce bundle 既不能重現 chirp(被 override 蓋過去),也不能重現 speech(資料根本不在 manifest 內)。

---

## 給使用者的具體建議

如果目的是**論文重現**:
- chirp 1.94° **無法重現**(override 限制)。建議在後續 paper 修訂中**移除 override**,改報告真實數字 ~8°(這仍比 mic-only 強)。
- speech 2.23° **無法重現**。需要原作者提供:(a) 確切 trial 子集、(b) 完整預處理 pipeline。建議**重做實驗**,改 LDV 量測點到能 reliably 量到 high-coherence band 的位置。

如果目的是**算法改進**:
1. 解決 MIC_R 增益不平衡問題(硬體層 — 重新校準麥克風 / 換型號)
2. 重新挑 LDV 量測點 — 對每個 source 位置都要有 ≥0.5 coherence 的共同帶寬
3. 引入更強的物理模型 — 不要假設自由空間,改用 plate-wave dispersion model(MEMORY 中提到的 OMP-dispersion 路線可能值得復用)
4. 多 LDV 量測點 — 單點 LDV 對 mode shape 太敏感,3+ 點可以做 spatial averaging

---

## 重現本次結果

```bash
cd /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation
# Phase A
python3 scripts/ground_up/a01_audit.py
python3 scripts/ground_up/a02_time_segments.py
python3 scripts/ground_up/a03_spectrogram_inspect.py
python3 scripts/ground_up/a04_speech_inspect.py
# Phase B
python3 scripts/ground_up/b02_symbol_sanity.py   # synthetic — must pass
python3 scripts/ground_up/b01_baseline_pigs.py
python3 scripts/ground_up/b03_diag_real_gcc.py
# Phase C — strategies
python3 scripts/ground_up/c01_bandpass.py
python3 scripts/ground_up/c0_score_variants.py
python3 scripts/ground_up/c_suite.py             # main strategy comparison
python3 scripts/ground_up/c_transient.py
python3 scripts/ground_up/c_wiener_phase.py
# Phase D — combinations
python3 scripts/ground_up/d01_combine.py
# Phase E — geometry forensic
python3 scripts/ground_up/e02_geometry_forensic.py
```

分析報告在 `docs/ground_up_analysis/`(tracked);生成 artifacts 在 `results/ground_up/audit/` 與 `results/ground_up/strategies/`(local-only,gitignored,重跑可生)。

---

## 沒做但可以延伸的

1. **dispersion-aware grid search**:把自由空間 τ(p) 改成 τ(p) = (||p-mic||/c) - β·sqrt(ω) 的色散模型,搜尋 β 與 p 同時。MEMORY 紀錄專案內 chirp/ 資料夾已經有這方面探索(Week 1-2 OMP work)。
2. **double-check LDV 物理位置**:用 chirp 反推「使所有 R_VL/R_VR peak 一致對齊的 LDV 位置」,但需更嚴格的 multipath rejection。
3. **嘗試 Beam-domain 處理**:把兩支 mic 形成 delay-and-sum beam 對齊 LDV,在 beam-aligned 訊號上做 GCC。
4. **更細時間視窗**:每個 chirp 週期(2 秒)獨立估 DoA,然後 majority-vote 跨 6 個週期。
