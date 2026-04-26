# 完整寫作戰略大綱：漏斗式敘事引導法 (The Funnel Narrative)

這份大綱的最高指導原則是**「認知降載 (Cognitive Load Reduction)」**與**「模組化隔離 (Modular Isolation)」**。
我們要在每一段落精準狙殺審稿人腦中的一個疑慮，並在處理完後**永遠不再提起它**。這會讓原本極度複雜的多重物理干擾（衰減、牆壁相位扭曲、Jammer、雷利多路徑），變成一條極其乾淨、毫無負擔的單向邏輯鏈，最終將讀者無縫引導至我們的終極解法：**PI-GS**。

---

## 總體「模組化隔離」心法
1.  **Sec 2.1 處死純麥克風陣列後，後面再也不提它。** (除了 Sec 5 拿出來鞭屍對比)
2.  **Sec 2.2 用物理阻抗殺死 Jammer 後，Sec 3.1 和 3.2 的數學推導絕對不混入雜訊干擾。** (讓推導維持在無限大 SNR 的純淨狀態，降載 50%)
3.  **Sec 3.1 證明完牆壁局部相位完美抵銷 ($|U_v|^2$) 後，Sec 3.2 就不再提局部介質問題。**
4.  **Sec 3.2 專心對抗最後也是最大的魔王：大面積牆壁產生的「雷利模糊 (Rayleigh Blur) / 同調陷阱 (Coherence Trap)」。**

---

## 逐章節邏輯鋪陳與認知降載策略

### Section 1: Introduction (漏斗開口 - 痛點與工具)
*   **敘事目標**：打破「穿牆只是小聲一點」的迷思，點出真正的物理死穴。
*   **段落流佈**：
    1.  **場景導入**：搜救/監聽場景。大家都在用麥克風陣列算 TDoA。
    2.  **指出痛點 (Cognitive Hook)**：穿牆 DoA 無法靠「提高麥克風靈敏度」解決，因為牆壁會導致嚴重的「空間相位錯位 (Spatial Mismatch)」。
    3.  **引入工具 (降低抗拒心理)**：介紹 LDV。但要立刻強調：我們不是要用天價設備取代麥克風，LDV 只是一個「實體的跨介質錨點 (Anchor)」。
    4.  **拋出主菜 (大轉折)**：即使有 LDV，牆壁依然會產生可怕的「同調陷阱 (Coherence Trap)」。我們不使用會產生過度擬合 (Overfitting) 的黑盒子神經網路，而是提出一套 Zero-shot (免訓練) 的 **Physics-Informed Geometric Search (PI-GS)** 演算法，直接注射物理先驗來解題。

### Section 2: System Model (第一層過濾 - 殺死 Jammer)
*   **敘事目標**：用最簡單直白的物理常識，將「噪音干擾」這個最煩人的變數從後續的幾何推導中徹底抹除。
*   **段落流佈**：
    1.  **2.1 麥克風的物理死刑 (Two-Stage Propagation)**：
        *   不再用「一團黑箱濾波器」唬弄讀者。明確定義：聲源 $\rightarrow$ 打到牆上 $\rightarrow$ 重新輻射到左右麥克風。
        *   **核心公式**：$\Delta\Phi_{\text{err}} \neq 0$。因為左右麥克風看牆壁的入射角不同，相位一定不同。這是純麥克風陣列注定失敗的鐵證。
    2.  **2.2 LDV 聲阻抗護罩 (Impedance Mismatch)**：
        *   **認知降載**：用一句話解釋完 Jammer 的消除——「空氣與牆壁的聲阻抗差異極大，室內的 Jammer 聲音撞到牆壁會產生 $>99\%$ 的反射」。
        *   結論：LDV 測量的是純淨的 Target 震動，Jammer 被物理硬體天然隔離。後面的數學推導**不用再考慮 Jammer**。

### Section 3: Proposed Physics-Informed Geometric Search (第二層與終極過濾)
*   **敘事目標**：用極簡的數學證明完美抵銷，並順勢帶出最後的幾何解法。
*   **段落流佈**：
    1.  **3.1 跨模態相位完美抵銷**：
        *   **認知降載**：大方承認我們無法還原丟失的振幅 (Magnitude)，所以我們直接用 GCC-PHAT 把它丟掉，只看相位。
        *   **核心公式**：證明 LDV 與 Mic 共享同一個牆點的傳遞函數 $U_v$。相乘時 $U_v \cdot U_v^* = |U_v|^2$。這是一個多麼優美的數學結果！複雜的牆壁介質相位**無條件完美抵銷**。
        *   **引出終極大魔王**：我們雖然抵銷了 LDV 那一「點」的相位，但牆壁「其他地方」的震動傳到麥克風（雷利積分 Eq. 6），會產生無數的假峰值。我們稱之為 **Coherence Trap (同調陷阱)**。
    2.  **3.2 物理啟發的幾何搜尋 (PI-GS)**：
        *   **認知降載**：不再提任何 MLP, Neural Network, Training 的字眼。
        *   **解法**：為了解決假峰值，我們把剛性的幾何理論差 $\Delta\tau_{\text{theory}}$ 直接注射進搜尋演算法。
        *   **核心公式**：聯合目標函數 (Eq. 10)。我們不是盲目挑選最高峰，而是強制在 2D 空間中尋找讓左右耳峰值能「建設性重合 (Constructive interference)」的物理座標。假峰值只會被互相抵銷。

---

## Section 4 & Section 5: Experiments, Results & Discussion 攻防戰

這個章節的使命，是用幾張極度直觀、不用動腦思考的圖表，暴力地把我們在 Sec 1~3 吹的牛皮，全部化為冰冷的實驗數據。

### Section 4: Experimental Setup (實驗設定)
*   **認知降載策略**：
    *   **強調「嚴苛性」**：近場 25cm、寬間距麥克風 (1.4m)、紙板 (TL ≈ 10-20 dB)。越嚴苛的幾何形狀越能凸顯物理推論的正確性與 PI-GS (免訓練) 的強度。
    *   **定義兩種信號源 (Chirp vs. Speech)**：這是為了證明 PI-GS 不依賴信號特徵 (Robust by design)。不管是寬頻掃頻還是男童語音，只要是同一個位置，都能算得準。
    *   **定義殘酷指標 (嚴重的結構性誤差)**：我們不只要比誤差，還要特別標記傳統陣列如何被牆壁同調陷阱鎖定。這是在告訴讀者：麥克風陣列在遭受遮擋時，**甚至連提供隨機假答案的能力都沒有**，它會直接自信地鎖定在錯誤的結構輻射點上，在訊號處理物理層面積底崩潰。

### Section 5: Results and Discussion (圖表與論証編排)

為了降低認知負擔，我們必須捨棄所有雜亂的曲線圖與合成變數，採用極致純淨的 **「一表一圖 (1 Table, 1 Figure)」** 鐵證架構，循序漸進地收網。

#### 【鐵證 1】Table 1: Systemic Bias Decoupling Matrix (跨信號與遮擋矩陣)
*   **表格內容**：將 Chirp 與 LibriSpeech 語音的 Block 與 Unblock 數據合併為一個包含 4 個維度的矩陣表 (Chirp Unblock/Block + Speech Unblock/Block)。全部統一使用 **MAE** 指標。
*   **要傳達的資訊與盲點防禦**：
    *   **無遮擋 (Unblock) 基準線**：展示純麥克風在無遮擋時皆能達到 $1.8^\circ \sim 1.97^\circ$ 的精準度，證明房間本身無共鳴干擾，消除審稿人對實驗設定的質疑。
    *   **主動聲明 LDV 數據缺失**：在 Speech Unblock 中，我們故意將 LDV-Mic 標示為 `--` (N/A)。因為無紙板時 LDV 無法/無須運作，這反向加強了實驗的物理合理性與學術誠信。
    *   **同調陷阱 (Coherence Trap)**：無論是 Chirp 還是 Speech，只要切換到 Block 條件，傳統麥克風陣列立即產生高達 $12^\circ \sim 14^\circ$ 的嚴重偏差。
    *   **PI-GS 無條件救援**：PI-GS 完全無視信號種類的差異，以純物理特徵將 Block 狀態下的 MAE 獨自壓抑回 $1.48^\circ$。
*   **認知降載 Caption**：
    *   *“Table 1: Cross-condition isolation of the Coherence Trap using Mean Absolute Error (MAE)... The unblocked controls strictly lack the LDV proxy (--) as the barrier physically drives the fusion...”*

#### 【鐵證 2】Figure 2: 真實空間頻譜 (True Spatial Score Curve)
*   **圖片內容設定**：替換掉原本心虛的 "Simulated MVDR"。繪製真實 `0224-block` 數據的 1D 空間得分曲線 (S3-joint score vs Mic-Mic GCC-PHAT)。
*   **要傳達的視覺衝擊**：用真實的物理波峰直接說話。讓讀者親眼看見 Mic-only 的波峰被「同調陷阱」死死吸在牆壁中央 (0度)，而 LDV-Mic S3-joint 則成功在真正的目標角度產生建設性干涉波峰。
*   **認知降載 Caption**：
    *   *“Figure 2: True spatial objective functions under blocked conditions. The conventional Mic-Mic array is violently hijacked by the wall's structural multipath, locking onto the geometric center. Our PI-GS mathematically unwraps this distortion, synthesizing a sharp peak at the true target azimuth.”*

---

## 🛡️ 審稿人防禦戰略 (Reviewer Defense Strategies)

在 Section 5 與 Section 6 中，為了避免被 Reviewer (特別是 Signal Processing 領域的挑剔學者) 攻擊，我們必須在敘事中**主動排雷**：

1.  **防禦「信號依賴性 (Signal Dependency)」攻擊**：
    *   **攻擊法**：「你們的演算法是不是只對 Chirp 有效？」
    *   **防禦法**：這就是為什麼我們必須**同時**放 Table 1 (Chirp, 寬頻) 與 Table 2 (Speech, 複雜語音)。我們主動在內文指出：因為 PI-GS 是純幾何驅動，所以它的結果與信號特徵完全脫鉤（Robust by design）。
2.  **防禦「為何選紙板 (Material Choice)」攻擊**：
    *   **攻擊法**：「紙板穿透損耗 (TL) 只有 10-20dB，為什麼不測金屬板？」
    *   **防禦法**：在 Section 6 主動說明，**紙板其實是最刁鑽的「中間地帶」**。金屬板會讓信號完全消失（這時麥克風輸出雜訊，很合理）；但紙板的 TL 恰好足以扭曲相位，卻又保留了足夠的能量，導致 GCC-PHAT 會充滿自信地給出一個**錯誤的偽峰 (高 PSR 結構輻射點)**。我們的 PI-GS 能解這個最棘手的陷阱，才是真功夫。

⚠️ **【最高風險警告：必須補齊的實驗】**
在查閱 `PAPER_NARRATIVE_v2.md` 時，我發現了一個會給 Reviewer **致命攻擊口實**的漏洞：
*   **問題**：在 `0224 Speech Unblock` 的實驗紀錄中，缺了 `+0.4m` 的數據 (因為缺少 MIC_R)。導致 Unblock 只有 4 個位置，而 Block 卻有 5 個。
*   **Reviewer 視角**：「為什麼唯獨漏了 +0.4m？是不是你們在 +0.4m 的基線表現太差，所以**選擇性隱瞞 (Cherry-picking)**？」
*   **解決方案**：強烈建議花 10 分鐘，將 `unblock-2 (+0.4m)` 的 210s 語音錄音**重跑一次**。讓三張 Tables 的實驗維度完全對稱 (5 vs 5 vs 5)，這是消滅 Reviewer 懷疑的最低成本做法。

---

### 【結語：為什麼這樣寫會贏？】
這種漏斗式寫法，每一頁都在為下一頁鋪路，每一張結果圖表都能在前面的理論公式中找到對應的**背書**：
1. **看到 Mic-Mic 無噪音也失效** 👉 讀者會回想起前頁的 Eq. 2 ($\Delta\Phi_{\text{err}}$)。
2. **看到 S3-joint 不依賴訓練也能贏** 👉 讀者會回想起前頁的 $\Delta\tau_{\text{theory}}$ 剛性約束。

這種首尾呼應、毫無廢話的寫法，能給予審稿人極大的**「掌控感與安全感」**。只要能降低他們的認知耗損，被接受的機率就會呈指數級上升。
