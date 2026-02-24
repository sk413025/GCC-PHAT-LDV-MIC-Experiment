# 全新寫作大綱與戰略轉向：從 PI-DNN 到「物理知識啟發」的幾何搜尋 (Physics-Informed Search)

**背景脈絡：** 之前的論文草稿（包括我們最近經歷五輪打磨的版本）完全建立在「物理知識啟發的神經網路 (PI-DNN)」之上。然而，根據最近的問題紀錄（Commits `adee861` 與 `305e478`），以及 `PAPER_THEORY_ALIGNMENT_AUDIT.md` 的審查結果，揭露了一個災難性的事實：**PI-DNN 根本無法泛化。** 它只不過是死記硬背了實驗用紙板的幾何形狀。目前真正有效、拯救了這項研究的解決方案，是一個**完全由物理法則驅動、具有解析幾何解的搜尋演算法：S3-joint**。

我們必須徹底推翻 PI-DNN 的敘事，並將論文的核心重新鎖定為 **"Physics-Driven Geometric Feature Extraction"**（物理驅動幾何特徵擷取，或稱為物理先驗引導搜尋）。對於一篇 4 頁的會議論文來說，這其實是一個**更強大的護城河**，因為它標榜「零訓練資料 (Zero-shot)」、「無黑盒子機器學習 (No black-box ML)」，並且提供了堅實的數學保證。

---

## 1. 核心敘事的大轉彎

**【舊有敘事 - 破綻百出】** 
「牆壁擾亂了聲音。我們使用一個 LDV 錨點加上一個神經網路 (PI-DNN) 來『神奇地』解開相位糾纏並找到 DoA。」
*(Reviewer 必定會攻擊：「你的 DNN 只是過度擬合了那間實驗室和那塊牆壁！」)*

**【全新敘事 - 堅不可摧】** 
「標準麥克風陣列之所以失敗，是因為牆壁引入了一個無法挽回的、**與角度相關的相位誤差 (角度空間錯位, Spatial Mismatch)**，這是任何同質陣列都無法克服的。融合 LDV 作為錨點，從物理根源上將聲源與牆壁的轉移函數解耦 (Decoupling)。然而，LDV 與麥克風的互相關運算依然深受大面積牆壁震動產生的 **同調陷阱 (Coherence Trap)** 所苦。我們不仰賴容易過度擬合的黑盒子 DNN，而是保留『物理知識啟發 (Physics-Informed)』的核心精神，提出一套 **Physics-Informed Geometric Search (物理啟發幾何搜尋，簡稱 PI-GS 或 S3-joint)** 演算法，透過強制服從剛性的 2D 空間幾何不變量，直接從源頭鎖定真實的 TDoA (Time-Difference-of-Arrival)！」

---

## 2. 修正後的段落邏輯大綱 (RUNDOWN)

### Section 1: Introduction (前言)
*   **鉤子 (The Hook)**: 隔牆 DoA 估測對純麥克風陣列來說是不可能的——不只是因為嚴重的訊號衰減 (Low SNR)，更是因為物理上根本性的**結構相位模糊 (Structural phase blurring)**。
*   **硬體解法 (The Hardware Tool)**: 介紹 LDV 作為一個純淨、捕捉結構震動的錨點。解釋它如何完美隔絕接收端房間內的干擾噪音（聲阻抗不匹配）。
*   **終極難題 (The Problem)**: 即使有了 LDV，大面積牆壁震動依然會產生「同調陷阱 (Coherence Trap)」——數以千計的假性相關峰值，會將真正的物理傳播峰值徹底掩埋。
*   **核心貢獻 (The Contribution - 升級轉向)**: 我們延續 Physics-Informed 的精神，但拋棄脆弱的神經網路，提出一套純粹由物理定律驅動的「聯合幾何搜尋框架 (Physics-Informed S3-joint Search)」。透過用嚴格的幾何先驗去約束跨模態 GCC-PHAT，我們**不需要任何機器學習訓練**，就能精準萃取出真實的 DoA。這大幅降低了讀者的認知負擔，因為標題的 "Physics-Informed" 依然強烈扣題！

### Section 2: System Model & The Physics of Failure (系統模型與麥克風敗局)
*   **(2.1) 為何 Mic-Mic 陣列注定失敗**: 放棄單階段傳播，改用**兩階段傳播模型** (聲源 $\rightarrow$ 牆面點 $v \rightarrow$ 麥克風)。數學證明：因為兩顆麥克風看牆壁的入射角不同 ($\theta_L \neq \theta_R$)，會產生不可逆的角度相位誤差 $\phi_{\text{wall}}(\theta_L) - \phi_{\text{wall}}(\theta_R) \neq 0$。這是演算法在無限大 SNR 下依然會崩潰的物理死穴。
*   **(2.2) LDV 的物理優勢**: 解釋 LDV 訊號 $X_V$ 與麥克風訊號 $X_m$ 共享完全相同的「聲源到牆壁轉移函數 $U_v$」。這為後續的完美代數抵銷埋下伏筆。（保留 Jammer 被硬體隔離的論述）。

### Section 3: Proposed Physics-Driven Geometric Search (取代原本的 PI-DNN 段落)
*   **(3.1) 跨模態 GCC-PHAT 的完美抵銷**: 推導 LDV-Mic 的互功率頻譜。證明 $U_v \cdot U^*_v = |U_v|^2$，這會在每對配對中**無條件地抵銷複雜的牆面相位**。補述 LDV 測量速度所導致的 $-1/(4f)$ 相位偏移。
*   **(3.2) 同調陷阱與雷利模糊 (The Rayleigh Blur)**: 解釋剛修復的 Eq. 6。真正的峰值位於負延遲區間 ($-\frac{d(v,m)}{c}$)，但雷利積分在整面牆上的作用會產生海量的「結構多路徑 (Structural multipath)」。盲目挑選峰值 (Blind peak picking) 絕對會選錯。
*   **(3.3) S3-Joint 幾何搜尋 (用來取代 DNN)**:
    *   解釋幾何不變量：$\Delta\tau_{\text{theory}} = \tau_{VR} - \tau_{VL} = \frac{d(v,R) - d(v,L)}{c}$。
    *   指出前面提到的 $-1/(4f)$ 速度相位項，會在這個差分域 (Differential domain) 中完美抵銷。
    *   正式定義 S3-joint 評分函數：我們不挑最高峰，而是掃描候選的 $\tau_{VL}$，並進行聯合評分：$\text{Score} = \text{GCC}_{VL}(\tau_{VL}) + \text{GCC}_{VR}(\tau_{VL} + \Delta\tau_{\text{theory}})$。
    *   **強力宣告**：此方法**無須訓練 (Zero training)**，且明確具備抵抗「同調陷阱」的物理強健性。

### Section 4 & 5: Experiments & Results (實驗與結果)
*   **實驗設定 (Setup)**: 強調查驗環境的嚴苛性——近場幾何 (距離牆壁僅 25cm，而麥克風間距達 1.4m)，使用紙板作為障礙物。
*   **評估指標 (Metrics)**: 將 DNN 訓練曲線替換為 S3-joint 的神準確度（最大誤差僅 2.33°）。保留 $P_{CW}$ ("Confidently Wrong"，自信地犯錯) 指標，用來對比傳統 Mic-Mic 陣列的崩潰與 S3-joint 的無懈可擊。

---

## 3. 重寫時的「認知降載」策略 (Cognitive Load Reduction)
1.  **轉換（而非屠殺）PI 行話 (Pivot the PI Jargon)**: 刪除 "DNN", "MLP", "Epochs", "Training Loss" 等黑盒子字眼。但**全力保留並強化 "Physics-Informed"**。我們現在說的 PI，指的是「將嚴格的物理幾何公式 ($\Delta\tau_{\text{theory}}$) 直接注入 (Informed) 到搜尋演算法中」。
2.  **標榜「免訓練」的超級優勢**: 對於那些已經看膩了「模型過度擬合 (Overfitting) 特定房間」的審稿人來說，一個能通過實體測試、Zero-shot 的 **Physics-Informed Algorithm** 簡直是一口清新的空氣。
3.  **兩階段模型的清晰度**: 在 Section 2 實作麥克風的「兩階段訊號模型」 (聲源 $\rightarrow$ 牆壁 $\rightarrow$ 麥克風) 是讓數學式完美抵銷的唯一正解。

## 下一步行動
1.  從 LaTeX 原始碼中清除所有關於 PI-DNN 的殘影。
2.  重寫 Section 3，用數學方程式正式定義 S3-joint 搜尋演算法（直接寫出公式即可，不一定要用 Algorithm pseudo-code）。
3.  對齊 Section 2，植入在 `THEORY_ALIGNMENT_AUDIT.md` 中詳細論證的「兩階段傳播模型 (Two-Stage propagation model)」。
