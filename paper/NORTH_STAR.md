# 北極星（North Star）：最終目標情境、主張邊界、與「認知降載」推導寫作策略

這份文件是之後所有論文改稿的**主軸**。我們希望做到兩件事：

1) 先把「我們到底在解哪個情境」講死，避免審稿人把我們當成在寫普適物理定理。  
2) 再把「推導與敘事怎麼寫才最省腦、最不容易被抓漏洞」講死，降低讀者與審稿人的認知負擔。

一句話定位：本論文是**工程取向、條件式成立（condition-based）**的方法論。只要在我們的 testbed 與相近條件下，存在可量測的 *LDV-coherent* 成分（LDV 與麥克風之間仍有一段可用的相干 lag 結構），PI-GS 就能穩定工作；我們**不**宣稱普遍的「牆體反演／物理解耦保證」。

---

## 1) 我們最終想達成的目標與情境（Target Scenario）

### 1.1 任務定義（我們真正要解的問題）
- **Through-barrier speech DoA / tracking**：牆後（障礙物後）有一個主導的語音目標源，要估 DoA（或等價的 2D 位置投影/追蹤）。
- **接收端室內有強干擾**：在麥克風那一側的房間有 jammer（babble/noise），而且強到足以拖垮 mic-only。
- **近場牆體效應主導**：牆不是「多一個 LTI 濾波器」而已；它會像一個**大面積二次輻射體**，造成 correlation landscape 被 multipath/結構反射的假峰值淹沒。

### 1.2 感測器設定（我們靠什麼破局）
- **Mic L / Mic R**：接收端兩支麥克風。
- **LDV**：量牆面某個固定點 `r_ldv` 的結構振動（點量測）。

### 1.3 我們要打爆的 baseline 失敗模式（The Villain）
讀者要先吃進去一個直覺：**牆後的 mic 不是在聽「目標源的一個乾淨延遲拷貝」**，而是在聽：
- 整片牆面不同點的再輻射疊加（相位/延遲分佈很亂），再加上室內 jammer。

結果就是：
- mic–mic 的 GCC(-PHAT) 會出現大量很大的 spurious peaks（coherence trap / multipath blur）。
- peak 很容易「自信地錯」（鎖到邊緣繞射/結構反射，而不是目標幾何）。

### 1.4 我們依賴的成功條件（不是魔法，是條件）
- **LDV 是 anchor**：在我們 setup 裡，jammer 能量通常更直接支配 mic pressure；而耦合到牆振動的效率相對弱，讓 LDV 更可能保留可用的目標相干成分。
- 在選定的 analysis band + time-frequency aggregation 後，**LDV–Mic 仍保有可用的 LDV-coherent lag 結構**（即使被大量 residual 污染）。

---

## 2) 主張邊界（Claims Boundary）：我們要主張什麼、不主張什麼

### 2.1 我們明確主張（We DO claim）
- PI-GS 是一個 **zero-shot（不訓練）** 的幾何搜尋方法，在上述情境下可穩定工作。
- LDV–Mic 的 cross-modal GCC-PHAT 在 *LDV-coherent component* 不小時，能提供**有用的空間線索**。
- 兩路（LDV–MicL / LDV–MicR）加上 **joint geometric consistency** 能降低被 spurious peak 捕獲的機率。
- 在我們的 testbed 上，能達到論文報告的誤差/抗干擾結果（以實驗為主張核心）。

### 2.2 我們明確不主張（We DO NOT claim）
- 不宣稱普適的「牆體反演／解耦」或對任意材質/幾何都有保證的物理定理。
- 不宣稱 LDV 對干擾「完全免疫／完美隔離」。
- 不宣稱從 Rayleigh re-radiation（面源積分）完整推導出一個對所有牆都精準的 `τ(p)` closed-form forward model。

### 2.3 讀者契約（Reader Contract，一句話）
PI-GS 是**工程方法**：它「physics-informed」在 **異質感測 + 幾何一致性約束**，而不是在於一個普適的牆體物理解析解。

---

## 3) 推導主線（最小、但對齊故事且站得住的 physics）

這一段的寫法要遵守：**先定性 → 再定量 → 最後一句白話定調**。

### 3.1 先定性：為什麼 mic–mic 會輸（先把大反派塑造好）
- mic 在牆後聽到的是「牆面多點再輻射疊加」而不是「乾淨延遲」。
- 因此 mic–mic correlation 會被 coherence trap 支配，產生大量假峰值。
- 所以 blocked + jammer 時 mic-only 會 collapse（鎖錯峰）。

### 3.2 再定性：為什麼 LDV 能當 anchor（但不要寫成神）
- LDV 是**牆面單點量測**，不是面積分。
- 在我們 setup 中，jammer 對 mic pressure 的支配性更強；相對地，耦合進牆振動常常較弱，因此 LDV 常能提供較乾淨的參考。

### 3.3 再定量：只用一個最小分解就夠（coherent + rest）
對每支麥克風，在 TF 領域用最小分解：
- `X_m(f) = (LDV-coherent component) + (everything else)`

白話解釋：
- coherent component：跟 LDV spot 鄰域同源/同相干、能反映幾何的那一部分。
- everything else：牆上其他點的再輻射 + 室內干擾 + 不相干/多路徑（coherence trap）。

這個分解支持我們的條件式主張：
- PHAT-GCC **不保證**消牆；但只要 coherent component 不小，它就能提供可用的 lag 結構。

### 3.4 最後定調：PI-GS 的 disambiguation 來源不是「牆相消」，是「兩路 joint consistency」
PI-GS 真正做的事（用白話講）：
- 算兩條 cross-modal 特徵：`R_VL(τ)`、`R_VR(τ)`（PHAT-GCC）。
- 用 travel-time template（free-space surrogate）建 `τ_Vm(p)=τ_m(p)-τ_V(p)` 當幾何約束。
- 用 joint objective 找同時在兩路都一致的 `p`（objective 用 `|R|` 或 `|R|^2`，避免複數相位慣例的坑）。

一句話收斂：
我們不是宣稱「牆被數學消掉」，我們是用 **“兩路同時對齊”** 讓大量假峰值更難同時騙過系統。

---

## 4) 全文「認知降載」策略（怎麼寫才最省審稿人腦）

### 4.1 一個小節只允許一個 mental model
- System model：牆=大面積二次輻射體 → mic–mic 被 coherence trap 淹沒。
- LDV：單點 anchor → 更可能保留可用相干訊號。
- PI-GS：兩路特徵 + 幾何一致性 → 排掉不一致假峰。

### 4.2 假設要放在 Method 一開始（避免 reviewer whiplash）
Method 一開始要主動講清楚：
- single dominant source（至少是我們 evaluation 的目標假設）。
- 不做 universal barrier inversion。
- `τ(p)` 是 template / surrogate（工程約束），不是完整 forward model。
- 成功條件是 *LDV-coherent component* 在 aggregation 後仍可用。

### 4.3 數學符號與實作要 100% 對齊（不留暗坑）
- 文字說 correlation 可能是複數 → objective 就不能直接加 `R`，要加 `|R|` 或明確定義 `Re{R}` 的操作。
- 不要把「負延遲」講成物理必然；把它當成 sign convention（或說我們搜全 τ）。
- 不要在 caption/討論段落偷偷升級主張（caption 的強度不能超過 main text）。

### 4.4 避免觸發審稿人防衛的過度保證詞
盡量不用：
- guarantee / perfectly / immune / decouple / strictly / completely independent

改用：
- 在我們 setup / 在觀測到的 coherence 條件下 / can / tends to / reduces sensitivity / engineering surrogate

### 4.5 圖表承擔直覺，不要承擔新主張
Caption 只做三件事：
- 再講一次 villain（coherence trap）。
- 給讀者直覺：LDV anchor 的對比效果。
- 不要突然引入新的“物理必然句”（例如「>99% reflection」「完全隔離」這種）。

---

## 5) 審稿人常問問題（提前防禦，低成本回答模板）

### Q1：你們用 free-space travel-time template，但 mic 是牆再輻射，合理嗎？
答法核心：我們把 `τ_Vm(p)` 當作**幾何動機的 template constraint**。PI-GS 成功條件不是「template = 精確 forward model」，而是：在我們 setup 中，LDV-coherent lag structure 隨 `p` 的變化在兩支麥克風上夠一致，使 joint consistency 能排掉大量不一致假峰。

### Q2：PHAT-GCC 是不是把牆消掉了？
答法核心：不是。PHAT 主要是降低對幅度（magnitude coloration）的敏感度；牆造成的 multipath/coherence trap 仍存在。PI-GS 是靠**兩路 joint consistency**在做 disambiguation。

### Q3：LDV 是否永遠不受 jammer 影響？
答法核心：不保證。我們只主張：在我們 testbed 與相近條件下，jammer 對 mic pressure 更具支配性，而耦合進牆振動常常相對弱，所以 LDV 能作為更乾淨的 anchor。

### Q4：為什麼 objective 用 `|R|`？
答法核心：跨模態（速度類 vs 壓力類）相關量可能為複數。用 `|R|` 得到一致的實數打分，避免脆弱的相位慣例/符號問題。

---

## 6) 未來改稿自檢清單（Guardrails）

每次改 `paper/main.tex` 前先檢查：
- 有沒有把 paper 寫成「普適物理定理」或「保證解耦」？
- 有沒有出現 “guarantee / immune / decouple / perfectly” 之類的過強字眼？
- `R_{Vm}(τ)` 的用法有沒有跟 objective 對齊（complex → `|R|`）？
- `τ_Vm(p)` 有沒有被寫成“strict analytic prior”（應該是 template/surrogate）？
- 敘事順序是否仍是：**大反派（mic–mic coherence trap）→ anchor（LDV）→ joint consistency（PI-GS）**？
- Caption 有沒有偷塞比本文更強的物理主張？

