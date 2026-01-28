# MUSIC 陣列訊號處理：從第一性原理理解子空間方法

> 本文從零開始建構 MUSIC（Multiple Signal Classification）演算法的數學直覺，適合非陣列訊號處理領域的讀者。

---

## 1. 問題的起點：用多個麥克風定位聲源

假設有 $M$ 個麥克風排成一列，有一個聲源從某個角度 $\theta$ 發出訊號。聲音到達每個麥克風的時間不同——如果第一個麥克風收到的是 $s(t)$，第二個麥克風因為距離差，收到的會是 $s(t - \tau_1)$，第三個是 $s(t - \tau_2)$，以此類推。

**核心問題**：從 $M$ 個麥克風收到的資料，怎麼反推聲源的方向 $\theta$？

---

## 2. 導引向量：把幾何關係編碼成數學

### 2.1 窄頻假設下的簡化

假設訊號是窄頻的（頻率集中在 $f_0$ 附近）。這個假設讓數學變簡單，因為對於窄頻訊號，時間延遲 $\tau$ 等價於相位偏移 $e^{-j 2\pi f_0 \tau}$。

為什麼？因為 $s(t-\tau)$ 的傅立葉轉換是 $S(f) e^{-j2\pi f \tau}$。如果 $S(f)$ 只在 $f_0$ 附近有值，那整個效果就是乘上一個固定相位 $e^{-j2\pi f_0 \tau}$。

### 2.2 陣列接收模型

$M$ 個麥克風收到的訊號可以寫成向量：

$$\mathbf{x}(t) = \begin{bmatrix} 1 \\ e^{j\phi_1} \\ e^{j\phi_2} \\ \vdots \\ e^{j\phi_{M-1}} \end{bmatrix} s(t) + \mathbf{n}(t)$$

其中 $\phi_k = -2\pi f_0 \tau_k$，而 $\tau_k$ 由幾何決定。

### 2.3 導引向量的完整形式

那個向量 $\mathbf{a}(\theta) = [1, e^{j\phi_1}, ..., e^{j\phi_{M-1}}]^T$ 稱為**導引向量**（steering vector），它編碼了「如果聲音從 $\theta$ 方向來，各麥克風之間的相位關係應該長什麼樣」。

展開來看，假設麥克風等間距排列（間距 $d$），聲速為 $c$，波長 $\lambda = c/f_0$：

$$\phi_k = -\frac{2\pi d}{\lambda} k \sin\theta$$

所以完整的導引向量是：

$$\mathbf{a}(\theta) = \begin{bmatrix} 1 \\ e^{-j\frac{2\pi d}{\lambda}\sin\theta} \\ e^{-j\frac{2\pi d}{\lambda} \cdot 2\sin\theta} \\ \vdots \\ e^{-j\frac{2\pi d}{\lambda}(M-1)\sin\theta} \end{bmatrix}$$

$\theta$ 藏在每個相位項裡面。不同的入射角度會產生不同的導引向量。

### 2.4 複數指數的模恆為 1

導引向量的每個元素都是 $e^{j\phi}$ 的形式。根據尤拉公式：

$$e^{j\phi} = \cos\phi + j\sin\phi$$

複數的模（magnitude）定義為 $|z| = \sqrt{a^2 + b^2}$（對於 $z = a + jb$），所以：

$$|e^{j\phi}| = \sqrt{\cos^2\phi + \sin^2\phi} = 1$$

這意味著 $\mathbf{a}^H \mathbf{a} = \sum_{k=0}^{M-1} |e^{j\phi_k}|^2 = M$。

---

## 3. 共變異數矩陣：把未知訊號「平均掉」

### 3.1 為什麼需要共變異數矩陣？

我們的模型是 $\mathbf{x}(t) = \mathbf{a}(\theta) s(t) + \mathbf{n}(t)$，想找 $\theta$，但 $s(t)$ 是未知的、隨時間變化的訊號。

關鍵洞察：雖然 $s(t)$ 未知，但 $\mathbf{a}(\theta)$ 是固定的。如果我們能找到某種操作，把 $s(t)$ 的影響「平均掉」，只留下 $\mathbf{a}(\theta)$ 的資訊...

這就是為什麼要計算**共變異數矩陣** $R = E[\mathbf{x}(t)\mathbf{x}^H(t)]$。

### 3.2 共變異數矩陣的結構

展開計算：

$$R = E[(\mathbf{a}s + \mathbf{n})(\mathbf{a}s + \mathbf{n})^H]$$

假設訊號和雜訊不相關（交叉項期望值為零）：

$$R = E[|s|^2] \mathbf{a}\mathbf{a}^H + E[\mathbf{n}\mathbf{n}^H] = \sigma_s^2 \mathbf{a}\mathbf{a}^H + \sigma_n^2 \mathbf{I}$$

其中 $\sigma_s^2$ 是訊號功率，$\sigma_n^2$ 是雜訊功率。

### 3.3 矩陣各元素的意義

$R$ 的 $(i,j)$ 元素是 $E[x_i(t) x_j^*(t)]$——第 $i$ 和第 $j$ 個麥克風訊號的互相關：

- **對角線** $R_{ii}$：第 $i$ 個麥克風的功率
- **非對角線** $R_{ij}$：麥克風 $i$ 和 $j$ 之間的相關性（包含相位資訊）

如果只有雜訊（無訊號），而且雜訊在各麥克風獨立，$R = \sigma_n^2 \mathbf{I}$，只有對角線有值。有訊號時，因為同一個 $s(t)$ 到達所有麥克風（只是相位不同），$R$ 會有非對角線的結構——這正是訊號存在的證據。

### 3.4 雜訊共變異數為何是 $\sigma_n^2 \mathbf{I}$？

這不是數學定理，而是對雜訊的**假設**（稱為「空間白雜訊」）：

1. **每個麥克風的雜訊功率相同**：$E[|n_i|^2] = \sigma_n^2$（對角線都等於 $\sigma_n^2$）
2. **不同麥克風的雜訊互不相關**：$E[n_i n_j^*] = 0$ 當 $i \neq j$（非對角線都是零）

這個假設在許多情況下合理（電子雜訊獨立、漫射背景雜訊），但如果雜訊有空間結構，MUSIC 效果會變差。

---

## 4. 特徵分解：自然分離訊號與雜訊子空間

### 4.1 外積矩陣的秩

先看 $\mathbf{a}\mathbf{a}^H$ 這個矩陣。它是一個向量乘以自己的共軛轉置，這種形式叫做**外積**（outer product）。

**命題**：外積矩陣的秩為 1。

**證明**：$\mathbf{a}\mathbf{a}^H$ 的第 $i$ 列是 $a_i \cdot \mathbf{a}^H = a_i \cdot [a_1^*, a_2^*, ..., a_M^*]$。所有列都是同一個向量 $\mathbf{a}^H$ 的純量倍，所以所有列線性相關，只有一個獨立方向，秩 = 1。

### 4.2 單一訊號源的特徵結構

對於 $R = \sigma_s^2 \mathbf{a}\mathbf{a}^H + \sigma_n^2 \mathbf{I}$：

先分析 $\mathbf{a}\mathbf{a}^H$ 的特徵值和特徵向量：

- 設 $\mathbf{v} = \mathbf{a}$，則 $\mathbf{a}\mathbf{a}^H \mathbf{a} = \mathbf{a} (\mathbf{a}^H \mathbf{a}) = \mathbf{a} \cdot M$
- 所以 $\mathbf{a}$ 是特徵向量，對應特徵值 $M$
- 其他 $M-1$ 個特徵向量正交於 $\mathbf{a}$，對應特徵值 0

加上 $\sigma_n^2 \mathbf{I}$ 的效果是每個特徵值都加上 $\sigma_n^2$，所以 $R$ 的特徵結構是：

| 特徵值 | 數量 | 對應特徵向量 |
|--------|------|--------------|
| $\sigma_s^2 M + \sigma_n^2$（大） | 1 | 平行於 $\mathbf{a}$ |
| $\sigma_n^2$（小） | $M-1$ | 都正交於 $\mathbf{a}$ |

### 4.3 訊號子空間與雜訊子空間

這個特徵結構自然地將空間分成兩部分：

- **訊號子空間** $E_s$：大特徵值對應的特徵向量（就是 $\mathbf{a}$ 的方向）
- **雜訊子空間** $E_n$：小特徵值對應的特徵向量們（都正交於 $\mathbf{a}$）

這個分解不是人為強加的，而是 $R$ 本身結構的必然結果。

---

## 5. 多訊號源的推廣

### 5.1 多訊號源模型

假設有 $K$ 個訊號源，從不同角度 $\theta_1, \theta_2, ..., \theta_K$ 來：

$$\mathbf{x}(t) = \mathbf{a}(\theta_1) s_1(t) + \mathbf{a}(\theta_2) s_2(t) + ... + \mathbf{a}(\theta_K) s_K(t) + \mathbf{n}(t)$$

寫成矩陣形式：

$$\mathbf{x}(t) = A \mathbf{s}(t) + \mathbf{n}(t)$$

其中 $A = [\mathbf{a}(\theta_1), \mathbf{a}(\theta_2), ..., \mathbf{a}(\theta_K)]$ 是 $M \times K$ 的矩陣。

### 5.2 共變異數矩陣的新結構

$$R = A R_s A^H + \sigma_n^2 I$$

其中 $R_s = E[\mathbf{s}\mathbf{s}^H]$ 是訊號的共變異數矩陣。

如果 $K$ 個導引向量線性獨立且 $R_s$ 滿秩，則 $A R_s A^H$ 的秩是 $K$。

### 5.3 特徵值的分布

$R$ 的 $M$ 個特徵值會呈現：

- **$K$ 個大特徵值**：$\lambda_1, ..., \lambda_K > \sigma_n^2$，對應的特徵向量張成 $A$ 的行空間
- **$M-K$ 個小特徵值**：$\lambda_{K+1}, ..., \lambda_M \approx \sigma_n^2$，對應的特徵向量正交於 $A$ 的行空間

實際操作時，把特徵值從大到小排列會看到明顯的「落差」——這個落差告訴我們有幾個訊號源。

```
特徵值
  |
  |  *
  |     *
  |        *
  |           * * * * * *  ← 這些差不多大，都接近雜訊功率
  +------------------------
     1  2  3  4  5  6  7 ...
```

---

## 6. MUSIC 演算法：掃描尋找正交性

### 6.1 核心公式

$$P_{MUSIC}(\theta) = \frac{1}{\mathbf{a}^H(\theta) E_n E_n^H \mathbf{a}(\theta)}$$

### 6.2 逐項解讀

**$E_n E_n^H$ 是什麼**：

$E_n$ 的每一行是一個正交於真實導引向量的特徵向量。$E_n E_n^H$ 是這些向量張成的子空間的**投影矩陣**——任何向量乘上它，結果就是那個向量在雜訊子空間的投影。

**$\mathbf{a}^H(\theta) E_n E_n^H \mathbf{a}(\theta)$ 是什麼**：

這是「假設角度 $\theta$」對應的導引向量，投影到雜訊子空間後的長度平方。

數學上，設 $\mathbf{v} = E_n^H \mathbf{a}(\theta)$，則 $\mathbf{a}^H(\theta) E_n E_n^H \mathbf{a}(\theta) = \mathbf{v}^H \mathbf{v} = |\mathbf{v}|^2$。

**為什麼取倒數**：

- 當 $\theta = \theta_{true}$ 時，$\mathbf{a}(\theta_{true})$ 正交於 $E_n$，分母趨近零，$P_{MUSIC}$ 產生峰值
- 當 $\theta \neq \theta_{true}$ 時，分母有值，$P_{MUSIC}$ 維持在一般水平

### 6.3 二次型的幾何意義

$\mathbf{a}^H M \mathbf{a}$ 這個形式叫做**二次型**。對於向量 $\mathbf{a}$ 和矩陣 $M$，它計算的是：把 $\mathbf{a}$ 用 $M$ 轉換後，和原本的 $\mathbf{a}$ 有多「對齊」。

如果 $M$ 是投影矩陣，$\mathbf{a}^H M \mathbf{a}$ 就是 $\mathbf{a}$ 投影到那個子空間的長度平方。

共軛轉置的出現是因為複數向量的內積定義：$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^H \mathbf{v}$，這樣長度平方 $\mathbf{v}^H \mathbf{v}$ 才會是正實數。

---

## 7. 完整演算法流程

### 步驟一：從資料估計共變異數矩陣

$$\hat{R} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{x}(t) \mathbf{x}^H(t)$$

### 步驟二：特徵分解

$$\hat{R} = E \Lambda E^H$$

得到特徵值 $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_M$ 和對應的特徵向量。

### 步驟三：分離子空間

根據特徵值的「落差」判斷訊號源數量 $K$：
- 前 $K$ 個特徵向量組成 $E_s$
- 後 $M-K$ 個特徵向量組成 $E_n$

### 步驟四：掃描計算 MUSIC 頻譜

對所有可能的 $\theta$ 計算 $P_{MUSIC}(\theta)$，找峰值位置即為訊號源方向。

---

## 8. MUSIC 與傳統互相關方法的比較

### 8.1 GCC-PHAT 回顧

兩個麥克風訊號 $x_1(t)$ 和 $x_2(t)$，GCC-PHAT 在頻域計算：

$$GCC\text{-}PHAT(\tau) = \int \frac{X_1(f) X_2^*(f)}{|X_1(f) X_2^*(f)|} e^{j2\pi f\tau} df$$

除以模是為了只保留相位資訊，找 $\tau$ 使得結果最大。

### 8.2 兩者比較

| 面向 | GCC-PHAT | MUSIC |
|------|----------|-------|
| 輸入 | 兩麥克風的頻譜 | 共變異數矩陣 |
| 處理方式 | 相位加權的互相關 | 特徵分解 + 子空間投影 |
| 輸出 | 延遲 $\tau$ | 角度 $\theta$（可換算成 $\tau$）|
| 多訊號源 | 難以處理 | 可分辨（需 $K < M$）|
| 計算複雜度 | 低 | 高（需特徵分解）|

### 8.3 為什麼 MUSIC 要「繞一圈」？

當 $M = 2$ 且只有一個訊號源時，GCC-PHAT 和 MUSIC 在找的東西本質相同：哪個延遲/角度最能解釋兩個麥克風訊號之間的相位關係。

但當 $M > 2$ 且有多個訊號源時，互相關方法不直接推廣——它只能處理「一對」麥克風。MUSIC 的子空間方法可以同時利用所有麥克風對的資訊，並在理論上分辨多個訊號源。

---

## 9. 重要限制與實務考量

### 9.1 訊號源數量限制

**$K < M$**：訊號源數量必須小於麥克風數量，否則雜訊子空間維度變成零，MUSIC 失效。

### 9.2 空間白雜訊假設

如果雜訊有空間結構（例如另一個干擾源），$E[\mathbf{n}\mathbf{n}^H] \neq \sigma_n^2 \mathbf{I}$，MUSIC 的效果會下降。

### 9.3 窄頻假設

MUSIC 基於窄頻假設。對於寬頻訊號，需要使用分頻帶處理或其他寬頻 DOA 方法。

### 9.4 相干訊號問題

如果多個訊號源高度相關（例如多路徑反射），$R_s$ 會變成奇異矩陣，標準 MUSIC 會失效。需要使用空間平滑等預處理技術。

---

## 10. 總結：邏輯鏈

1. **聲音從某角度來** → 各麥克風收到的訊號有固定相位關係 → **導引向量** $\mathbf{a}(\theta)$

2. **想找 $\theta$ 但不知道 $s(t)$** → 算共變異數把 $s(t)$ 平均掉 → 得到 **$R$**

3. **$R$ 的結構** = 秩-$K$ 訊號部分 + 對角雜訊部分 → 特徵分解**自然分出**兩個子空間

4. **真實的導引向量正交於雜訊子空間** → 掃描 $\theta$ 看誰最正交 → **MUSIC 峰值**

MUSIC 的威力在於它把「定位」問題轉化成「正交性檢驗」問題，而正交性可以透過共變異數矩陣的特徵結構自然獲得。

---

## 參考文獻

1. Schmidt, R. O. (1986). Multiple emitter location and signal parameter estimation. *IEEE Transactions on Antennas and Propagation*, 34(3), 276-280.

2. Krim, H., & Viberg, M. (1996). Two decades of array signal processing research: the parametric approach. *IEEE Signal Processing Magazine*, 13(4), 67-94.