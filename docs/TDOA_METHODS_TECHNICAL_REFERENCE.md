# TDoA/DoA 估計方法技術參考手冊

> **Project:** exp-tdoa-cross-correlation / exp-interspeech-GRU2
> **Date:** 2026-01-28
> **Author:** Jenner
> **Purpose:** 從第一性原理完整解釋所有使用的時延/方向估計方法

---

## 目錄

1. [前言與問題定義](#1-前言與問題定義)
2. [信號模型與數學基礎](#2-信號模型與數學基礎)
3. [Cross-Correlation 方法](#3-cross-correlation-方法)
   - 3.1 Standard Cross-Correlation (CC)
   - 3.2 Normalized Cross-Correlation (NCC)
   - 3.3 GCC-PHAT
4. [MUSIC 演算法](#4-music-演算法)
5. [OMP-DTmin 頻率補償](#5-omp-dtmin-頻率補償)
6. [方法比較與選擇指南](#6-方法比較與選擇指南)
7. [附錄：數學推導](#7-附錄數學推導)

---

## 1. 前言與問題定義

### 1.1 什麼是 TDoA？

**Time Difference of Arrival (TDoA)** 是指同一聲源信號到達兩個不同感測器的時間差。

```
聲源 S ──────┬──→ 感測器 A (到達時間 t_A)
             │
             └──→ 感測器 B (到達時間 t_B)

TDoA = τ = t_B - t_A
```

### 1.2 什麼是 DoA？

**Direction of Arrival (DoA)** 是聲源相對於感測器陣列的方向角。

對於線性陣列：
```
          聲源 S
             ╲
              ╲ θ (DoA)
               ╲
    ○──────────○ (感測器陣列)
    A    d     B
```

TDoA 與 DoA 的關係：
```
τ = (d × sin(θ)) / c

其中：
  d = 感測器間距 (m)
  c = 聲速 (約 343 m/s)
  θ = DoA 角度
```

### 1.3 MIC-LDV 系統的特殊性

本專案使用 **麥克風 (MIC)** 與 **雷射都卜勒振動計 (LDV)** 組成的異質雙感測器系統：

| 感測器 | 測量物理量 | 特性 |
|--------|-----------|------|
| **MIC** | 空氣聲壓 | 直接測量聲波 |
| **LDV** | 表面振動速度 | 非接觸式，測量振動面 |

**關鍵挑戰：頻率色散 (Dispersion)**

不同頻率的聲波經歷不同的有效延遲：
```
τ(f) = τ₀ + Δτ(f)
```

原因：
1. 多路徑傳播（直接聲 + 反射聲干涉）
2. LDV 目標面的機械共振（不同頻率響應不同）
3. 空氣吸收的頻率相依性

---

## 2. 信號模型與數學基礎

### 2.1 連續時間信號模型

設 MIC 信號為 x(t)，LDV 信號為 y(t)：

```
y(t) = α × x(t - τ) + n(t)
```

其中：
- α：振幅係數（可能因傳播損耗而異）
- τ：時間延遲 (TDoA)
- n(t)：噪聲

### 2.2 頻域表示

對信號做傅立葉轉換：

```
X(f) = ∫ x(t) × e^(-j2πft) dt

Y(f) = α × X(f) × e^(-j2πfτ) + N(f)
```

**時延在頻域表現為線性相位**：
```
∠[Y(f)/X(f)] = -2πfτ + ∠α
```

這是所有頻域 TDoA 估計方法的理論基礎。

### 2.3 離散時間處理

實際處理使用 STFT (Short-Time Fourier Transform)：

```
X[k, n] = Σ x[m] × w[m - nH] × e^(-j2πkm/N)
           m

其中：
  k = 頻率 bin 索引 (0 to N/2)
  n = 時間幀索引
  H = hop length (幀間跳躍樣本數)
  w[·] = 窗函數 (如 Hann window)
```

---

## 3. Cross-Correlation 方法

Cross-correlation 是最基本的時延估計方法，利用信號的相似性來找出最佳對齊位置。

### 3.1 Standard Cross-Correlation (CC)

#### 3.1.1 定義

```
R_xy(τ) = ∫ x(t) × y(t + τ) dt
```

離散形式：
```
R_xy[m] = Σ x[n] × y[n + m]
           n
```

#### 3.1.2 頻域計算（高效實作）

利用 **卷積定理**：
```
R_xy = IFFT{ X*(f) × Y(f) }
```

其中 X*(f) 是 X(f) 的共軛。

#### 3.1.3 TDoA 估計

```
τ̂ = argmax_τ |R_xy(τ)|
```

找到互相關峰值的位置即為 TDoA 估計。

#### 3.1.4 物理意義

CC 本質上是 **振幅加權平均**：

```
τ_cc ≈ Σ |X(f)|² × |Y(f)|² × τ(f) / Σ |X(f)|² × |Y(f)|²
```

- 高能量頻率貢獻較大
- 對語音信號：低頻（基頻、諧波）主導估計

#### 3.1.5 優缺點

| 優點 | 缺點 |
|------|------|
| 計算簡單 | 受振幅影響 |
| 對低 SNR 有一定魯棒性 | 頻率色散下偏向低頻 |
| 物理直觀 | 峰值可能不銳利 |

---

### 3.2 Normalized Cross-Correlation (NCC)

#### 3.2.1 定義

```
NCC(τ) = R_xy(τ) / √(R_xx(0) × R_yy(0))
```

其中 R_xx(0) 和 R_yy(0) 是各自的自相關在零延遲處的值（即能量）。

#### 3.2.2 範圍

```
NCC ∈ [-1, +1]
```

- NCC = 1：完全正相關
- NCC = -1：完全負相關
- NCC = 0：無相關

#### 3.2.3 與 CC 的關係

NCC 只是對 CC 做 **全域正規化**：
```
NCC = CC / (常數)
```

因此：
- **峰值位置與 CC 相同**
- 峰值大小標準化，便於比較不同信號對

#### 3.2.4 優缺點

| 優點 | 缺點 |
|------|------|
| 峰值大小有物理意義（相關係數） | 與 CC 估計相同 |
| 便於跨信號比較 | 仍受頻率色散影響 |
| 對增益變化不敏感 | 計算量略增 |

---

### 3.3 GCC-PHAT (Generalized Cross-Correlation with Phase Transform)

#### 3.3.1 動機

在混響環境或頻率色散下，CC 的峰值可能被分散或出現多峰。GCC-PHAT 透過 **相位變換 (Phase Transform)** 來銳化峰值。

#### 3.3.2 定義

```
R_PHAT(τ) = IFFT{ X*(f) × Y(f) / |X*(f) × Y(f)| }
          = IFFT{ e^(j∠[X*(f)×Y(f)]) }
```

關鍵：只保留 **相位資訊**，振幅設為 1。

#### 3.3.3 幾何解釋

想像頻域中的交叉頻譜 X*(f)×Y(f) 是一個複數向量：

```
Without PHAT:  每個頻率貢獻 |X||Y| × e^(jφ)
With PHAT:     每個頻率貢獻 1 × e^(jφ)
```

PHAT 讓所有頻率 **等權重** 貢獻。

#### 3.3.4 物理意義

```
τ_PHAT ≈ (1/N) × Σ τ(f)
```

GCC-PHAT 估計的是 **跨頻率平均延遲**，而非能量加權平均。

#### 3.3.5 為什麼 PHAT 能銳化峰值？

考慮理想情況（無頻率色散）：
```
所有頻率的相位延遲相同：φ(f) = -2πfτ₀

IFFT{ e^(j2πfτ₀) } = δ(τ - τ₀)  (理想脈衝)
```

峰值理論上是 **完美脈衝**，比 CC 更銳利。

#### 3.3.6 頻率色散下的行為

當存在頻率色散時：
```
φ(f) = -2πfτ(f)，其中 τ(f) 隨頻率變化

IFFT 結果不再是完美脈衝，而是展寬的峰值
```

但因為等權重，GCC-PHAT 仍比 CC 更能反映「真實」的平均延遲。

#### 3.3.7 實作細節

```python
def gcc_phat(x, y, fs, eps=1e-12):
    # FFT
    nfft = next_power_of_2(2 * len(x))
    X = np.fft.rfft(x, n=nfft)
    Y = np.fft.rfft(y, n=nfft)

    # Cross-spectrum
    R = np.conj(X) * Y

    # Phase transform (PHAT weighting)
    R_phat = R / (np.abs(R) + eps)

    # IFFT
    cc = np.fft.irfft(R_phat, n=nfft)

    # Find peak
    tau_samples = np.argmax(np.abs(cc))
    tau_seconds = tau_samples / fs

    return tau_seconds
```

#### 3.3.8 優缺點

| 優點 | 缺點 |
|------|------|
| 峰值銳利 | 對噪聲敏感（低 SNR 頻率等權重） |
| 混響魯棒 | 頻率色散下峰值展寬 |
| 等權重所有頻率 | 需要帶通濾波預處理 |
| 物理可解釋 | |

---

## 4. MUSIC 演算法

MUSIC (Multiple Signal Classification) 是一種 **子空間分解** 方法，原設計用於雷達/天線陣列，但可應用於聲學陣列。

### 4.1 基本原理

#### 4.1.1 信號模型

對於 M 個感測器接收 K 個信號源：

```
x(t) = A(θ) × s(t) + n(t)
```

其中：
- x(t)：M×1 接收信號向量
- A(θ)：M×K 導引矩陣 (steering matrix)
- s(t)：K×1 信號源向量
- n(t)：M×1 噪聲向量

#### 4.1.2 協方差矩陣

空間協方差矩陣：
```
R = E[x × x^H] = A × R_s × A^H + σ²I
```

其中 R_s = E[s × s^H] 是信號協方差，σ² 是噪聲功率。

#### 4.1.3 特徵分解

對 R 做特徵分解：
```
R = U × Λ × U^H

其中：
  Λ = diag(λ₁, λ₂, ..., λ_M)，λ₁ ≥ λ₂ ≥ ... ≥ λ_M
  U = [u₁, u₂, ..., u_M]
```

#### 4.1.4 子空間分離

關鍵洞察：**信號子空間與噪聲子空間正交**

```
信號子空間：U_s = [u₁, ..., u_K]    (對應最大的 K 個特徵值)
噪聲子空間：U_n = [u_{K+1}, ..., u_M] (對應最小的 M-K 個特徵值)
```

對於真實的信號方向 θ_true：
```
a(θ_true)^H × U_n = 0  (導引向量與噪聲子空間正交)
```

### 4.2 MUSIC 頻譜

定義 MUSIC 空間頻譜：
```
P_MUSIC(θ) = 1 / |a(θ)^H × U_n × U_n^H × a(θ)|
```

- 當 θ = θ_true 時，分母趨近於 0，P_MUSIC 趨近於 ∞
- 因此信號方向對應於 P_MUSIC 的峰值

### 4.3 導引向量 (Steering Vector)

對於線性等距陣列 (ULA)，頻率 f 的導引向量：

```
a(θ, f) = [1, e^(-jφ₁), e^(-jφ₂), ..., e^(-jφ_{M-1})]^T

其中：
  φ_m = 2π × f × m × d × sin(θ) / c
  d = 感測器間距
  c = 聲速
```

### 4.4 雙元素 MUSIC (MIC-LDV 應用)

對於本專案的 2 元素陣列 (M=2, K=1)：

#### 4.4.1 協方差矩陣

```
R = | R_mic_mic    R_mic_ldv  |  (2×2 矩陣)
    | R_ldv_mic    R_ldv_ldv  |
```

#### 4.4.2 特徵分解

```
R = U × Λ × U^H

Λ = | λ₁  0  |，λ₁ > λ₂
    | 0   λ₂ |

U = [u₁, u₂]
```

- 信號子空間：u₁ (對應 λ₁)
- 噪聲子空間：u₂ (對應 λ₂)

#### 4.4.3 MUSIC 頻譜

```
P_MUSIC(θ) = 1 / |a(θ)^H × u₂ × u₂^H × a(θ)|
```

#### 4.4.4 SNR 估計

從特徵值比估計訊噪比：
```
SNR_eigen = 10 × log₁₀(λ₁/λ₂ - 1)  [dB]
```

原理：λ₁ ≈ P_signal + P_noise，λ₂ ≈ P_noise

### 4.5 寬頻 MUSIC (Incoherent Wideband)

語音是寬頻信號，需要處理多個頻率：

```python
P_MUSIC_wideband(θ) = Σ P_MUSIC(θ, f)  # 對所有頻率求和
                       f
```

這是 **非相干寬頻 MUSIC**，假設不同頻率獨立處理後再平均。

### 4.6 實作細節

```python
def music_2element(X_mic, X_ldv, freqs, spacing, c=343):
    """
    X_mic, X_ldv: STFT (n_freq, n_frames)
    """
    n_freq, n_frames = X_mic.shape
    angles = np.arange(-90, 91, 1)  # 掃描角度
    P_music = np.zeros(len(angles))

    for f_idx, f in enumerate(freqs):
        if f < freq_min or f > freq_max:
            continue

        # 建立協方差矩陣
        x = np.vstack([X_mic[f_idx], X_ldv[f_idx]])  # 2 × n_frames
        R = (x @ x.conj().T) / n_frames              # 2 × 2

        # 特徵分解
        eigvals, eigvecs = np.linalg.eigh(R)
        U_n = eigvecs[:, 0:1]  # 噪聲子空間 (最小特徵值)

        # 掃描角度
        for i, theta in enumerate(angles):
            theta_rad = np.deg2rad(theta)
            tau = spacing * np.sin(theta_rad) / c
            a = np.array([[1], [np.exp(-1j * 2 * np.pi * f * tau)]])

            denom = np.abs(a.conj().T @ U_n @ U_n.conj().T @ a)
            P_music[i] += 1 / (denom + eps)

    # DoA 估計
    doa = angles[np.argmax(P_music)]
    return doa, P_music
```

### 4.7 優缺點

| 優點 | 缺點 |
|------|------|
| 超解析度（理論上） | 需要多元素陣列 |
| 提供 SNR 估計 | 計算複雜度高 |
| 輸出空間頻譜（視覺化） | 2 元素解析度有限 |
| 理論基礎扎實 | 需要已知陣列幾何 |

### 4.8 MUSIC 的限制

對於 2 元素陣列：
- **最多解析 M-1 = 1 個信號源**
- **存在 ±θ 模糊**（線性陣列無法區分前後）
- **解析度受限**：與 GCC-PHAT 相比沒有顯著優勢

---

## 5. OMP-DTmin 頻率補償

OMP-DTmin 是一種 **頻率色散補償** 方法，透過 Orthogonal Matching Pursuit (OMP) 演算法估計每個頻率的時延，然後進行相位校正。

### 5.1 問題背景

在 MIC-LDV 系統中，頻率色散導致：
```
Y(f) = H(f) × X(f)

其中 H(f) = |H(f)| × e^(-j2πfτ(f))

τ(f) 隨頻率變化 → 無法用單一時延描述
```

### 5.2 OMP 演算法基礎

#### 5.2.1 稀疏表示問題

目標：用字典 D 的少量原子表示目標信號 y：
```
y ≈ D × α，其中 ||α||₀ 最小（非零元素最少）
```

#### 5.2.2 OMP 迭代步驟

```
初始化：
  殘差 r = y
  支持集 S = ∅

For k = 1 to K_max:
  1. 匹配：找最相關原子
     i* = argmax_i |<r, d_i>|

  2. 更新支持集
     S = S ∪ {i*}

  3. 投影：最小二乘求係數
     α_S = argmin_α ||y - D_S × α||²

  4. 更新殘差
     r = y - D_S × α_S

  5. 停止條件檢查
```

### 5.3 DTmin 的 Lag 字典

#### 5.3.1 字典構建

對於每個頻率 f，建立時延字典：
```
D[f] = { x[t - k] | k ∈ [L_min, L_max] }
```

其中 x[t-k] 是 MIC 信號延遲 k 幀的版本。

#### 5.3.2 第一步 Lag 選擇

OMP 第一步選擇的 lag 索引即為該頻率的時延估計：
```
ℓ̂(f) = argmax_k |<y_f, x_f[t-k]>|
```

轉換為時間延遲：
```
τ̂(f) = ℓ̂(f) × (hop_length / fs)
```

### 5.4 相位均衡器構建

#### 5.4.1 均衡器公式

```
G_DTmin(f) = e^(+j2πf×τ̂(f))
```

這是一個 **全通濾波器**：
- 幅度 |G(f)| = 1（不改變能量）
- 相位 ∠G(f) = +2πf×τ̂(f)（補償延遲）

#### 5.4.2 應用補償

```
Y_compensated(f) = Y(f) × G_DTmin(f)
                 = Y(f) × e^(+j2πf×τ̂(f))
```

### 5.5 實作細節 (generate_lag_omp.py)

```python
def run_omp_lag_capture(X_history, y_target, K_max=16, Lag_Min=-32, Lag_Max=32):
    """
    X_history: (L, F) MIC STFT history
    y_target: (Tw, F) LDV STFT target window

    Returns:
        Full_Corrs: (F, K_max, M) 相關值
        Full_Actions: (F, K_max) 選擇的 lag 索引
        Full_Reductions: (F, K_max) 能量減少比例
    """

    Tw, F = y_target.shape
    Lags = range(Lag_Min, Lag_Max + 1)  # e.g., -32 to +32
    M = len(Lags)  # 字典大小 (e.g., 65)

    # 建立字典 (F, Tw, M)
    Dict_tensor = build_lag_dictionary(X_history, Lags, Tw)

    # 正規化字典
    Dict_Norm = Dict_tensor / ||Dict_tensor||

    # OMP 迭代
    for k in range(K_max):
        # 1. 計算相關
        Corrs = Dict_Norm^H @ Residuals  # (F, M)

        # 2. 選擇最佳 lag
        Best_Lags = argmax(|Corrs|, dim=1)  # (F,)

        # 3. 更新係數 (最小二乘)
        Weights = lstsq(Dict_active, Targets)

        # 4. 更新殘差
        Residuals = Targets - Dict_active @ Weights

    return Full_Corrs, Full_Actions, Full_Reductions
```

### 5.6 Penalty-OMP (自適應停止)

標準 OMP 需要預設 K_max。Penalty-OMP 引入 **成本懲罰**，自動決定停止時機：

```
當 ΔE_k < λ × E_0 時停止

其中：
  ΔE_k = E_{k-1} - E_k (第 k 步的能量減少)
  λ = 成本係數 (如 3e-4)
  E_0 = 初始能量
```

### 5.7 相位均衡效果

#### 5.7.1 色散度量：tau_band_spread

```
tau_band_spread = max(τ_band) - min(τ_band)

其中 τ_band 是各子頻帶的 GCC-PHAT 估計
```

#### 5.7.2 典型改善

| 條件 | tau_band_spread |
|------|-----------------|
| 無補償 | ~10 ms |
| DTmin 補償後 | ~7 ms (-30%) |

### 5.8 為什麼 DTmin 有效？

#### 5.8.1 物理解釋

DTmin 的第一步 lag 選擇本質上是 **逐頻帶 GCC-PHAT**：
- 找到該頻率的最佳時延
- 不受其他頻率干擾
- 與稀疏表示目標一致

#### 5.8.2 與群延遲的比較

傳統群延遲估計：
```
τ_group = -dφ/dω  (相位對頻率的導數)
```

問題：多路徑下，群延遲是 **混合平均**，不代表任何單一路徑。

DTmin 逐頻帶估計則直接找到 **最強路徑** 的延遲。

### 5.9 優缺點

| 優點 | 缺點 |
|------|------|
| 物理可解釋 | 計算複雜度較高 |
| 頻率色散補償效果好 | 需要足夠的 STFT 幀 |
| 與下游處理目標一致 | Lag 範圍需要調整 |
| 自適應停止（Penalty-OMP） | |

---

## 6. 方法比較與選擇指南

### 6.1 計算複雜度

| 方法 | 複雜度 | 說明 |
|------|--------|------|
| CC/NCC | O(N log N) | FFT-based |
| GCC-PHAT | O(N log N) | FFT-based + 正規化 |
| MUSIC | O(F × A × M²) | F=頻率數, A=角度數, M=元素數 |
| OMP-DTmin | O(F × K × L) | K=步數, L=lag範圍 |

### 6.2 適用場景

| 場景 | 推薦方法 | 原因 |
|------|---------|------|
| 即時處理 | GCC-PHAT | 計算快，品質好 |
| 低 SNR | CC | 振幅加權抑制噪聲 |
| 需要 SNR 資訊 | MUSIC | 提供特徵值比 |
| 頻率色散嚴重 | GCC-PHAT + DTmin | 補償後效果最佳 |
| 研究/分析 | 全部 | 比較不同方法 |

### 6.3 組合使用建議

最佳實踐流程：
```
1. MIC/LDV 信號載入
       ↓
2. STFT 轉換
       ↓
3. [Optional] DTmin 估計 τ(f) → 相位補償
       ↓
4. GCC-PHAT 估計 TDoA
       ↓
5. [Optional] MUSIC 估計 DoA + SNR
       ↓
6. 輸出結果
```

### 6.4 品質指標

| 指標 | 定義 | 意義 |
|------|------|------|
| **PSR** | peak / median(sidelobe) | 峰值品質 |
| **tau_std** | std(tau_estimates) | 估計穩定性 |
| **SNR_eigen** | 10log₁₀(λ₁/λ₂-1) | 訊噪比 |
| **tau_band_spread** | max(τ_band) - min(τ_band) | 色散程度 |

---

## 7. 附錄：數學推導

### 7.1 CC 頻域等價證明

時域定義：
```
R_xy(τ) = ∫ x(t) × y(t + τ) dt
```

對 y(t+τ) 做傅立葉轉換：
```
F{y(t + τ)} = Y(f) × e^(j2πfτ)
```

利用帕塞瓦爾定理：
```
R_xy(τ) = ∫ X*(f) × Y(f) × e^(j2πfτ) df
        = F⁻¹{X*(f) × Y(f)}
```

### 7.2 MUSIC 正交性證明

設 A 是導引矩陣，R_s 是信號協方差：
```
R = A × R_s × A^H + σ²I
```

對於信號子空間的特徵向量 u_i (i ≤ K)：
```
R × u_i = λ_i × u_i
(A × R_s × A^H + σ²I) × u_i = λ_i × u_i
```

對於噪聲子空間的特徵向量 u_j (j > K)：
```
λ_j = σ²（噪聲特徵值）
```

由於 A 的列向量張成信號子空間：
```
span(A) ⊥ span(U_n)
→ A^H × U_n = 0
→ a(θ_true)^H × U_n = 0
```

### 7.3 GCC-PHAT 白化解釋

GCC-PHAT 的加權函數：
```
W_PHAT(f) = 1 / |X(f) × Y*(f)|
```

這等價於對交叉頻譜做 **預白化**：
- 消除頻譜著色
- 讓所有頻率等權重貢獻

與 ML (Maximum Likelihood) 加權比較：
```
W_ML(f) = 1 / (|N_x(f)|² × |Y(f)|² + |N_y(f)|² × |X(f)|²)
```

PHAT 假設噪聲為白噪聲，是 ML 的特例。

---

## 參考文獻

1. Knapp, C., & Carter, G. C. (1976). "The generalized correlation method for estimation of time delay." IEEE Trans. ASSP.

2. Schmidt, R. O. (1986). "Multiple emitter location and signal parameter estimation." IEEE Trans. AP.

3. Pati, Y. C., Rezaiifar, R., & Krishnaprasad, P. S. (1993). "Orthogonal matching pursuit: Recursive function approximation with applications to wavelet decomposition." Asilomar Conference.

4. Benesty, J., Chen, J., & Huang, Y. (2008). "Microphone Array Signal Processing." Springer.

5. Internal documentation: exp-interspeech-GRU2, E4o DTmin phase equalization.

---

*Document version: 1.0*
*Last updated: 2026-01-28*
