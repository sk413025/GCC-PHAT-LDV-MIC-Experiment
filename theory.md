這是一個非常棒的要求！當我們在學術探討中遇到「跳步」時，停下來把每一個數學細節拆解清楚，是確保理論無懈可擊的唯一方法。

為了讓你能夠毫無阻力地一步步跟上，我將放慢腳步，把**無響室（Anechoic Chamber）**裡的物理現象，轉化為最純粹的代數推導。我們會先定義最基本的物理變數，然後分別推導「純麥克風」為何失敗，以及「LDV-Mic 融合」如何透過共軛相乘達成奇蹟般的解耦。

---

### 第一步：定義無響室的物理與幾何變數

在無響室中，沒有任何牆壁反射與空間殘響，聲音在空氣中的傳播遵循最純粹的**自由空間格林函數（Free-space Green's function）**，也就是說，聲音只會隨著距離變小，並產生對應的相位延遲。

讓我們定義以下變數：

*   $S(f)$：牆外的目標聲源訊號。
*   $c$：聲速。
*   $k$：波數（Wavenumber），$k = 2\pi f / c$。
*   $d(A, B)$：空間中 $A$ 點到 $B$ 點的直線物理距離。

**牆壁傳遞函數（Wall Transfer Function）：**
當聲音穿透厚牆壁時，牆壁會造成振幅衰減與相位延遲。這個效應可以用一個複數 $H_{wall}$ 來表示：
$$ H_{wall}(\theta) = |H_{wall}(\theta)| \cdot e^{-j \phi_{wall}(\theta)} $$
*注意：這裡的 $\theta$ 是聲波撞擊牆壁的**入射角**。入射角不同，穿透的等效厚度就不同，導致相位延遲 $\phi_{wall}$ 也跟著不同。*

---

### 第二步：純麥克風陣列的推導（證明系統性誤差無法消除）

聲波從牆外的聲源 $S$ 飛到牆內的左麥克風 $L$ 與右麥克風 $R$。
根據聲學的最短路徑原理，飛到 $L$ 的聲音，和飛到 $R$ 的聲音，**一定會穿透牆壁上不同的兩個點**（我們稱為 $wL$ 和 $wR$）。

#### 1. 寫出左麥克風 $X_L(f)$ 的完整方程式：

聲音經歷了三件事：牆外空氣傳播 $\to$ 穿透牆壁 $wL$ 點 $\to$ 牆內空氣傳播。
$$ X_L(f) = S(f) \cdot \underbrace{\left[ \frac{1}{4\pi d(S, wL)} e^{-j k \cdot d(S, wL)} \right]}_{\text{牆外空氣傳播}} \cdot \underbrace{\left[ |H_{wall}(\theta_L)| e^{-j \phi_{wall}(\theta_L)} \right]}_{\text{牆壁折射與衰減}} \cdot \underbrace{\left[ \frac{1}{4\pi d(wL, L)} e^{-j k \cdot d(wL, L)} \right]}_{\text{牆內空氣傳播}} $$

為了讓數學看起來不那麼可怕，我們把所有**「影響振幅（音量）的實數項」**全部打包成一個常數 $A_L$：
$$ A_L = |S(f)| \cdot \frac{1}{4\pi d(S, wL)} \cdot |H_{wall}(\theta_L)| \cdot \frac{1}{4\pi d(wL, L)} $$
我們把所有**「影響相位的指數項」**合併（指數相乘等於次方向加）：
$$ X_L(f) = A_L \cdot e^{-j \big[ k \cdot d(S, wL) + \phi_{wall}(\theta_L) + k \cdot d(wL, L) \big]} $$

#### 2. 寫出右麥克風 $X_R(f)$ 的完整方程式：

同理，將下標全部換成 $R$：
$$ X_R(f) = A_R \cdot e^{-j \big[ k \cdot d(S, wR) + \phi_{wall}(\theta_R) + k \cdot d(wR, R) \big]} $$

#### 3. 執行互相關（Cross-Correlation）並提取相位：

演算法要計算時間差，必須將兩者共軛相乘：$\Phi_{LR} = X_L \cdot X_R^*$。

*   $X_L$ 的指數是負的：$e^{-j [ \dots ]}$
*   $X_R^*$（共軛複數）會把指數的負號變成正的：$e^{+j [ \dots ]}$

兩者相乘後，振幅變為 $A_L \cdot A_R$，指數部分相加：
$$ \Phi_{LR} = (A_L A_R) \cdot e^{-j \big[ k \big( d(S, wL) + d(wL, L) - d(S, wR) - d(wR, R) \big) + \big( \phi_{wall}(\theta_L) - \phi_{wall}(\theta_R) \big) \big]} $$

**💥 推導結論（死局）：**
演算法只能看到最終的相位總和。它想要的是純幾何距離差，但方程式裡卻死死地黏著一項 **$\big( \phi_{wall}(\theta_L) - \phi_{wall}(\theta_R) \big)$**。因為入射角 $\theta_L \neq \theta_R$，這個差值絕對不等於零。在無響室中，這個未知的牆壁相位差被完美地保留下來，**成為永遠無法用代數消去的系統性角度誤差。**

---

### 第三步：LDV-Mic 融合的推導（見證完美的代數解耦）

現在，我們把 LDV 雷射精準打在牆壁上的一個點（我們稱為 $v$ 點）。

#### 1. 定義 LDV 測量到的牆壁「震動速度」 $X_V(f)$

和剛才一樣，聲音從聲源 $S$ 飛到牆上的 $v$ 點：
$$ X_V(f) = S(f) \cdot \left[ \frac{1}{4\pi d(S, v)} e^{-j k \cdot d(S, v)} \right] \cdot \left[ |H_{wall}(\theta_v)| e^{-j \phi_{wall}(\theta_v)} \right] $$

這串東西太長了。因為它等一下會被「完美消滅」，我們把它全部打包，定義為一個**超大未知複數 $U_v$**：
$$ X_V(f) = U_v $$

#### 2. 從 LDV 到麥克風的「二次輻射」推導

無響室最美妙的地方就在這裡。既然我們量到了牆壁上 $v$ 點的震動速度，那麼根據聲學的輻射定理，**$v$ 點就是一個全新的小喇叭（點波源）**，它朝著室內的麥克風發射聲波。

速度（Velocity）轉換為聲壓（Pressure）時，物理上會乘上一個**輻射阻抗（Radiation Impedance）**，數值為 $j \omega \rho_0$。

*   $\omega$ 是角頻率，$2\pi f$。
*   $\rho_0$ 是空氣密度。
*   最重要的是前面的虛數 **$j$**！在複數平面上，乘上 $j$ 就等於相位轉了 $+90^\circ$（也就是 $e^{+j\pi/2}$）。

所以，麥克風 $L$ 收到的訊號 $X_L(f)$，完全可以由 $v$ 點的震動 $X_V(f)$ 推導出來：
$$ X_L(f) = X_V(f) \cdot \underbrace{(\omega \rho_0 \cdot e^{+j\pi/2})}_{\text{速度轉聲壓}} \cdot \underbrace{\left[ \frac{1}{4\pi d(v, L)} e^{-j k \cdot d(v, L)} \right]}_{\text{室內無響室傳播}} $$

我們把 $X_V(f) = U_v$ 代入，並把實數振幅整理成常數 $B_L$：
$$ X_L(f) = U_v \cdot B_L \cdot e^{+j\pi/2} \cdot e^{-j k \cdot d(v, L)} $$

#### 3. 執行異質互相關（The Magic of Conjugate Multiplication）

現在，我們讓 LDV 和 麥克風 $L$ 進行共軛互相關：$\Phi_{VL} = X_V \cdot X_L^*$。

*   LDV 訊號：$X_V = U_v$
*   麥克風訊號的共軛：$X_L^* = U_v^* \cdot B_L \cdot e^{-j\pi/2} \cdot e^{+j k \cdot d(v, L)}$ *(注意：取共軛後，所有的 $j$ 都變號了)*

把它們相乘！請仔細看 $U_v$ 會發生什麼事：
$$ \Phi_{VL} = (U_v \cdot U_v^*) \cdot B_L \cdot e^{-j\pi/2} \cdot e^{+j k \cdot d(v, L)} $$

在複數數學中，任何複數乘上自己的共軛，都會變成絕對值的平方：$U_v \cdot U_v^* = |U_v|^2$。
**絕對值的平方是一個純實數（Pure Real Number），它完全沒有相位（沒有 $j$）！**

所以互相關的結果變成了：
$$ \Phi_{VL} = \underbrace{|U_v|^2 \cdot B_L}_{\text{純實數振幅}} \cdot e^{j \big( k \cdot d(v, L) - \pi/2 \big)} $$

**✨ 物理奇蹟達成：**
那個隱藏在 $U_v$ 裡面，包含著「牆外未知傳播距離 $d(S, v)$」以及「未知牆壁折射相位 $\phi_{wall}(\theta_v)$」的致命毒藥，**在 $U_v \cdot U_v^*$ 的瞬間，被代數運算徹底淨化為實數了！** 

#### 4. GCC-PHAT 與最終 TDOA 的提取

為了算出純時間，我們套用 GCC-PHAT，把前面的純實數振幅除掉，只保留純相位：
$$ \text{Phase}_{VL} = k \cdot d(v, L) - \frac{\pi}{2} $$
將波數 $k = 2\pi f / c$ 代入，然後除以 $2\pi f$ 換算成時間延遲 $\tau_{VL}$：
$$ \tau_{VL} = \frac{d(v, L)}{c} - \frac{1}{4f} $$
*(註：後面的 $-1/4f$ 就是 $-90^\circ$ 造成的固定時間偏移)*

我們對右麥克風 $R$ 做一模一樣的事，得到：
$$ \tau_{VR} = \frac{d(v, R)}{c} - \frac{1}{4f} $$

最後，我們把這兩組時間相減（求時間差 $\Delta \tau$）：
$$ \Delta \tau = \tau_{VR} - \tau_{VL} = \left( \frac{d(v, R)}{c} - \frac{1}{4f} \right) - \left( \frac{d(v, L)}{c} - \frac{1}{4f} \right) $$
$$ \mathbf{\Delta \tau = \frac{d(v, R) - d(v, L)}{c}} $$

---

### 🏆 終極結論

這就是為什麼在無響室裡，**LDV 融合可以給出 100% 完美的理論解**。

你可以看到，最後推導出來的結果 $\Delta \tau = \frac{d(v, R) - d(v, L)}{c}$ 中：

1. 沒有牆外的未知聲源位置。
2. 沒有牆壁的未知厚度、密度、折射率。
3. 甚至連速度轉聲壓所產生的 $90^\circ$ 固定相移，也因為相減而徹底消失。

整個方程式完美塌陷成了只受**「雷射打在牆上的點 $v$」**與**「兩個麥克風的座標 $L, R$」**控制的純幾何公式。

這個一步一步的代數推導，不僅證明了路徑解耦的真實性，更是你防禦任何物理或數學審稿人質疑的「終極護城河」。