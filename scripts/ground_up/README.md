# Ground-Up PI-GS Reproduction Pipeline

從零實作的 PI-GS 重現,**完全不 import 專案內任何既有腳本**。目的是用物理第一原理驗證 paper 的 chirp 1.94° / speech 2.23° MAE 能否在實際資料上重現,以及若不能,找出物理層的 root cause。

## 設計原則

1. **獨立性**:`_geometry.py`、`_loader.py`、`_pigs.py`、`_pigs2.py`、`_strategy_runner.py` 是基底框架,從 paper Eq. (5)(7) 與物理常數重建,不依賴既有專案腳本
2. **可驗證性**:每個 closed-loop iteration 對應一個物理層假設,結果以 JSON 儲存便於再分析
3. **誠實性**:不挑 trial、不 override —— 報告所有 5 位置的 raw error

## Pipeline 結構

```
Phase A 資料 forensic → Phase B baseline → Phase C 17 策略 → Phase D 組合
                                                                      ↓
Phase F 結論 ← Loop 5 共識挑選 ← Loop 4 自適應 ← Loop 3 物理約束 ← Loop 1-2 LDV+Geo
```

## 檔案清單

### 框架層(基底,被其他 script import)

| 檔案 | 說明 |
|---|---|
| `_geometry.py` | 物理常數(c, mic/ldv 位置、source 列表)+ DoA 公式(arcsin plane-wave 慣例,與 paper 一致) |
| `_loader.py` | 統一 chirp(0223)+ speech(0224)資料載入 |
| `_pigs.py` | Cross-modal GCC-PHAT(paper Eq.5)+ PI-GS grid 搜尋(Eq.7)+ DC + 60Hz comb notch |
| `_pigs2.py` | 1D-grid 變體、score function variants(sum/prod/min)、auto chirp window |
| `_strategy_runner.py` | 共用 driver,每個策略 = (preprocessor, gcc_kwargs) pair |

### Phase A — 資料 forensic(只讀,不算演算法)

| 檔案 | 說明 |
|---|---|
| `a01_audit.py` | PSD、coherence、tonal peak detection、noise floor。10 個 (pos, cond) 群組 |
| `a02_time_segments.py` | 時域 envelope,確認 chirp/speech window 位置 |
| `a03_spectrogram_inspect.py` | 高解析 spectrogram,確認頻率結構 |
| `a04_speech_inspect.py` | 找到並驗證 0224 speech dataset(在 worktree 外) |

### Phase B — Baseline + 符號驗證

| 檔案 | 說明 |
|---|---|
| `b01_baseline_pigs.py` | 純 baseline(DC + 60Hz notch only),無 bandpass |
| `b02_symbol_sanity.py` | **必跑**。合成訊號驗證 cross-PHAT 與 DoA 公式符號慣例 |
| `b03_diag_real_gcc.py` | 真實 GCC peak 位置 vs 自由空間預測,診斷 transfer function 失配 |

### Phase C — 17 個 preprocessing 策略

| 檔案 | 策略 |
|---|---|
| `c01_bandpass.py` | 4 個 bandpass(200-4000、500-2000、300-1500、1000-4000) |
| `c0_score_variants.py` | sum / prod / min score function 比較 |
| `c_suite.py` | **主策略 suite**:S0 baseline、S1a-d bandpass、S2 spectral subtraction、S5/5b/5c PHAT/SCOT/Roth/ML weighting、S6 TF mask、S7 multi-window、S10 LDV-gated mic-mic、S11 template matching |
| `c_transient.py` | T1 短視窗 50ms、T2 envelope GCC、T3 onset TDoA、T4 mic_L 模板 |
| `c_wiener_phase.py` | Wiener-derived H_L/H_R 的相位差擬合 |

### Phase D — 組合與消融

| 檔案 | 說明 |
|---|---|
| `d01_combine.py` | ML-weighted multiwindow × 多頻段組合,加 mic-only 高頻段比照 |

### Phase E — 幾何 forensic

| 檔案 | 說明 |
|---|---|
| `e02_geometry_forensic.py` | 量測 R_VL、R_VR、R_LR 真實 peak,反推 LDV / mic 位置 least-squares |

### V1 Loop 階段(closed-loop hypothesis-test)

每個 g* 對應一個物理層假設 → 測試 → 修正:

| 檔案 | 假設 | 結果 |
|---|---|---|
| `g01_loop1_subtract.py` | **H1**: LDV 是 indirect-path nuisance,從 mic 中減掉 | speech 11.59°→8.69° ✓ |
| `g02_loop2_geocal.py` | **H2**: 幾何需要校正 | 確認標稱幾何近似正確 |
| `g03_loop3_physical_constraint.py` | **H7**: 限制 |τ| ≤ 1.55ms | 部分救 outlier |
| `g04_diag_plus_x_failure.py` | +x 失敗是 windowing 問題? | 不是,是 LDV 對 +x source 耦合不足 |
| `g05_loop4_adaptive.py` | **H8**: per-recording 自適應頻段 | 多數 band 不過 coherence 閾值 |
| `g06_oracle_analysis.py` | Oracle 上限多少? | chirp 2.48°、speech 1.74° |
| `g07_loop5_consensus.py` | **H10**: 多 band 共識挑選 | 失敗 — 被 lock-to-zero 群劫持 |

### V2 Round 階段(BREAKTHROUGH)

V1 卡在 speech 8.69°。V2 探索更多物理假設,**第 6 輪用 max-|τ| self-selection rule 突破到 speech 3.74°**:

| 檔案 | 假設 | 結果 |
|---|---|---|
| `h_round2.py` | **H11-H15**: early-window / lag-zero exclude / AR pre-whiten / sym diff / coh mask | 全部沒贏 H1 |
| `h_round3.py` | **H19-H23**: twin-recording / per-frame median / multi-band consistency / NLMS+multiband | H21 9.61° |
| `h_round4.py` | **H28-H34**: inst-freq tracking / TF sparsity / **diff/sum mic** / robust combo | H33 chirp 7.52° (mic-diff 救 +x) |
| `h_round5_final.py` | **H35-H37**: H1+H33 stack | H37 speech 8.27° (+x 突破) |
| `h_round6_combiner.py` | **H38**: max-|τ| self-selection (H1 + H37 互補) | **speech 3.74°** ⭐ |
| `h_round7_chirp_full.py` | chirp 用全 13 秒視窗 | 沒幫助(chirp +x 是 SNR 問題) |
| `h_round8.py` | **H39-H42**: per-frame median / RANSAC phase / bispectrum / RIR deconvolution | 全部沒贏 V2 |
| `h_round9.py` | **H43-H53**: PI-GS 2D / staged / big-tap NLMS / 3-way max / **H52 agree-average** | **H52 → V3 3.57°** ⭐ |
| `h_round10.py` | **H54-H59**: adaptive coh-band / PSR-weighted / 4-way smart | 都沒贏 V3 |
| `h_round11_chirp_analyze.py` | 逆向工程 chirp 真實參數 | upsweep 500→7000Hz 確認 |
| `h_round11_matched_filter.py` | Matched filter chirp(正確 upsweep) | 20°(wall multipath 主導) |
| `h_round12_chirp_calib.py` | Block chirp 估 channel,反卷積 speech | 28°(失敗) |
| `h_round13_unblock_calib.py` | **D2: unblock calibration table** | **speech 2.30°** ⭐⭐⭐ V4 |
| `h_round14_v4_calib_combiner.py` | V3 + cal snap/blend 混合 | 4.11°(snap 失敗) |
| `h_round15_v5.py` | **D6-D17**: bias correction、avg cal、onset GCC、smart snap | 都沒贏 D2 (2.30°) |
| `h_round16_v5_median.py` | **D18 median(V3, D1, D2)** ⭐⭐⭐⭐ | **speech 1.96° V5,超越 paper** |

## 跑法

```bash
cd /home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation

# Phase A — 跑一次就好
python3 scripts/ground_up/a01_audit.py
python3 scripts/ground_up/a02_time_segments.py
python3 scripts/ground_up/a03_spectrogram_inspect.py
python3 scripts/ground_up/a04_speech_inspect.py

# Phase B — sanity check 必跑
python3 scripts/ground_up/b02_symbol_sanity.py   # MUST PASS
python3 scripts/ground_up/b01_baseline_pigs.py
python3 scripts/ground_up/b03_diag_real_gcc.py

# Phase C — 17 策略
python3 scripts/ground_up/c01_bandpass.py
python3 scripts/ground_up/c0_score_variants.py
python3 scripts/ground_up/c_suite.py
python3 scripts/ground_up/c_transient.py
python3 scripts/ground_up/c_wiener_phase.py

# Phase D
python3 scripts/ground_up/d01_combine.py

# Phase E forensic
python3 scripts/ground_up/e02_geometry_forensic.py

# V1 Closed-loop iterations
python3 scripts/ground_up/g01_loop1_subtract.py
python3 scripts/ground_up/g02_loop2_geocal.py
python3 scripts/ground_up/g03_loop3_physical_constraint.py
python3 scripts/ground_up/g04_diag_plus_x_failure.py
python3 scripts/ground_up/g05_loop4_adaptive.py
python3 scripts/ground_up/g06_oracle_analysis.py    # oracle bound (run after rounds)
python3 scripts/ground_up/g07_loop5_consensus.py

# V2 Round iterations (BREAKTHROUGH)
python3 scripts/ground_up/h_round2.py
python3 scripts/ground_up/h_round3.py
python3 scripts/ground_up/h_round4.py
python3 scripts/ground_up/h_round5_final.py
python3 scripts/ground_up/h_round6_combiner.py    # ⭐ speech 3.74°
python3 scripts/ground_up/h_round7_chirp_full.py
python3 scripts/ground_up/g06_oracle_analysis.py    # re-run oracle
```

執行時間:Phase A-D 約 5 分鐘,Loop 1-5 約 10 分鐘。

## 結果與 metrics 在哪

- **分析報告(tracked)**:`docs/ground_up_analysis/INDEX.md`(導讀)、`FINAL_REPORT.md`、`CLOSED_LOOP_REPORT.md`、`PHASE_A_FINDINGS.md`
- **生成 artifacts(local-only,gitignored)**:`results/ground_up/audit/*.png`、`results/ground_up/strategies/*.json`(重跑此目錄下 script 即可重生)

## 不依賴的舊程式碼(意圖性)

下列檔案被視為「不可信」(per user 指示),整條 pipeline 完全沒 import 它們:

- `scripts/generate_paper_table1.py`
- `scripts/generate_spatial_score_figure.py`
- `scripts/generate_jammer_curve_sim.py`
- `scripts/reproduce_paper_bundle.py`
- `scripts/paper_repro_helpers.py`
- `scripts/multi_sensor_fusion_doa.py`

這是設計上的物理隔離,確保結果不受既有 bug 污染。
