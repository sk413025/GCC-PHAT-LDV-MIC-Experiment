# Ground-Up PI-GS Reproduction — Results & Analysis

## 重點數字 (V2 update with Round 5+6 breakthrough)

| 指標 | Paper 宣稱 | V1 ground-up | **V2 ground-up (H38a)** |
|---|---|---|---|
| Chirp MAE (block) | 1.94° (override) | 8.43° / 2.48° oracle | 10.31° / **0.93° oracle** |
| Speech MAE (block) | 2.23° | 8.69° / 1.74° oracle | **3.74°** ⭐ / 1.59° oracle |
| Mic unblock chirp | 4.03° | 2.16° | (same) |
| Mic unblock speech | 4.04° | ~3° | (same) |

**V1 → V2 突破**(speech):從 8.69° 改到 **3.74°**(改善 57%)。關鍵是發現 H1 與 H37 互補的失敗模式,用 **max-|τ| self-selection rule** 自動選非失敗的估計器。完整討論見 [`ROUND_5_6_BREAKTHROUGH.md`](ROUND_5_6_BREAKTHROUGH.md)。

**核心物理直覺**:Lock-to-zero 是演算法 failure 的 signature(|τ|≈0),正確估計則 |τ|≥0.2 ms。max-|τ| rule 利用此**拓撲特性**自然挑出沒失敗的演算法,不需 PSR、不需 ground truth。

## 文件導讀

依閱讀順序:

1. **[`PHASE_A_FINDINGS.md`](PHASE_A_FINDINGS.md)** —— 資料層 forensic 發現:
   - F1 真正的 speech dataset 不在 worktree 內
   - F2 60Hz 諧波系統污染
   - F3 LDV-Mic coherence 帶寬隨位置劇烈變動
   - F4 LDV 訊號強度因位置變 30 倍
   - F7 MIC_R 增益是 MIC_L 的 50%
   - 物理層補充推論(為何 LDV 抗 jammer、為何 PHAT 在此場景反而傷害效能)

2. **[`FINAL_REPORT.md`](FINAL_REPORT.md)** —— Phase A-F 全程結果:
   - 17 種策略效能 ranking
   - 重大資料層發現(0223 chirp vs 0224 speech 分離、幾何 forensic、MIC asymmetry、60Hz hum 污染、coherence 帶寬可變)
   - 演算法層發現(17 策略均無法逼近 paper 數字)

3. **[`CLOSED_LOOP_REPORT.md`](CLOSED_LOOP_REPORT.md)** —— V1 Loop 1-5 closed-loop:
   - 5 個物理層假設的測試與修正循環
   - **Loop 1 (H1)**: LDV 是 nuisance reference → speech 改善 25%
   - **Loop 2 (H2)**: 幾何 calibration → 確認標稱幾何 OK
   - **Loop 3 (H7)**: 物理約束 |τ|≤1.55ms → 部分救 outlier
   - **Loop 4-5**: 自適應 / 共識挑選 → 被 lock-to-zero 共同失敗模式劫持
   - **V1 結論**:資料層 OK、演算法庫 OK,缺的是 self-selection 機制

4. **[`ROUND_5_6_BREAKTHROUGH.md`](ROUND_5_6_BREAKTHROUGH.md)** ⭐ —— V2 突破:
   - **Round 4 H33**: differential mic (mic_L-mic_R 對 mic_L+mic_R) 救 +x chirp
   - **Round 5 H37**: NLMS + diff/sum + multi-band median + reject-zero → +x 解放
   - **Round 6 H38a**: max-|τ| self-selection 把 H1 (-x) + H37 (+x) 互補性發揮 → **speech 3.74°**
   - 30+ 策略完整 ranking,V1 → V2 per-position 改善表

## 結果檔案分類(local-only artifacts,gitignored;重跑可生)

> 路徑相對於 repo root。如未存在,執行 `scripts/ground_up/` 對應 script 即可重生。

### `results/ground_up/audit/` — Phase A 訊號層 forensic

每位置 (5 source × 2 condition) 一張 PSD + coherence 圖:

```
+0.0_block.png  +0.0_unblock.png   ← x=0.0m source
+0.4_block.png  +0.4_unblock.png
+0.8_block.png  +0.8_unblock.png
-0.4_block.png  -0.4_unblock.png
-0.8_block.png  -0.8_unblock.png
```

額外:
- `envelopes.png` — chirp dataset 5 位置時域 envelope
- `speech_envelopes.png` — speech dataset 同樣
- `spec_*.png` — 高解析 spectrogram
- `diag_R_*.png` — cross-modal GCC peak 位置診斷
- `audit_data.json` — 機讀統計
- `audit_summary.md` — 機讀文字摘要

### `results/ground_up/strategies/` — Phase B-G 各策略 metrics

每個 JSON 含 `mae_chirp / mae_speech / per-position rows`:

| 檔名 | 內容 |
|---|---|
| `B01_baseline.json` | DC + 60Hz notch only (起點) |
| `C0_score_variants.json` | sum/prod/min score |
| `C1a-d_bp*.json` | 4 個 bandpass 變體 |
| `C_suite_summary.json` | 主 suite (S0-S11) 共 12 策略 |
| `C_transient_summary.json` | 4 個 transient/onset 變體 |
| `C_wiener_phase.json` | Wiener phase fit |
| `D_combinations.json` | Phase D 組合 + mic-only 高頻段 |
| `G_loop1_wiener_subtract.json` | Loop 1 LDV-subtract (H1 驗證) |
| `G_loop2_geocal.json` | Loop 2 幾何 forensic |
| `G_loop3_physical_constraint.json` | Loop 3 物理約束 |
| `G_loop4_adaptive.json` | Loop 4 自適應頻段 |
| `G_loop5_consensus.json` | Loop 5 多 band 共識 |

## 使用建議

讀完 3 份 markdown 報告後,如果要驗證某個結論,直接看對應的 JSON 即可:

```bash
# 例如查看 Loop 1 H1 假設驗證的 per-position 結果
python3 -c "
import json
data = json.load(open('../../results/ground_up/strategies/G_loop1_wiener_subtract.json'))
for sid in data['strategies']:
    if 'G1b_nlms_300_4000' in sid:
        for sig, rows in data['strategies'][sid]['rows'].items():
            print(f'{sid} {sig}:')
            for pos, r in rows.items():
                print(f'  x={pos}: err={r[\"err\"]:.2f}°')
"
```

## 重跑

完整 pipeline 重跑見 [`scripts/ground_up/README.md`](../../scripts/ground_up/README.md)。
