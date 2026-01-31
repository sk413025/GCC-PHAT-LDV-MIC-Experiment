# LDV Perfect Geometry Validation Experiment

## 實驗目標

驗證 OMP (Orthogonal Matching Pursuit) 對齊方法，使 LDV (Laser Doppler Vibrometer) 信號能夠替代麥克風進行 TDoA/DoA 估計。

## 實驗設計

採用四階段驗證流程：

| Stage | 驗證項目 | 通過標準 |
|-------|---------|---------|
| Stage 1 | 能量縮減驗證 | OMP 殘差 < Random baseline |
| Stage 2 | 目標麥克風相似度 | GCC-PHAT τ → 0 |
| Stage 3 | 跨麥克風 TDoA | OMP τ ≈ baseline τ |
| Stage 4 | DoA 多方法驗證 | 4種方法 × 4種信號配對 |

## 實驗結果摘要

**Stage 4 最終驗證 - 5/5 speakers PASS**

| Speaker | Baseline τ (ms) | OMP τ (ms) | Error (ms) | Status |
|---------|----------------|------------|------------|--------|
| 18 | -1.229 | -1.229 | 0.000 | ✓ |
| 19 | -1.583 | -1.563 | 0.021 | ✓ |
| 20 | -1.688 | -1.688 | 0.000 | ✓ |
| 21 | -1.625 | -1.625 | 0.000 | ✓ |
| 22 | -1.313 | -1.313 | 0.000 | ✓ |

## 目錄結構

```
experiments/ldv-perfect-geometry/
├── README.md                          # 本文件
├── EXPERIMENT_PLAN.md                 # 初始實驗計劃
├── STAGE_VALIDATION_PLAN.md           # 詳細驗證流程設計
├── STAGE_VALIDATION_RESULTS.md        # 完整驗證結果報告
├── GCC-PHAT_LDV_MIC_完整實驗報告.md   # 中文完整報告
├── full_analysis.py                   # 完整分析腳本
├── scripts/
│   ├── stage1_energy_reduction.py     # Stage 1 驗證腳本
│   ├── stage2_target_similarity.py    # Stage 2 驗證腳本
│   ├── stage3_tdoa_evaluation.py      # Stage 3 單段驗證
│   ├── stage3_multi_segment.py        # Stage 3 多段驗證
│   └── stage4_doa_validation.py       # Stage 4 DoA 驗證
└── validation-results/
    ├── stage1_energy_reduction/       # Stage 1 結果 (5 speakers)
    ├── stage2_target_similarity/      # Stage 2 結果 (5 speakers)
    ├── stage3_tdoa_evaluation/        # Stage 3 單段結果
    ├── stage3_multi_segment/          # Stage 3 多段結果
    └── stage4_doa_validation/         # Stage 4 DoA 結果
```

## 關鍵技術發現

1. **OMP 對齊有效性**：OMP 能夠成功重建 LDV 信號使其與目標麥克風對齊
2. **TDoA 保持精度**：經 OMP 處理的 LDV 信號保持與 baseline 麥克風相同的 TDoA 估計
3. **多方法一致性**：GCC-PHAT、CC、NCC、MUSIC 四種方法結果一致
4. **幾何簽署修正**：τ = (d_right - d_left) / c 是正確的 TDoA 符號約定

## 注意事項

此目錄只包含驗證結果的摘要 JSON 檔案和腳本。大型 WAV 檔案和完整的頻譜資料存放在本地 worktree 中，未包含在 git repository 中。

## 相關文件

- 完整的驗證結果請見 `STAGE_VALIDATION_RESULTS.md`
- 實驗設計細節請見 `STAGE_VALIDATION_PLAN.md`
- 中文報告請見 `GCC-PHAT_LDV_MIC_完整實驗報告.md`
