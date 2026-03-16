# Paper Reproduction

這份說明是目前 paper 的正式重現入口。重點不是叫你自己猜資料該放哪，而是讓你照著 repo 內建路徑走，就能把 paper 的核心數據產物重新生出來。

## 這套流程可以做什麼

- 新機器 clone 後，直接拉 `0223` 資料並重現 paper
- 只局部更新 `Table 1`、`spatial score` 或 jammer curve
- 把 paper 依賴的中間產物同步到 `paper/generated/`
- 編譯最新版 `paper/main.tex`
- 留下 run metadata，方便之後回頭追溯

## 情境 1：同團隊第一次在新機器重現

1. 建立 Python 環境：

```bash
conda env create -f environment.paper-repro.yml
conda activate paper-repro
```

2. clone repo 後拉 `0223` LFS 資料：

```bash
git lfs pull --include="dataset/0223/**"
```

3. 一條指令重現 paper：

```bash
python scripts/reproduce_paper_bundle.py --data_root dataset/0223
```

跑完後你應該看到：

- `results/paper_repro_<timestamp>/`
- `paper/generated/table1_latex.tex`
- `paper/generated/table1_values.json`
- `paper/generated/spatial_score_curves.dat`
- `paper/generated/jammer_resilience_curve_sim.dat`
- `results/paper_repro_<timestamp>/paper_main.pdf`

## 情境 2：只想更新單一資產

```bash
python scripts/reproduce_paper_bundle.py \
  --data_root dataset/0223 \
  --only table1 \
  --skip_pdf
```

可選值：

- `--only table1`
- `--only spatial_score`
- `--only jammer`

## 情境 3：同一台機器開新 worktree

這個 repo 的 worktree 共用同一份 `.git/lfs/objects`。白話講，只要：

- 這個 branch 裡有 `dataset/0223` 的 LFS pointer
- 你這台機器曾經做過一次 `git lfs pull --include="dataset/0223/**"`

那新 worktree 通常就能直接用，不需要再手動複製一份 `0223` 資料夾。

## 官方資料根長怎樣

```text
dataset/0223
├── 0223-block/
├── 0223-block-1/
├── 0223-block-6(high)/
├── 0223-unblock-3(high)/
├── 0223-unblock-4(high)/
├── 0223-unblock-5(high)/
└── 0223-unblock-6(high)/
```

這就是 paper 現在的 canonical dataset root。`paper/repro_asset_manifest.json` 內所有資料相對路徑，都是以這一層為準。

## Paper 圖表對應

- `Table 1`：`scripts/generate_paper_table1.py`
- `Fig. 3 spatial score`：`scripts/generate_spatial_score_figure.py`
- `Fig. 4 jammer curve`：`scripts/generate_jammer_curve_sim.py`

## 常見失敗原因

- 還沒先跑 `git lfs pull --include="dataset/0223/**"`
- `--data_root` 指錯層級，不是 `dataset/0223`
- 本機沒有 `pdflatex` / `bibtex`
- 想直接編譯 `paper/main.tex`，但還沒先生成 `paper/generated/`

## 成功判準

- `results/paper_repro_<timestamp>/run_manifest.json` 存在
- `results/paper_repro_<timestamp>/dataset_manifest.json` 能列出實際使用到的 WAV 與 hash
- `paper/generated/` 底下三個 paper 資產已更新
- `results/paper_repro_<timestamp>/paper_main.pdf` 存在
