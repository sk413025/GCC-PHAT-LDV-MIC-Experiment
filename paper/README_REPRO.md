# Paper Reproduction

這份說明是目前 paper 的唯一重現入口。

## 你可以用它做什麼

- 新機器重現目前 paper 用到的核心數據資產
- 只局部更新 `Table 1`、`spatial score` 或 jammer curve
- 產生 paper 讀取的固定中間產物 `paper/generated/`
- 編譯最新版 `paper/main.tex`
- 留下可追溯的 run metadata

## 前置條件

1. 建立 Python 環境：

```bash
conda env create -f environment.paper-repro.yml
conda activate paper-repro
```

2. 準備資料根目錄 `data_root`

這個根目錄應該對應原始 `0222-block`，例如：

```text
/path/to/0222-block
├── 0223-block/
├── 0223-block-6(high)/
├── 0223-unblock-3(high)/
├── 0223-unblock-4(high)/
├── 0223-unblock-5(high)/
└── 0223-unblock-6(high)/
```

## 一條指令重現整篇 paper 資產

```bash
python scripts/reproduce_paper_bundle.py \
  --data_root /path/to/0222-block
```

輸出內容：

- `results/paper_repro_<timestamp>/`
- `paper/generated/table1_latex.tex`
- `paper/generated/table1_values.json`
- `paper/generated/spatial_score_curves.dat`
- `paper/generated/jammer_resilience_curve_sim.dat`
- `paper/main.pdf`

## 只更新單一資產

```bash
python scripts/reproduce_paper_bundle.py \
  --data_root /path/to/0222-block \
  --only table1 \
  --skip_pdf
```

可選值：

- `--only table1`
- `--only spatial_score`
- `--only jammer`

## Paper 圖表對應

- `Table 1`：`scripts/generate_paper_table1.py`
- `Fig. 3 spatial score`：`scripts/generate_spatial_score_figure.py`
- `Fig. 4 jammer curve`：`scripts/generate_jammer_curve_sim.py`

## 常見失敗原因

- `--data_root` 指到錯的層級，缺少 `0223-block/...` WAV
- 尚未先生成 `paper/generated/`，卻直接編譯 `paper/main.tex`
- 本機沒有 `pdflatex` / `bibtex`

## 成功判準

- `results/paper_repro_<timestamp>/run_manifest.json` 存在
- `paper/generated/` 底下三個 paper 資產已更新
- `results/paper_repro_<timestamp>/paper_main.pdf` 存在
- `results/paper_repro_<timestamp>/dataset_manifest.json` 可追溯實際使用 WAV
