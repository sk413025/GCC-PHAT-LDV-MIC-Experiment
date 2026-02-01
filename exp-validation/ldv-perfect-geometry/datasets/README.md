# Datasets（Policy A：不把長音檔 commit 進 repo）

本 repo 的實驗結果會被 commit（`exp-validation/ldv-perfect-geometry/validation-results/**`），但**原始語音長音檔**（Speaker 18–22）不 commit。

## Speech 18–22（長音檔；需要自行放置）

本實驗流程預期你在 repo 根目錄放置以下資料夾（或用 `--data_root` 指到你的資料位置）：

```
18-0.1V/
19-0.1V/
20-0.1V/
21-0.1V/
22-0.1V/
```

校驗（hash 必須一致）：
```bash
sha256sum -c exp-validation/ldv-perfect-geometry/datasets/LDVPG_speech_18-22_root.sha256
```

## Chirp 23/24（小檔；會被 commit）

Chirp 會保留在 `_old_reports/23-chirp(-0.8m)/` 與 `_old_reports/24-chirp(-0.4m)/`，作為 Stage 4-B/4-C 的 truth-ref 來源。

