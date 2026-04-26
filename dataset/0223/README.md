# 0223 Dataset

這裡是目前 paper 重現流程的正式資料根。你可以把它想成「repo 內建的 canonical 0223 錄音包」。

## 這批資料是拿來做什麼

- `Table 1` 的 blocked / unblocked 誤差
- `Fig. 3` 的 representative spatial score slice
- `Fig. 4` 的 jammer resilience curve

如果你只是要重現 paper，官方用法就是：

```bash
git lfs pull --include="dataset/0223/**"
python scripts/reproduce_paper_bundle.py --data_root dataset/0223
```

## 目錄結構

- `0223-block/`
  - 內含 `0223-block-2`、`0223-block-3(high)`、`0223-block-4(high)`、`0223-block-5(high)`、`0223-block-7(high)` 與 `0223-unblock-7(high)`
- `0223-block-1/`
  - 較早期的 near/far 命名版本，保留作 provenance
- `0223-block-6(high)/`
- `0223-unblock-3(high)/`
- `0223-unblock-4(high)/`
- `0223-unblock-5(high)/`
- `0223-unblock-6(high)/`

## 情境 1：新機器第一次拉資料

```bash
git lfs pull --include="dataset/0223/**"
```

跑完後，`dataset/0223/manifest.json` 會是你核對完整性的主清單。

## 情境 2：想知道 paper 實際用了哪些檔案

看這兩個地方：

- `paper/repro_asset_manifest.json`
- `dataset/0223/manifest.json`

前者告訴你 paper 會吃哪幾個 WAV，後者告訴你整批 `0223` 有哪些檔案、hash 是什麼、哪些是 paper 直接用到的。

## 使用邊界

- 這裡只正式收 `0223` 系列，不包含更早的 `block/`、`noblock/`
- 目前 paper 的官方資料根就是 `dataset/0223`
- 若未來新增新的 paper 圖表，也應該以這裡為基底擴充 manifest，而不是再回到 `/home/.../0222-block`
