# LDVPG-20260209-S4C-fullchain-repro-62a51617

## 背景 / 動機

`exp-validation/ldv-perfect-geometry/STAGE_VALIDATION_RESULTS.md`（commit `62a51617f62e08ff93f53d2be67ecd548b51cf30`）中的 Stage 4-C 有兩筆關鍵表格列：

- `21-0.1V`（對應 `chirp 24 mic truth-ref (5s)`）
- `22-0.1V`（對應 `chirp 23 mic truth-ref (5s)`）

雖然該 commit 已包含結果，但為了可審計、可追溯、可重跑，需要補一份「從輸入資料到最終表格」的 full-chain 重現紀錄，而不是只重跑最後一步（直接餵固定 truth-ref 常數）。

## 實驗目的

完整重現 Stage 4-C 表格兩列數據，並驗證以下流程可再現：

1. 先重跑 Stage 4-B chirp（1s scan）取得 chirp 中心時間。
2. 用相同中心重算 5s chirp truth-ref（`tau_ref/theta_ref`）。
3. 以該 truth-ref 重跑 speech Stage 4-C（5s + guided peak）。
4. 驗證最終結果四捨五入到小數第 2 位與 commit 表格完全一致。

## 範圍

- In scope：`21-0.1V`, `22-0.1V` 的 Stage 4-C（GCC-PHAT）兩列。
- Out of scope：重寫演算法、調參找更好結果、覆蓋 Stage 1-4 全部資料集。

## 前置條件與環境

- Repo root：`/home/sbplab/jiawei/data`
- 目標 commit：`62a51617f62e08ff93f53d2be67ecd548b51cf30`
- Python：`python`（3.12）
- 套件：`numpy`, `scipy`
- 輸出路徑：`/tmp/ldvpg-62a5161-repro`（避免修改 repo tracked artifacts）

## 輸入資料

### Speech（18-22）

- 路徑：repo root 的 `18-0.1V/` ... `22-0.1V/`
- 校驗檔：`exp-validation/ldv-perfect-geometry/datasets/LDVPG_speech_18-22_root.sha256`

### Chirp（23/24）

- `_old_reports/23-chirp(-0.8m)/`
- `_old_reports/24-chirp(-0.4m)/`

## 重現步驟（可直接貼上執行）

### Step 0: Pin commit 與資料完整性

```bash
cd /home/sbplab/jiawei/data
git rev-parse HEAD
sha256sum -c exp-validation/ldv-perfect-geometry/datasets/LDVPG_speech_18-22_root.sha256
find "_old_reports/23-chirp(-0.8m)" -maxdepth 1 -type f -name '*.wav' | sort
find "_old_reports/24-chirp(-0.4m)" -maxdepth 1 -type f -name '*.wav' | sort
```

預期：
- `HEAD == 62a51617...`
- hash 全部 `OK`
- chirp 兩資料夾各 3 個 wav

### Step 1: 重跑 Stage 4-B（1s scan）取得 chirp centers

```bash
set -euo pipefail
RUN_ROOT=/tmp/ldvpg-62a5161-repro
OUT_S4B=$RUN_ROOT/stage4b_scan1s
mkdir -p "$OUT_S4B"

python exp-validation/ldv-perfect-geometry/scripts/stage4_doa_validation.py \
  --data_root _old_reports \
  --speaker '23-chirp(-0.8m)' --speaker_key 22 \
  --output_dir "$OUT_S4B" \
  --segment_mode scan --n_segments 1 --scan_start_sec 0 \
  --eval_window_sec 1 --analysis_slice_sec 5 \
  --gcc_bandpass_low 0 --gcc_bandpass_high 0 \
  --ldv_prealign gcc_phat \
  --scan_hop_sec 0.1 --scan_sort_by psr \
  --scan_psr_min_db 5 --scan_ldv_micl_psr_min_db 4 --scan_tau_err_max_ms 0.3

python exp-validation/ldv-perfect-geometry/scripts/stage4_doa_validation.py \
  --data_root _old_reports \
  --speaker '24-chirp(-0.4m)' --speaker_key 21 \
  --output_dir "$OUT_S4B" \
  --segment_mode scan --n_segments 1 --scan_start_sec 0 \
  --eval_window_sec 1 --analysis_slice_sec 5 \
  --gcc_bandpass_low 0 --gcc_bandpass_high 0 \
  --ldv_prealign gcc_phat \
  --scan_hop_sec 0.1 --scan_sort_by psr \
  --scan_psr_min_db 5 --scan_ldv_micl_psr_min_db 4 --scan_tau_err_max_ms 0.3

C23=$(jq '.segment_centers_sec[0]' "$OUT_S4B/23-chirp(-0.8m)/summary.json")
C24=$(jq '.segment_centers_sec[0]' "$OUT_S4B/24-chirp(-0.4m)/summary.json")
echo "C23=$C23"
echo "C24=$C24"
```

本次實測：
- `C23=11.399999999999975`
- `C24=13.599999999999968`

### Step 2: 用 Step 1 centers 重算 5s chirp truth-ref

```bash
set -euo pipefail
RUN_ROOT=/tmp/ldvpg-62a5161-repro
OUT_S4B=$RUN_ROOT/stage4b_scan1s
OUT_TRUTH5=$RUN_ROOT/chirp_truthref_5s
mkdir -p "$OUT_TRUTH5"

C23=$(jq '.segment_centers_sec[0]' "$OUT_S4B/23-chirp(-0.8m)/summary.json")
C24=$(jq '.segment_centers_sec[0]' "$OUT_S4B/24-chirp(-0.4m)/summary.json")

python exp-validation/ldv-perfect-geometry/scripts/stage4_doa_validation.py \
  --data_root _old_reports \
  --speaker '23-chirp(-0.8m)' --speaker_key 22 \
  --output_dir "$OUT_TRUTH5" \
  --segment_mode scan --n_segments 1 \
  --scan_start_sec "$C23" --scan_end_sec "$C23" --scan_hop_sec 1 \
  --eval_window_sec 5 --analysis_slice_sec 5 \
  --gcc_bandpass_low 0 --gcc_bandpass_high 0 \
  --ldv_prealign gcc_phat \
  --scan_psr_min_db -20 --scan_tau_err_max_ms 10

python exp-validation/ldv-perfect-geometry/scripts/stage4_doa_validation.py \
  --data_root _old_reports \
  --speaker '24-chirp(-0.4m)' --speaker_key 21 \
  --output_dir "$OUT_TRUTH5" \
  --segment_mode scan --n_segments 1 \
  --scan_start_sec "$C24" --scan_end_sec "$C24" --scan_hop_sec 1 \
  --eval_window_sec 5 --analysis_slice_sec 5 \
  --gcc_bandpass_low 0 --gcc_bandpass_high 0 \
  --ldv_prealign gcc_phat \
  --scan_psr_min_db -20 --scan_tau_err_max_ms 10

T23_TAU=$(jq '.results["GCC-PHAT"]["MicL-MicR"].tau_median_ms' "$OUT_TRUTH5/23-chirp(-0.8m)/summary.json")
T23_THETA=$(jq '.results["GCC-PHAT"]["MicL-MicR"].theta_median_deg' "$OUT_TRUTH5/23-chirp(-0.8m)/summary.json")
T24_TAU=$(jq '.results["GCC-PHAT"]["MicL-MicR"].tau_median_ms' "$OUT_TRUTH5/24-chirp(-0.4m)/summary.json")
T24_THETA=$(jq '.results["GCC-PHAT"]["MicL-MicR"].theta_median_deg' "$OUT_TRUTH5/24-chirp(-0.4m)/summary.json")
echo "T23_TAU=$T23_TAU"
echo "T23_THETA=$T23_THETA"
echo "T24_TAU=$T24_TAU"
echo "T24_THETA=$T24_THETA"
```

本次實測：
- `T23_TAU=-1.3072570860049215`
- `T23_THETA=-18.67973713241359`
- `T24_TAU=-0.6345287406996999`
- `T24_THETA=-8.943449115610564`

### Step 3: 重跑 Stage 4-C speech（5s + guided + truth override）

```bash
set -euo pipefail
RUN_ROOT=/tmp/ldvpg-62a5161-repro
OUT_TRUTH5=$RUN_ROOT/chirp_truthref_5s
OUT_S4C=$RUN_ROOT/stage4c_speech_guided_5s
mkdir -p "$OUT_S4C"

T23_TAU=$(jq '.results["GCC-PHAT"]["MicL-MicR"].tau_median_ms' "$OUT_TRUTH5/23-chirp(-0.8m)/summary.json")
T23_THETA=$(jq '.results["GCC-PHAT"]["MicL-MicR"].theta_median_deg' "$OUT_TRUTH5/23-chirp(-0.8m)/summary.json")
T24_TAU=$(jq '.results["GCC-PHAT"]["MicL-MicR"].tau_median_ms' "$OUT_TRUTH5/24-chirp(-0.4m)/summary.json")
T24_THETA=$(jq '.results["GCC-PHAT"]["MicL-MicR"].theta_median_deg' "$OUT_TRUTH5/24-chirp(-0.4m)/summary.json")

python exp-validation/ldv-perfect-geometry/scripts/stage4_doa_validation.py \
  --data_root . --speaker 21-0.1V \
  --output_dir "$OUT_S4C" \
  --segment_mode scan --n_segments 1 \
  --scan_start_sec 100 --scan_end_sec 600 --scan_hop_sec 1 \
  --scan_sort_by tau_err --scan_psr_min_db -20 --scan_tau_err_max_ms 0.3 \
  --eval_window_sec 5 --analysis_slice_sec 5 \
  --gcc_bandpass_low 0 --gcc_bandpass_high 0 \
  --ldv_prealign gcc_phat --gcc_guided_peak_radius_ms 0.3 \
  --truth_tau_ms "$T24_TAU" --truth_theta_deg "$T24_THETA" \
  --truth_label 'chirp 24 mic truth-ref (5s)'

python exp-validation/ldv-perfect-geometry/scripts/stage4_doa_validation.py \
  --data_root . --speaker 22-0.1V \
  --output_dir "$OUT_S4C" \
  --segment_mode scan --n_segments 1 \
  --scan_start_sec 100 --scan_end_sec 600 --scan_hop_sec 1 \
  --scan_sort_by tau_err --scan_psr_min_db -20 --scan_tau_err_max_ms 0.3 \
  --eval_window_sec 5 --analysis_slice_sec 5 \
  --gcc_bandpass_low 0 --gcc_bandpass_high 0 \
  --ldv_prealign gcc_phat --gcc_guided_peak_radius_ms 0.3 \
  --truth_tau_ms "$T23_TAU" --truth_theta_deg "$T23_THETA" \
  --truth_label 'chirp 23 mic truth-ref (5s)'
```

### Step 4: 抽取欄位並驗收（2 位小數）

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('/tmp/ldvpg-62a5161-repro/stage4c_speech_guided_5s')
expected = {
    '21-0.1V': (279.00, -8.94, -6.77, -9.14, -9.00, 0.20, 0.06, True),
    '22-0.1V': (395.00, -18.68, -22.28, -17.44, -22.30, 1.24, 3.62, False),
}

def r2(x):
    return round(float(x) + 1e-12, 2)

ok = True
for spk in ('21-0.1V', '22-0.1V'):
    d = json.loads((root / spk / 'summary.json').read_text())
    got = (
        r2(d['segment_centers_sec'][0]),
        r2(d['truth_reference']['theta_ref_deg']),
        r2(d['results']['GCC-PHAT']['MicL-MicR']['theta_median_deg']),
        r2(d['results']['GCC-PHAT']['Raw_LDV']['theta_median_deg']),
        r2(d['results']['GCC-PHAT']['OMP_LDV']['theta_median_deg']),
        r2(d['results']['GCC-PHAT']['Raw_LDV']['theta_error_median_deg']),
        r2(d['results']['GCC-PHAT']['OMP_LDV']['theta_error_median_deg']),
        bool(d['pass_conditions']['GCC-PHAT']['passed']),
    )
    print(spk, 'got=', got, 'expected=', expected[spk])
    if got != expected[spk]:
        ok = False

print('OVERALL_MATCH=', ok)
PY
```

本次實測：
- `OVERALL_MATCH=True`

## 最終重現結果（2 位小數）

| Speech | Chirp truth-ref | center (s) | θ_ref (°) | Mic θ (°) | Raw θ (°) | OMP θ (°) | abs(Raw-Ref θ) (°) | abs(OMP-Ref θ) (°) | GCC-PHAT |
|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| 21-0.1V | chirp 24 mic truth-ref (5s) | 279.00 | -8.94 | -6.77 | -9.14 | -9.00 | 0.20 | 0.06 | ✅ |
| 22-0.1V | chirp 23 mic truth-ref (5s) | 395.00 | -18.68 | -22.28 | -17.44 | -22.30 | 1.24 | 3.62 | ❌ |

## Artifacts

- Stage 4-B（1s scan）：
  - `/tmp/ldvpg-62a5161-repro/stage4b_scan1s/23-chirp(-0.8m)/summary.json`
  - `/tmp/ldvpg-62a5161-repro/stage4b_scan1s/24-chirp(-0.4m)/summary.json`
- Chirp truth-ref（5s）：
  - `/tmp/ldvpg-62a5161-repro/chirp_truthref_5s/23-chirp(-0.8m)/summary.json`
  - `/tmp/ldvpg-62a5161-repro/chirp_truthref_5s/24-chirp(-0.4m)/summary.json`
- Stage 4-C speech（5s + guided）：
  - `/tmp/ldvpg-62a5161-repro/stage4c_speech_guided_5s/21-0.1V/summary.json`
  - `/tmp/ldvpg-62a5161-repro/stage4c_speech_guided_5s/22-0.1V/summary.json`

## 常見失敗點 / 排查

1. `segment_mode=scan selected 0 segments`
   - 確認 chirp step 使用 `--scan_start_sec 0`
   - 放寬 scan 條件（`scan_psr_min_db`, `scan_tau_err_max_ms`）
2. speech 跑不出 `center=279/395`
   - 確認 Stage 4-C 使用 `--scan_sort_by tau_err --scan_psr_min_db -20 --scan_tau_err_max_ms 0.3`
   - 確認有設定 `--gcc_guided_peak_radius_ms 0.3`
3. 數值微小浮點差異
   - 以 2 位小數驗收（報告表格等級）

## Lineage

- Parent Exp: `LDVPG-20260201-S4C-speech-vs-chirp-truthref`
- Target Report Commit: `62a51617f62e08ff93f53d2be67ecd548b51cf30`
