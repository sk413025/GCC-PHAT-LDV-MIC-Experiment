"""C0 — Quick check: does score=prod or score=min, with 1D grid (y=0 fixed),
help over the baseline? This isn't one of the 9 strategies — it's a sanity
test of the score function itself.

Also tests auto-windowing for chirp.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.signal import butter, filtfilt

from _loader import chirp_groups, speech_groups, load_group
from _pigs import preprocess, estimate_doa_micmic
from _pigs2 import estimate_doa_pigs_1d, auto_window_chirp
from _geometry import expected_doa_deg, REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"


def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def run_score_test(score, signal_type, band_hz=(500, 2000)):
    groups = chirp_groups() if signal_type == "chirp" else speech_groups()
    rows = []
    for (pos, cond), paths in sorted(groups.items()):
        if cond != "block":
            continue
        chans, sr = load_group(paths)
        # Window
        if signal_type == "chirp":
            t0, t1 = auto_window_chirp(chans["mic_l"], sr, dur_s=1.5)
        else:
            t0, t1 = 5.0, 25.0
        n0, n1 = int(t0 * sr), int(t1 * sr)
        chans = {ch: x[n0:n1] for ch, x in chans.items()}
        chans = {ch: preprocess(x, sr) for ch, x in chans.items()}
        chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}

        out = estimate_doa_pigs_1d(chans["mic_l"], chans["mic_r"], chans["ldv"], sr,
                                  band_hz=band_hz, score=score)
        theta_true = expected_doa_deg(float(pos))
        rows.append({"pos": pos, "true": theta_true, "pi": out["theta_deg"],
                    "err": abs(out["theta_deg"] - theta_true), "p": out["p_hat"],
                    "win": (t0, t1)})
    mae = float(np.mean([r["err"] for r in rows]))
    return mae, rows


def main():
    summary = {}
    for score in ("sum", "prod", "min"):
        for sig in ("chirp", "speech"):
            mae, rows = run_score_test(score, sig)
            errs = " ".join(f"{r['pos']}={r['err']:.1f}" for r in rows)
            print(f"score={score:5s} | {sig:6s} | MAE={mae:6.2f}°  errs: {errs}")
            summary[f"{sig}_{score}"] = {"mae": mae, "rows": rows}
    (OUT_DIR / "C0_score_variants.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
