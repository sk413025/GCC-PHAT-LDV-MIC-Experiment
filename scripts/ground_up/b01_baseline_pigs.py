"""Phase B.1 — baseline PI-GS on real data.

Mandatory preprocessing only (DC removal + 60Hz comb notch). No bandpass,
no spectral subtraction, no special weighting. Establishes the floor.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from _loader import chirp_groups, speech_groups, load_group
from _pigs import (estimate_doa_pigs, estimate_doa_micmic, preprocess)
from _geometry import expected_doa_deg, REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-signal-type time windows (seconds), based on Phase A inspection
WINDOWS = {
    "chirp": (0.0, 4.0),    # ~2 chirp bursts
    "speech": (5.0, 25.0),  # 20 sec inside long speech recording
}


def slice_window(x: np.ndarray, sr: int, t0: float, t1: float):
    n0 = int(round(t0 * sr))
    n1 = int(round(t1 * sr))
    return x[n0:min(n1, len(x))]


def run_one(paths_by_pos, signal_type, label, max_lag_s=0.007, band_hz=None,
            apply_preprocess=True):
    rows = []
    t0, t1 = WINDOWS[signal_type]
    for (pos, cond), paths in sorted(paths_by_pos.items()):
        chans, sr = load_group(paths)
        # Window
        chans_w = {ch: slice_window(x, sr, t0, t1) for ch, x in chans.items()}
        # Preprocess
        if apply_preprocess:
            chans_w = {ch: preprocess(x, sr) for ch, x in chans_w.items()}

        x_l = chans_w["mic_l"]
        x_r = chans_w["mic_r"]
        theta_true = expected_doa_deg(float(pos))

        # Mic-only
        theta_mm, tau_mm, _, _ = estimate_doa_micmic(x_l, x_r, sr,
                                                    max_lag_s=0.005,
                                                    band_hz=band_hz)
        err_mm = abs(theta_mm - theta_true)

        # PI-GS only when LDV present (block condition)
        if "ldv" in chans_w:
            x_v = chans_w["ldv"]
            out = estimate_doa_pigs(x_l, x_r, x_v, sr, max_lag_s=max_lag_s,
                                   band_hz=band_hz)
            theta_pi = out["theta_deg"]
            err_pi = abs(theta_pi - theta_true)
            p_hat = out["p_hat"]
        else:
            theta_pi = None
            err_pi = None
            p_hat = None

        rows.append({
            "signal_type": signal_type,
            "pos": pos,
            "cond": cond,
            "theta_true": theta_true,
            "theta_mic": theta_mm,
            "err_mic": err_mm,
            "theta_pigs": theta_pi,
            "err_pigs": err_pi,
            "p_hat": p_hat,
        })
    return rows


def summarize(rows, label):
    # MAE for mic-only (block + unblock separately)
    block = [r for r in rows if r["cond"] == "block"]
    unblock = [r for r in rows if r["cond"] == "unblock"]
    mae_mic_block = float(np.mean([r["err_mic"] for r in block]))
    mae_mic_unblock = float(np.mean([r["err_mic"] for r in unblock])) if unblock else None
    mae_pigs_block = float(np.mean([r["err_pigs"] for r in block if r["err_pigs"] is not None]))
    return {
        "label": label,
        "mae_mic_block": mae_mic_block,
        "mae_mic_unblock": mae_mic_unblock,
        "mae_pigs_block": mae_pigs_block,
        "rows": rows,
    }


def main():
    summary = {"label": "B01 baseline (DC+60Hz notch only)", "experiments": []}
    for signal_type, groups_fn in (("chirp", chirp_groups), ("speech", speech_groups)):
        print(f"\n=== {signal_type.upper()} ===")
        groups = groups_fn()
        rows = run_one(groups, signal_type, f"baseline_{signal_type}")
        s = summarize(rows, f"baseline_{signal_type}")
        # Print per-position
        print(f"{'pos':>6} {'cond':>8} {'true':>7} {'mic_θ':>7} {'mic_err':>8} {'pi_θ':>7} {'pi_err':>8}")
        for r in rows:
            pi = f"{r['theta_pigs']:+6.2f}" if r['theta_pigs'] is not None else "  N/A "
            pe = f"{r['err_pigs']:7.3f}" if r['err_pigs'] is not None else "  N/A  "
            print(f"{r['pos']:>6} {r['cond']:>8} {r['theta_true']:+6.2f}° "
                  f"{r['theta_mic']:+6.2f}° {r['err_mic']:7.3f}° "
                  f"{pi}° {pe}°")
        print(f"  MAE mic-only block:    {s['mae_mic_block']:7.3f}°")
        if s['mae_mic_unblock'] is not None:
            print(f"  MAE mic-only unblock:  {s['mae_mic_unblock']:7.3f}°")
        print(f"  MAE PI-GS block:       {s['mae_pigs_block']:7.3f}°")
        summary["experiments"].append(s)

    out_path = OUT_DIR / "B01_baseline.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
