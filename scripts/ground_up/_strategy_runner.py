"""Shared driver to run a strategy (which is just a (preprocessor, gcc_kwargs) pair)
on chirp + speech datasets and produce comparable summary JSON.

A "preprocessor" takes (chans_dict, sr) -> chans_dict (transforms in place).
A "gcc_kwargs" is a dict passed to estimate_doa_pigs.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from _loader import chirp_groups, speech_groups, load_group
from _pigs import (estimate_doa_pigs, estimate_doa_micmic, preprocess as basic_preprocess)
from _geometry import expected_doa_deg, REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = {
    "chirp": (0.0, 2.0),    # one chirp burst (~1.5s active)
    "speech": (5.0, 25.0),
}


def slice_window(x, sr, t0, t1):
    n0 = int(round(t0 * sr))
    n1 = int(round(t1 * sr))
    return x[n0:min(n1, len(x))]


def default_preprocessor(chans, sr):
    """DC removal + 60 Hz comb notch (mandatory)."""
    return {ch: basic_preprocess(x, sr) for ch, x in chans.items()}


def run_strategy(strategy_id: str, preprocessor=None, pigs_kwargs=None,
                 micmic_kwargs=None, signal_types=("chirp", "speech")):
    preprocessor = preprocessor or default_preprocessor
    pigs_kwargs = pigs_kwargs or {}
    micmic_kwargs = micmic_kwargs or {}

    summary = {"strategy_id": strategy_id, "experiments": []}
    for signal_type in signal_types:
        groups = chirp_groups() if signal_type == "chirp" else speech_groups()
        t0, t1 = WINDOWS[signal_type]
        rows = []
        for (pos, cond), paths in sorted(groups.items()):
            chans, sr = load_group(paths)
            chans = {ch: slice_window(x, sr, t0, t1) for ch, x in chans.items()}
            chans = preprocessor(chans, sr)

            theta_true = expected_doa_deg(float(pos))
            x_l, x_r = chans["mic_l"], chans["mic_r"]
            theta_mm, _, _, _ = estimate_doa_micmic(x_l, x_r, sr, **micmic_kwargs)
            err_mm = abs(theta_mm - theta_true)

            if "ldv" in chans:
                out = estimate_doa_pigs(x_l, x_r, chans["ldv"], sr, **pigs_kwargs)
                theta_pi = out["theta_deg"]
                err_pi = abs(theta_pi - theta_true)
                p_hat = out["p_hat"]
            else:
                theta_pi = None; err_pi = None; p_hat = None

            rows.append({
                "pos": pos, "cond": cond, "theta_true": theta_true,
                "theta_mic": theta_mm, "err_mic": err_mm,
                "theta_pigs": theta_pi, "err_pigs": err_pi,
                "p_hat": p_hat,
            })
        block = [r for r in rows if r["cond"] == "block"]
        unblock = [r for r in rows if r["cond"] == "unblock"]
        summary["experiments"].append({
            "signal_type": signal_type,
            "mae_mic_block": float(np.mean([r["err_mic"] for r in block])),
            "mae_mic_unblock": float(np.mean([r["err_mic"] for r in unblock])) if unblock else None,
            "mae_pigs_block": float(np.mean([r["err_pigs"] for r in block if r["err_pigs"] is not None])),
            "rows": rows,
        })
    out_path = OUT_DIR / f"{strategy_id}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def print_strategy(summary):
    print(f"\n=== {summary['strategy_id']} ===")
    for exp in summary["experiments"]:
        st = exp["signal_type"]
        print(f"  {st:>6} | mic_block={exp['mae_mic_block']:6.2f}° "
              f"mic_unblk={exp['mae_mic_unblock']:6.2f}° "
              f"pigs_block={exp['mae_pigs_block']:6.2f}°")
