"""Round 7 — chirp specifically: use FULL 13s recording (6 chirp bursts) instead of
just the first burst. More data = better NLMS convergence + more SNR for GCC.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from _loader import chirp_groups, load_group
from _pigs import (MIC_L, MIC_R, C_MPS, MIC_SPACING,
                   preprocess as basic_preprocess)
from _geometry import expected_doa_deg, REPO_ROOT
from h_round6_combiner import (h38a_max_abs_tau, h38c_physical_gating,
                              h38d_three_way, h38b_psr_weighted)

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"


def main():
    results = {}
    groups = chirp_groups()
    for (pos, cond), paths in sorted(groups.items()):
        if cond != "block":
            continue
        chans, sr = load_group(paths)
        # FULL window — all 6 chirp bursts
        chans = {ch: basic_preprocess(x, sr) for ch, x in chans.items()}
        theta_true = expected_doa_deg(float(pos))
        for sid, fn in [
            ("H38a_full13s", h38a_max_abs_tau),
            ("H38c_full13s", h38c_physical_gating),
            ("H38d_full13s", h38d_three_way),
            ("H38b_full13s", h38b_psr_weighted),
        ]:
            try:
                theta = fn(chans, sr, "chirp")
            except Exception as e:
                theta = None
            err = abs(theta - theta_true) if theta is not None else None
            results.setdefault(sid, {}).setdefault("chirp", {})[pos] = {
                "true": theta_true, "est": theta, "err": err}

    print(f"\n{'strategy':<28} | {'chirp MAE':>10}")
    print("-" * 45)
    for sid in results:
        c_errs = [v["err"] for v in results[sid].get("chirp", {}).values()
                  if v["err"] is not None]
        c_mae = float(np.mean(c_errs)) if c_errs else None
        c_str = f"{c_mae:6.2f}°" if c_mae is not None else "  N/A "
        print(f"{sid:<28} | {c_str:>10}")

    print("\nPer-position breakdown:")
    for sid in results:
        print(f"\n=== {sid} ===")
        for pos in sorted(results[sid]["chirp"]):
            r = results[sid]["chirp"][pos]
            if r["err"] is None:
                print(f"  x={pos}: NA")
                continue
            print(f"  x={pos}: true={r['true']:+6.2f}° est={r['est']:+6.2f}° err={r['err']:5.2f}°")

    out = OUT_DIR / "H_round7_chirp_full.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
