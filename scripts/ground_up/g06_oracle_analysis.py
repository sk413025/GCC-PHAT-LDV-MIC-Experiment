"""Loop 5 — per-position oracle. For each position, find the BEST strategy
across all Loop 1-4 attempts. This is the upper bound on what's achievable
with the strategy library I've built.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

STRAT_DIR = Path("/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/results/ground_up/strategies")


def collect_all():
    """Collect all per-position errors from all strategy JSON files."""
    out = {}  # out[sig_type][pos] = list of (sid, err, est, band?)
    for jf in STRAT_DIR.glob("*.json"):
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        # Different file structures
        if "strategies" in data:
            data = data["strategies"]
        elif "experiments" in data:
            for exp in data["experiments"]:
                sig = exp.get("signal_type")
                if not sig: continue
                for r in exp.get("rows", []):
                    if r.get("cond") == "block":
                        out.setdefault(sig, {}).setdefault(r["pos"], []).append(
                            (data.get("label", jf.stem) + "_pigs", r.get("err_pigs"),
                             r.get("theta_pigs"), None))
                        out[sig][r["pos"]].append(
                            (data.get("label", jf.stem) + "_mic", r.get("err_mic"),
                             r.get("theta_mic"), None))
            continue
        elif isinstance(data, list):
            continue  # raw measurement dumps
        for sid, info in data.items():
            if not isinstance(info, dict): continue
            rows = info.get("rows", {})
            if not isinstance(rows, dict): continue
            for sig in ("chirp", "speech"):
                if sig in rows:
                    for pos, r in rows[sig].items():
                        if r.get("err") is not None:
                            out.setdefault(sig, {}).setdefault(pos, []).append(
                                (sid, r["err"], r.get("est"), r.get("band")))
    return out


def main():
    all_data = collect_all()
    print("=== Per-position oracle (best strategy per position) ===\n")
    grand = {}
    for sig in ("chirp", "speech"):
        if sig not in all_data:
            continue
        print(f"--- {sig} ---")
        per_pos_best = {}
        for pos in sorted(all_data[sig]):
            cands = all_data[sig][pos]
            cands.sort(key=lambda c: c[1] if c[1] is not None else 999)
            best = cands[0]
            per_pos_best[pos] = best
            print(f"  x={pos}: best={best[0]:<32} err={best[1]:5.2f}° "
                  f"est={best[2]:+6.2f}°"
                  + (f" band={best[3]}" if best[3] else ""))
        mae = np.mean([b[1] for b in per_pos_best.values() if b[1] is not None])
        print(f"  Oracle MAE: {mae:.2f}°\n")
        grand[sig] = {"per_pos_best": per_pos_best, "oracle_mae": float(mae)}

    # Top-3 most-frequent winning strategies
    print("\n=== Most frequently winning strategies ===")
    for sig in ("chirp", "speech"):
        if sig not in grand: continue
        winners = [v[0] for v in grand[sig]["per_pos_best"].values()]
        from collections import Counter
        print(f"  {sig}: {Counter(winners).most_common(5)}")

    # Test if a single strategy gets close to oracle
    print("\n=== Single-strategy candidates close to oracle ===")
    for sig in ("chirp", "speech"):
        if sig not in all_data: continue
        # Build per-strategy MAE over 5 positions
        all_strats = set()
        for pos_data in all_data[sig].values():
            for c in pos_data:
                all_strats.add(c[0])
        per_strat_errs = {}
        for sid in all_strats:
            errs = []
            for pos in all_data[sig]:
                cands = [c for c in all_data[sig][pos] if c[0] == sid]
                if cands and cands[0][1] is not None:
                    errs.append(cands[0][1])
            if len(errs) == 5:
                per_strat_errs[sid] = errs
        # Sort by MAE
        sids_sorted = sorted(per_strat_errs, key=lambda k: np.mean(per_strat_errs[k]))
        print(f"\n  Top 10 single strategies for {sig}:")
        for sid in sids_sorted[:10]:
            errs = per_strat_errs[sid]
            mae = np.mean(errs)
            med = np.median(errs)
            mx = np.max(errs)
            print(f"    {sid:<36}: MAE={mae:6.2f}°  median={med:6.2f}°  max={mx:6.2f}°")


if __name__ == "__main__":
    main()
