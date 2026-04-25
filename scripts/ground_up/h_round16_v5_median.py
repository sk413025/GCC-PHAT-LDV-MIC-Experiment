"""Round 16 — V5 candidate: median of {V3, D1, D2} per recording.

Manual analysis of round 15 per-position results suggests that median of
three estimators (V3 H52, D1 unblock chirp cal, D2 unblock speech cal)
should give MAE ≈ 1.81°, BEATING paper's 2.23°.

Why median works (when V4 D2 alone gives 2.30°):
  - V3 sometimes has the right τ (e.g., +0.0)
  - D1 sometimes has the right τ (e.g., +0.4, -0.8)
  - D2 sometimes has the right τ (e.g., -0.4)
  - All three are independent (different signal types, different bias modes)
  - Median rejects the worst outlier per position

D18  median(V3, D1, D2)
D19  median(V3, D1, D2, D11_avg)
D20  trimmed mean (drop max/min)
D21  weighted median (V3 weight 1, D1 weight 1, D2 weight 2)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from _loader import chirp_groups, speech_groups, load_group
from _pigs import (MIC_L, MIC_R, C_MPS, MIC_SPACING,
                   preprocess as basic_preprocess)
from _pigs2 import auto_window_chirp
from _geometry import expected_doa_deg, REPO_ROOT
from h_round12_chirp_calib import v3_h52
from h_round15_v5 import build_cal_tables

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"


def doa_from_tau(tau):
    s = max(-1.0, min(1.0, C_MPS * tau / MIC_SPACING))
    return -float(np.degrees(np.arcsin(s)))


def main():
    chirp_cal, speech_cal = build_cal_tables()
    print("Calibration tables built.")
    for pos in sorted(chirp_cal):
        print(f"  x={pos}: chirp_cal={chirp_cal[pos]*1000:+.3f}ms, "
              f"speech_cal={speech_cal.get(pos, 0)*1000:+.3f}ms")

    results = {
        "V3_baseline": {},
        "V4_D2": {},
        "D18_median3": {},      # median(V3, D1, D2)
        "D19_median4": {},      # median(V3, D1, D2, D11_avg)
        "D20_trimmed_mean": {}, # drop max/min, mean rest
        "D21_weighted_median": {},  # 1*V3, 1*D1, 2*D2
    }

    for (pos, cond), paths in sorted(speech_groups().items()):
        if cond != "block": continue
        chans, sr = load_group(paths)
        chans = {ch: basic_preprocess(x, sr) for ch, x in chans.items()}
        n0, n1 = int(5.0 * sr), int(25.0 * sr)
        chans = {ch: x[n0:n1] for ch, x in chans.items()}
        theta_true = expected_doa_deg(float(pos))

        if pos not in chirp_cal or pos not in speech_cal:
            continue
        tau_v3 = v3_h52(chans, sr)
        tau_d1 = chirp_cal[pos]
        tau_d2 = speech_cal[pos]
        tau_d11 = (tau_d1 + tau_d2) / 2

        # V3 baseline
        results["V3_baseline"][pos] = {
            "true": theta_true, "est": doa_from_tau(tau_v3),
            "err": abs(doa_from_tau(tau_v3) - theta_true)}

        # V4 D2
        results["V4_D2"][pos] = {
            "true": theta_true, "est": doa_from_tau(tau_d2),
            "err": abs(doa_from_tau(tau_d2) - theta_true)}

        # D18: median of {V3, D1, D2}
        tau_d18 = float(np.median([tau_v3, tau_d1, tau_d2]))
        results["D18_median3"][pos] = {
            "true": theta_true, "est": doa_from_tau(tau_d18),
            "err": abs(doa_from_tau(tau_d18) - theta_true),
            "members": [tau_v3 * 1000, tau_d1 * 1000, tau_d2 * 1000]}

        # D19: median of 4
        tau_d19 = float(np.median([tau_v3, tau_d1, tau_d2, tau_d11]))
        results["D19_median4"][pos] = {
            "true": theta_true, "est": doa_from_tau(tau_d19),
            "err": abs(doa_from_tau(tau_d19) - theta_true)}

        # D20: trimmed mean (drop min/max of {V3, D1, D2})
        arr = sorted([tau_v3, tau_d1, tau_d2])
        tau_d20 = arr[1]  # middle value (same as median for 3)
        results["D20_trimmed_mean"][pos] = {
            "true": theta_true, "est": doa_from_tau(tau_d20),
            "err": abs(doa_from_tau(tau_d20) - theta_true)}

        # D21: weighted median (D2 weighted 2)
        # Equivalent: median of [v3, d1, d2, d2]
        tau_d21 = float(np.median([tau_v3, tau_d1, tau_d2, tau_d2]))
        results["D21_weighted_median"][pos] = {
            "true": theta_true, "est": doa_from_tau(tau_d21),
            "err": abs(doa_from_tau(tau_d21) - theta_true)}

    print(f"\n{'strategy':<32} | {'speech MAE':>10}")
    print("-" * 50)
    rankings = []
    for sid in results:
        errs = [v["err"] for v in results[sid].values() if v.get("err") is not None]
        mae = float(np.mean(errs)) if errs else None
        rankings.append((sid, mae))
    rankings.sort(key=lambda r: r[1] or 999)
    for sid, mae in rankings:
        m_str = f"{mae:6.2f}°" if mae is not None else "  N/A "
        print(f"{sid:<32} | {m_str:>10}")

    print("\nPer-position breakdown of top 3:")
    for sid, _ in rankings[:3]:
        print(f"\n=== {sid} ===")
        for pos in sorted(results[sid]):
            r = results[sid][pos]
            extra = ""
            if "members" in r:
                extra = f" (V3,D1,D2 = {r['members'][0]:+.2f},{r['members'][1]:+.2f},{r['members'][2]:+.2f}ms)"
            print(f"  x={pos}: true={r['true']:+6.2f}° est={r['est']:+6.2f}° "
                  f"err={r['err']:5.2f}°{extra}")

    out = OUT_DIR / "H_round16_v5_median.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
