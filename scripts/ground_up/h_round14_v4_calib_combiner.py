"""Round 14 — V4: V3 + unblock-chirp calibration combiner.

Round 13 confirmed that unblock-derived τ_LR is highly accurate (2.3° MAE
when using unblock_speech, 3.3° with unblock_chirp). V3 alone gives 3.57°.

V4 idea: instead of REPLACING V3 with calibrated τ, COMBINE them:
  - V3 is a noisy continuous estimator (works for any source signal)
  - Unblock chirp gives a discrete set of accurate calibration points
  - Snap V3 estimate to nearest calibration point
  - But only if V3 estimate is "close enough" to a calibration point;
    otherwise trust V3 (which handles unseen positions)

Strategies:
  V4a  Snap V3 to nearest unblock chirp τ (always)
  V4b  Snap only if |V3 − nearest cal| < 1ms (else trust V3)
  V4c  Weighted average: λ·V3 + (1−λ)·nearest_cal, λ from confidence
  V4d  V3 as classifier: pick position whose unblock_chirp τ is closest
       to V3 τ; output that position's calibrated τ
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.signal import butter, filtfilt

from _loader import chirp_groups, speech_groups, load_group
from _pigs import (MIC_L, MIC_R, C_MPS, MIC_SPACING,
                   preprocess as basic_preprocess)
from _pigs2 import auto_window_chirp
from _geometry import expected_doa_deg, REPO_ROOT
from h_round12_chirp_calib import v3_h52, gcc

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"


def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def doa_from_tau(tau):
    s = max(-1.0, min(1.0, C_MPS * tau / MIC_SPACING))
    return -float(np.degrees(np.arcsin(s)))


def get_unblock_chirp_taus(sr, band=(500, 5000)):
    """Compute unblock chirp τ_LR per position. Used as calibration table."""
    taus = {}
    for (pos, cond), paths in sorted(chirp_groups().items()):
        if cond != "unblock": continue
        chans, sr_check = load_group(paths)
        chans = {ch: basic_preprocess(x, sr_check) for ch, x in chans.items()}
        t0, t1 = auto_window_chirp(chans["mic_l"], sr_check, dur_s=1.6)
        n0, n1 = int(t0 * sr_check), int(t1 * sr_check)
        chans = {ch: x[n0:n1] for ch, x in chans.items()}
        chans_b = {ch: bp(x, sr_check, *band) for ch, x in chans.items()}
        tau = gcc(chans_b["mic_l"], chans_b["mic_r"], sr_check, band)
        taus[pos] = tau
    return taus


def main():
    cal_taus = get_unblock_chirp_taus(48000)
    print(f"Calibration table from unblock chirp:")
    for pos in sorted(cal_taus):
        print(f"  x={pos}: τ_cal = {cal_taus[pos]*1000:+7.3f} ms = "
              f"{doa_from_tau(cal_taus[pos]):+6.2f}°")
    cal_arr = np.array([cal_taus[p] for p in sorted(cal_taus)])
    print()

    results = {"V3_baseline": {}, "V4a_snap_always": {},
               "V4b_snap_if_close": {}, "V4c_weighted_blend": {},
               "V4d_v3_classifier": {}}

    for (pos, cond), paths in sorted(speech_groups().items()):
        if cond != "block": continue
        chans, sr = load_group(paths)
        chans = {ch: basic_preprocess(x, sr) for ch, x in chans.items()}
        n0, n1 = int(5.0 * sr), int(25.0 * sr)
        chans = {ch: x[n0:n1] for ch, x in chans.items()}
        theta_true = expected_doa_deg(float(pos))

        # V3 baseline τ
        tau_v3 = v3_h52(chans, sr)
        theta_v3 = doa_from_tau(tau_v3)
        results["V3_baseline"][pos] = {
            "true": theta_true, "est": theta_v3,
            "err": abs(theta_v3 - theta_true)}

        # V4a: snap V3 to nearest cal τ
        nearest_idx = int(np.argmin(np.abs(cal_arr - tau_v3)))
        tau_snap = cal_arr[nearest_idx]
        theta_v4a = doa_from_tau(tau_snap)
        results["V4a_snap_always"][pos] = {
            "true": theta_true, "est": theta_v4a,
            "err": abs(theta_v4a - theta_true),
            "v3_tau_ms": tau_v3 * 1000,
            "snapped_to_ms": tau_snap * 1000}

        # V4b: snap only if |V3 − nearest| < 0.8ms
        if abs(tau_v3 - tau_snap) < 0.8e-3:
            theta_v4b = theta_v4a  # snap
        else:
            theta_v4b = theta_v3   # trust V3
        results["V4b_snap_if_close"][pos] = {
            "true": theta_true, "est": theta_v4b,
            "err": abs(theta_v4b - theta_true)}

        # V4c: weighted blend (lambda based on distance to nearest)
        dist = abs(tau_v3 - tau_snap)
        # Weight: more confidence in cal if dist is small
        lam = max(0.0, min(1.0, 1.0 - dist / 1.0e-3))  # 0 if dist≥1ms, 1 if dist=0
        tau_blend = lam * tau_snap + (1 - lam) * tau_v3
        theta_v4c = doa_from_tau(tau_blend)
        results["V4c_weighted_blend"][pos] = {
            "true": theta_true, "est": theta_v4c,
            "err": abs(theta_v4c - theta_true)}

        # V4d: V3 as classifier - assign to nearest position from cal table
        # Then output that position's cal τ
        sorted_positions = sorted(cal_taus.keys())
        nearest_pos = sorted_positions[nearest_idx]
        tau_class = cal_taus[nearest_pos]
        theta_v4d = doa_from_tau(tau_class)
        results["V4d_v3_classifier"][pos] = {
            "true": theta_true, "est": theta_v4d,
            "err": abs(theta_v4d - theta_true),
            "predicted_pos": nearest_pos}

    print(f"{'strategy':<32} | {'speech MAE':>10}")
    print("-" * 48)
    for sid in results:
        errs = [v["err"] for v in results[sid].values() if v.get("err") is not None]
        mae = float(np.mean(errs)) if errs else None
        m_str = f"{mae:6.2f}°" if mae is not None else "  N/A "
        print(f"{sid:<32} | {m_str:>10}")

    print("\nPer-position breakdown:")
    for sid in results:
        print(f"\n=== {sid} ===")
        for pos in sorted(results[sid]):
            r = results[sid][pos]
            extra = ""
            if "v3_tau_ms" in r:
                extra = f" v3={r['v3_tau_ms']:+.2f}ms→cal={r['snapped_to_ms']:+.2f}ms"
            if "predicted_pos" in r:
                extra = f" predicted_pos={r['predicted_pos']}"
            print(f"  x={pos}: true={r['true']:+6.2f}° est={r['est']:+6.2f}° "
                  f"err={r['err']:5.2f}°{extra}")

    out = OUT_DIR / "H_round14_v4.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
