"""Loop 4 diag — investigate +x failure mode.

Findings so far:
  - chirp +x side mic RMS is ~30x smaller than -x side
  - Maybe windowing using mic_L threshold misses chirp burst on +x
  - Maybe the chirp recording at +x is qualitatively different

Test plan:
  1. Show per-position mic_L, mic_R, ldv RMS for chirp block
  2. Re-do windowing using LDV (which has reliable signal everywhere)
  3. Re-run mic-mic GCC + Wiener + physical constraint with new windowing
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.signal import butter, filtfilt
from scipy import signal as sp

from _loader import chirp_groups, load_group
from _pigs import (cross_phat, MIC_L, MIC_R, C_MPS, MIC_SPACING,
                   preprocess as basic_preprocess)
from _geometry import expected_doa_deg, REPO_ROOT


def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def auto_window_ldv(x_v, sr, dur_s=1.6):
    """Find first chirp burst using LDV (always strong)."""
    e = np.abs(sp.hilbert(x_v))
    n = int(sr * 0.02)
    e = np.convolve(e, np.ones(n) / n, mode="same")
    thresh = 0.3 * e.max()
    n0 = int(np.argmax(e > thresh))
    return n0 / sr, (n0 + int(dur_s * sr)) / sr


def main():
    groups = chirp_groups()
    print(f"{'pos':>5} | {'mic_L_RMS':>11} {'mic_R_RMS':>11} {'ldv_RMS':>11} | "
          f"{'win (mic)':>14} {'win (ldv)':>14}")
    print("-" * 80)
    for (pos, cond), paths in sorted(groups.items()):
        if cond != "block":
            continue
        chans, sr = load_group(paths)
        # Mic-derived window
        from _pigs2 import auto_window_chirp
        m_t0, m_t1 = auto_window_chirp(chans["mic_l"], sr, dur_s=1.6)
        # LDV-derived window
        l_t0, l_t1 = auto_window_ldv(chans["ldv"], sr)
        # RMS over LDV window (most reliable)
        n0, n1 = int(l_t0 * sr), int(l_t1 * sr)
        rms_L = np.sqrt(np.mean(chans["mic_l"][n0:n1] ** 2))
        rms_R = np.sqrt(np.mean(chans["mic_r"][n0:n1] ** 2))
        rms_V = np.sqrt(np.mean(chans["ldv"][n0:n1] ** 2))
        print(f"{pos:>5} | {rms_L:11.6f} {rms_R:11.6f} {rms_V:11.6f} | "
              f"{m_t0:5.2f}-{m_t1:5.2f}  {l_t0:5.2f}-{l_t1:5.2f}")

    # Now re-test G1b NLMS using LDV-derived window
    print("\n=== Re-test G1b NLMS 300-4000 with LDV-derived window ===")
    from _pigs import preprocess as pp
    PHYS_MAX = MIC_SPACING / C_MPS * 1.05
    band = (300, 4000)
    rows = []
    for (pos, cond), paths in sorted(groups.items()):
        if cond != "block":
            continue
        chans, sr = load_group(paths)
        l_t0, l_t1 = auto_window_ldv(chans["ldv"], sr)
        n0, n1 = int(l_t0 * sr), int(l_t1 * sr)
        chans_w = {ch: x[n0:n1] for ch, x in chans.items()}
        chans_w = {ch: pp(x, sr) for ch, x in chans_w.items()}
        chans_w = {ch: bp(x, sr, *band) for ch, x in chans_w.items()}
        # NLMS subtract
        from g01_loop1_subtract import nlms_subtract
        e_L, _ = nlms_subtract(chans_w["ldv"], chans_w["mic_l"], taps=256)
        e_R, _ = nlms_subtract(chans_w["ldv"], chans_w["mic_r"], taps=256)
        # Constrained mic-mic GCC
        from g03_loop3_physical_constraint import constrained_micmic_gcc, doa_from_tau
        tau, psr = constrained_micmic_gcc(e_L, e_R, sr, band, max_lag_s=PHYS_MAX,
                                         return_psr=True)
        theta = doa_from_tau(tau)
        true_th = expected_doa_deg(float(pos))
        rows.append({"pos": pos, "true": true_th, "est": theta, "err": abs(theta - true_th),
                    "tau_ms": tau * 1000, "psr": psr})
        print(f"  x={pos}: τ={tau*1000:+.3f}ms PSR={psr:5.1f} | "
              f"true={true_th:+6.2f}° est={theta:+6.2f}° err={abs(theta-true_th):5.2f}°")
    mae = float(np.mean([r["err"] for r in rows]))
    print(f"\n  MAE: {mae:.2f}°  (was 13.51° with mic-derived window)")


if __name__ == "__main__":
    main()
