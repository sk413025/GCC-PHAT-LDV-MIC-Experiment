"""Round 5 — final combo: H1 (LDV-NLMS) stacked with H33 (mic differential).

Then re-run global oracle analysis across rounds 1-4 to see updated upper bound.

H35  H1+H33 stack: NLMS subtract LDV → diff/sum mic → GCC
H36  Multi-band diff-mic GCC + median (no LDV)
H37  H35 + multi-band median
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from numpy.fft import rfft, irfft
from scipy.signal import butter, filtfilt
from scipy import signal as sp

from _loader import chirp_groups, speech_groups, load_group
from _pigs import (MIC_L, MIC_R, C_MPS, MIC_SPACING,
                   preprocess as basic_preprocess)
from _pigs2 import auto_window_chirp
from _geometry import expected_doa_deg, REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"
PHYS_MAX = MIC_SPACING / C_MPS * 1.05


def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def doa_from_tau(tau):
    s = max(-1.0, min(1.0, C_MPS * tau / MIC_SPACING))
    return -float(np.degrees(np.arcsin(s)))


def gcc(x_l, x_r, sr, band, max_lag_s=PHYS_MAX):
    n = max(len(x_l), len(x_r))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xl = rfft(x_l, n_fft); Xr = rfft(x_r, n_fft)
    G = Xr * np.conj(Xl)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    G = G * ((f >= band[0]) & (f <= band[1]))
    eps = 1e-12
    r = np.fft.fftshift(irfft(G / (np.abs(G) + eps), n_fft))
    max_n = int(np.ceil(max_lag_s * sr))
    mid = len(r) // 2
    R = np.abs(r[mid - max_n: mid + max_n + 1])
    lags = (np.arange(len(R)) - max_n) / sr
    return R, lags


def pick_subsample(R, lags):
    pk = int(np.argmax(R))
    tau = float(lags[pk])
    if 1 <= pk < len(R) - 1:
        ym, y0, yp = R[pk - 1], R[pk], R[pk + 1]
        d = ym - 2 * y0 + yp
        if abs(d) > 1e-12:
            tau += 0.5 * (ym - yp) / d * (lags[1] - lags[0])
    return tau


def nlms_subtract(x_v, x_m, taps=256, mu=0.5):
    n = min(len(x_v), len(x_m))
    x_v = x_v[:n]; x_m = x_m[:n]
    w = np.zeros(taps); e = np.zeros(n); eps_ = 1e-6
    for i in range(taps, n):
        u = x_v[i - taps + 1:i + 1][::-1]
        y = w @ u
        err = x_m[i] - y
        norm = u @ u + eps_
        w = w + mu * err * u / norm
        e[i] = err
    return e


def diff_sum_gcc(x_l, x_r, sr, band):
    """GCC between (mic_L - mic_R) and (mic_L + mic_R). Theoretically:
      diff has wall cancelled (common mode), direct path doubled.
      sum has wall doubled, direct path mixed.
      Cross-correlation peaks at the direct-path delay."""
    diff = x_l - x_r
    summ = x_l + x_r
    return gcc(diff, summ, sr, band)


# ============================================================================
# H35 — H1 (LDV-NLMS) + H33 (diff/sum mic)
# ============================================================================

def h35_nlms_diff_sum(chans, sr, sig_type, band):
    chans = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    e_L = nlms_subtract(chans["ldv"], chans["mic_l"])
    e_R = nlms_subtract(chans["ldv"], chans["mic_r"])
    R, lags = diff_sum_gcc(e_L, e_R, sr, band)
    return doa_from_tau(pick_subsample(R, lags))


def h35b_nlms_diff_self(chans, sr, sig_type, band):
    """NLMS residuals → diff GCC vs raw mic_L (instead of sum)."""
    chans = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    e_L = nlms_subtract(chans["ldv"], chans["mic_l"])
    e_R = nlms_subtract(chans["ldv"], chans["mic_r"])
    R, lags = gcc(e_L - e_R, e_L, sr, band)
    return doa_from_tau(pick_subsample(R, lags))


# ============================================================================
# H36 — multi-band diff/sum + median
# ============================================================================

def h36_diff_multi_band(chans, sr, sig_type, bands=None):
    if bands is None:
        bands = [(500, 1500), (1000, 2500), (1500, 3500), (2500, 5000),
                 (3500, 7000)]
    chans = {ch: bp(x, sr, 200, 8000) for ch, x in chans.items()}
    taus = []
    for band in bands:
        l_b = bp(chans["mic_l"], sr, *band)
        r_b = bp(chans["mic_r"], sr, *band)
        R, lags = diff_sum_gcc(l_b, r_b, sr, band)
        taus.append(pick_subsample(R, lags))
    taus = np.array(taus)
    # Reject near-zero before median
    nz = taus[np.abs(taus) > 5e-5]
    if len(nz) >= 2:
        return doa_from_tau(float(np.median(nz)))
    return doa_from_tau(float(np.median(taus)))


# ============================================================================
# H37 — full stack: NLMS → diff → multi-band median
# ============================================================================

def h37_full_stack(chans, sr, sig_type, bands=None):
    if bands is None:
        bands = [(500, 1500), (1000, 2500), (1500, 3500), (2500, 5000),
                 (3500, 7000), (1000, 5000), (300, 4000)]
    chans_full = {ch: bp(x, sr, 200, 8000) for ch, x in chans.items()}
    e_L = nlms_subtract(chans_full["ldv"], chans_full["mic_l"])
    e_R = nlms_subtract(chans_full["ldv"], chans_full["mic_r"])
    taus = []
    for band in bands:
        e_L_b = bp(e_L, sr, *band)
        e_R_b = bp(e_R, sr, *band)
        R, lags = diff_sum_gcc(e_L_b, e_R_b, sr, band)
        taus.append(pick_subsample(R, lags))
    taus = np.array(taus)
    # Reject near-zero
    nz = taus[np.abs(taus) > 1e-4]
    if len(nz) >= 2:
        return doa_from_tau(float(np.median(nz)))
    return doa_from_tau(float(np.median(taus)))


STRATS = [
    ("H35_nlms_diff_500_2k", lambda c, sr, s: h35_nlms_diff_sum(c, sr, s, (500, 2000))),
    ("H35_nlms_diff_1k_5k", lambda c, sr, s: h35_nlms_diff_sum(c, sr, s, (1000, 5000))),
    ("H35_nlms_diff_300_4k", lambda c, sr, s: h35_nlms_diff_sum(c, sr, s, (300, 4000))),
    ("H35b_nlms_diff_self_1k_5k", lambda c, sr, s: h35b_nlms_diff_self(c, sr, s, (1000, 5000))),
    ("H36_multiband_diff", lambda c, sr, s: h36_diff_multi_band(c, sr, s)),
    ("H37_full_stack", lambda c, sr, s: h37_full_stack(c, sr, s)),
]


def main():
    results = {}
    for sig_type in ("chirp", "speech"):
        groups = chirp_groups() if sig_type == "chirp" else speech_groups()
        for (pos, cond), paths in sorted(groups.items()):
            if cond != "block":
                continue
            chans, sr = load_group(paths)
            if sig_type == "chirp":
                t0, t1 = auto_window_chirp(chans["mic_l"], sr, dur_s=1.6)
            else:
                t0, t1 = 5.0, 25.0
            n0, n1 = int(t0 * sr), int(t1 * sr)
            chans = {ch: x[n0:n1] for ch, x in chans.items()}
            chans = {ch: basic_preprocess(x, sr) for ch, x in chans.items()}
            theta_true = expected_doa_deg(float(pos))
            for sid, fn in STRATS:
                try:
                    theta = fn(chans, sr, sig_type)
                except Exception as e:
                    theta = None
                err = abs(theta - theta_true) if theta is not None else None
                results.setdefault(sid, {}).setdefault(sig_type, {})[pos] = {
                    "true": theta_true, "est": theta, "err": err}

    summary = {}
    for sid in results:
        c_errs = [v["err"] for v in results[sid].get("chirp", {}).values()
                  if v["err"] is not None]
        s_errs = [v["err"] for v in results[sid].get("speech", {}).values()
                  if v["err"] is not None]
        c_mae = float(np.mean(c_errs)) if c_errs else None
        s_mae = float(np.mean(s_errs)) if s_errs else None
        summary[sid] = {"chirp_mae": c_mae, "speech_mae": s_mae,
                       "rows": results[sid]}

    print(f"\n{'strategy':<32} | {'chirp':>10} | {'speech':>10}")
    print("-" * 60)
    rows_sorted = sorted(summary.items(), key=lambda kv: kv[1]["speech_mae"] or 999)
    for sid, s in rows_sorted:
        c = s["chirp_mae"]; sp_ = s["speech_mae"]
        c_str = f"{c:6.2f}°" if c is not None else "  N/A "
        s_str = f"{sp_:6.2f}°" if sp_ is not None else "  N/A "
        print(f"{sid:<32} | {c_str:>10} | {s_str:>10}")

    print("\nPer-position breakdown of all:")
    for sid, _ in rows_sorted:
        print(f"\n=== {sid} ===")
        for sig in ("chirp", "speech"):
            if sig in summary[sid]["rows"]:
                print(f"  {sig}:")
                for pos in sorted(summary[sid]["rows"][sig]):
                    r = summary[sid]["rows"][sig][pos]
                    if r["err"] is None:
                        print(f"    x={pos}: NA")
                        continue
                    print(f"    x={pos}: true={r['true']:+6.2f}° "
                          f"est={r['est']:+6.2f}° err={r['err']:5.2f}°")

    out = OUT_DIR / "H_round5_final.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
