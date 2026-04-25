"""Loop 3 — physical max-lag constraint + LDV-subtraction stack.

Hypothesis H7: Many block-condition GCC failures pick a peak at |τ| > 1.5ms,
which is PHYSICALLY IMPOSSIBLE for a source on the source plane (mic spacing
1.4m / c=343m/s gives max τ_LR = ±1.5ms - I'll use 1.55ms cushion).

Multipath peaks (from reverb / wall reflections) can fall outside this. By
constraining peak search to |τ| ≤ 1.55 ms, we force the algorithm to pick the
best physically plausible peak, rejecting multipath confusers.

Variants tested:
  H1+phys   : LDV-subtraction (G1b NLMS or G1a STFT-Wiener) + |τ|≤1.55ms
  H7_only   : raw mic-mic + |τ|≤1.55ms
  H7+psr    : same + check Peak-to-Sidelobe Ratio (reject if PSR < threshold)
  H7+sub_pk : H1+phys + sub-sample parabolic interpolation
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.signal import butter, filtfilt
from scipy import signal as sp

from _loader import chirp_groups, speech_groups, load_group
from _pigs import (cross_phat, MIC_L, MIC_R, C_MPS, MIC_SPACING,
                   preprocess as basic_preprocess)
from _pigs2 import auto_window_chirp
from _geometry import expected_doa_deg, REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"

PHYS_MAX_LAG_S = MIC_SPACING / C_MPS * 1.05  # 1.547 ms — small cushion


def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def stft_wiener_subtract(x_v, x_m, sr, nperseg=2048, n_ovl=1536, ridge_eps=1e-6):
    f, t, Zv = sp.stft(x_v, fs=sr, nperseg=nperseg, noverlap=n_ovl, window="hann")
    _, _, Zm = sp.stft(x_m, fs=sr, nperseg=nperseg, noverlap=n_ovl, window="hann")
    Pvv = np.mean(np.abs(Zv) ** 2, axis=1)
    Pvm = np.mean(Zm * np.conj(Zv), axis=1)
    H = Pvm / (Pvv + ridge_eps * Pvv.max())
    Zm_resid = Zm - H[:, None] * Zv
    _, x_resid = sp.istft(Zm_resid, fs=sr, nperseg=nperseg, noverlap=n_ovl, window="hann")
    return x_resid[:len(x_m)]


def nlms_subtract(x_v, x_m, taps=256, mu=0.5):
    n = min(len(x_v), len(x_m))
    x_v = x_v[:n]; x_m = x_m[:n]
    w = np.zeros(taps)
    e = np.zeros(n)
    eps = 1e-6
    for i in range(taps, n):
        u = x_v[i - taps + 1:i + 1][::-1]
        y = w @ u
        err = x_m[i] - y
        norm = u @ u + eps
        w = w + mu * err * u / norm
        e[i] = err
    return e


def constrained_micmic_gcc(x_l, x_r, sr, band_hz, max_lag_s=PHYS_MAX_LAG_S,
                          subsample=True, return_psr=False):
    """Mic-mic GCC-PHAT with physically-bounded peak search."""
    n = max(len(x_l), len(x_r))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xl = np.fft.rfft(x_l, n_fft)
    Xr = np.fft.rfft(x_r, n_fft)
    G = Xr * np.conj(Xl)
    if band_hz:
        f = np.fft.rfftfreq(n_fft, 1 / sr)
        G = G * ((f >= band_hz[0]) & (f <= band_hz[1]))
    eps = 1e-12
    r = np.fft.fftshift(np.fft.irfft(G / (np.abs(G) + eps), n_fft))
    max_n = int(np.ceil(max_lag_s * sr))
    mid = len(r) // 2
    R = np.abs(r[mid - max_n: mid + max_n + 1])
    lags = (np.arange(len(R)) - max_n) / sr
    pk = int(np.argmax(R))
    tau = float(lags[pk])
    if subsample and 1 <= pk < len(R) - 1:
        # Parabolic interpolation
        y_m, y0, y_p = R[pk - 1], R[pk], R[pk + 1]
        denom = y_m - 2 * y0 + y_p
        if abs(denom) > 1e-12:
            offset = 0.5 * (y_m - y_p) / denom
            tau = tau + offset / sr
    if return_psr:
        # PSR: peak / median of off-peak window
        off_mask = np.ones(len(R), dtype=bool)
        off_mask[max(0, pk - 3):min(len(R), pk + 4)] = False
        psr = R[pk] / (np.median(R[off_mask]) + 1e-12) if off_mask.any() else 0.0
        return tau, psr
    return tau, None


def doa_from_tau(tau):
    s = max(-1.0, min(1.0, C_MPS * tau / MIC_SPACING))
    return -float(np.degrees(np.arcsin(s)))


def strat_h7_raw(chans, sr, sig_type, band_hz):
    chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    tau, _ = constrained_micmic_gcc(chans["mic_l"], chans["mic_r"], sr, band_hz)
    return doa_from_tau(tau)


def strat_h1_phys_nlms(chans, sr, sig_type, band_hz):
    chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    e_L = nlms_subtract(chans["ldv"], chans["mic_l"], taps=256)
    e_R = nlms_subtract(chans["ldv"], chans["mic_r"], taps=256)
    tau, _ = constrained_micmic_gcc(e_L, e_R, sr, band_hz)
    return doa_from_tau(tau)


def strat_h1_phys_wiener(chans, sr, sig_type, band_hz):
    chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    e_L = stft_wiener_subtract(chans["ldv"], chans["mic_l"], sr)
    e_R = stft_wiener_subtract(chans["ldv"], chans["mic_r"], sr)
    tau, _ = constrained_micmic_gcc(e_L, e_R, sr, band_hz)
    return doa_from_tau(tau)


def strat_h1h7_combined_minband(chans, sr, sig_type, band_hz):
    """Try BOTH wiener and NLMS, use whichever has higher PSR."""
    chans_b = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    # NLMS
    e_L_nl = nlms_subtract(chans_b["ldv"], chans_b["mic_l"], taps=256)
    e_R_nl = nlms_subtract(chans_b["ldv"], chans_b["mic_r"], taps=256)
    tau_nl, psr_nl = constrained_micmic_gcc(e_L_nl, e_R_nl, sr, band_hz, return_psr=True)
    # Wiener
    e_L_w = stft_wiener_subtract(chans_b["ldv"], chans_b["mic_l"], sr)
    e_R_w = stft_wiener_subtract(chans_b["ldv"], chans_b["mic_r"], sr)
    tau_w, psr_w = constrained_micmic_gcc(e_L_w, e_R_w, sr, band_hz, return_psr=True)
    # Pick higher PSR
    if psr_nl >= psr_w:
        return doa_from_tau(tau_nl)
    return doa_from_tau(tau_w)


STRATS = [
    ("H7_raw_500_2000", lambda c, sr, s: strat_h7_raw(c, sr, s, (500, 2000))),
    ("H7_raw_1000_5000", lambda c, sr, s: strat_h7_raw(c, sr, s, (1000, 5000))),
    ("H7_raw_300_4000", lambda c, sr, s: strat_h7_raw(c, sr, s, (300, 4000))),
    ("H7_raw_2000_8000", lambda c, sr, s: strat_h7_raw(c, sr, s, (2000, 8000))),
    ("H1H7_nlms_500_2000", lambda c, sr, s: strat_h1_phys_nlms(c, sr, s, (500, 2000))),
    ("H1H7_nlms_1000_5000", lambda c, sr, s: strat_h1_phys_nlms(c, sr, s, (1000, 5000))),
    ("H1H7_nlms_300_4000", lambda c, sr, s: strat_h1_phys_nlms(c, sr, s, (300, 4000))),
    ("H1H7_nlms_500_5000", lambda c, sr, s: strat_h1_phys_nlms(c, sr, s, (500, 5000))),
    ("H1H7_wiener_500_2000", lambda c, sr, s: strat_h1_phys_wiener(c, sr, s, (500, 2000))),
    ("H1H7_wiener_1000_5000", lambda c, sr, s: strat_h1_phys_wiener(c, sr, s, (1000, 5000))),
    ("H1H7_wiener_300_4000", lambda c, sr, s: strat_h1_phys_wiener(c, sr, s, (300, 4000))),
    ("H1H7_combined_500_2000", lambda c, sr, s: strat_h1h7_combined_minband(c, sr, s, (500, 2000))),
    ("H1H7_combined_300_4000", lambda c, sr, s: strat_h1h7_combined_minband(c, sr, s, (300, 4000))),
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

    print(f"\n{'strategy':<32} | {'chirp':>10} | {'speech':>10}")
    print("-" * 60)
    summary = {}
    for sid in results:
        c_errs = [v["err"] for v in results[sid].get("chirp", {}).values() if v["err"] is not None]
        s_errs = [v["err"] for v in results[sid].get("speech", {}).values() if v["err"] is not None]
        c_mae = float(np.mean(c_errs)) if c_errs else None
        s_mae = float(np.mean(s_errs)) if s_errs else None
        summary[sid] = {"chirp_mae": c_mae, "speech_mae": s_mae,
                       "rows": results[sid]}
    rows_sorted = sorted(summary.items(), key=lambda kv: kv[1]["speech_mae"] or 999)
    for sid, s in rows_sorted:
        c = s["chirp_mae"]; sp_ = s["speech_mae"]
        c_str = f"{c:6.2f}°" if c is not None else "  N/A "
        s_str = f"{sp_:6.2f}°" if sp_ is not None else "  N/A "
        print(f"{sid:<32} | {c_str:>10} | {s_str:>10}")

    # Show per-position for top
    best_sid = rows_sorted[0][0]
    print(f"\nPer-position breakdown of best ({best_sid}):")
    for sig in ("chirp", "speech"):
        if sig in summary[best_sid]["rows"]:
            print(f"  {sig}:")
            for pos in sorted(summary[best_sid]["rows"][sig]):
                r = summary[best_sid]["rows"][sig][pos]
                print(f"    x={pos}: true={r['true']:+6.2f}° est={r['est']:+6.2f}° err={r['err']:5.2f}°")

    out = OUT_DIR / "G_loop3_physical_constraint.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
