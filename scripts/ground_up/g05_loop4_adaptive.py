"""Loop 4 — adaptive per-recording band selection + multi-band ensemble.

Hypothesis H8: Each recording has its own usable LDV-Mic coherence band (per
Phase A audit, varies wildly across positions). A SINGLE fixed bandpass
forces some positions into a band where they have no SNR. Adaptive selection
should help.

Hypothesis H9: When uncertain, use ENSEMBLE — run multiple bands, weight by
PSR, pick winner.

Strategies:
  AD1   per-rec auto-band : pick contiguous γ²>0.3 band with max coherent power
  AD2   multi-band ensemble: run 4 bands, take PSR-weighted median τ
  AD3   per-rec auto + LDV-NLMS subtraction + |τ|≤1.55ms
  AD4   ENSEMBLE of AD3 across 4 bands, PSR-pick
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
from _pigs import (MIC_L, MIC_R, C_MPS, MIC_SPACING,
                   preprocess as basic_preprocess)
from _pigs2 import auto_window_chirp
from _geometry import expected_doa_deg, REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"

PHYS_MAX = MIC_SPACING / C_MPS * 1.05


def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def stft_wiener(x_v, x_m, sr, nperseg=2048, n_ovl=1536, ridge=1e-6):
    f, t, Zv = sp.stft(x_v, fs=sr, nperseg=nperseg, noverlap=n_ovl, window="hann")
    _, _, Zm = sp.stft(x_m, fs=sr, nperseg=nperseg, noverlap=n_ovl, window="hann")
    Pvv = np.mean(np.abs(Zv) ** 2, axis=1)
    Pvm = np.mean(Zm * np.conj(Zv), axis=1)
    H = Pvm / (Pvv + ridge * Pvv.max())
    Zm_resid = Zm - H[:, None] * Zv
    _, x_resid = sp.istft(Zm_resid, fs=sr, nperseg=nperseg, noverlap=n_ovl, window="hann")
    return x_resid[:len(x_m)]


def nlms(x_v, x_m, taps=256, mu=0.5):
    n = min(len(x_v), len(x_m))
    x_v = x_v[:n]; x_m = x_m[:n]
    w = np.zeros(taps); e = np.zeros(n)
    eps = 1e-6
    for i in range(taps, n):
        u = x_v[i - taps + 1:i + 1][::-1]
        y = w @ u
        err = x_m[i] - y
        norm = u @ u + eps
        w = w + mu * err * u / norm
        e[i] = err
    return e


def auto_band_from_coherence(x_v, x_m, sr, threshold=0.3, min_width_hz=300,
                            search_range=(200, 8000)):
    """Find widest contiguous band where MS-coherence > threshold."""
    f, c = sp.coherence(x_v, x_m, fs=sr, nperseg=2048, noverlap=1024)
    in_range = (f >= search_range[0]) & (f <= search_range[1])
    f, c = f[in_range], c[in_range]
    above = c >= threshold
    runs = []
    start = None
    for i, a in enumerate(above):
        if a and start is None:
            start = i
        elif not a and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(above)))
    if not runs:
        return None
    best = None
    best_score = 0
    for s, e in runs:
        width = f[e - 1] - f[s]
        if width < min_width_hz:
            continue
        # Score = mean coherence × width
        score = np.mean(c[s:e]) * width
        if score > best_score:
            best_score = score
            best = (float(f[s]), float(f[e - 1]))
    return best


def auto_band_consensus(x_v, x_l, x_r, sr, threshold=0.3, min_width_hz=300):
    """Find band where BOTH LDV-MicL and LDV-MicR have coherence."""
    f, c_l = sp.coherence(x_v, x_l, fs=sr, nperseg=2048, noverlap=1024)
    _, c_r = sp.coherence(x_v, x_r, fs=sr, nperseg=2048, noverlap=1024)
    c_min = np.minimum(c_l, c_r)
    in_range = (f >= 200) & (f <= 8000)
    f, c_min = f[in_range], c_min[in_range]
    above = c_min >= threshold
    runs = []
    start = None
    for i, a in enumerate(above):
        if a and start is None: start = i
        elif not a and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(above)))
    best = None; best_score = 0
    for s, e in runs:
        width = f[e - 1] - f[s]
        if width < min_width_hz:
            continue
        score = np.mean(c_min[s:e]) * width
        if score > best_score:
            best_score = score; best = (float(f[s]), float(f[e - 1]))
    return best


def constrained_gcc_psr(x_l, x_r, sr, band_hz, max_lag_s=PHYS_MAX):
    n = max(len(x_l), len(x_r))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xl = np.fft.rfft(x_l, n_fft); Xr = np.fft.rfft(x_r, n_fft)
    G = Xr * np.conj(Xl)
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
    if 1 <= pk < len(R) - 1:
        y_m, y0, y_p = R[pk - 1], R[pk], R[pk + 1]
        denom = y_m - 2 * y0 + y_p
        if abs(denom) > 1e-12:
            tau += 0.5 * (y_m - y_p) / denom / sr
    off_mask = np.ones(len(R), dtype=bool)
    off_mask[max(0, pk - 3):min(len(R), pk + 4)] = False
    psr = R[pk] / (np.median(R[off_mask]) + 1e-12) if off_mask.any() else 0.0
    return tau, psr


def doa_from_tau(tau):
    s = max(-1.0, min(1.0, C_MPS * tau / MIC_SPACING))
    return -float(np.degrees(np.arcsin(s)))


def strat_ad1_auto_band(chans, sr, sig_type, subtraction="nlms"):
    """Auto-select band, then do mic-mic GCC + LDV subtraction."""
    band = auto_band_consensus(chans["ldv"], chans["mic_l"], chans["mic_r"], sr)
    if band is None:
        band = (500, 4000)
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    if subtraction == "nlms":
        e_L = nlms(chans_b["ldv"], chans_b["mic_l"])
        e_R = nlms(chans_b["ldv"], chans_b["mic_r"])
    elif subtraction == "wiener":
        e_L = stft_wiener(chans_b["ldv"], chans_b["mic_l"], sr)
        e_R = stft_wiener(chans_b["ldv"], chans_b["mic_r"], sr)
    else:
        e_L, e_R = chans_b["mic_l"], chans_b["mic_r"]
    tau, psr = constrained_gcc_psr(e_L, e_R, sr, band)
    return doa_from_tau(tau), band, psr


def strat_ad2_ensemble(chans, sr, sig_type, bands=None):
    """Run multiple bands, pick highest-PSR result."""
    bands = bands or [(500, 2000), (1000, 5000), (2000, 8000), (300, 4000)]
    best_psr = -1; best_tau = 0; best_band = None
    for band in bands:
        chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
        e_L = nlms(chans_b["ldv"], chans_b["mic_l"])
        e_R = nlms(chans_b["ldv"], chans_b["mic_r"])
        tau, psr = constrained_gcc_psr(e_L, e_R, sr, band)
        if psr > best_psr:
            best_psr = psr; best_tau = tau; best_band = band
    return doa_from_tau(best_tau), best_band, best_psr


def strat_ad3_individual_band(chans, sr, sig_type):
    """Use INDIVIDUAL LDV-MicL band for L-channel, LDV-MicR band for R-channel.
    This treats each pair independently.
    """
    band_L = auto_band_from_coherence(chans["ldv"], chans["mic_l"], sr)
    band_R = auto_band_from_coherence(chans["ldv"], chans["mic_r"], sr)
    if band_L is None and band_R is None:
        return None, None, None
    # Use intersection or union?
    if band_L and band_R:
        # Use union (widest combined coverage)
        lo = min(band_L[0], band_R[0])
        hi = max(band_L[1], band_R[1])
        band = (lo, hi)
    elif band_L:
        band = band_L
    else:
        band = band_R
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    e_L = nlms(chans_b["ldv"], chans_b["mic_l"])
    e_R = nlms(chans_b["ldv"], chans_b["mic_r"])
    tau, psr = constrained_gcc_psr(e_L, e_R, sr, band)
    return doa_from_tau(tau), band, psr


STRATS = [
    ("AD1_auto_band_nlms", lambda c, sr, s: strat_ad1_auto_band(c, sr, s, "nlms")),
    ("AD1b_auto_band_wiener", lambda c, sr, s: strat_ad1_auto_band(c, sr, s, "wiener")),
    ("AD1c_auto_band_no_sub", lambda c, sr, s: strat_ad1_auto_band(c, sr, s, "none")),
    ("AD2_ensemble_PSR", lambda c, sr, s: strat_ad2_ensemble(c, sr, s)),
    ("AD3_union_LR_band", lambda c, sr, s: strat_ad3_individual_band(c, sr, s)),
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
                    res = fn(chans, sr, sig_type)
                    theta, band, psr = res
                except Exception as e:
                    theta = None; band = None; psr = None
                err = abs(theta - theta_true) if theta is not None else None
                results.setdefault(sid, {}).setdefault(sig_type, {})[pos] = {
                    "true": theta_true, "est": theta, "err": err,
                    "band": band, "psr": psr,
                }

    print(f"\n{'strategy':<26} | {'chirp':>10} | {'speech':>10}")
    print("-" * 50)
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
        print(f"{sid:<26} | {c_str:>10} | {s_str:>10}")

    best_sid = rows_sorted[0][0]
    print(f"\nPer-position breakdown of best ({best_sid}):")
    for sig in ("chirp", "speech"):
        if sig in summary[best_sid]["rows"]:
            print(f"  {sig}:")
            for pos in sorted(summary[best_sid]["rows"][sig]):
                r = summary[best_sid]["rows"][sig][pos]
                band_s = f"{r['band']}" if r['band'] else "(none)"
                print(f"    x={pos}: band={band_s} | true={r['true']:+6.2f}° "
                      f"est={r['est']:+6.2f}° err={r['err']:5.2f}° PSR={r['psr']:5.1f}" if r['psr'] else
                     f"    x={pos}: band={band_s} | NA")

    out = OUT_DIR / "G_loop4_adaptive.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
