"""D01 — Combine top-3 strategies from Phase C.

Top performers:
  - chirp: S5c_ml (11.7°), S1c_bp1000_5000 (19.96°), S7_multiwin (15.94°)
  - speech: S1c_bp1000_5000 (13.55°), S0_baseline (23.5°), S5b_roth (21.95°)

Try combinations:
  D1  ml + multi-window averaging at band 1000-5000
  D2  ml + multi-window + spectral sub
  D3  ml weighting at band 700-3000 (intermediate)
  D4  mic-only band 1000-5000 (no LDV) — compare to PI-GS results
  D5  per-position best band (cheating; for upper bound)
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
from _pigs import (cross_phat, expected_tau_VM, MIC_L, MIC_R, C_MPS,
                   MIC_SPACING, interp_R, preprocess as basic_preprocess,
                   estimate_doa_micmic)
from _pigs2 import auto_window_chirp
from _geometry import expected_doa_deg, REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"


def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def cross_ml_weighted(x_v, x_m, sr, max_lag_s, band_hz):
    """ML-style coherence-weighted cross-correlation."""
    n = max(len(x_v), len(x_m))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xv = rfft(x_v, n_fft); Xm = rfft(x_m, n_fft)
    G = Xm * np.conj(Xv)
    f_c, c = sp.coherence(x_v, x_m, fs=sr, nperseg=2048, noverlap=1024)
    c_interp = np.interp(np.fft.rfftfreq(n_fft, 1 / sr), f_c, c)
    gamma2 = np.clip(c_interp, 1e-3, 0.999)
    eps = 1e-12
    W = (gamma2 / (1.0 - gamma2)) / (np.abs(G) + eps)
    if band_hz:
        f = np.fft.rfftfreq(n_fft, 1 / sr)
        mask = (f >= band_hz[0]) & (f <= band_hz[1])
        G = G * mask; W = W * mask
    r = np.fft.fftshift(irfft(G * W, n_fft))
    max_n = int(np.ceil(max_lag_s * sr))
    mid = len(r) // 2
    R = r[mid - max_n: mid + max_n + 1]
    lags = (np.arange(len(R)) - max_n) / sr
    return lags, R


def doa_from_R_pair(R_VL, lags_VL, R_VR, lags_VR, score="sum"):
    xs = np.arange(-1.5, 1.5 + 1e-9, 0.005)
    pts = np.stack([xs, np.zeros_like(xs)], axis=1)
    tau_VL = np.array([expected_tau_VM(p, MIC_L) for p in pts])
    tau_VR = np.array([expected_tau_VM(p, MIC_R) for p in pts])
    s_l = interp_R(lags_VL, np.abs(R_VL), tau_VL)
    s_r = interp_R(lags_VR, np.abs(R_VR), tau_VR)
    if score == "sum":
        S = s_l + s_r
    elif score == "prod":
        S = s_l * s_r
    elif score == "min":
        S = np.minimum(s_l, s_r)
    xi = int(np.argmax(S))
    p_x = float(xs[xi])
    tau_lr = (np.hypot(p_x - MIC_R[0], -MIC_R[1])
              - np.hypot(p_x - MIC_L[0], -MIC_L[1])) / C_MPS
    s = max(-1.0, min(1.0, C_MPS * tau_lr / MIC_SPACING))
    return -float(np.degrees(np.arcsin(s))), S, xs


def d_ml_multiwin(chans, sr, signal_type, band_hz, score="sum"):
    """ML-weighted GCC averaged over 0.5s windows."""
    chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    n_win = int(0.5 * sr); n_hop = int(0.25 * sr)
    R_VL_acc = R_VR_acc = None
    lags_VL = lags_VR = None
    n_count = 0
    for start in range(0, max(1, len(chans["mic_l"]) - n_win), n_hop):
        sl = slice(start, start + n_win)
        if (sl.stop - sl.start) < n_win // 2:
            continue
        lags_VL, R_VL = cross_ml_weighted(chans["ldv"][sl], chans["mic_l"][sl],
                                         sr, max_lag_s=0.007, band_hz=band_hz)
        lags_VR, R_VR = cross_ml_weighted(chans["ldv"][sl], chans["mic_r"][sl],
                                         sr, max_lag_s=0.007, band_hz=band_hz)
        if R_VL_acc is None:
            R_VL_acc = np.abs(R_VL); R_VR_acc = np.abs(R_VR)
        else:
            R_VL_acc += np.abs(R_VL); R_VR_acc += np.abs(R_VR)
        n_count += 1
    if n_count == 0:
        return None
    R_VL_acc /= n_count; R_VR_acc /= n_count
    theta, _, _ = doa_from_R_pair(R_VL_acc, lags_VL, R_VR_acc, lags_VR, score)
    return theta


def d_mic_only_wide(chans, sr, signal_type, band_hz=(1000, 5000)):
    chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    theta, _, _, _ = estimate_doa_micmic(chans["mic_l"], chans["mic_r"], sr,
                                        max_lag_s=0.005, band_hz=band_hz)
    return theta


STRATS = [
    ("D1_ml_multiwin_1000_5000_sum", lambda c, sr, s: d_ml_multiwin(c, sr, s, (1000, 5000), "sum")),
    ("D1b_ml_multiwin_1000_5000_min", lambda c, sr, s: d_ml_multiwin(c, sr, s, (1000, 5000), "min")),
    ("D1c_ml_multiwin_1000_5000_prod", lambda c, sr, s: d_ml_multiwin(c, sr, s, (1000, 5000), "prod")),
    ("D2_ml_multiwin_500_2000_sum", lambda c, sr, s: d_ml_multiwin(c, sr, s, (500, 2000), "sum")),
    ("D3_ml_multiwin_700_3000", lambda c, sr, s: d_ml_multiwin(c, sr, s, (700, 3000), "sum")),
    ("D4_ml_multiwin_300_4000", lambda c, sr, s: d_ml_multiwin(c, sr, s, (300, 4000), "sum")),
    ("D5_micmic_1000_5000", lambda c, sr, s: d_mic_only_wide(c, sr, s, (1000, 5000))),
    ("D5b_micmic_500_2000", lambda c, sr, s: d_mic_only_wide(c, sr, s, (500, 2000))),
    ("D5c_micmic_2000_8000", lambda c, sr, s: d_mic_only_wide(c, sr, s, (2000, 8000))),
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
        c = s["chirp_mae"]; sp = s["speech_mae"]
        c_str = f"{c:6.2f}°" if c is not None else "  N/A "
        s_str = f"{sp:6.2f}°" if sp is not None else "  N/A "
        print(f"{sid:<32} | {c_str:>10} | {s_str:>10}")

    out = OUT_DIR / "D_combinations.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
