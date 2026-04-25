"""C — Transient/onset-based strategies. Key insight from forensics:
direct-path GCC peaks are obscured by board ringing. Try:
  T1  short window  : take only first 50 ms of chirp burst (no ringing)
  T2  envelope GCC  : cross-correlate Hilbert envelopes (slower features, less noise)
  T3  onset detect  : find onset in mic_L, mic_R, ldv separately, use TDoA
  T4  template_v2   : refine S11 template_xc with better template parameters
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy import signal as sp

from _loader import chirp_groups, speech_groups, load_group
from _pigs import (cross_phat, expected_tau_VM, MIC_L, MIC_R, LDV, C_MPS,
                   MIC_SPACING, interp_R, preprocess as basic_preprocess)
from _pigs2 import auto_window_chirp
from _geometry import expected_doa_deg, REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"


def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def find_onset(x, sr, smooth_ms=2.0):
    e = np.abs(hilbert(x))
    n = int(sr * smooth_ms / 1000)
    e = np.convolve(e, np.ones(n) / n, mode="same")
    thresh = 0.2 * e.max()
    return int(np.argmax(e > thresh)) / sr


def t1_short_window(chans, sr, signal_type, win_s=0.05, band=(500, 5000)):
    """First win_s of the chirp burst only."""
    if signal_type != "chirp":
        return None
    chans = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    n = int(win_s * sr)
    chans = {ch: x[:n] for ch, x in chans.items()}
    lags_VL, R_VL = cross_phat(chans["ldv"], chans["mic_l"], sr,
                              max_lag_s=0.007, band_hz=band)
    lags_VR, R_VR = cross_phat(chans["ldv"], chans["mic_r"], sr,
                              max_lag_s=0.007, band_hz=band)
    return _doa_from_R(R_VL, lags_VL, R_VR, lags_VR)


def t2_envelope_gcc(chans, sr, signal_type, band=(500, 4000)):
    chans = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    chans = {ch: np.abs(hilbert(x)) for ch, x in chans.items()}
    chans = {ch: x - np.mean(x) for ch, x in chans.items()}
    lags_VL, R_VL = cross_phat(chans["ldv"], chans["mic_l"], sr, max_lag_s=0.007)
    lags_VR, R_VR = cross_phat(chans["ldv"], chans["mic_r"], sr, max_lag_s=0.007)
    return _doa_from_R(R_VL, lags_VL, R_VR, lags_VR)


def t3_onset(chans, sr, signal_type, band=(500, 4000)):
    chans = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    t_v = find_onset(chans["ldv"], sr)
    t_l = find_onset(chans["mic_l"], sr)
    t_r = find_onset(chans["mic_r"], sr)
    tau_lr = t_r - t_l  # in seconds
    s = max(-1.0, min(1.0, C_MPS * tau_lr / MIC_SPACING))
    theta = -float(np.degrees(np.arcsin(s)))
    return theta


def t4_template(chans, sr, signal_type, band=(500, 5000)):
    if signal_type != "chirp":
        return None
    # Use mic_L itself as the template (assume it's the cleanest source-bearing signal)
    chans = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    template = chans["mic_l"][:int(0.2 * sr)]
    template -= np.mean(template)
    # Cross-correlate mic_R with template
    n = max(len(chans["mic_r"]), len(template))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xr = np.fft.rfft(chans["mic_r"], n_fft)
    Xt = np.fft.rfft(template, n_fft)
    G = Xr * np.conj(Xt)
    eps = 1e-12
    r_phat = np.fft.fftshift(np.fft.irfft(G / (np.abs(G) + eps), n_fft))
    max_n = int(0.005 * sr)
    mid = len(r_phat) // 2
    R = r_phat[mid - max_n: mid + max_n + 1]
    lags = (np.arange(len(R)) - max_n) / sr
    tau_lr = float(lags[int(np.argmax(np.abs(R)))])
    s = max(-1.0, min(1.0, C_MPS * tau_lr / MIC_SPACING))
    theta = -float(np.degrees(np.arcsin(s)))
    return theta


def _doa_from_R(R_VL, lags_VL, R_VR, lags_VR):
    xs = np.arange(-1.5, 1.5 + 1e-9, 0.005)
    pts = np.stack([xs, np.zeros_like(xs)], axis=1)
    tau_VL = np.array([expected_tau_VM(p, MIC_L) for p in pts])
    tau_VR = np.array([expected_tau_VM(p, MIC_R) for p in pts])
    s_l = interp_R(lags_VL, np.abs(R_VL), tau_VL)
    s_r = interp_R(lags_VR, np.abs(R_VR), tau_VR)
    S = s_l + s_r
    xi = int(np.argmax(S))
    p_x = float(xs[xi])
    tau_lr = (np.hypot(p_x - MIC_R[0], -MIC_R[1])
              - np.hypot(p_x - MIC_L[0], -MIC_L[1])) / C_MPS
    s = max(-1.0, min(1.0, C_MPS * tau_lr / MIC_SPACING))
    return -float(np.degrees(np.arcsin(s)))


STRATS = [
    ("T1_short_50ms", t1_short_window),
    ("T2_envelope_gcc", t2_envelope_gcc),
    ("T3_onset_TDoA", t3_onset),
    ("T4_template_micL", t4_template),
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
                    "theta_true": theta_true, "theta_est": theta, "err": err,
                }

    print(f"\n{'strategy':<22} | {'chirp MAE':>10} | {'speech MAE':>10}")
    print("-" * 50)
    summary = {}
    for sid in results:
        c_errs = [v["err"] for v in results[sid].get("chirp", {}).values() if v["err"] is not None]
        s_errs = [v["err"] for v in results[sid].get("speech", {}).values() if v["err"] is not None]
        c_mae = float(np.mean(c_errs)) if c_errs else None
        s_mae = float(np.mean(s_errs)) if s_errs else None
        summary[sid] = {"chirp_mae": c_mae, "speech_mae": s_mae,
                       "rows": results[sid]}
        c_str = f"{c_mae:6.2f}°" if c_mae else "  N/A "
        s_str = f"{s_mae:6.2f}°" if s_mae else "  N/A "
        print(f"{sid:<22} | {c_str:>10} | {s_str:>10}")

    out = OUT_DIR / "C_transient_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
