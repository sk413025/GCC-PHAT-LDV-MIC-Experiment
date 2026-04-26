"""Loop 1 — LDV-subtracted mic-mic GCC.

Hypothesis H1: LDV measures the wall-radiated indirect path that contaminates
both mics with high coherence at τ=0 (since the wall is roughly equidistant
from both mics). This is why mic-mic GCC locks to 0 in block conditions.

If we subtract the LDV-coherent component from each mic (Wiener filter with
LDV as reference, mic as desired), the residual contains the direct path with
direction information, and mic-mic GCC on residuals should recover true τ_LR.

Variants tested:
  G1a  STFT Wiener: per-bin H_VL(f), H_VR(f) estimated from welch CSD,
       subtract H * X_V from mic spectra, IFFT, then mic-mic GCC.
  G1b  Time-domain LMS: adaptive filter w[n] convolved with x_V[n] subtracted
       from each mic. NLMS, length 256 taps.
  G1c  STFT magnitude subtraction (over-aggressive): subtract |H_VL X_V| from
       |X_L| keeping mic phase. This is for ablation only.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.signal import butter, filtfilt, stft, istft, csd, welch
from scipy import signal as sp

from _loader import chirp_groups, speech_groups, load_group
from _pigs import (estimate_doa_micmic, MIC_L, MIC_R, C_MPS, MIC_SPACING,
                   preprocess as basic_preprocess)
from _pigs2 import auto_window_chirp
from _geometry import expected_doa_deg, REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"


def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def stft_wiener_subtract(x_v, x_m, sr, nperseg=2048, n_ovl=1536, ridge_eps=1e-6):
    """Estimate H_VM(f) = S_VM/S_VV and subtract H_VM * X_V from X_M, return residual."""
    f, t, Zv = sp.stft(x_v, fs=sr, nperseg=nperseg, noverlap=n_ovl, window="hann")
    _, _, Zm = sp.stft(x_m, fs=sr, nperseg=nperseg, noverlap=n_ovl, window="hann")
    # Per-frequency H estimate via Welch averaging (over time):
    # H = E[Zm Zv*] / E[|Zv|^2]
    Pvv = np.mean(np.abs(Zv) ** 2, axis=1)  # shape (n_freq,)
    Pvm = np.mean(Zm * np.conj(Zv), axis=1)
    H = Pvm / (Pvv + ridge_eps * Pvv.max())
    # Subtract H[k] * Zv from Zm
    Zm_resid = Zm - H[:, None] * Zv
    _, x_resid = sp.istft(Zm_resid, fs=sr, nperseg=nperseg, noverlap=n_ovl, window="hann")
    return x_resid[:len(x_m)], H, f


def nlms_subtract(x_v, x_m, taps=256, mu=0.5):
    """Time-domain NLMS adaptive filter: estimate filter w that maps x_v -> x_m,
    return residual e = x_m - w * x_v.
    """
    n = min(len(x_v), len(x_m))
    x_v = x_v[:n]; x_m = x_m[:n]
    w = np.zeros(taps)
    e = np.zeros(n)
    eps = 1e-6
    for i in range(taps, n):
        u = x_v[i - taps + 1:i + 1][::-1]  # newest first
        y = w @ u
        err = x_m[i] - y
        norm = u @ u + eps
        w = w + mu * err * u / norm
        e[i] = err
    return e, w


def g1a_stft_wiener(chans, sr, signal_type, band_hz=(500, 5000)):
    chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    e_L, _, _ = stft_wiener_subtract(chans["ldv"], chans["mic_l"], sr)
    e_R, _, _ = stft_wiener_subtract(chans["ldv"], chans["mic_r"], sr)
    theta, _, _, _ = estimate_doa_micmic(e_L, e_R, sr, max_lag_s=0.005, band_hz=band_hz)
    return theta


def g1b_nlms(chans, sr, signal_type, band_hz=(500, 5000), taps=256, mu=0.5):
    chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    e_L, _ = nlms_subtract(chans["ldv"], chans["mic_l"], taps=taps, mu=mu)
    e_R, _ = nlms_subtract(chans["ldv"], chans["mic_r"], taps=taps, mu=mu)
    theta, _, _, _ = estimate_doa_micmic(e_L, e_R, sr, max_lag_s=0.005, band_hz=band_hz)
    return theta


def g1c_stft_wiener_then_mlcoh(chans, sr, signal_type, band_hz=(500, 5000)):
    """G1a residuals + ML-coherence-weighted mic-mic GCC."""
    from _pigs import cross_phat
    chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    e_L, _, _ = stft_wiener_subtract(chans["ldv"], chans["mic_l"], sr)
    e_R, _, _ = stft_wiener_subtract(chans["ldv"], chans["mic_r"], sr)
    # ML coherence weighting between residuals
    n = max(len(e_L), len(e_R))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    El = np.fft.rfft(e_L, n_fft); Er = np.fft.rfft(e_R, n_fft)
    G = Er * np.conj(El)
    f_c, c = sp.coherence(e_L, e_R, fs=sr, nperseg=2048, noverlap=1024)
    c_interp = np.interp(np.fft.rfftfreq(n_fft, 1 / sr), f_c, c)
    gamma2 = np.clip(c_interp, 1e-3, 0.999)
    eps = 1e-12
    W = (gamma2 / (1.0 - gamma2)) / (np.abs(G) + eps)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    mask = (f >= band_hz[0]) & (f <= band_hz[1])
    G = G * mask; W = W * mask
    r = np.fft.fftshift(np.fft.irfft(G * W, n_fft))
    max_n = int(0.005 * sr); mid = len(r) // 2
    R = r[mid - max_n: mid + max_n + 1]
    lags = (np.arange(len(R)) - max_n) / sr
    tau = float(lags[int(np.argmax(np.abs(R)))])
    s = max(-1.0, min(1.0, C_MPS * tau / MIC_SPACING))
    return -float(np.degrees(np.arcsin(s)))


def coh_diagnostic(chans, sr, signal_type, band_hz=(500, 5000)):
    """Diagnostic: how much LDV-coherent content is removed in each channel?"""
    chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    e_L, _, _ = stft_wiener_subtract(chans["ldv"], chans["mic_l"], sr)
    e_R, _, _ = stft_wiener_subtract(chans["ldv"], chans["mic_r"], sr)
    rms = lambda x: np.sqrt(np.mean(x ** 2))
    return {
        "rms_L_orig": rms(chans["mic_l"]),
        "rms_L_resid": rms(e_L),
        "rms_R_orig": rms(chans["mic_r"]),
        "rms_R_resid": rms(e_R),
        "frac_L_removed": 1 - rms(e_L) / rms(chans["mic_l"]),
        "frac_R_removed": 1 - rms(e_R) / rms(chans["mic_r"]),
    }


STRATS = [
    ("G1a_wiener_500_5000", lambda c, sr, s: g1a_stft_wiener(c, sr, s, (500, 5000))),
    ("G1a_wiener_300_4000", lambda c, sr, s: g1a_stft_wiener(c, sr, s, (300, 4000))),
    ("G1a_wiener_1000_5000", lambda c, sr, s: g1a_stft_wiener(c, sr, s, (1000, 5000))),
    ("G1a_wiener_500_2000", lambda c, sr, s: g1a_stft_wiener(c, sr, s, (500, 2000))),
    ("G1b_nlms_500_5000_t256", lambda c, sr, s: g1b_nlms(c, sr, s, (500, 5000), 256)),
    ("G1b_nlms_500_5000_t512", lambda c, sr, s: g1b_nlms(c, sr, s, (500, 5000), 512)),
    ("G1b_nlms_300_4000_t256", lambda c, sr, s: g1b_nlms(c, sr, s, (300, 4000), 256)),
    ("G1c_wiener+mlcoh_500_5000", lambda c, sr, s: g1c_stft_wiener_then_mlcoh(c, sr, s, (500, 5000))),
    ("G1c_wiener+mlcoh_300_4000", lambda c, sr, s: g1c_stft_wiener_then_mlcoh(c, sr, s, (300, 4000))),
]


def main():
    results = {}
    diag_acc = []
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

            # Diagnostic: how much energy got subtracted?
            d = coh_diagnostic(chans, sr, sig_type)
            d.update({"pos": pos, "sig": sig_type})
            diag_acc.append(d)

            for sid, fn in STRATS:
                try:
                    theta = fn(chans, sr, sig_type)
                except Exception as e:
                    theta = None
                err = abs(theta - theta_true) if theta is not None else None
                results.setdefault(sid, {}).setdefault(sig_type, {})[pos] = {
                    "true": theta_true, "est": theta, "err": err}

    # Print diagnostic first
    print("\nWiener subtraction diagnostic (fraction of mic energy that's LDV-coherent):")
    print(f"{'sig':>7} {'pos':>5} | {'frac_L':>7} {'frac_R':>7} | {'rms_L_orig':>11} {'rms_L_resid':>12}")
    for d in diag_acc:
        print(f"{d['sig']:>7} {d['pos']:>5} | {d['frac_L_removed']:7.3f} {d['frac_R_removed']:7.3f} | "
              f"{d['rms_L_orig']:11.5f} {d['rms_L_resid']:12.5f}")

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

    out = OUT_DIR / "G_loop1_wiener_subtract.json"
    out.write_text(json.dumps({"strategies": summary, "diagnostic": diag_acc},
                              indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
