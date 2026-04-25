"""Phase E.2 — geometry forensics.

For each chirp recording (5 positions, blocked), measure the actual peak τ in
R_VL and R_VR (using best-bandpass per audit), and compare with free-space
prediction. If there's a systematic offset, back-compute true LDV position.

Also do MIC-MIC for sanity (we expect mic-mic τ to closely match expected
since mic positions are well-defined).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt

from _loader import chirp_groups, load_group
from _pigs import (cross_phat, expected_tau_VM, MIC_L, MIC_R, LDV, C_MPS,
                   preprocess as basic_preprocess)
from _pigs2 import auto_window_chirp


def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def cross_phat_micmic(x_l, x_r, sr, max_lag_s, band_hz):
    n = max(len(x_l), len(x_r))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xl = np.fft.rfft(x_l, n_fft); Xr = np.fft.rfft(x_r, n_fft)
    G = Xr * np.conj(Xl)
    if band_hz:
        f = np.fft.rfftfreq(n_fft, 1 / sr)
        G = G * ((f >= band_hz[0]) & (f <= band_hz[1]))
    eps = 1e-12
    r = np.fft.fftshift(np.fft.irfft(G / (np.abs(G) + eps), n_fft))
    max_n = int(np.ceil(max_lag_s * sr))
    mid = len(r) // 2
    R = r[mid - max_n: mid + max_n + 1]
    lags = (np.arange(len(R)) - max_n) / sr
    return lags, R


def main():
    groups = chirp_groups()
    band = (500, 5000)  # broader band for chirp
    print(f"Using bandpass {band} Hz, max_lag = 12 ms\n")
    print(f"{'pos':>5} | {'τ_VL_exp':>9} {'τ_VL_meas':>10} {'Δ_VL':>7} | "
          f"{'τ_VR_exp':>9} {'τ_VR_meas':>10} {'Δ_VR':>7} | "
          f"{'τ_LR_exp':>9} {'τ_LR_meas':>10} {'Δ_LR':>7}")
    print("-" * 110)
    rows = []
    for (pos, cond), paths in sorted(groups.items()):
        if cond != "block":
            continue
        chans, sr = load_group(paths)
        t0, t1 = auto_window_chirp(chans["mic_l"], sr, dur_s=1.6)
        n0, n1 = int(t0 * sr), int(t1 * sr)
        chans = {ch: x[n0:n1] for ch, x in chans.items()}
        chans = {ch: basic_preprocess(x, sr) for ch, x in chans.items()}
        chans = {ch: bp(x, sr, *band) for ch, x in chans.items()}

        x_pos = float(pos)
        # R_VL
        lags, R = cross_phat(chans["ldv"], chans["mic_l"], sr,
                            max_lag_s=0.012, band_hz=band)
        tau_vl_meas = float(lags[int(np.argmax(np.abs(R)))])
        tau_vl_exp = expected_tau_VM((x_pos, 0.0), MIC_L)
        # R_VR
        lags, R = cross_phat(chans["ldv"], chans["mic_r"], sr,
                            max_lag_s=0.012, band_hz=band)
        tau_vr_meas = float(lags[int(np.argmax(np.abs(R)))])
        tau_vr_exp = expected_tau_VM((x_pos, 0.0), MIC_R)
        # Mic-mic
        lags, R = cross_phat_micmic(chans["mic_l"], chans["mic_r"], sr,
                                   max_lag_s=0.005, band_hz=band)
        tau_lr_meas = float(lags[int(np.argmax(np.abs(R)))])
        tau_lr_exp = (np.hypot(x_pos - MIC_R[0], -MIC_R[1])
                      - np.hypot(x_pos - MIC_L[0], -MIC_L[1])) / C_MPS
        rows.append({
            "pos": pos, "x": x_pos,
            "tau_vl_exp": tau_vl_exp, "tau_vl_meas": tau_vl_meas,
            "tau_vr_exp": tau_vr_exp, "tau_vr_meas": tau_vr_meas,
            "tau_lr_exp": tau_lr_exp, "tau_lr_meas": tau_lr_meas,
        })
        print(f"{pos:>5} | {tau_vl_exp*1000:+7.3f}ms {tau_vl_meas*1000:+8.3f}ms "
              f"{(tau_vl_meas-tau_vl_exp)*1000:+5.2f}ms | "
              f"{tau_vr_exp*1000:+7.3f}ms {tau_vr_meas*1000:+8.3f}ms "
              f"{(tau_vr_meas-tau_vr_exp)*1000:+5.2f}ms | "
              f"{tau_lr_exp*1000:+7.3f}ms {tau_lr_meas*1000:+8.3f}ms "
              f"{(tau_lr_meas-tau_lr_exp)*1000:+5.2f}ms")

    # If mic-mic offset is systematic, that's the ADC channel timing offset
    print(f"\nMic-mic Δ (system offset?) mean: "
          f"{np.mean([r['tau_lr_meas']-r['tau_lr_exp'] for r in rows])*1000:+.3f} ms, "
          f"std: {np.std([r['tau_lr_meas']-r['tau_lr_exp'] for r in rows])*1000:.3f} ms")

    # Back-compute LDV position by fitting τ_VL_meas, τ_VR_meas data
    # For LDV at (a, b), expected τ_VL(x_s) = (||(x_s,0)-MIC_L|| - ||(x_s,0)-(a,b)||)/c
    # We have 5 measurements per pair → 10 equations, 2 unknowns (LDV)
    def residuals(params):
        a, b = params
        res = []
        for r in rows:
            x_s = r["x"]
            d_ldv = np.hypot(x_s - a, 0 - b)
            d_l = np.hypot(x_s - MIC_L[0], 0 - MIC_L[1])
            d_r = np.hypot(x_s - MIC_R[0], 0 - MIC_R[1])
            tau_vl_pred = (d_l - d_ldv) / C_MPS
            tau_vr_pred = (d_r - d_ldv) / C_MPS
            res.append(tau_vl_pred - r["tau_vl_meas"])
            res.append(tau_vr_pred - r["tau_vr_meas"])
        return res

    x0 = [0.0, 0.25]  # paper geometry
    sol = least_squares(residuals, x0, bounds=([-1.5, -0.5], [1.5, 2.0]))
    print(f"\nBest-fit LDV position: ({sol.x[0]:+.3f}, {sol.x[1]:+.3f}) m")
    print(f"  Paper says: {LDV}")
    print(f"  Residual norm: {np.linalg.norm(sol.fun)*1000:.3f} ms")

    # Try fitting LDV + per-recording timing offset (one offset per recording)
    def residuals2(params):
        a, b = params[0], params[1]
        offsets = params[2:]  # 5 offsets, one per source position
        res = []
        for i, r in enumerate(rows):
            x_s = r["x"]
            d_ldv = np.hypot(x_s - a, 0 - b)
            d_l = np.hypot(x_s - MIC_L[0], 0 - MIC_L[1])
            d_r = np.hypot(x_s - MIC_R[0], 0 - MIC_R[1])
            tau_vl_pred = (d_l - d_ldv) / C_MPS + offsets[i]
            tau_vr_pred = (d_r - d_ldv) / C_MPS + offsets[i]
            res.append(tau_vl_pred - r["tau_vl_meas"])
            res.append(tau_vr_pred - r["tau_vr_meas"])
        return res

    x0_2 = [0.0, 0.25] + [0.0] * len(rows)
    sol2 = least_squares(residuals2, x0_2)
    print(f"\nWith per-recording timing offsets:")
    print(f"  Best-fit LDV: ({sol2.x[0]:+.3f}, {sol2.x[1]:+.3f}) m")
    print(f"  Per-pos offsets (ms): {[f'{o*1000:+.2f}' for o in sol2.x[2:]]}")
    print(f"  Residual norm: {np.linalg.norm(sol2.fun)*1000:.3f} ms")


if __name__ == "__main__":
    main()
