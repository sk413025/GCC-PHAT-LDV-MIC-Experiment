"""Loop 2 — geometry self-calibration + +x asymmetry investigation.

Hypothesis H2: stated mic positions (±0.7, 2.0) and LDV (0, 0.25) may be off.
Hypothesis H2b: +x side systematically locks to τ=0 in mic-mic GCC. Investigate
  whether this happens in UNBLOCK condition too. If yes → mic gain/geometry issue.
  If no → barrier-specific (block-only).

Procedure:
  1. For unblock condition, run mic-mic GCC at various bandpass settings, record
     measured τ_LR(x_s) for each source position.
  2. If unblock works for all positions → mic geometry roughly OK; +x failure is
     block-specific (i.e., barrier-induced).
  3. Fit (mic_L_x, mic_R_x) from 5-point unblock τ vs free-space prediction.
  4. Use refined geometry in Loop 1 to see if +x improves.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.optimize import least_squares

from _loader import chirp_groups, speech_groups, load_group
from _pigs import (cross_phat, MIC_L, MIC_R, C_MPS, MIC_SPACING,
                   preprocess as basic_preprocess, estimate_doa_micmic)
from _pigs2 import auto_window_chirp
from _geometry import expected_doa_deg, REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"


def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def measure_tau_lr(chans, sr, band_hz, max_lag_s=0.005):
    """Run mic-mic GCC-PHAT, return measured τ_LR (sec) and PSR."""
    chans_b = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    n = max(len(chans_b["mic_l"]), len(chans_b["mic_r"]))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xl = np.fft.rfft(chans_b["mic_l"], n_fft)
    Xr = np.fft.rfft(chans_b["mic_r"], n_fft)
    G = Xr * np.conj(Xl)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    G = G * ((f >= band_hz[0]) & (f <= band_hz[1]))
    eps = 1e-12
    r = np.fft.fftshift(np.fft.irfft(G / (np.abs(G) + eps), n_fft))
    max_n = int(max_lag_s * sr); mid = len(r) // 2
    R = np.abs(r[mid - max_n: mid + max_n + 1])
    lags = (np.arange(len(R)) - max_n) / sr
    pk = int(np.argmax(R))
    tau = float(lags[pk])
    # PSR: peak / median
    psr = float(R[pk] / (np.median(R) + 1e-12))
    return tau, psr, R, lags


def main():
    # Step 1: measure τ_LR for ALL positions, both block & unblock, multiple bands
    bands = [(500, 2000), (1000, 5000), (2000, 8000), (300, 4000)]
    records = []
    for sig_type in ("chirp", "speech"):
        groups = chirp_groups() if sig_type == "chirp" else speech_groups()
        for (pos, cond), paths in sorted(groups.items()):
            chans, sr = load_group(paths)
            if sig_type == "chirp":
                t0, t1 = auto_window_chirp(chans["mic_l"], sr, dur_s=1.6)
            else:
                t0, t1 = 5.0, 25.0
            n0, n1 = int(t0 * sr), int(t1 * sr)
            chans = {ch: x[n0:n1] for ch, x in chans.items()}
            chans = {ch: basic_preprocess(x, sr) for ch, x in chans.items()}
            for band in bands:
                tau, psr, _, _ = measure_tau_lr(chans, sr, band)
                records.append({
                    "sig": sig_type, "pos": pos, "cond": cond,
                    "band": band, "tau_meas": tau, "psr": psr,
                })

    # Step 2: print unblock asymmetry table
    print("\nUNBLOCK condition mic-mic τ_LR (truth value should match expected):")
    print(f"{'sig':>6} {'pos':>5} {'band':>13} | {'τ_exp':>9} {'τ_meas':>10} {'Δ':>7} {'PSR':>5}")
    print("-" * 70)
    for r in records:
        if r["cond"] != "unblock":
            continue
        x_pos = float(r["pos"])
        tau_exp = (np.hypot(x_pos - MIC_R[0], -MIC_R[1])
                   - np.hypot(x_pos - MIC_L[0], -MIC_L[1])) / C_MPS
        tau_m = r["tau_meas"]
        delta = tau_m - tau_exp
        print(f"{r['sig']:>6} {r['pos']:>5} {str(r['band']):>13} | "
              f"{tau_exp*1000:+7.3f}ms {tau_m*1000:+8.3f}ms "
              f"{delta*1000:+5.2f}ms {r['psr']:5.1f}")

    print("\nBLOCK condition mic-mic τ_LR (this is where it locks to 0):")
    for r in records:
        if r["cond"] != "block":
            continue
        x_pos = float(r["pos"])
        tau_exp = (np.hypot(x_pos - MIC_R[0], -MIC_R[1])
                   - np.hypot(x_pos - MIC_L[0], -MIC_L[1])) / C_MPS
        tau_m = r["tau_meas"]
        delta = tau_m - tau_exp
        print(f"{r['sig']:>6} {r['pos']:>5} {str(r['band']):>13} | "
              f"{tau_exp*1000:+7.3f}ms {tau_m*1000:+8.3f}ms "
              f"{delta*1000:+5.2f}ms {r['psr']:5.1f}")

    # Step 3: pick best unblock band per signal_type to fit mic positions
    # Use chirp unblock 1-5kHz (cleanest)
    fit_records = [r for r in records
                   if r["cond"] == "unblock" and r["sig"] == "chirp"
                   and r["band"] == (1000, 5000)]
    if len(fit_records) >= 3:
        # Fit (mic_L_x, mic_R_x, mic_y) — assume y same for both, source at (x_s, 0)
        def residuals(params):
            mLx, mRx, my = params
            res = []
            for r in fit_records:
                x_s = float(r["pos"])
                d_L = np.hypot(x_s - mLx, my)
                d_R = np.hypot(x_s - mRx, my)
                tau_pred = (d_R - d_L) / C_MPS
                res.append(tau_pred - r["tau_meas"])
            return res

        x0 = [MIC_L[0], MIC_R[0], MIC_L[1]]
        sol = least_squares(residuals, x0,
                           bounds=([-1.5, 0.0, 0.5], [0.0, 1.5, 4.0]))
        print(f"\nMic-position fit from unblock chirp 1-5kHz:")
        print(f"  Stated:   MIC_L=({MIC_L[0]:+.3f}, {MIC_L[1]:.3f}), MIC_R=({MIC_R[0]:+.3f}, {MIC_R[1]:.3f})")
        print(f"  Fitted:   MIC_L=({sol.x[0]:+.3f}, {sol.x[2]:.3f}), MIC_R=({sol.x[1]:+.3f}, {sol.x[2]:.3f})")
        print(f"  Spacing:  stated={MIC_SPACING:.3f}m, fitted={sol.x[1]-sol.x[0]:.3f}m")
        print(f"  Residual norm: {np.linalg.norm(sol.fun)*1000:.4f} ms")

        # Step 4: Also fit per-source-position offset (sensor latency or ADC drift)
        def residuals2(params):
            mLx, mRx, my = params[:3]
            offsets = params[3:]  # one offset per source position
            res = []
            for i, r in enumerate(fit_records):
                x_s = float(r["pos"])
                d_L = np.hypot(x_s - mLx, my)
                d_R = np.hypot(x_s - mRx, my)
                tau_pred = (d_R - d_L) / C_MPS + offsets[i]
                res.append(tau_pred - r["tau_meas"])
            return res

        x0_2 = [MIC_L[0], MIC_R[0], MIC_L[1]] + [0.0] * len(fit_records)
        sol2 = least_squares(residuals2, x0_2)
        print(f"\nWith per-recording timing offsets:")
        print(f"  Fitted:   MIC_L=({sol2.x[0]:+.3f}, {sol2.x[2]:.3f}), MIC_R=({sol2.x[1]:+.3f}, ...)")
        print(f"  Spacing:  {sol2.x[1]-sol2.x[0]:.3f}m")
        print(f"  Per-pos offsets (ms): {[f'{o*1000:+.3f}' for o in sol2.x[3:]]}")
        print(f"  Residual norm: {np.linalg.norm(sol2.fun)*1000:.4f} ms")

    out = OUT_DIR / "G_loop2_geocal.json"
    out.write_text(json.dumps(records, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
