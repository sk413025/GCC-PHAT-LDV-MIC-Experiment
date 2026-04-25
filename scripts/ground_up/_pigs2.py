"""Refined PI-GS variants: multiplicative score, 1D grid (y=0 fixed), auto-window.

These are options that the strategy runner can flip on. Kept separate from _pigs.py
so the baseline remains a faithful paper Eq. 5/7 implementation.
"""
from __future__ import annotations
import numpy as np

from _pigs import (cross_phat, expected_tau_VM, MIC_L, MIC_R, LDV, C_MPS,
                   MIC_SPACING, interp_R)


def auto_window_chirp(x_l, sr, dur_s=1.5):
    """Find first high-energy burst in mic_L and return its (t0, t1) window."""
    n = int(sr * 0.02)
    e = np.sqrt(np.convolve(x_l ** 2, np.ones(n) / n, mode="same"))
    thresh = 0.3 * e.max()
    above = e > thresh
    n0 = int(np.argmax(above))  # first sample above threshold
    n1 = min(len(x_l), n0 + int(dur_s * sr))
    return n0 / sr, n1 / sr


def estimate_doa_pigs_1d(x_l, x_r, x_v, sr, max_lag_s=0.007, band_hz=None,
                        x_lo=-1.5, x_hi=1.5, x_step=0.005, y_fixed=0.0,
                        score="sum"):
    """1-D grid search over x at fixed y=y_fixed.

    score:
      "sum"  — |R_VL| + |R_VR| (paper Eq.7)
      "prod" — |R_VL| * |R_VR| (geometric intersection)
      "min"  — min(|R_VL|, |R_VR|) (worst-pair punishment)
    """
    xs = np.arange(x_lo, x_hi + 1e-9, x_step)
    pts = np.stack([xs, np.full_like(xs, y_fixed)], axis=1)

    lags_VL, R_VL = cross_phat(x_v, x_l, sr, max_lag_s=max_lag_s, band_hz=band_hz)
    lags_VR, R_VR = cross_phat(x_v, x_r, sr, max_lag_s=max_lag_s, band_hz=band_hz)

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
    else:
        raise ValueError(score)

    xi = int(np.argmax(S))
    p_hat = (float(xs[xi]), y_fixed)
    tau_lr = (np.hypot(p_hat[0] - MIC_R[0], p_hat[1] - MIC_R[1])
              - np.hypot(p_hat[0] - MIC_L[0], p_hat[1] - MIC_L[1])) / C_MPS
    s = max(-1.0, min(1.0, C_MPS * tau_lr / MIC_SPACING))
    theta_hat = -float(np.degrees(np.arcsin(s)))
    return {
        "theta_deg": theta_hat,
        "p_hat": p_hat,
        "S": S, "xs": xs,
        "lags_VL": lags_VL, "R_VL": R_VL,
        "lags_VR": lags_VR, "R_VR": R_VR,
    }
