"""Core PI-GS implementation written from scratch from paper Eq. (5) & (7).

No imports from project's existing scripts. Functions:

  cross_phat(x_v, x_m, sr, max_lag_s) -> (lags_s, R) — paper Eq. (5)
  build_grid(...) -> (XX, YY, points)
  pigs_score(grid, lags_VL, R_VL, lags_VR, R_VR, sr) -> S(p)
  pigs_estimate_doa(...) -> (theta_deg, p_hat, S_grid)

Sign convention (declared explicitly to avoid bugs, verified by b02_symbol_sanity.py):
  We compute G = X_M * conj(X_V), then PHAT-normalize and IFFT.
  Peak of R_VM at τ̂ means: x_m arrives τ̂ seconds after x_v.
  So expected τ̂ = (||p - mic|| - ||p - ldv||) / c (positive when mic farther).
  (Paper Eq. 5 writes X_v X_m^* — same up to mirror; we use the order that puts
  positive τ at "mic later" which is more intuitive.)
"""
from __future__ import annotations
import numpy as np
from numpy.fft import rfft, irfft
from scipy.signal import iirnotch, filtfilt


def remove_dc_and_hum(x: np.ndarray, sr: int, hum_freqs=(60, 120, 180, 240, 300), q=30):
    x = x - np.mean(x)
    for f0 in hum_freqs:
        if f0 < sr / 2 - 5:
            b, a = iirnotch(f0 / (sr / 2), q)
            x = filtfilt(b, a, x)
    return x


def preprocess(x: np.ndarray, sr: int):
    """Mandatory pre-processing applied to baseline (DC + 60Hz comb notch)."""
    return remove_dc_and_hum(x, sr)


def cross_phat(x_v: np.ndarray, x_m: np.ndarray, sr: int, max_lag_s: float = 0.01,
               band_hz: tuple | None = None):
    """Cross-modal GCC-PHAT between LDV(x_v) and mic(x_m).

    Returns (lags_s, R) where lags_s is symmetric around 0, length 2*max_lag_samples+1.
    """
    n = max(len(x_v), len(x_m))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xv = rfft(x_v, n_fft)
    Xm = rfft(x_m, n_fft)
    G = Xm * np.conj(Xv)  # peak of IFFT(G/|G|) at τ = t_m - t_v
    if band_hz is not None:
        f = np.fft.rfftfreq(n_fft, 1 / sr)
        mask = (f >= band_hz[0]) & (f <= band_hz[1])
        G = G * mask
    eps = 1e-12
    Gphat = G / (np.abs(G) + eps)
    r_full = irfft(Gphat, n_fft)
    # Lag axis: [0, +1, ..., +n_fft/2-1, -n_fft/2, ..., -1] -> shift to [-N/2, +N/2-1]
    r = np.fft.fftshift(r_full)
    lags = np.arange(-n_fft // 2, n_fft // 2) / sr
    # Trim to ±max_lag
    max_n = int(np.ceil(max_lag_s * sr))
    mid = len(r) // 2
    sl = slice(mid - max_n, mid + max_n + 1)
    return lags[sl], r[sl]


def euclid(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


# Geometry constants reused
C_MPS = 343.0
MIC_L = (-0.7, 2.0)
MIC_R = (+0.7, 2.0)
LDV = (0.0, 0.25)
MIC_SPACING = abs(MIC_R[0] - MIC_L[0])


def expected_tau_VM(p, mic):
    """Expected lag τ_VM for source at p; positive means mic later than ldv."""
    return (euclid(p, mic) - euclid(p, LDV)) / C_MPS


def expected_tau_LR(p):
    return (euclid(p, MIC_R) - euclid(p, MIC_L)) / C_MPS


def build_grid(x_lo=-1.5, x_hi=1.5, x_step=0.01, y_lo=-0.1, y_hi=1.0, y_step=0.05):
    xs = np.arange(x_lo, x_hi + 1e-9, x_step)
    ys = np.arange(y_lo, y_hi + 1e-9, y_step)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    return xs, ys, pts


def interp_R(lags_s, R, query_lags_s):
    return np.interp(query_lags_s, lags_s, R, left=0.0, right=0.0)


def pigs_score(pts, lags_VL, R_VL, lags_VR, R_VR):
    tau_VL = np.array([expected_tau_VM(p, MIC_L) for p in pts])
    tau_VR = np.array([expected_tau_VM(p, MIC_R) for p in pts])
    s_l = interp_R(lags_VL, np.abs(R_VL), tau_VL)
    s_r = interp_R(lags_VR, np.abs(R_VR), tau_VR)
    return s_l + s_r


def pigs_doa_from_grid(xs, ys, S_grid):
    flat = S_grid.argmax()
    yi, xi = np.unravel_index(flat, S_grid.shape)
    p_hat = (float(xs[xi]), float(ys[yi]))
    tau_LR = expected_tau_LR(p_hat)
    s = max(-1.0, min(1.0, C_MPS * tau_LR / MIC_SPACING))
    theta_hat = -float(np.degrees(np.arcsin(s)))
    return theta_hat, p_hat


def estimate_doa_pigs(x_l, x_r, x_v, sr, max_lag_s=0.007, band_hz=None,
                     grid_kwargs=None):
    """Full PI-GS DoA estimate."""
    grid_kwargs = grid_kwargs or {}
    xs, ys, pts = build_grid(**grid_kwargs)
    lags_VL, R_VL = cross_phat(x_v, x_l, sr, max_lag_s=max_lag_s, band_hz=band_hz)
    lags_VR, R_VR = cross_phat(x_v, x_r, sr, max_lag_s=max_lag_s, band_hz=band_hz)
    S = pigs_score(pts, lags_VL, R_VL, lags_VR, R_VR)
    S_grid = S.reshape(len(ys), len(xs))
    theta_hat, p_hat = pigs_doa_from_grid(xs, ys, S_grid)
    return {
        "theta_deg": theta_hat,
        "p_hat": p_hat,
        "S_grid": S_grid,
        "xs": xs, "ys": ys,
        "lags_VL": lags_VL, "R_VL": R_VL,
        "lags_VR": lags_VR, "R_VR": R_VR,
    }


def estimate_doa_micmic(x_l, x_r, sr, max_lag_s=0.005, band_hz=None):
    """Mic-only baseline: classic GCC-PHAT between mic_L and mic_R."""
    n = max(len(x_l), len(x_r))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xl = rfft(x_l, n_fft)
    Xr = rfft(x_r, n_fft)
    G = Xr * np.conj(Xl)  # peak of IFFT(G/|G|) at τ = t_r - t_l
    if band_hz is not None:
        f = np.fft.rfftfreq(n_fft, 1 / sr)
        mask = (f >= band_hz[0]) & (f <= band_hz[1])
        G = G * mask
    eps = 1e-12
    Gphat = G / (np.abs(G) + eps)
    r = np.fft.fftshift(irfft(Gphat, n_fft))
    max_n = int(np.ceil(max_lag_s * sr))
    mid = len(r) // 2
    R = r[mid - max_n: mid + max_n + 1]
    lags = (np.arange(len(R)) - max_n) / sr
    peak = int(np.argmax(np.abs(R)))
    tau_hat = float(lags[peak])
    s = max(-1.0, min(1.0, C_MPS * tau_hat / MIC_SPACING))
    theta_hat = -float(np.degrees(np.arcsin(s)))
    return theta_hat, tau_hat, lags, R
