"""Round 9 — refinements of V2 H38a self-selection rule.

V2 H38a: speech 3.74°. Oracle 1.59°. Gap is dominated by:
  -0.4: 5.37° (oracle 0.60° via simple bandpass — H38a too aggressive)
  +0.8: 3.68° (oracle 0.72° via H37)
  -0.8: 2.97° (oracle 2.33° via Wiener)
  +0.0: 1.93° (oracle 0° — H38a picked too-large |τ|)

H43  PI-GS 2D grid using H1 / H37 residuals
H47  Staged: H1 default → switch to H37 if H1 |τ| < 0.1 ms
H49  NLMS bigger taps (1024)
H51  3-way max-|τ| including low-band mic-only (rescues -0.4)
H52  Smart rule: avg if H1 & H37 agree, else max-|τ|
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
from _pigs import (MIC_L, MIC_R, LDV, C_MPS, MIC_SPACING,
                   preprocess as basic_preprocess, expected_tau_VM, interp_R)
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


def gcc(x_l, x_r, sr, band, max_lag_s=PHYS_MAX, return_R=False, return_psr=False):
    n = max(len(x_l), len(x_r))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xl = rfft(x_l, n_fft); Xr = rfft(x_r, n_fft)
    G = Xr * np.conj(Xl)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    G = G * ((f >= band[0]) & (f <= band[1]))
    eps = 1e-12
    r_full = np.fft.fftshift(irfft(G / (np.abs(G) + eps), n_fft))
    max_n = int(np.ceil(max_lag_s * sr))
    mid = len(r_full) // 2
    R = np.abs(r_full[mid - max_n: mid + max_n + 1])
    lags = (np.arange(len(R)) - max_n) / sr
    pk = int(np.argmax(R))
    tau = float(lags[pk])
    if 1 <= pk < len(R) - 1:
        ym, y0, yp = R[pk - 1], R[pk], R[pk + 1]
        d = ym - 2 * y0 + yp
        if abs(d) > 1e-12:
            tau += 0.5 * (ym - yp) / d * (lags[1] - lags[0])
    if return_R:
        return tau, R, lags
    if return_psr:
        off = np.ones(len(R), dtype=bool)
        off[max(0, pk - 3): min(len(R), pk + 4)] = False
        psr = R[pk] / (np.median(R[off]) + 1e-12) if off.any() else 0
        return tau, psr
    return tau


def diff_sum_gcc(x_l, x_r, sr, band, **kwargs):
    return gcc(x_l - x_r, x_l + x_r, sr, band, **kwargs)


def nlms(x_v, x_m, taps=256, mu=0.5):
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


# ============================================================================
# Component estimators
# ============================================================================

def est_h1(chans, sr, band=(300, 4000), taps=256):
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    e_L = nlms(chans_b["ldv"], chans_b["mic_l"], taps=taps)
    e_R = nlms(chans_b["ldv"], chans_b["mic_r"], taps=taps)
    tau, psr = gcc(e_L, e_R, sr, band, return_psr=True)
    return tau, psr


def est_h37(chans, sr, bands=None, taps=256):
    if bands is None:
        bands = [(500, 1500), (1000, 2500), (1500, 3500), (2500, 5000),
                 (3500, 7000), (1000, 5000), (300, 4000)]
    chans_full = {ch: bp(x, sr, 200, 8000) for ch, x in chans.items()}
    e_L = nlms(chans_full["ldv"], chans_full["mic_l"], taps=taps)
    e_R = nlms(chans_full["ldv"], chans_full["mic_r"], taps=taps)
    taus = []; psrs = []
    for band in bands:
        l_b = bp(e_L, sr, *band); r_b = bp(e_R, sr, *band)
        tau, psr = diff_sum_gcc(l_b, r_b, sr, band, return_psr=True)
        taus.append(tau); psrs.append(psr)
    taus = np.array(taus); psrs = np.array(psrs)
    nz = np.abs(taus) > 1e-4
    if nz.sum() >= 2:
        return float(np.median(taus[nz])), float(np.median(psrs[nz]))
    return float(np.median(taus)), float(np.median(psrs))


def est_low_band_micmic(chans, sr, band=(300, 1500)):
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    tau, psr = gcc(chans_b["mic_l"], chans_b["mic_r"], sr, band, return_psr=True)
    return tau, psr


# ============================================================================
# H43 — PI-GS 2D grid using H1 residuals
# ============================================================================

def h43_pigs_2d_residuals(chans, sr, sig_type, band=(500, 4000)):
    """Use NLMS LDV-subtract residuals, then PI-GS-style 2D grid search using
    cross-modal R_VL, R_VR (NOT mic-mic). This is paper PI-GS but on residuals.
    """
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    e_L = nlms(chans_b["ldv"], chans_b["mic_l"])
    e_R = nlms(chans_b["ldv"], chans_b["mic_r"])
    # Use the original LDV (not residual!) for cross-modal anchor
    x_v = chans_b["ldv"]
    # cross-modal GCC R_V-eL, R_V-eR
    n = max(len(e_L), len(e_R), len(x_v))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xv = rfft(x_v, n_fft); El = rfft(e_L, n_fft); Er = rfft(e_R, n_fft)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    eps = 1e-12
    G_VL = El * np.conj(Xv); G_VL = G_VL * ((f >= band[0]) & (f <= band[1]))
    G_VR = Er * np.conj(Xv); G_VR = G_VR * ((f >= band[0]) & (f <= band[1]))
    R_VL = np.fft.fftshift(irfft(G_VL / (np.abs(G_VL) + eps), n_fft))
    R_VR = np.fft.fftshift(irfft(G_VR / (np.abs(G_VR) + eps), n_fft))
    max_lag_s = 0.012
    max_n = int(max_lag_s * sr); mid = len(R_VL) // 2
    R_VL_w = np.abs(R_VL[mid - max_n: mid + max_n + 1])
    R_VR_w = np.abs(R_VR[mid - max_n: mid + max_n + 1])
    lags = (np.arange(len(R_VL_w)) - max_n) / sr
    # 1D source x grid (fix y=0)
    xs = np.arange(-1.5, 1.5 + 1e-9, 0.01)
    best_x = 0.0; best_score = -1
    for x_s in xs:
        p = (x_s, 0.0)
        tau_VL = expected_tau_VM(p, MIC_L)
        tau_VR = expected_tau_VM(p, MIC_R)
        s_l = float(np.interp(tau_VL, lags, R_VL_w, left=0, right=0))
        s_r = float(np.interp(tau_VR, lags, R_VR_w, left=0, right=0))
        score = s_l * s_r  # multiplicative for stricter agreement
        if score > best_score:
            best_score = score
            best_x = x_s
    # Convert source x to DoA
    tau_lr = (np.hypot(best_x - MIC_R[0], -MIC_R[1])
              - np.hypot(best_x - MIC_L[0], -MIC_L[1])) / C_MPS
    return doa_from_tau(tau_lr)


# ============================================================================
# H47 — Staged: H1 → H37 only if H1 locked-to-zero
# ============================================================================

def h47_staged(chans, sr, sig_type, lock_thresh_us=80):
    tau_h1, _ = est_h1(chans, sr)
    if abs(tau_h1) > lock_thresh_us * 1e-6:
        return doa_from_tau(tau_h1)
    tau_h37, _ = est_h37(chans, sr)
    return doa_from_tau(tau_h37)


def h47b_staged_with_low_band(chans, sr, sig_type, lock_thresh_us=80):
    """Staged: H1 → H37 → low-band mic-only."""
    tau_h1, _ = est_h1(chans, sr)
    if abs(tau_h1) > lock_thresh_us * 1e-6:
        return doa_from_tau(tau_h1)
    tau_h37, _ = est_h37(chans, sr)
    if abs(tau_h37) > lock_thresh_us * 1e-6:
        return doa_from_tau(tau_h37)
    tau_lb, _ = est_low_band_micmic(chans, sr)
    return doa_from_tau(tau_lb)


# ============================================================================
# H49 — Larger NLMS taps
# ============================================================================

def h49_max_abs_bigtap(chans, sr, sig_type, taps=1024):
    tau_h1, _ = est_h1(chans, sr, taps=taps)
    tau_h37, _ = est_h37(chans, sr, taps=taps)
    return doa_from_tau(tau_h37 if abs(tau_h37) > abs(tau_h1) else tau_h1)


# ============================================================================
# H51 — 3-way max-|τ|: H1, H37, low-band mic-only
# ============================================================================

def h51_three_max_abs(chans, sr, sig_type):
    tau_h1, _ = est_h1(chans, sr)
    tau_h37, _ = est_h37(chans, sr)
    tau_lb, _ = est_low_band_micmic(chans, sr, (300, 1500))
    candidates = [tau_h1, tau_h37, tau_lb]
    return doa_from_tau(max(candidates, key=abs))


def h51b_three_max_abs_v2(chans, sr, sig_type):
    """3-way: H1 (300-4k), H37 (multi-band), bp 500-2000 mic-only."""
    tau_h1, _ = est_h1(chans, sr)
    tau_h37, _ = est_h37(chans, sr)
    tau_500_2k, _ = est_low_band_micmic(chans, sr, (500, 2000))
    return doa_from_tau(max([tau_h1, tau_h37, tau_500_2k], key=abs))


# ============================================================================
# H52 — Smart rule: agree → average, disagree → max-|τ|
# ============================================================================

def h52_agree_average(chans, sr, sig_type, agree_tol_ms=0.2):
    tau_h1, _ = est_h1(chans, sr)
    tau_h37, _ = est_h37(chans, sr)
    same_sign = (tau_h1 * tau_h37) > 0
    close = abs(tau_h1 - tau_h37) < agree_tol_ms * 1e-3
    if same_sign and close:
        return doa_from_tau((tau_h1 + tau_h37) / 2)
    return doa_from_tau(tau_h37 if abs(tau_h37) > abs(tau_h1) else tau_h1)


# ============================================================================
# H53 — Adaptive: pick by |τ| AND PSR
# ============================================================================

def h53_psr_priority(chans, sr, sig_type):
    """Pick estimator with highest PSR among non-locked-zero candidates."""
    tau_h1, psr_h1 = est_h1(chans, sr)
    tau_h37, psr_h37 = est_h37(chans, sr)
    tau_lb, psr_lb = est_low_band_micmic(chans, sr, (500, 2000))
    candidates = [(tau_h1, psr_h1, "h1"), (tau_h37, psr_h37, "h37"),
                  (tau_lb, psr_lb, "lb")]
    # Filter out locked-to-zero
    non_locked = [c for c in candidates if abs(c[0]) > 8e-5]
    if not non_locked:
        # All locked; trust the one with highest PSR
        return doa_from_tau(max(candidates, key=lambda x: x[1])[0])
    return doa_from_tau(max(non_locked, key=lambda x: x[1])[0])


STRATS = [
    ("H43_pigs_2d_residuals", h43_pigs_2d_residuals),
    ("H47_staged_h1h37", h47_staged),
    ("H47b_staged_3way", h47b_staged_with_low_band),
    ("H49_max_abs_taps1024", lambda c, sr, s: h49_max_abs_bigtap(c, sr, s, 1024)),
    ("H49b_max_abs_taps512", lambda c, sr, s: h49_max_abs_bigtap(c, sr, s, 512)),
    ("H51_three_max_abs_lb_300_1500", h51_three_max_abs),
    ("H51b_three_max_abs_lb_500_2000", h51b_three_max_abs_v2),
    ("H52_agree_average", h52_agree_average),
    ("H53_psr_priority_3way", h53_psr_priority),
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

    print(f"\n{'strategy':<36} | {'chirp':>10} | {'speech':>10}")
    print("-" * 64)
    rows_sorted = sorted(summary.items(), key=lambda kv: kv[1]["speech_mae"] or 999)
    for sid, s in rows_sorted:
        c = s["chirp_mae"]; sp_ = s["speech_mae"]
        c_str = f"{c:6.2f}°" if c is not None else "  N/A "
        s_str = f"{sp_:6.2f}°" if sp_ is not None else "  N/A "
        print(f"{sid:<36} | {c_str:>10} | {s_str:>10}")

    print("\nPer-position breakdown of top 5:")
    for sid, _ in rows_sorted[:5]:
        print(f"\n=== {sid} ===")
        for sig in ("chirp", "speech"):
            if sig in summary[sid]["rows"]:
                print(f"  {sig}:")
                for pos in sorted(summary[sid]["rows"][sig]):
                    r = summary[sid]["rows"][sig][pos]
                    if r["err"] is None:
                        print(f"    x={pos}: NA"); continue
                    print(f"    x={pos}: true={r['true']:+6.2f}° "
                          f"est={r['est']:+6.2f}° err={r['err']:5.2f}°")

    out = OUT_DIR / "H_round9.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
