"""Round 3 — deeper physics: dispersion, twin-recording, per-frame robustness.

Round 1+2 maxed out at speech 8.69° / chirp 8.4° MAE. This round attempts
fundamentally different angles:

H19  twin-recording calibration : block = α·unblock + wall_residual
       Recover direct-path component by deconvolving with unblock as ground truth
H21  per-frame TDoA + median   : τ per ~50ms frame; robust median over all frames
                                  rejects lock-to-zero on individual frames
H22  multi-narrow-band consistency : direct path has CONSTANT τ across bands
                                      (no air dispersion); wall has frequency-
                                      dependent τ. Pick band-invariant τ.
H23  H1 + H22 stacked          : LDV-NLMS subtract then multi-band consistency
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
from _pigs import (MIC_L, MIC_R, C_MPS, MIC_SPACING,
                   preprocess as basic_preprocess)
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


def gcc_R(x_l, x_r, sr, band, max_lag_s=PHYS_MAX):
    n = max(len(x_l), len(x_r))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xl = rfft(x_l, n_fft); Xr = rfft(x_r, n_fft)
    G = Xr * np.conj(Xl)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    G = G * ((f >= band[0]) & (f <= band[1]))
    eps = 1e-12
    r = np.fft.fftshift(irfft(G / (np.abs(G) + eps), n_fft))
    max_n = int(np.ceil(max_lag_s * sr))
    mid = len(r) // 2
    R = np.abs(r[mid - max_n: mid + max_n + 1])
    lags = (np.arange(len(R)) - max_n) / sr
    return R, lags


def pick_peak_subsample(R, lags):
    pk = int(np.argmax(R))
    tau = float(lags[pk])
    if 1 <= pk < len(R) - 1:
        ym, y0, yp = R[pk - 1], R[pk], R[pk + 1]
        d = ym - 2 * y0 + yp
        if abs(d) > 1e-12:
            tau += 0.5 * (ym - yp) / d * (lags[1] - lags[0])
    return tau


def nlms_subtract(x_v, x_m, taps=256, mu=0.5):
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
# H19 — twin-recording calibration
# ============================================================================

def h19_twin_recording(chans_block, chans_unblock, sr, band):
    """Find α (frequency-domain attenuation) such that block ≈ α·unblock + wall.
    Recover direct part as α·unblock and run mic-mic GCC on it.
    """
    # Match lengths
    n = min(len(chans_block["mic_l"]), len(chans_unblock["mic_l"]))
    bl = chans_block["mic_l"][:n]; br = chans_block["mic_r"][:n]
    ul = chans_unblock["mic_l"][:n]; ur = chans_unblock["mic_r"][:n]
    bl = bp(bl, sr, *band); br = bp(br, sr, *band)
    ul = bp(ul, sr, *band); ur = bp(ur, sr, *band)
    # Estimate per-bin α via Wiener: α(f) = E[B U*] / E[|U|^2]
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Bl = rfft(bl, n_fft); Br = rfft(br, n_fft)
    Ul = rfft(ul, n_fft); Ur = rfft(ur, n_fft)
    eps = 1e-12
    alpha_L = (Bl * np.conj(Ul)) / (np.abs(Ul) ** 2 + eps * np.abs(Ul).max() ** 2)
    alpha_R = (Br * np.conj(Ur)) / (np.abs(Ur) ** 2 + eps * np.abs(Ur).max() ** 2)
    # Direct-path estimates
    direct_L = irfft(alpha_L * Ul, n_fft)[:n]
    direct_R = irfft(alpha_R * Ur, n_fft)[:n]
    R, lags = gcc_R(direct_L, direct_R, sr, band)
    return doa_from_tau(pick_peak_subsample(R, lags))


# ============================================================================
# H21 — per-frame TDoA + median
# ============================================================================

def h21_per_frame_median(chans, sr, sig_type, band, frame_ms=80, hop_ms=40,
                       reject_zero_us=200):
    """Compute mic-mic τ per frame, return median across frames (excluding |τ|<reject_zero)."""
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    n_frame = int(frame_ms / 1000 * sr)
    n_hop = int(hop_ms / 1000 * sr)
    taus = []
    for s0 in range(0, len(chans_b["mic_l"]) - n_frame, n_hop):
        sl = slice(s0, s0 + n_frame)
        R, lags = gcc_R(chans_b["mic_l"][sl], chans_b["mic_r"][sl], sr, band)
        # Optional zero rejection
        if reject_zero_us > 0:
            zr = int(reject_zero_us * 1e-6 * sr)
            mid = len(R) // 2
            R_m = R.copy()
            R_m[max(0, mid - zr): min(len(R), mid + zr + 1)] = 0
            tau = pick_peak_subsample(R_m, lags)
        else:
            tau = pick_peak_subsample(R, lags)
        taus.append(tau)
    if not taus:
        return None
    return doa_from_tau(float(np.median(taus)))


def h21_per_frame_with_nlms(chans, sr, sig_type, band, frame_ms=80, hop_ms=40):
    """H1 LDV-NLMS first, then per-frame median."""
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    e_L = nlms_subtract(chans_b["ldv"], chans_b["mic_l"])
    e_R = nlms_subtract(chans_b["ldv"], chans_b["mic_r"])
    n_frame = int(frame_ms / 1000 * sr)
    n_hop = int(hop_ms / 1000 * sr)
    taus = []
    for s0 in range(0, len(e_L) - n_frame, n_hop):
        sl = slice(s0, s0 + n_frame)
        R, lags = gcc_R(e_L[sl], e_R[sl], sr, band)
        taus.append(pick_peak_subsample(R, lags))
    return doa_from_tau(float(np.median(taus))) if taus else None


# ============================================================================
# H22 — multi-narrow-band TDoA consistency
# ============================================================================

def h22_multiband_consistent(chans, sr, sig_type, narrow_bands=None,
                             tolerance_ms=0.3):
    """Compute τ in each narrow band; cluster; pick most populous cluster's τ.

    Direct path → constant τ across bands (air is non-dispersive)
    Wall multipath → τ varies with band (board bending wave is dispersive)
    """
    if narrow_bands is None:
        narrow_bands = [(300, 800), (500, 1200), (800, 1800), (1200, 2500),
                        (1800, 3500), (2500, 5000), (3500, 6500), (5000, 8000)]
    chans_full = chans
    taus = []
    for band in narrow_bands:
        chans_b = {ch: bp(x, sr, *band) for ch, x in chans_full.items()}
        R, lags = gcc_R(chans_b["mic_l"], chans_b["mic_r"], sr, band)
        taus.append(pick_peak_subsample(R, lags))
    taus = np.array(taus)
    # Cluster: for each tau, count how many other taus are within tolerance
    tol = tolerance_ms / 1000
    best_score = -1; best_tau = 0
    for tau_c in taus:
        score = np.sum(np.abs(taus - tau_c) <= tol)
        if score > best_score or (score == best_score and abs(tau_c) > abs(best_tau)):
            # Tie-break: prefer non-zero
            best_score = score
            best_tau = tau_c
    # Refine by averaging within-cluster taus
    inliers = taus[np.abs(taus - best_tau) <= tol]
    refined_tau = float(np.mean(inliers))
    return doa_from_tau(refined_tau)


def h23_nlms_then_multiband(chans, sr, sig_type, narrow_bands=None,
                            tolerance_ms=0.3):
    """H1 LDV-NLMS → multi-narrow-band consistency on residuals."""
    if narrow_bands is None:
        narrow_bands = [(300, 800), (500, 1200), (800, 1800), (1200, 2500),
                        (1800, 3500), (2500, 5000), (3500, 6500), (5000, 8000)]
    # NLMS on a wide band first
    chans_b = {ch: bp(x, sr, 300, 8000) for ch, x in chans.items()}
    e_L = nlms_subtract(chans_b["ldv"], chans_b["mic_l"])
    e_R = nlms_subtract(chans_b["ldv"], chans_b["mic_r"])
    taus = []
    for band in narrow_bands:
        e_L_b = bp(e_L, sr, *band)
        e_R_b = bp(e_R, sr, *band)
        R, lags = gcc_R(e_L_b, e_R_b, sr, band)
        taus.append(pick_peak_subsample(R, lags))
    taus = np.array(taus)
    tol = tolerance_ms / 1000
    best_score = -1; best_tau = 0
    for tau_c in taus:
        score = np.sum(np.abs(taus - tau_c) <= tol)
        if score > best_score:
            best_score = score; best_tau = tau_c
    inliers = taus[np.abs(taus - best_tau) <= tol]
    refined_tau = float(np.mean(inliers))
    return doa_from_tau(refined_tau), taus


def h23b_per_frame_multiband(chans, sr, sig_type, narrow_bands=None):
    """Per-frame × per-band τ matrix; pick globally-consistent τ."""
    if narrow_bands is None:
        narrow_bands = [(300, 800), (500, 1200), (1000, 2000), (1500, 3500),
                        (2500, 5000), (4000, 7000)]
    chans_full = chans
    n_frame = int(0.08 * sr); n_hop = int(0.04 * sr)
    all_taus = []
    for s0 in range(0, len(chans_full["mic_l"]) - n_frame, n_hop):
        for band in narrow_bands:
            chans_b = {ch: bp(x[s0:s0+n_frame], sr, *band)
                       for ch, x in chans_full.items()}
            R, lags = gcc_R(chans_b["mic_l"], chans_b["mic_r"], sr, band)
            all_taus.append(pick_peak_subsample(R, lags))
    all_taus = np.array(all_taus)
    # 2D mode-ish: histogram with 0.1ms bins, pick highest non-zero bin
    bins = np.arange(-2.0, 2.0, 0.1) * 1e-3
    hist, edges = np.histogram(all_taus, bins=bins)
    # Find peak bin (could include zero); but require width >= 2 bins?
    pk = int(np.argmax(hist))
    tau = float((edges[pk] + edges[pk + 1]) / 2)
    # Refine by averaging within ±0.1ms
    inliers = all_taus[np.abs(all_taus - tau) <= 1e-4]
    if len(inliers) > 5:
        tau = float(np.median(inliers))
    return doa_from_tau(tau)


# ============================================================================
# Driver
# ============================================================================

STRATS_PER_REC = [
    ("H21_frame80_500_2k", lambda c, sr, s: h21_per_frame_median(c, sr, s, (500, 2000))),
    ("H21_frame80_300_4k", lambda c, sr, s: h21_per_frame_median(c, sr, s, (300, 4000))),
    ("H21_frame80_1k_5k", lambda c, sr, s: h21_per_frame_median(c, sr, s, (1000, 5000))),
    ("H21_NLMS_frame80_300_4k", lambda c, sr, s: h21_per_frame_with_nlms(c, sr, s, (300, 4000))),
    ("H22_multiband_default", lambda c, sr, s: h22_multiband_consistent(c, sr, s)),
    ("H22_multiband_tol0.5ms", lambda c, sr, s: h22_multiband_consistent(c, sr, s, tolerance_ms=0.5)),
    ("H22_multiband_8bands_tol0.2", lambda c, sr, s: h22_multiband_consistent(c, sr, s, tolerance_ms=0.2)),
    ("H23_NLMS_multiband", lambda c, sr, s: h23_nlms_then_multiband(c, sr, s)[0]),
    ("H23b_perframe_multiband", lambda c, sr, s: h23b_per_frame_multiband(c, sr, s)),
]


def run_per_rec_strats():
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
            for sid, fn in STRATS_PER_REC:
                try:
                    theta = fn(chans, sr, sig_type)
                except Exception as e:
                    theta = None
                err = abs(theta - theta_true) if theta is not None else None
                results.setdefault(sid, {}).setdefault(sig_type, {})[pos] = {
                    "true": theta_true, "est": theta, "err": err}
    return results


def run_h19():
    """H19 needs paired block + unblock. Mics-only (unblock has no LDV)."""
    results = {}
    for sig_type in ("chirp", "speech"):
        groups = chirp_groups() if sig_type == "chirp" else speech_groups()
        for (pos, cond), paths in sorted(groups.items()):
            if cond != "block":
                continue
            unblock_key = (pos, "unblock")
            if unblock_key not in groups:
                continue
            block_chans, sr = load_group(paths)
            unblock_chans, _ = load_group(groups[unblock_key])
            # Both use same time window
            if sig_type == "chirp":
                t0, t1 = auto_window_chirp(block_chans["mic_l"], sr, dur_s=1.6)
            else:
                t0, t1 = 5.0, 25.0
            n0, n1 = int(t0 * sr), int(t1 * sr)
            block_chans = {ch: x[n0:n1] for ch, x in block_chans.items()}
            unblock_chans = {ch: x[n0:n1] for ch, x in unblock_chans.items()
                             if ch in ("mic_l", "mic_r")}
            block_chans = {ch: basic_preprocess(x, sr)
                           for ch, x in block_chans.items()}
            unblock_chans = {ch: basic_preprocess(x, sr)
                             for ch, x in unblock_chans.items()}
            theta_true = expected_doa_deg(float(pos))
            for band_label, band in [("500_2k", (500, 2000)),
                                      ("1k_5k", (1000, 5000)),
                                      ("300_4k", (300, 4000))]:
                sid = f"H19_twin_{band_label}"
                try:
                    theta = h19_twin_recording(block_chans, unblock_chans,
                                              sr, band)
                except Exception as e:
                    theta = None
                err = abs(theta - theta_true) if theta is not None else None
                results.setdefault(sid, {}).setdefault(sig_type, {})[pos] = {
                    "true": theta_true, "est": theta, "err": err}
    return results


def summarize(results):
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
    return summary


def main():
    r1 = run_per_rec_strats()
    r2 = run_h19()
    r1.update(r2)
    summary = summarize(r1)

    print(f"\n{'strategy':<32} | {'chirp':>10} | {'speech':>10}")
    print("-" * 60)
    rows_sorted = sorted(summary.items(), key=lambda kv: kv[1]["speech_mae"] or 999)
    for sid, s in rows_sorted:
        c = s["chirp_mae"]; sp_ = s["speech_mae"]
        c_str = f"{c:6.2f}°" if c is not None else "  N/A "
        s_str = f"{sp_:6.2f}°" if sp_ is not None else "  N/A "
        print(f"{sid:<32} | {c_str:>10} | {s_str:>10}")

    print("\nPer-position breakdown of top 5 by speech MAE:")
    for sid, _ in rows_sorted[:5]:
        print(f"\n=== {sid} ===")
        for sig in ("chirp", "speech"):
            if sig in summary[sid]["rows"]:
                print(f"  {sig}:")
                for pos in sorted(summary[sid]["rows"][sig]):
                    r = summary[sid]["rows"][sig][pos]
                    if r["err"] is None:
                        print(f"    x={pos}: NA")
                        continue
                    print(f"    x={pos}: true={r['true']:+6.2f}° "
                          f"est={r['est']:+6.2f}° err={r['err']:5.2f}°")

    out = OUT_DIR / "H_round3.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
