"""Round 2 — physics hypotheses I missed in round 1.

Each hypothesis is a NEW physical/signal-processing idea, independent of round 1.

H11 early-time window — direct path arrives before plate rings up
H12 lag-zero exclude — common-mode wall radiation peaks at τ=0; reject and pick next
H13 AR pre-whitening — wall ringing is AR process; inverse-filter it out
H14 symmetry differential — wall reflection same for +x and -x; subtract paired GCCs
H15 mic-mic coherence mask — only use frequencies where mic-mic γ² is high (direct dominant)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from numpy.fft import rfft, irfft
from scipy.signal import butter, filtfilt, hilbert, lfilter
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


# ============================================================================
# H12 — lag-zero exclusion + second-peak selection
# ============================================================================

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


def pick_with_zero_excl(R, lags, exclude_us=300):
    """Exclude |τ| < exclude_us microseconds from peak search."""
    excl_n = int(exclude_us * 1e-6 / (lags[1] - lags[0]))
    mid = len(R) // 2
    R_masked = R.copy()
    R_masked[max(0, mid - excl_n): min(len(R), mid + excl_n + 1)] = 0
    pk = int(np.argmax(R_masked))
    tau = float(lags[pk])
    if 1 <= pk < len(R) - 1:
        ym, y0, yp = R[pk - 1], R[pk], R[pk + 1]
        d = ym - 2 * y0 + yp
        if abs(d) > 1e-12:
            tau += 0.5 * (ym - yp) / d / (1 / (lags[1] - lags[0]))
    return tau


def h12_lag_zero_excl(chans, sr, sig_type, band, exclude_us=300):
    chans = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    R, lags = gcc_R(chans["mic_l"], chans["mic_r"], sr, band)
    tau = pick_with_zero_excl(R, lags, exclude_us)
    return doa_from_tau(tau)


# ============================================================================
# H11 — early-time chirp window (pre-ringup)
# ============================================================================

def h11_early_window(chans, sr, sig_type, band, win_ms=15, n_bursts=6):
    """For chirp only: take first win_ms of each detected burst, average GCC."""
    if sig_type != "chirp":
        return None
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    # Detect bursts via mic_L envelope crossings
    e = np.abs(hilbert(chans_b["mic_l"]))
    e = np.convolve(e, np.ones(int(0.005 * sr)) / int(0.005 * sr), mode="same")
    thresh = 0.3 * e.max()
    above = e > thresh
    # Find rising edges
    rising = np.where(np.diff(above.astype(int)) > 0)[0]
    n_win = int(win_ms / 1000 * sr)
    if len(rising) == 0:
        return None
    R_acc = None; lags_ = None
    cnt = 0
    for r0 in rising[:n_bursts]:
        sl = slice(r0, r0 + n_win)
        if sl.stop > len(chans_b["mic_l"]):
            break
        R, lags_ = gcc_R(chans_b["mic_l"][sl], chans_b["mic_r"][sl], sr, band)
        R_acc = R if R_acc is None else R_acc + R
        cnt += 1
    if cnt == 0:
        return None
    R_acc /= cnt
    pk = int(np.argmax(R_acc))
    tau = float(lags_[pk])
    if 1 <= pk < len(R_acc) - 1:
        ym, y0, yp = R_acc[pk - 1], R_acc[pk], R_acc[pk + 1]
        d = ym - 2 * y0 + yp
        if abs(d) > 1e-12:
            tau += 0.5 * (ym - yp) / d / sr
    return doa_from_tau(tau)


# ============================================================================
# H13 — AR inverse-filter pre-whitening
# ============================================================================

def ar_whiten(x, order=64):
    """Fit AR(order) to x via Yule-Walker, return inverse-filtered residual.
    Uses FFT-based autocorrelation: O(n log n) instead of O(n^2).
    """
    n = len(x)
    if n < 4 * order:
        return x.copy()
    # FFT-based autocorrelation
    x_centered = x - np.mean(x)
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x_centered, n_fft)
    r = np.fft.irfft(np.abs(X) ** 2, n_fft)[:order + 1] / n
    if r[0] == 0:
        return x.copy()
    # Levinson-Durbin
    a = np.zeros(order + 1); a[0] = 1
    e = r[0]
    for i in range(1, order + 1):
        k = -np.sum(a[:i] * r[i:0:-1]) / e
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        a = a_new
        e = e * (1 - k * k)
        if e <= 0:
            break
    return lfilter(a, [1.0], x)


def h13_ar_whiten(chans, sr, sig_type, band, ar_order=64):
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    chans_w = {ch: ar_whiten(x, order=ar_order) for ch, x in chans_b.items()}
    R, lags = gcc_R(chans_w["mic_l"], chans_w["mic_r"], sr, band)
    pk = int(np.argmax(R))
    tau = float(lags[pk])
    return doa_from_tau(tau)


def h13_ar_zeroexcl(chans, sr, sig_type, band, ar_order=64):
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    chans_w = {ch: ar_whiten(x, order=ar_order) for ch, x in chans_b.items()}
    R, lags = gcc_R(chans_w["mic_l"], chans_w["mic_r"], sr, band)
    tau = pick_with_zero_excl(R, lags, exclude_us=300)
    return doa_from_tau(tau)


# ============================================================================
# H14 — symmetry differential (paired +x vs -x)
# ============================================================================

def h14_symmetry_diff_compute_all(groups, sig_type, band):
    """Compute GCC R(τ) for all positions, then for each (+x_s, -x_s) pair,
    compute differential R_+ - R_- and locate peak. Both members of pair share
    the result (since |x_s| same → physically same magnitude τ).

    Returns dict {pos: theta}.
    """
    Rs = {}
    lags_ = None
    for (pos, cond), paths in sorted(groups.items()):
        if cond != "block":
            continue
        chans, _sr = load_group(paths)
        if sig_type == "chirp":
            t0, t1 = auto_window_chirp(chans["mic_l"], _sr, dur_s=1.6)
        else:
            t0, t1 = 5.0, 25.0
        n0, n1 = int(t0 * _sr), int(t1 * _sr)
        chans = {ch: x[n0:n1] for ch, x in chans.items()}
        chans = {ch: basic_preprocess(x, _sr) for ch, x in chans.items()}
        chans_b = {ch: bp(x, _sr, *band) for ch, x in chans.items()}
        R, lags_ = gcc_R(chans_b["mic_l"], chans_b["mic_r"], _sr, band)
        Rs[pos] = R

    out = {}
    for pos in sorted(Rs):
        x_s = float(pos)
        partner = f"{-x_s:+.1f}".replace("+0.0", "+0.0").replace("-0.0", "+0.0")
        # Manual pairing: +0.4 ↔ -0.4, +0.8 ↔ -0.8, +0.0 alone
        if x_s == 0.0:
            # No symmetric pair; use raw R
            R_use = Rs[pos]
        else:
            # Look for partner key with opposite sign
            partner_R = None
            for k in Rs:
                if abs(float(k) + x_s) < 1e-6:
                    partner_R = Rs[k]
                    break
            if partner_R is None:
                R_use = Rs[pos]
            else:
                # Differential: R_+ - R_-(reversed)
                # If true source is at +x: R_+ has direct peak at -|τ|, wall at 0
                #                          R_- has direct peak at +|τ|, wall at 0
                # R_+(τ) - R_-(-τ) → wall at 0 cancels, direct adds
                R_minus_rev = partner_R[::-1]
                R_use = Rs[pos] - R_minus_rev
        pk = int(np.argmax(R_use))
        tau = float(lags_[pk])
        if 1 <= pk < len(R_use) - 1:
            ym, y0, yp = R_use[pk - 1], R_use[pk], R_use[pk + 1]
            d = ym - 2 * y0 + yp
            if abs(d) > 1e-12:
                tau += 0.5 * (ym - yp) / d / 48000.0  # assume 48k
        out[pos] = doa_from_tau(tau)
    return out


# ============================================================================
# H15 — mic-mic coherence mask (use only freqs where γ² high)
# ============================================================================

def h15_micmic_coh_mask(chans, sr, sig_type, band, coh_thresh=0.4):
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    f_c, c = sp.coherence(chans_b["mic_l"], chans_b["mic_r"], fs=sr,
                         nperseg=2048, noverlap=1024)
    n = max(len(chans_b["mic_l"]), len(chans_b["mic_r"]))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    f_full = np.fft.rfftfreq(n_fft, 1 / sr)
    coh_interp = np.interp(f_full, f_c, c)
    Xl = rfft(chans_b["mic_l"], n_fft)
    Xr = rfft(chans_b["mic_r"], n_fft)
    G = Xr * np.conj(Xl)
    eps = 1e-12
    band_mask = ((f_full >= band[0]) & (f_full <= band[1])).astype(float)
    coh_mask = (coh_interp > coh_thresh).astype(float)
    weight = band_mask * coh_mask
    Gw = G / (np.abs(G) + eps) * weight
    r = np.fft.fftshift(irfft(Gw, n_fft))
    max_n = int(np.ceil(PHYS_MAX * sr))
    mid = len(r) // 2
    R = np.abs(r[mid - max_n: mid + max_n + 1])
    lags = (np.arange(len(R)) - max_n) / sr
    pk = int(np.argmax(R))
    tau = float(lags[pk])
    return doa_from_tau(tau)


# ============================================================================
# Driver
# ============================================================================

STRATS = [
    # H12
    ("H12_zero_excl_300us_500_2k", lambda c, sr, s: h12_lag_zero_excl(c, sr, s, (500, 2000), 300)),
    ("H12_zero_excl_300us_300_4k", lambda c, sr, s: h12_lag_zero_excl(c, sr, s, (300, 4000), 300)),
    ("H12_zero_excl_300us_1k_5k", lambda c, sr, s: h12_lag_zero_excl(c, sr, s, (1000, 5000), 300)),
    ("H12_zero_excl_500us_500_2k", lambda c, sr, s: h12_lag_zero_excl(c, sr, s, (500, 2000), 500)),
    ("H12_zero_excl_100us_500_2k", lambda c, sr, s: h12_lag_zero_excl(c, sr, s, (500, 2000), 100)),
    # H11
    ("H11_early15ms_300_4k", lambda c, sr, s: h11_early_window(c, sr, s, (300, 4000), 15)),
    ("H11_early30ms_500_5k", lambda c, sr, s: h11_early_window(c, sr, s, (500, 5000), 30)),
    ("H11_early50ms_300_4k", lambda c, sr, s: h11_early_window(c, sr, s, (300, 4000), 50)),
    # H13
    ("H13_ar32_500_2k", lambda c, sr, s: h13_ar_whiten(c, sr, s, (500, 2000), 32)),
    ("H13_ar64_500_2k", lambda c, sr, s: h13_ar_whiten(c, sr, s, (500, 2000), 64)),
    ("H13_ar128_300_4k", lambda c, sr, s: h13_ar_whiten(c, sr, s, (300, 4000), 128)),
    ("H13_ar64_zexcl_500_2k", lambda c, sr, s: h13_ar_zeroexcl(c, sr, s, (500, 2000), 64)),
    # H15
    ("H15_coh4_500_2k", lambda c, sr, s: h15_micmic_coh_mask(c, sr, s, (500, 2000), 0.4)),
    ("H15_coh3_300_4k", lambda c, sr, s: h15_micmic_coh_mask(c, sr, s, (300, 4000), 0.3)),
    ("H15_coh5_500_5k", lambda c, sr, s: h15_micmic_coh_mask(c, sr, s, (500, 5000), 0.5)),
]


def run_per_recording():
    """Run STRATS (which only need single recording) on all 5 positions."""
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
    return results


def run_h14():
    """Run H14 symmetry differential — needs pairs of recordings."""
    out = {}
    for sig_type in ("chirp", "speech"):
        groups = chirp_groups() if sig_type == "chirp" else speech_groups()
        for band_label, band in [("500_2k", (500, 2000)), ("300_4k", (300, 4000)),
                                  ("1k_5k", (1000, 5000))]:
            sid = f"H14_symdiff_{band_label}"
            doas = h14_symmetry_diff_compute_all(groups, sig_type, band)
            for pos, theta in doas.items():
                theta_true = expected_doa_deg(float(pos))
                err = abs(theta - theta_true)
                out.setdefault(sid, {}).setdefault(sig_type, {})[pos] = {
                    "true": theta_true, "est": theta, "err": err}
    return out


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
    r1 = run_per_recording()
    r2 = run_h14()
    r1.update(r2)
    summary = summarize(r1)

    print(f"\n{'strategy':<32} | {'chirp':>10} | {'speech':>10}")
    print("-" * 60)
    rows_sorted = sorted(summary.items(),
                         key=lambda kv: kv[1]["speech_mae"] or 999)
    for sid, s in rows_sorted:
        c = s["chirp_mae"]; sp_ = s["speech_mae"]
        c_str = f"{c:6.2f}°" if c is not None else "  N/A "
        s_str = f"{sp_:6.2f}°" if sp_ is not None else "  N/A "
        print(f"{sid:<32} | {c_str:>10} | {s_str:>10}")

    # Per-position for top 3
    print("\nPer-position breakdown of top 3 by speech MAE:")
    for sid, _ in rows_sorted[:3]:
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

    out = OUT_DIR / "H_round2.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
