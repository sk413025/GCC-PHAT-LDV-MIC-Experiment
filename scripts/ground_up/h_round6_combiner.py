"""Round 6 — exploit H1/H37 complementarity via |τ|-based self-selection.

Discovery from round 5: H1 (NLMS direct) wins -x positions, H37 (NLMS+diff/sum+
multi-band median+reject-zero) wins +x positions. They never both lock to
zero on the same position. So: pick whichever produces larger |τ|.

Physics rationale: lock-to-zero produces |τ| ≈ 0; correct direct-path detection
produces |τ| ≥ 0.2 ms for non-broadside sources. Picking max(|τ_H1|, |τ_H37|)
is a physically-motivated, ground-truth-free decision rule.

Variants:
  H38a   max-|τ| rule
  H38b   PSR-weighted choice (require PSR > thresh)
  H38c   physical-range gating (require 0.05 < |τ| < 1.55 ms)
  H38d   stack of three estimators (H1, H37, mic-only D5) → max-|τ|
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


def gcc(x_l, x_r, sr, band, max_lag_s=PHYS_MAX, return_psr=False):
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
    pk = int(np.argmax(R))
    tau = float(lags[pk])
    if 1 <= pk < len(R) - 1:
        ym, y0, yp = R[pk - 1], R[pk], R[pk + 1]
        d = ym - 2 * y0 + yp
        if abs(d) > 1e-12:
            tau += 0.5 * (ym - yp) / d * (lags[1] - lags[0])
    if return_psr:
        off_mask = np.ones(len(R), dtype=bool)
        off_mask[max(0, pk - 3): min(len(R), pk + 4)] = False
        psr = R[pk] / (np.median(R[off_mask]) + 1e-12) if off_mask.any() else 0
        return tau, psr
    return tau


def diff_sum_gcc(x_l, x_r, sr, band, return_psr=False):
    return gcc(x_l - x_r, x_l + x_r, sr, band, return_psr=return_psr)


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
# Component estimators
# ============================================================================

def est_h1(chans, sr, band=(300, 4000)):
    """H1 NLMS subtract → direct mic-mic GCC (no diff/sum)."""
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    e_L = nlms_subtract(chans_b["ldv"], chans_b["mic_l"])
    e_R = nlms_subtract(chans_b["ldv"], chans_b["mic_r"])
    tau, psr = gcc(e_L, e_R, sr, band, return_psr=True)
    return tau, psr


def est_h37(chans, sr, bands=None):
    """H37 NLMS → diff/sum × multi-band median + reject zero."""
    if bands is None:
        bands = [(500, 1500), (1000, 2500), (1500, 3500), (2500, 5000),
                 (3500, 7000), (1000, 5000), (300, 4000)]
    chans_full = {ch: bp(x, sr, 200, 8000) for ch, x in chans.items()}
    e_L = nlms_subtract(chans_full["ldv"], chans_full["mic_l"])
    e_R = nlms_subtract(chans_full["ldv"], chans_full["mic_r"])
    taus = []
    psrs = []
    for band in bands:
        e_L_b = bp(e_L, sr, *band)
        e_R_b = bp(e_R, sr, *band)
        tau, psr = diff_sum_gcc(e_L_b, e_R_b, sr, band, return_psr=True)
        taus.append(tau); psrs.append(psr)
    taus = np.array(taus)
    psrs = np.array(psrs)
    nz = np.abs(taus) > 1e-4
    if nz.sum() >= 2:
        tau_combined = float(np.median(taus[nz]))
        psr_combined = float(np.median(psrs[nz]))
    else:
        tau_combined = float(np.median(taus))
        psr_combined = float(np.median(psrs))
    return tau_combined, psr_combined


def est_d5(chans, sr, band=(1000, 5000)):
    """D5 mic-only direct, no LDV (good for chirp -x)."""
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    tau, psr = gcc(chans_b["mic_l"], chans_b["mic_r"], sr, band, return_psr=True)
    return tau, psr


# ============================================================================
# Combiners
# ============================================================================

def h38a_max_abs_tau(chans, sr, sig_type):
    """Pick estimator with largest |τ|."""
    tau1, _ = est_h1(chans, sr)
    tau37, _ = est_h37(chans, sr)
    if abs(tau37) > abs(tau1):
        return doa_from_tau(tau37)
    return doa_from_tau(tau1)


def h38b_psr_weighted(chans, sr, sig_type, psr_thresh=4.0):
    """Pick by PSR, but only if PSR exceeds threshold; else max-|τ|."""
    tau1, psr1 = est_h1(chans, sr)
    tau37, psr37 = est_h37(chans, sr)
    if psr1 >= psr_thresh and psr37 >= psr_thresh:
        return doa_from_tau(tau37 if abs(tau37) > abs(tau1) else tau1)
    if psr1 >= psr_thresh:
        return doa_from_tau(tau1)
    if psr37 >= psr_thresh:
        return doa_from_tau(tau37)
    # Both low confidence — pick max |τ|
    return doa_from_tau(tau37 if abs(tau37) > abs(tau1) else tau1)


def h38c_physical_gating(chans, sr, sig_type):
    """Reject |τ|<0.05ms (lock-to-zero) AND |τ|>1.55ms (multipath)."""
    tau1, _ = est_h1(chans, sr)
    tau37, _ = est_h37(chans, sr)
    valid1 = 0.05e-3 < abs(tau1) < PHYS_MAX
    valid37 = 0.05e-3 < abs(tau37) < PHYS_MAX
    if valid1 and valid37:
        return doa_from_tau(tau37 if abs(tau37) > abs(tau1) else tau1)
    if valid1:
        return doa_from_tau(tau1)
    if valid37:
        return doa_from_tau(tau37)
    # Both invalid: take max-|τ| from the two
    return doa_from_tau(tau37 if abs(tau37) > abs(tau1) else tau1)


def h38d_three_way(chans, sr, sig_type):
    """3 estimators (H1, H37, D5 mic-only); pick max |τ|."""
    tau1, _ = est_h1(chans, sr)
    tau37, _ = est_h37(chans, sr)
    taud5, _ = est_d5(chans, sr)
    candidates = [(abs(tau1), tau1), (abs(tau37), tau37), (abs(taud5), taud5)]
    candidates.sort(reverse=True)
    return doa_from_tau(candidates[0][1])


def h38e_majority_within_phys(chans, sr, sig_type):
    """Run H1, H37, D5; if any 2 agree within 0.3ms, take their mean.
    Otherwise max-|τ|."""
    tau1, _ = est_h1(chans, sr)
    tau37, _ = est_h37(chans, sr)
    taud5, _ = est_d5(chans, sr)
    taus = [tau1, tau37, taud5]
    tol = 0.3e-3
    # Any pair within tolerance?
    pairs = [(0, 1), (0, 2), (1, 2)]
    best_pair = None; best_diff = float("inf")
    for i, j in pairs:
        if abs(taus[i] - taus[j]) <= tol:
            if abs(taus[i] - taus[j]) < best_diff:
                best_diff = abs(taus[i] - taus[j])
                best_pair = (i, j)
    if best_pair:
        i, j = best_pair
        # Reject if both are near zero AND truth could be non-zero
        avg = (taus[i] + taus[j]) / 2
        if abs(avg) < 1e-4:
            # Pair-near-zero — fall back to max non-zero
            taus_arr = np.array(taus)
            nz = np.abs(taus_arr) > 1e-4
            if nz.any():
                return doa_from_tau(taus_arr[nz][np.argmax(np.abs(taus_arr[nz]))])
        return doa_from_tau(avg)
    # No agreement — max |τ|
    return doa_from_tau(max(taus, key=abs))


STRATS = [
    ("H38a_max_abs_tau", h38a_max_abs_tau),
    ("H38b_psr_weighted", h38b_psr_weighted),
    ("H38c_physical_gating", h38c_physical_gating),
    ("H38d_three_way_max_abs", h38d_three_way),
    ("H38e_majority_within_phys", h38e_majority_within_phys),
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

    print(f"\n{'strategy':<32} | {'chirp':>10} | {'speech':>10}")
    print("-" * 60)
    rows_sorted = sorted(summary.items(), key=lambda kv: kv[1]["speech_mae"] or 999)
    for sid, s in rows_sorted:
        c = s["chirp_mae"]; sp_ = s["speech_mae"]
        c_str = f"{c:6.2f}°" if c is not None else "  N/A "
        s_str = f"{sp_:6.2f}°" if sp_ is not None else "  N/A "
        print(f"{sid:<32} | {c_str:>10} | {s_str:>10}")

    print("\nPer-position breakdown of all:")
    for sid, _ in rows_sorted:
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

    out = OUT_DIR / "H_round6_combiner.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
