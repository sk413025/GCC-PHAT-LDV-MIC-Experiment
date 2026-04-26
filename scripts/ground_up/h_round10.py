"""Round 10 — last refinements + adaptive-band insight from oracle analysis.

V3 (round 9 H52): speech 3.57°. Per-position oracle is 1.59°. Key gaps:
  -0.4 speech: 5.37° (oracle 0.60° via plain bp 300-1500, no LDV)
  +0.8 speech: 3.68° (oracle 0.72° via H14 sym-diff with paired data)

H54  Adaptive-band mic-mic GCC: pick band where LDV-mic coh is high (both pairs),
     run plain mic-mic GCC at that band. Add to H38 pool.
H56  H52 with 4 candidates: H1, H37, low-band-bp, adaptive-band
H57  PSR-tiebreak when |τ| close
H58  Multi-tau fusion with weighted average by PSR among non-zero candidates
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
        off = np.ones(len(R), dtype=bool)
        off[max(0, pk - 3): min(len(R), pk + 4)] = False
        psr = R[pk] / (np.median(R[off]) + 1e-12) if off.any() else 0
        return tau, psr
    return tau


def diff_sum_gcc(x_l, x_r, sr, band, return_psr=False):
    return gcc(x_l - x_r, x_l + x_r, sr, band, return_psr=return_psr)


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


def find_consensus_band(chans, sr, threshold=0.25, search=(200, 8000),
                       min_width=200):
    """Find widest contiguous band where BOTH LDV-MicL and LDV-MicR have
    coherence > threshold. Returns (lo, hi) or None."""
    f, c_l = sp.coherence(chans["ldv"], chans["mic_l"], fs=sr, nperseg=2048,
                         noverlap=1024)
    _, c_r = sp.coherence(chans["ldv"], chans["mic_r"], fs=sr, nperseg=2048,
                         noverlap=1024)
    in_range = (f >= search[0]) & (f <= search[1])
    f, c_l, c_r = f[in_range], c_l[in_range], c_r[in_range]
    above = (c_l > threshold) & (c_r > threshold)
    runs = []
    start = None
    for i, a in enumerate(above):
        if a and start is None: start = i
        elif not a and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(above)))
    best = None; best_w = 0
    for s, e in runs:
        w = f[e - 1] - f[s]
        if w >= min_width and w > best_w:
            best_w = w; best = (float(f[s]), float(f[e - 1]))
    return best


# ============================================================================
# Estimators
# ============================================================================

def est_h1(chans, sr, band=(300, 4000)):
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    e_L = nlms(chans_b["ldv"], chans_b["mic_l"])
    e_R = nlms(chans_b["ldv"], chans_b["mic_r"])
    return gcc(e_L, e_R, sr, band, return_psr=True)


def est_h37(chans, sr):
    bands = [(500, 1500), (1000, 2500), (1500, 3500), (2500, 5000),
             (3500, 7000), (1000, 5000), (300, 4000)]
    chans_full = {ch: bp(x, sr, 200, 8000) for ch, x in chans.items()}
    e_L = nlms(chans_full["ldv"], chans_full["mic_l"])
    e_R = nlms(chans_full["ldv"], chans_full["mic_r"])
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


def est_low_band_bp(chans, sr, band=(300, 1500)):
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    return gcc(chans_b["mic_l"], chans_b["mic_r"], sr, band, return_psr=True)


def est_adaptive_band(chans, sr):
    """Find LDV-mic consensus coherence band, run mic-mic GCC there."""
    band = find_consensus_band(chans, sr, threshold=0.3)
    if band is None:
        band = find_consensus_band(chans, sr, threshold=0.2)
    if band is None:
        band = (500, 4000)
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    return gcc(chans_b["mic_l"], chans_b["mic_r"], sr, band, return_psr=True)


# ============================================================================
# Combiners
# ============================================================================

def h54_h38_with_adaptive(chans, sr, sig_type):
    """V2 H38a + adaptive-band candidate."""
    tau_h1, _ = est_h1(chans, sr)
    tau_h37, _ = est_h37(chans, sr)
    tau_ad, _ = est_adaptive_band(chans, sr)
    return doa_from_tau(max([tau_h1, tau_h37, tau_ad], key=abs))


def h56_four_candidate_smart(chans, sr, sig_type, agree_tol_ms=0.2):
    """4 candidates: H1, H37, low-band 300-1500, adaptive coh-band.
    Smart rule: same-sign pair within tol → average; else max-|τ|."""
    tau_h1, _ = est_h1(chans, sr)
    tau_h37, _ = est_h37(chans, sr)
    tau_lb, _ = est_low_band_bp(chans, sr, (300, 1500))
    tau_ad, _ = est_adaptive_band(chans, sr)
    cands = np.array([tau_h1, tau_h37, tau_lb, tau_ad])
    # Find pair with smallest distance
    n = len(cands)
    best_pair = None; best_diff = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(cands[i] - cands[j])
            if cands[i] * cands[j] > 0 and d < best_diff:
                best_diff = d; best_pair = (i, j)
    if best_pair and best_diff < agree_tol_ms * 1e-3:
        return doa_from_tau((cands[best_pair[0]] + cands[best_pair[1]]) / 2)
    return doa_from_tau(max(cands.tolist(), key=abs))


def h57_psr_tiebreak(chans, sr, sig_type, abs_tol_ms=0.1):
    """If |τ_H1| and |τ_H37| close, pick higher PSR; else max-|τ|."""
    tau_h1, psr_h1 = est_h1(chans, sr)
    tau_h37, psr_h37 = est_h37(chans, sr)
    if abs(abs(tau_h1) - abs(tau_h37)) < abs_tol_ms * 1e-3:
        return doa_from_tau(tau_h1 if psr_h1 > psr_h37 else tau_h37)
    return doa_from_tau(tau_h37 if abs(tau_h37) > abs(tau_h1) else tau_h1)


def h58_psr_weighted_fusion(chans, sr, sig_type):
    """Weighted average of all non-locked candidates, weighted by PSR."""
    cands = []
    tau_h1, psr_h1 = est_h1(chans, sr)
    if abs(tau_h1) > 8e-5:
        cands.append((tau_h1, psr_h1))
    tau_h37, psr_h37 = est_h37(chans, sr)
    if abs(tau_h37) > 8e-5:
        cands.append((tau_h37, psr_h37))
    tau_ad, psr_ad = est_adaptive_band(chans, sr)
    if abs(tau_ad) > 8e-5:
        cands.append((tau_ad, psr_ad))
    if not cands:
        # All locked — use max PSR
        all_cands = [(tau_h1, psr_h1), (tau_h37, psr_h37), (tau_ad, psr_ad)]
        return doa_from_tau(max(all_cands, key=lambda x: x[1])[0])
    # Same-sign filter (consensus on direction)
    pos = [c for c in cands if c[0] > 0]
    neg = [c for c in cands if c[0] < 0]
    if len(pos) > len(neg):
        cands = pos
    elif len(neg) > len(pos):
        cands = neg
    # Weighted mean
    weights = np.array([c[1] for c in cands])
    taus = np.array([c[0] for c in cands])
    return doa_from_tau(float(np.sum(weights * taus) / np.sum(weights)))


def h59_v3_plus_adaptive(chans, sr, sig_type, agree_tol_ms=0.2):
    """V3 H52 (agree-average) + adaptive band as third option."""
    tau_h1, _ = est_h1(chans, sr)
    tau_h37, _ = est_h37(chans, sr)
    tau_ad, _ = est_adaptive_band(chans, sr)
    # Prefer agreement among any 2
    pairs = [(tau_h1, tau_h37), (tau_h1, tau_ad), (tau_h37, tau_ad)]
    for a, b in pairs:
        if a * b > 0 and abs(a - b) < agree_tol_ms * 1e-3 and abs((a+b)/2) > 5e-5:
            return doa_from_tau((a + b) / 2)
    return doa_from_tau(max([tau_h1, tau_h37, tau_ad], key=abs))


STRATS = [
    ("H54_h38_adaptive_3way", h54_h38_with_adaptive),
    ("H56_four_smart", h56_four_candidate_smart),
    ("H56b_four_smart_tol0.3", lambda c, sr, s: h56_four_candidate_smart(c, sr, s, 0.3)),
    ("H57_psr_tiebreak", h57_psr_tiebreak),
    ("H58_psr_weighted", h58_psr_weighted_fusion),
    ("H59_v3_plus_adaptive", h59_v3_plus_adaptive),
    ("H59b_v3_plus_adaptive_tol0.3", lambda c, sr, s: h59_v3_plus_adaptive(c, sr, s, 0.3)),
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

    print("\nPer-position breakdown of top 3:")
    for sid, _ in rows_sorted[:3]:
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

    out = OUT_DIR / "H_round10.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
