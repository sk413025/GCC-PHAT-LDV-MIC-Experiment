"""Loop 5 — multi-band consensus self-selection.

Hypothesis H10: The data has per-position usable bands, but there's no a-priori
way to pick. Solution: run a BANK of bands, score each by:
  (a) PSR (peak-to-sidelobe ratio) — local quality
  (b) τ-consistency across multiple bands — global quality
  (c) Sub-bands within main band agree

Self-selection algorithm:
  1. Run 10+ candidate (band × processing) pairs.
  2. For each, compute (τ_lr, psr).
  3. Cluster τ values; pick the cluster with highest mass × mean PSR.
  4. Output τ = mean of winning cluster.

This avoids "cheating" because we never look at ground truth — we just pick the
estimate that's most consistent across our preprocessing variants.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
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


def stft_wiener(x_v, x_m, sr, nperseg=2048, n_ovl=1536, ridge=1e-6):
    f, t, Zv = sp.stft(x_v, fs=sr, nperseg=nperseg, noverlap=n_ovl, window="hann")
    _, _, Zm = sp.stft(x_m, fs=sr, nperseg=nperseg, noverlap=n_ovl, window="hann")
    Pvv = np.mean(np.abs(Zv) ** 2, axis=1)
    Pvm = np.mean(Zm * np.conj(Zv), axis=1)
    H = Pvm / (Pvv + ridge * Pvv.max())
    Zm_resid = Zm - H[:, None] * Zv
    _, x_resid = sp.istft(Zm_resid, fs=sr, nperseg=nperseg, noverlap=n_ovl, window="hann")
    return x_resid[:len(x_m)]


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


def gcc_with_psr(x_l, x_r, sr, band, max_lag_s=PHYS_MAX):
    n = max(len(x_l), len(x_r))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xl = np.fft.rfft(x_l, n_fft); Xr = np.fft.rfft(x_r, n_fft)
    G = Xr * np.conj(Xl)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    G = G * ((f >= band[0]) & (f <= band[1]))
    eps = 1e-12
    r = np.fft.fftshift(np.fft.irfft(G / (np.abs(G) + eps), n_fft))
    max_n = int(np.ceil(max_lag_s * sr)); mid = len(r) // 2
    R = np.abs(r[mid - max_n: mid + max_n + 1])
    lags = (np.arange(len(R)) - max_n) / sr
    pk = int(np.argmax(R))
    tau = float(lags[pk])
    if 1 <= pk < len(R) - 1:
        ym, y0, yp = R[pk - 1], R[pk], R[pk + 1]
        d = ym - 2 * y0 + yp
        if abs(d) > 1e-12:
            tau += 0.5 * (ym - yp) / d / sr
    off = np.ones(len(R), dtype=bool)
    off[max(0, pk - 3):min(len(R), pk + 4)] = False
    psr = R[pk] / (np.median(R[off]) + 1e-12) if off.any() else 0.0
    return tau, psr


# Bank of (proc_name, band, subtraction_kind)
BANK = [
    # No subtraction, various bands
    ("raw", (300, 1500), None),
    ("raw", (500, 2000), None),
    ("raw", (700, 3000), None),
    ("raw", (1000, 5000), None),
    ("raw", (1500, 6000), None),
    ("raw", (2000, 8000), None),
    ("raw", (300, 4000), None),
    # NLMS subtraction
    ("nlms", (300, 1500), "nlms"),
    ("nlms", (500, 2000), "nlms"),
    ("nlms", (1000, 5000), "nlms"),
    ("nlms", (300, 4000), "nlms"),
    ("nlms", (500, 5000), "nlms"),
    # Wiener
    ("wiener", (500, 2000), "wiener"),
    ("wiener", (1000, 5000), "wiener"),
    ("wiener", (300, 4000), "wiener"),
]


def evaluate_bank(chans, sr):
    """Run all bank entries, return list of (tau, psr, label)."""
    results = []
    for name, band, sub in BANK:
        chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
        if sub == "nlms":
            x_l = nlms(chans_b["ldv"], chans_b["mic_l"])
            x_r = nlms(chans_b["ldv"], chans_b["mic_r"])
        elif sub == "wiener":
            x_l = stft_wiener(chans_b["ldv"], chans_b["mic_l"], sr)
            x_r = stft_wiener(chans_b["ldv"], chans_b["mic_r"], sr)
        else:
            x_l = chans_b["mic_l"]; x_r = chans_b["mic_r"]
        tau, psr = gcc_with_psr(x_l, x_r, sr, band)
        label = f"{name}_{band[0]}_{band[1]}"
        results.append({"label": label, "tau": tau, "psr": psr, "band": band})
    return results


def consensus_select(results, dbscan_eps=0.0003, min_cluster=2):
    """Cluster τ values, pick cluster with max sum(psr).
    dbscan_eps in seconds (0.3ms = 1 sample at 3kHz).
    """
    if not results:
        return None, None
    # Sort by tau
    rs = sorted(results, key=lambda r: r["tau"])
    clusters = []
    cur = [rs[0]]
    for r in rs[1:]:
        if abs(r["tau"] - cur[-1]["tau"]) <= dbscan_eps:
            cur.append(r)
        else:
            clusters.append(cur)
            cur = [r]
    clusters.append(cur)
    # Score each cluster: sum(psr) × n_members, weighted by mean PSR
    best_cluster = max(clusters,
                      key=lambda c: sum(r["psr"] for r in c) if len(c) >= min_cluster else 0)
    if len(best_cluster) < min_cluster:
        # Fall back to highest single PSR
        best = max(results, key=lambda r: r["psr"])
        return best["tau"], [best]
    # Weighted average of cluster
    weights = np.array([r["psr"] for r in best_cluster])
    taus = np.array([r["tau"] for r in best_cluster])
    tau_w = float(np.sum(weights * taus) / np.sum(weights))
    return tau_w, best_cluster


def doa_from_tau(tau):
    s = max(-1.0, min(1.0, C_MPS * tau / MIC_SPACING))
    return -float(np.degrees(np.arcsin(s)))


def main():
    rows_out = {}
    for sig_type in ("chirp", "speech"):
        groups = chirp_groups() if sig_type == "chirp" else speech_groups()
        rows_out[sig_type] = []
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

            res = evaluate_bank(chans, sr)
            tau_consensus, cluster = consensus_select(res)
            theta_est = doa_from_tau(tau_consensus) if tau_consensus is not None else None
            err = abs(theta_est - theta_true) if theta_est is not None else None
            rows_out[sig_type].append({
                "pos": pos, "true": theta_true, "est": theta_est, "err": err,
                "tau_ms": tau_consensus * 1000 if tau_consensus is not None else None,
                "cluster_size": len(cluster) if cluster else 0,
                "cluster_taus": [c["tau"] * 1000 for c in cluster] if cluster else [],
                "cluster_psrs": [c["psr"] for c in cluster] if cluster else [],
                "cluster_labels": [c["label"] for c in cluster] if cluster else [],
            })

    print(f"\n{'sig':>6} {'pos':>5} | {'tau (ms)':>10} {'true':>7} {'est':>7} {'err':>7} | n_cluster | top labels")
    print("-" * 100)
    for sig in ("chirp", "speech"):
        for r in rows_out[sig]:
            tau_s = f"{r['tau_ms']:+7.3f}" if r['tau_ms'] is not None else "  N/A "
            est_s = f"{r['est']:+6.2f}°" if r['est'] is not None else "  N/A"
            err_s = f"{r['err']:5.2f}°" if r['err'] is not None else " N/A "
            top_labels = r['cluster_labels'][:3] if r['cluster_labels'] else []
            print(f"{sig:>6} {r['pos']:>5} | {tau_s:>10} {r['true']:+6.2f}° "
                  f"{est_s:>7} {err_s:>7} | {r['cluster_size']:^9} | {','.join(top_labels)}")
    for sig in ("chirp", "speech"):
        errs = [r["err"] for r in rows_out[sig] if r["err"] is not None]
        mae = float(np.mean(errs)) if errs else None
        print(f"\n  {sig} consensus MAE: {mae:.2f}°")

    out = OUT_DIR / "G_loop5_consensus.json"
    out.write_text(json.dumps(rows_out, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
