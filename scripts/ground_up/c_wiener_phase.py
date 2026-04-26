"""C-Wiener — use LDV as a coherent reference to estimate mic transfer functions
H_L(f), H_R(f), then derive mic-mic phase delay τ_LR from arg(H_L · conj(H_R)).

Physics: with source s(t), x_v = h_v * s + n_v ; x_L = h_L * s + n_L ; ...
  H_VL(f) := E[X_L · X_V*]/E[|X_V|²] = h_L(f)·h_v*(f)/|h_v(f)|² (LDV-coherent
    transfer from LDV to mic_L; phase is arg h_L - arg h_v)
  Similarly H_VR(f).
  H_VL/H_VR has phase = arg h_L - arg h_R, which is the mic-mic transfer phase
    purely from the source-driven component, jammer-free.
  Linear phase fit: τ_LR = -dφ/dω.
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
from _pigs import (cross_phat, MIC_L, MIC_R, C_MPS, MIC_SPACING,
                   preprocess as basic_preprocess)
from _pigs2 import auto_window_chirp
from _geometry import expected_doa_deg, REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"


def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def estimate_H(x_v, x_m, sr, nperseg=2048, n_ovl=1536):
    """Welch-style estimate H(f) = S_VM(f) / S_VV(f), and coherence γ²(f)."""
    f, Pvv = sp.welch(x_v, fs=sr, nperseg=nperseg, noverlap=n_ovl)
    _, Pmm = sp.welch(x_m, fs=sr, nperseg=nperseg, noverlap=n_ovl)
    _, Pvm = sp.csd(x_v, x_m, fs=sr, nperseg=nperseg, noverlap=n_ovl)
    H = Pvm / (Pvv + 1e-20)
    coh2 = np.abs(Pvm) ** 2 / (Pvv * Pmm + 1e-20)
    return f, H, coh2


def fit_tau_from_HLHR(f, H_L, H_R, coh_L, coh_R, band_hz=(500, 4000),
                     coh_thresh=0.2, max_lag_ms=10):
    """Fit linear phase to phase(H_L · conj(H_R)) over the band, weighted by coherence."""
    H_LR = H_L * np.conj(H_R)
    phase = np.unwrap(np.angle(H_LR))
    weight = np.minimum(coh_L, coh_R) ** 2
    in_band = (f >= band_hz[0]) & (f <= band_hz[1]) & (weight > coh_thresh)
    if in_band.sum() < 20:
        return None, None
    # Try a small set of candidate τ values, pick the one minimizing wrapped phase deviation
    best_tau = None; best_score = -np.inf
    omega = 2 * np.pi * f[in_band]
    phi = phase[in_band]
    w = weight[in_band]
    # Fine search over candidate τ
    for tau in np.linspace(-max_lag_ms * 1e-3, max_lag_ms * 1e-3, 4001):
        residual = phi + omega * tau  # H_LR has phase = -ωτ_LR if τ_LR is mic_R relative to mic_L
        residual = np.angle(np.exp(1j * residual))  # wrap
        score = -np.sum(w * residual ** 2)
        if score > best_score:
            best_score = score
            best_tau = tau
    return best_tau, best_score


def doa_from_tau(tau_lr):
    s = max(-1.0, min(1.0, C_MPS * tau_lr / MIC_SPACING))
    return -float(np.degrees(np.arcsin(s)))


def run_one(chans, sr, signal_type, band_hz, max_lag_ms=10):
    chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    f, H_L, coh_L = estimate_H(chans["ldv"], chans["mic_l"], sr)
    _, H_R, coh_R = estimate_H(chans["ldv"], chans["mic_r"], sr)
    tau_lr, _ = fit_tau_from_HLHR(f, H_L, H_R, coh_L, coh_R,
                                  band_hz=band_hz, max_lag_ms=max_lag_ms)
    if tau_lr is None:
        return None
    return doa_from_tau(tau_lr)


def main():
    bands = [
        ("WP_500_4000", (500, 4000)),
        ("WP_300_2000", (300, 2000)),
        ("WP_700_3000", (700, 3000)),
        ("WP_300_5000", (300, 5000)),
    ]
    summary = {}
    print(f"{'strategy':<22} | {'chirp':>10} | {'speech':>10}")
    print("-" * 50)
    for sid, band in bands:
        c_errs = []; s_errs = []
        rows = {}
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
                theta_est = run_one(chans, sr, sig_type, band)
                theta_true = expected_doa_deg(float(pos))
                if theta_est is None:
                    continue
                err = abs(theta_est - theta_true)
                rows.setdefault(sig_type, {})[pos] = {
                    "true": theta_true, "est": theta_est, "err": err}
                if sig_type == "chirp":
                    c_errs.append(err)
                else:
                    s_errs.append(err)
        c_mae = float(np.mean(c_errs)) if c_errs else None
        s_mae = float(np.mean(s_errs)) if s_errs else None
        summary[sid] = {"band": band, "chirp_mae": c_mae, "speech_mae": s_mae,
                       "rows": rows}
        c_str = f"{c_mae:6.2f}°" if c_mae else "  N/A "
        s_str = f"{s_mae:6.2f}°" if s_mae else "  N/A "
        print(f"{sid:<22} | {c_str:>10} | {s_str:>10}")

    out = OUT_DIR / "C_wiener_phase.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out}")
    # Detailed per-position for best strategy
    best = min(summary, key=lambda k: summary[k]["speech_mae"] or 999)
    print(f"\nBest strategy: {best}")
    for sig_type in ("chirp", "speech"):
        if sig_type in summary[best]["rows"]:
            print(f"  {sig_type}:")
            for pos in sorted(summary[best]["rows"][sig_type]):
                r = summary[best]["rows"][sig_type][pos]
                print(f"    x={pos}: true={r['true']:+6.2f}°, "
                      f"est={r['est']:+6.2f}°, err={r['err']:5.2f}°")


if __name__ == "__main__":
    main()
