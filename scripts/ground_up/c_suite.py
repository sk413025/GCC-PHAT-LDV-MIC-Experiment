"""Comprehensive strategy suite — runs many preprocessing/algorithm variants
and produces a single comparison table.

Strategies tested:
  S0  baseline      : DC+notch only
  S1  bp_500_2000   : paper default band
  S1b bp_300_1500   : low-band override
  S1c bp_1000_5000  : high-band override
  S2  specsub       : Wiener spectral subtraction (noise from low-energy frames)
  S5  scot          : SCOT weighting instead of PHAT (uses |X|^2 instead of |X*Y|)
  S5b roth          : Roth weighting (|X_L|^2 only)
  S6  tf_mask       : TF mask gated by LDV-mic coherence > threshold
  S7  multiwindow   : split into 0.5s windows, average |GCC|
  S10 ldv_gated_mm  : use LDV envelope as VAD for mic_L, mic_R, then mic-mic GCC
  S11 template_xc   : chirp only — cross-correlate with synthetic chirp template

For each strategy, runs on chirp + speech (block only) and outputs MAE.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from numpy.fft import rfft, irfft
from scipy.signal import butter, filtfilt, stft, istft, iirnotch
from scipy import signal as sp

from _loader import chirp_groups, speech_groups, load_group
from _pigs import (cross_phat, estimate_doa_pigs, estimate_doa_micmic,
                   expected_tau_VM, MIC_L, MIC_R, LDV, C_MPS, MIC_SPACING,
                   interp_R, preprocess as basic_preprocess, build_grid)
from _pigs2 import auto_window_chirp, estimate_doa_pigs_1d
from _geometry import expected_doa_deg, REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"


# ============================================================================
# Helpers
# ============================================================================

def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def apply_basic(chans, sr):
    return {ch: basic_preprocess(x, sr) for ch, x in chans.items()}


def get_window(signal_type, chans, sr):
    if signal_type == "chirp":
        return auto_window_chirp(chans["mic_l"], sr, dur_s=1.6)
    return 5.0, 25.0


def slice_chans(chans, sr, t0, t1):
    n0, n1 = int(t0 * sr), int(t1 * sr)
    return {ch: x[n0:min(n1, len(x))] for ch, x in chans.items()}


def cross_weighted(x_v, x_m, sr, weight_kind="phat", max_lag_s=0.007, band_hz=None):
    """Variants of cross-spectrum weighting.

    weight_kind:
      "phat"  : 1 / |X_M X_V*|     (paper)
      "scot"  : 1 / sqrt(|X_V|^2 |X_M|^2)
      "roth"  : 1 / |X_V|^2  (Roth, normalize by reference auto-power)
      "ml"    : MS coherence-weighted PHAT (γ² / |X_M X_V*| · (1-γ²) suppression)
    """
    n = max(len(x_v), len(x_m))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xv = rfft(x_v, n_fft); Xm = rfft(x_m, n_fft)
    G = Xm * np.conj(Xv)
    eps = 1e-12
    if weight_kind == "phat":
        W = 1.0 / (np.abs(G) + eps)
    elif weight_kind == "scot":
        W = 1.0 / (np.sqrt(np.abs(Xv) ** 2 * np.abs(Xm) ** 2) + eps)
    elif weight_kind == "roth":
        W = 1.0 / (np.abs(Xv) ** 2 + eps)
    elif weight_kind == "ml":
        # Approximate ML: weight = γ²/(1-γ²) · 1/|G|
        # Coherence requires Welch averaging — use STFT averaging over time
        f, t, Sv = sp.spectrogram(x_v, fs=sr, nperseg=2048, noverlap=1024)
        f, t, Sm = sp.spectrogram(x_m, fs=sr, nperseg=2048, noverlap=1024)
        # Coherence proxy via averaged cross/auto
        f, t, Svm = sp.spectrogram(x_v + 1j * 0, fs=sr, nperseg=2048, noverlap=1024,
                                  mode="complex")
        # Simpler: use mscoherence routine
        f_c, c = sp.coherence(x_v, x_m, fs=sr, nperseg=2048, noverlap=1024)
        c_interp = np.interp(np.fft.rfftfreq(n_fft, 1 / sr), f_c, c)
        gamma2 = np.clip(c_interp, 1e-3, 0.999)
        W = (gamma2 / (1.0 - gamma2)) / (np.abs(G) + eps)
    else:
        raise ValueError(weight_kind)
    if band_hz:
        f = np.fft.rfftfreq(n_fft, 1 / sr)
        mask = (f >= band_hz[0]) & (f <= band_hz[1])
        G = G * mask; W = W * mask
    r = np.fft.fftshift(irfft(G * W, n_fft))
    max_n = int(np.ceil(max_lag_s * sr))
    mid = len(r) // 2
    R = r[mid - max_n: mid + max_n + 1]
    lags = (np.arange(len(R)) - max_n) / sr
    return lags, R


def pigs_with_R_pair(R_VL, lags_VL, R_VR, lags_VR, score="sum",
                     x_lo=-1.5, x_hi=1.5, x_step=0.005):
    xs = np.arange(x_lo, x_hi + 1e-9, x_step)
    pts = np.stack([xs, np.zeros_like(xs)], axis=1)
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
    xi = int(np.argmax(S))
    p_hat = (float(xs[xi]), 0.0)
    tau_lr = (np.hypot(p_hat[0] - MIC_R[0], -MIC_R[1])
              - np.hypot(p_hat[0] - MIC_L[0], -MIC_L[1])) / C_MPS
    s = max(-1.0, min(1.0, C_MPS * tau_lr / MIC_SPACING))
    theta_hat = -float(np.degrees(np.arcsin(s)))
    return theta_hat, p_hat, S, xs


# ============================================================================
# Strategy implementations
# ============================================================================

def strat_baseline(chans, sr, signal_type, band_hz=None):
    out = estimate_doa_pigs(chans["mic_l"], chans["mic_r"], chans["ldv"], sr,
                           band_hz=band_hz, max_lag_s=0.007,
                           grid_kwargs=dict(x_lo=-1.5, x_hi=1.5, x_step=0.005,
                                            y_lo=0.0, y_hi=0.0, y_step=1.0))
    return out["theta_deg"]


def strat_bandpass(chans, sr, signal_type, band_hz):
    chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    return strat_baseline(chans, sr, signal_type, band_hz=band_hz)


def strat_specsub(chans, sr, signal_type, band_hz=(500, 2000)):
    """Spectral-subtraction style noise floor reduction."""
    out_chans = {}
    for ch, x in chans.items():
        # Estimate noise from lowest-energy 10% frames
        n_seg = 2048
        f, t, Z = sp.stft(x, fs=sr, nperseg=n_seg, noverlap=n_seg * 3 // 4,
                         window="hann")
        mag = np.abs(Z)
        # Frames with bottom 20% total power
        frame_p = (mag ** 2).sum(axis=0)
        thresh = np.quantile(frame_p, 0.2)
        noise_mask = frame_p <= thresh
        if noise_mask.sum() < 5:
            noise_mask = frame_p <= np.quantile(frame_p, 0.4)
        noise_psd = np.mean(mag[:, noise_mask] ** 2, axis=1) if noise_mask.any() else np.zeros(mag.shape[0])
        # Wiener-style attenuation
        sub = (mag ** 2 - 2 * noise_psd[:, None]).clip(min=mag ** 2 * 0.01)
        gain = np.sqrt(sub) / (mag + 1e-12)
        Z_clean = Z * gain
        _, x_clean = sp.istft(Z_clean, fs=sr, nperseg=n_seg, noverlap=n_seg * 3 // 4,
                             window="hann")
        out_chans[ch] = x_clean[:len(x)]
    out_chans = {ch: bp(x, sr, *band_hz) for ch, x in out_chans.items()}
    return strat_baseline(out_chans, sr, signal_type, band_hz=band_hz)


def strat_weighted(chans, sr, signal_type, weight_kind, band_hz):
    chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    lags_VL, R_VL = cross_weighted(chans["ldv"], chans["mic_l"], sr,
                                  weight_kind=weight_kind, band_hz=band_hz)
    lags_VR, R_VR = cross_weighted(chans["ldv"], chans["mic_r"], sr,
                                  weight_kind=weight_kind, band_hz=band_hz)
    theta, _, _, _ = pigs_with_R_pair(R_VL, lags_VL, R_VR, lags_VR, score="sum")
    return theta


def strat_tf_mask(chans, sr, signal_type, band_hz=(500, 2000), coh_thresh=0.3):
    """Mask out TF cells where LDV-Mic coherence is low."""
    n_seg = 1024; n_ovl = 768
    # Compute coherence per-frame between LDV and each mic
    f_axis = np.fft.rfftfreq(n_seg, 1 / sr)
    band_mask = (f_axis >= band_hz[0]) & (f_axis <= band_hz[1])
    masked_chans = {}
    fa, ta, Z_v = sp.stft(chans["ldv"], fs=sr, nperseg=n_seg, noverlap=n_ovl)
    out = {"ldv": chans["ldv"]}
    for mic_ch in ("mic_l", "mic_r"):
        _, _, Z_m = sp.stft(chans[mic_ch], fs=sr, nperseg=n_seg, noverlap=n_ovl)
        # Per-frame coherence proxy: cosine similarity of magnitudes in band
        Zv_b = np.abs(Z_v) * band_mask[:, None]
        Zm_b = np.abs(Z_m) * band_mask[:, None]
        # Energy correlation per frame
        num = (Zv_b * Zm_b).sum(axis=0)
        den = np.sqrt((Zv_b ** 2).sum(axis=0) * (Zm_b ** 2).sum(axis=0)) + 1e-12
        frame_coh = num / den
        keep_frames = frame_coh > coh_thresh
        # Apply mask
        Z_m_mask = Z_m * keep_frames[None, :]
        _, x_clean = sp.istft(Z_m_mask, fs=sr, nperseg=n_seg, noverlap=n_ovl)
        out[mic_ch] = x_clean[:len(chans[mic_ch])]
    out = {ch: bp(x, sr, *band_hz) for ch, x in out.items()}
    return strat_baseline(out, sr, signal_type, band_hz=band_hz)


def strat_multiwindow(chans, sr, signal_type, band_hz=(500, 2000), win_s=0.5,
                     hop_s=0.25):
    chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    n_win = int(win_s * sr); n_hop = int(hop_s * sr)
    R_VL_acc = None; R_VR_acc = None
    lags_VL = lags_VR = None
    n_count = 0
    for start in range(0, len(chans["mic_l"]) - n_win, n_hop):
        sl = slice(start, start + n_win)
        lags_VL, R_VL = cross_phat(chans["ldv"][sl], chans["mic_l"][sl], sr,
                                  max_lag_s=0.007, band_hz=band_hz)
        lags_VR, R_VR = cross_phat(chans["ldv"][sl], chans["mic_r"][sl], sr,
                                  max_lag_s=0.007, band_hz=band_hz)
        if R_VL_acc is None:
            R_VL_acc = np.abs(R_VL); R_VR_acc = np.abs(R_VR)
        else:
            R_VL_acc += np.abs(R_VL); R_VR_acc += np.abs(R_VR)
        n_count += 1
    R_VL_acc /= n_count; R_VR_acc /= n_count
    theta, _, _, _ = pigs_with_R_pair(R_VL_acc, lags_VL, R_VR_acc, lags_VR, score="sum")
    return theta


def strat_ldv_gated_micmic(chans, sr, signal_type, band_hz=(500, 2000),
                          gate_thresh=0.3):
    """Use LDV envelope as VAD to gate mic-mic GCC."""
    chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    # LDV envelope
    ldv_env = np.abs(chans["ldv"])
    n_smooth = int(0.02 * sr)
    ldv_env = np.convolve(ldv_env, np.ones(n_smooth) / n_smooth, mode="same")
    thresh = gate_thresh * ldv_env.max()
    gate = (ldv_env > thresh).astype(np.float64)
    # Smooth the gate
    n_fade = int(0.005 * sr)
    gate = np.convolve(gate, np.ones(n_fade) / n_fade, mode="same")
    x_l_g = chans["mic_l"] * gate
    x_r_g = chans["mic_r"] * gate
    theta, _, _, _ = estimate_doa_micmic(x_l_g, x_r_g, sr, band_hz=band_hz, max_lag_s=0.005)
    return theta


def strat_template_xc(chans, sr, signal_type, band_hz=(500, 6000)):
    """Chirp only: correlate mic_L and mic_R with synthetic chirp template,
    measure absolute arrival times, derive TDoA."""
    if signal_type != "chirp":
        return None
    # Synthesize template: down-sweep 6kHz->500Hz over 1.5s
    dur = 1.5
    t = np.arange(int(sr * dur)) / sr
    f0, f1 = 6000.0, 500.0
    template = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t ** 2))
    # Bandpass to match
    template = bp(template, sr, *band_hz)
    chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}
    # Cross-correlate
    def xc_peak(x):
        c = np.correlate(x, template, mode="full")
        return int(np.argmax(np.abs(c))) - (len(template) - 1)
    n_l = xc_peak(chans["mic_l"])
    n_r = xc_peak(chans["mic_r"])
    tau_lr = (n_r - n_l) / sr  # seconds
    s = max(-1.0, min(1.0, C_MPS * tau_lr / MIC_SPACING))
    theta = -float(np.degrees(np.arcsin(s)))
    return theta


# ============================================================================
# Driver
# ============================================================================

STRATEGIES = [
    ("S0_baseline", lambda c, sr, st: strat_baseline(c, sr, st)),
    ("S1a_bp500_2000", lambda c, sr, st: strat_bandpass(c, sr, st, (500, 2000))),
    ("S1b_bp300_1500", lambda c, sr, st: strat_bandpass(c, sr, st, (300, 1500))),
    ("S1c_bp1000_5000", lambda c, sr, st: strat_bandpass(c, sr, st, (1000, 5000))),
    ("S2_specsub_500_2000", lambda c, sr, st: strat_specsub(c, sr, st, (500, 2000))),
    ("S5_scot_500_2000", lambda c, sr, st: strat_weighted(c, sr, st, "scot", (500, 2000))),
    ("S5b_roth_500_2000", lambda c, sr, st: strat_weighted(c, sr, st, "roth", (500, 2000))),
    ("S5c_ml_500_2000", lambda c, sr, st: strat_weighted(c, sr, st, "ml", (500, 2000))),
    ("S6_tfmask_500_2000", lambda c, sr, st: strat_tf_mask(c, sr, st, (500, 2000))),
    ("S7_multiwin_500_2000", lambda c, sr, st: strat_multiwindow(c, sr, st, (500, 2000))),
    ("S10_ldv_gated_500_2000", lambda c, sr, st: strat_ldv_gated_micmic(c, sr, st, (500, 2000))),
    ("S11_template_xc", lambda c, sr, st: strat_template_xc(c, sr, st)),
]


def run_all():
    results = {}  # results[strategy_id][signal_type] = {pos: err, ..., MAE: ...}
    for sig_type in ("chirp", "speech"):
        groups = chirp_groups() if sig_type == "chirp" else speech_groups()
        for (pos, cond), paths in sorted(groups.items()):
            if cond != "block":
                continue
            chans, sr = load_group(paths)
            t0, t1 = get_window(sig_type, chans, sr)
            chans = slice_chans(chans, sr, t0, t1)
            chans = apply_basic(chans, sr)
            theta_true = expected_doa_deg(float(pos))
            for sid, fn in STRATEGIES:
                try:
                    theta = fn(chans, sr, sig_type)
                except Exception as e:
                    theta = None
                err = abs(theta - theta_true) if theta is not None else None
                key = sid
                results.setdefault(key, {}).setdefault(sig_type, {})[pos] = {
                    "theta_true": theta_true,
                    "theta_est": theta,
                    "err": err,
                }
    # Aggregate MAE
    summary = {}
    for sid, by_sig in results.items():
        summary[sid] = {}
        for sig_type, by_pos in by_sig.items():
            errs = [v["err"] for v in by_pos.values() if v["err"] is not None]
            summary[sid][sig_type] = {
                "mae": float(np.mean(errs)) if errs else None,
                "per_pos": by_pos,
            }
    return summary


def print_summary(summary):
    print(f"\n{'strategy':<28} | {'chirp MAE':>10} | {'speech MAE':>10}")
    print("-" * 56)
    rows = []
    for sid in summary:
        c_mae = summary[sid].get("chirp", {}).get("mae")
        s_mae = summary[sid].get("speech", {}).get("mae")
        rows.append((sid, c_mae, s_mae))
    # Sort by speech MAE (primary metric per plan)
    rows.sort(key=lambda r: (float("inf") if r[2] is None else r[2]))
    for sid, c, s in rows:
        c_str = f"{c:6.2f}°" if c is not None else "  N/A "
        s_str = f"{s:6.2f}°" if s is not None else "  N/A "
        print(f"{sid:<28} | {c_str:>10} | {s_str:>10}")


def main():
    summary = run_all()
    out_path = OUT_DIR / "C_suite_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print_summary(summary)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
