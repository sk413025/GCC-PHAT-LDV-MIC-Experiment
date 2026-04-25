"""Round 4 — final physics ideas before declaring plateau.

H28  Instantaneous-frequency tracking (chirp) — at time t, only use narrowband
     around f(t). Wall ringing has frequencies from earlier source content;
     this cleanly separates direct path (current freq) from wall (past freqs).

H31  TF max-energy sparsity — most TF cells are noise/wall; only highest-energy
     10-30% cells contain source. Restrict GCC to those.

H33  GCC-like via differential mic (mic_L - mic_R direct subtraction):
     wall component is common (cancels), direct path differential preserved.
     Then envelope correlation against original to recover delay.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from numpy.fft import rfft, irfft
from scipy.signal import butter, filtfilt, hilbert
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


# ============================================================================
# H28 — Instantaneous frequency tracking (chirp)
# ============================================================================

def detect_chirp_params(x, sr):
    """Estimate chirp instantaneous frequency from analytic signal phase."""
    x_a = hilbert(x)
    inst_phase = np.unwrap(np.angle(x_a))
    inst_freq = np.gradient(inst_phase) * sr / (2 * np.pi)
    return inst_freq


def h28_inst_freq_tracking(chans, sr, sig_type, half_bw_hz=300):
    """For each time, narrow-bandpass around instantaneous frequency and
    compute mic-mic GCC. Then global GCC over all time.
    """
    if sig_type != "chirp":
        return None
    chans = {ch: bp(x, sr, 200, 8000) for ch, x in chans.items()}
    inst_freq = detect_chirp_params(chans["mic_l"], sr)
    # Smooth inst_freq
    n_smooth = int(0.005 * sr)
    inst_freq_s = np.convolve(inst_freq, np.ones(n_smooth) / n_smooth, mode="same")
    # For each time block, bandpass mic signals around inst freq, GCC
    n_block = int(0.02 * sr)  # 20ms blocks
    n_hop = int(0.01 * sr)
    R_acc = None; lags_ = None
    cnt = 0
    for s0 in range(0, len(chans["mic_l"]) - n_block, n_hop):
        f_center = abs(np.median(inst_freq_s[s0: s0 + n_block]))
        if f_center < 200 or f_center > 8000:
            continue
        lo = max(200, f_center - half_bw_hz)
        hi = min(sr / 2 - 100, f_center + half_bw_hz)
        if hi <= lo:
            continue
        # Bandpass
        sl = slice(s0, s0 + n_block)
        try:
            l_b = bp(chans["mic_l"][sl], sr, lo, hi)
            r_b = bp(chans["mic_r"][sl], sr, lo, hi)
        except Exception:
            continue
        R, lags_ = gcc_R(l_b, r_b, sr, (lo, hi))
        R_acc = R if R_acc is None else R_acc + R
        cnt += 1
    if R_acc is None or cnt < 5:
        return None
    R_acc /= cnt
    return doa_from_tau(pick_peak_subsample(R_acc, lags_))


# ============================================================================
# H31 — TF max-energy sparsity
# ============================================================================

def h31_tf_sparsity(chans, sr, sig_type, band, top_pct=0.15):
    """Keep only top X% TF cells by energy in mic_L+mic_R envelope."""
    chans = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    n_seg = 1024; n_ovl = 768
    f, t, Zl = sp.stft(chans["mic_l"], fs=sr, nperseg=n_seg, noverlap=n_ovl)
    _, _, Zr = sp.stft(chans["mic_r"], fs=sr, nperseg=n_seg, noverlap=n_ovl)
    band_mask = (f >= band[0]) & (f <= band[1])
    # Energy per cell (combined)
    energy = np.abs(Zl) ** 2 + np.abs(Zr) ** 2
    # Threshold: keep top top_pct of in-band cells
    in_band = band_mask[:, None] * np.ones_like(energy[0])
    in_band_2d = band_mask[:, None] | np.zeros_like(energy, dtype=bool)
    flat_e = energy[in_band_2d]
    if len(flat_e) == 0:
        return None
    thresh = np.quantile(flat_e, 1 - top_pct)
    keep_mask = (energy >= thresh) & in_band_2d
    Zl_m = Zl * keep_mask
    Zr_m = Zr * keep_mask
    _, x_l_clean = sp.istft(Zl_m, fs=sr, nperseg=n_seg, noverlap=n_ovl)
    _, x_r_clean = sp.istft(Zr_m, fs=sr, nperseg=n_seg, noverlap=n_ovl)
    n = min(len(x_l_clean), len(x_r_clean))
    R, lags = gcc_R(x_l_clean[:n], x_r_clean[:n], sr, band)
    return doa_from_tau(pick_peak_subsample(R, lags))


# ============================================================================
# H33 — Differential mic + envelope correlation
# ============================================================================

def h33_diff_mic(chans, sr, sig_type, band):
    """mic_L - mic_R cancels the wall (common-mode); preserves differential.
    Cross-correlate with mic_L (or mic_R) reference to find delay."""
    chans = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    diff = chans["mic_l"] - chans["mic_r"]
    summ = chans["mic_l"] + chans["mic_r"]
    # diff has direct path 2*differential; summ has wall + small direct
    # Cross-correlate diff against summ
    n = max(len(diff), len(summ))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    D = rfft(diff, n_fft)
    S = rfft(summ, n_fft)
    G = S * np.conj(D)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    G = G * ((f >= band[0]) & (f <= band[1]))
    eps = 1e-12
    r = np.fft.fftshift(irfft(G / (np.abs(G) + eps), n_fft))
    max_n = int(np.ceil(PHYS_MAX * sr))
    mid = len(r) // 2
    R = np.abs(r[mid - max_n: mid + max_n + 1])
    lags = (np.arange(len(R)) - max_n) / sr
    return doa_from_tau(pick_peak_subsample(R, lags))


# ============================================================================
# H34 — robust τ estimation: median of multiple band-frame products + LDV-NLMS
# ============================================================================

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


def h34_robust_combo(chans, sr, sig_type, bands=None):
    """LDV-NLMS subtract → multiple band τ estimates → trimmed mean."""
    if bands is None:
        bands = [(500, 1500), (800, 2500), (1500, 4000), (2500, 6000), (1000, 5000)]
    chans_full = {ch: bp(x, sr, 200, 8000) for ch, x in chans.items()}
    e_L = nlms_subtract(chans_full["ldv"], chans_full["mic_l"])
    e_R = nlms_subtract(chans_full["ldv"], chans_full["mic_r"])
    taus = []
    for band in bands:
        e_L_b = bp(e_L, sr, *band)
        e_R_b = bp(e_R, sr, *band)
        R, lags = gcc_R(e_L_b, e_R_b, sr, band)
        taus.append(pick_peak_subsample(R, lags))
    taus = np.array(taus)
    # Trim outermost values, take mean
    taus_sorted = np.sort(taus)
    trimmed = taus_sorted[1:-1] if len(taus) > 3 else taus
    return doa_from_tau(float(np.mean(trimmed)))


def h34b_robust_combo_no_zeros(chans, sr, sig_type, bands=None):
    """H34 but EXCLUDE estimates with |τ|<0.05ms before averaging."""
    if bands is None:
        bands = [(500, 1500), (800, 2500), (1500, 4000), (2500, 6000),
                 (1000, 5000), (300, 4000)]
    chans_full = {ch: bp(x, sr, 200, 8000) for ch, x in chans.items()}
    e_L = nlms_subtract(chans_full["ldv"], chans_full["mic_l"])
    e_R = nlms_subtract(chans_full["ldv"], chans_full["mic_r"])
    taus = []
    for band in bands:
        e_L_b = bp(e_L, sr, *band)
        e_R_b = bp(e_R, sr, *band)
        R, lags = gcc_R(e_L_b, e_R_b, sr, band)
        taus.append(pick_peak_subsample(R, lags))
    taus = np.array(taus)
    # Exclude near-zero values
    nz = taus[np.abs(taus) > 5e-5]
    if len(nz) < 2:
        return doa_from_tau(float(np.median(taus)))
    return doa_from_tau(float(np.median(nz)))


STRATS = [
    ("H28_inst_freq_300hzbw", lambda c, sr, s: h28_inst_freq_tracking(c, sr, s, 300)),
    ("H28_inst_freq_500hzbw", lambda c, sr, s: h28_inst_freq_tracking(c, sr, s, 500)),
    ("H28_inst_freq_200hzbw", lambda c, sr, s: h28_inst_freq_tracking(c, sr, s, 200)),
    ("H31_top10pct_500_2k", lambda c, sr, s: h31_tf_sparsity(c, sr, s, (500, 2000), 0.10)),
    ("H31_top15pct_300_4k", lambda c, sr, s: h31_tf_sparsity(c, sr, s, (300, 4000), 0.15)),
    ("H31_top5pct_1k_5k", lambda c, sr, s: h31_tf_sparsity(c, sr, s, (1000, 5000), 0.05)),
    ("H33_diff_mic_500_2k", lambda c, sr, s: h33_diff_mic(c, sr, s, (500, 2000))),
    ("H33_diff_mic_300_4k", lambda c, sr, s: h33_diff_mic(c, sr, s, (300, 4000))),
    ("H33_diff_mic_1k_5k", lambda c, sr, s: h33_diff_mic(c, sr, s, (1000, 5000))),
    ("H34_robust_combo", lambda c, sr, s: h34_robust_combo(c, sr, s)),
    ("H34b_no_zero", lambda c, sr, s: h34b_robust_combo_no_zeros(c, sr, s)),
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

    print("\nPer-position breakdown of top 5:")
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

    out = OUT_DIR / "H_round4.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
