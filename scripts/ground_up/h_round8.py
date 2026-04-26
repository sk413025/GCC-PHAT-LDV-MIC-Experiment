"""Round 8 — robust statistics + RANSAC + bispectrum + RIR deconvolution.

H39  Per-frame H38a + median: max-|τ| at frame level, robust median over frames
H40  RANSAC phase-slope fit: mic-mic phase = -2πfτ; RANSAC rejects outliers
H41  Bispectrum-based τ: 4th-order cumulant rejects Gaussian reverb
H42  Chirp RIR deconvolution: extract direct path from impulse response
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from numpy.fft import rfft, irfft, fft, ifft
from scipy.signal import butter, filtfilt, hilbert, chirp as scipy_chirp
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


def gcc_phat(x_l, x_r, sr, band, max_lag_s=PHYS_MAX, return_psr=False):
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
    return gcc_phat(x_l - x_r, x_l + x_r, sr, band, return_psr=return_psr)


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
# H39 — Per-frame H38a + robust median
# ============================================================================

def h39_per_frame_max_abs(chans, sr, sig_type, frame_ms=80, hop_ms=40):
    """For each frame, run BOTH H1 and H37, pick max-|τ|, then median over frames."""
    band_h1 = (300, 4000)
    bands_h37 = [(500, 1500), (1000, 2500), (1500, 3500), (2500, 5000)]

    chans_full = {ch: bp(x, sr, 200, 8000) for ch, x in chans.items()}
    e_L = nlms(chans_full["ldv"], chans_full["mic_l"])
    e_R = nlms(chans_full["ldv"], chans_full["mic_r"])

    n_frame = int(frame_ms / 1000 * sr)
    n_hop = int(hop_ms / 1000 * sr)
    frame_taus = []
    for s0 in range(0, len(e_L) - n_frame, n_hop):
        sl = slice(s0, s0 + n_frame)
        # H1-style direct on residuals
        e_L_b = bp(e_L[sl], sr, *band_h1)
        e_R_b = bp(e_R[sl], sr, *band_h1)
        tau_h1 = gcc_phat(e_L_b, e_R_b, sr, band_h1)
        # H37-style multi-band diff/sum on residuals
        h37_taus = []
        for band in bands_h37:
            l_b = bp(e_L[sl], sr, *band)
            r_b = bp(e_R[sl], sr, *band)
            h37_taus.append(diff_sum_gcc(l_b, r_b, sr, band))
        h37_taus = np.array(h37_taus)
        nz = np.abs(h37_taus) > 1e-4
        if nz.sum() >= 2:
            tau_h37 = float(np.median(h37_taus[nz]))
        else:
            tau_h37 = float(np.median(h37_taus))
        # max-|τ|
        tau_frame = tau_h37 if abs(tau_h37) > abs(tau_h1) else tau_h1
        frame_taus.append(tau_frame)
    if not frame_taus:
        return None
    # Median over frames; reject outliers
    frame_taus = np.array(frame_taus)
    return doa_from_tau(float(np.median(frame_taus)))


def h39b_per_frame_trimmed_mean(chans, sr, sig_type, frame_ms=80, hop_ms=40,
                                trim=0.1):
    """Same as H39 but trimmed mean (more conservative robust statistic)."""
    band_h1 = (300, 4000)
    bands_h37 = [(500, 1500), (1000, 2500), (1500, 3500), (2500, 5000)]
    chans_full = {ch: bp(x, sr, 200, 8000) for ch, x in chans.items()}
    e_L = nlms(chans_full["ldv"], chans_full["mic_l"])
    e_R = nlms(chans_full["ldv"], chans_full["mic_r"])
    n_frame = int(frame_ms / 1000 * sr)
    n_hop = int(hop_ms / 1000 * sr)
    frame_taus = []
    for s0 in range(0, len(e_L) - n_frame, n_hop):
        sl = slice(s0, s0 + n_frame)
        e_L_b = bp(e_L[sl], sr, *band_h1)
        e_R_b = bp(e_R[sl], sr, *band_h1)
        tau_h1 = gcc_phat(e_L_b, e_R_b, sr, band_h1)
        h37_taus = []
        for band in bands_h37:
            l_b = bp(e_L[sl], sr, *band)
            r_b = bp(e_R[sl], sr, *band)
            h37_taus.append(diff_sum_gcc(l_b, r_b, sr, band))
        h37_taus = np.array(h37_taus)
        nz = np.abs(h37_taus) > 1e-4
        if nz.sum() >= 2:
            tau_h37 = float(np.median(h37_taus[nz]))
        else:
            tau_h37 = float(np.median(h37_taus))
        tau_frame = tau_h37 if abs(tau_h37) > abs(tau_h1) else tau_h1
        frame_taus.append(tau_frame)
    if not frame_taus:
        return None
    arr = np.sort(frame_taus)
    n_trim = int(len(arr) * trim)
    if n_trim > 0:
        arr = arr[n_trim: -n_trim]
    return doa_from_tau(float(np.mean(arr)))


# ============================================================================
# H40 — RANSAC phase-slope fit
# ============================================================================

def h40_ransac_phase(chans, sr, sig_type, band=(500, 5000), n_iter=200,
                    inlier_tol=0.3):
    """Mic-mic cross-spectrum phase = -2πfτ. RANSAC fits τ from per-bin phases."""
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    n = max(len(chans_b["mic_l"]), len(chans_b["mic_r"]))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xl = rfft(chans_b["mic_l"], n_fft)
    Xr = rfft(chans_b["mic_r"], n_fft)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    G = Xr * np.conj(Xl)
    in_band = (f >= band[0]) & (f <= band[1])
    f_b = f[in_band]
    phase = np.angle(G[in_band])
    mag = np.abs(G[in_band])
    # Candidate τ from grid (sub-ms resolution)
    tau_grid = np.linspace(-PHYS_MAX, PHYS_MAX, 401)
    # Score each candidate by # of inliers (phase residual within tol)
    omega = 2 * np.pi * f_b
    best_tau = 0.0; best_score = -1
    for tau_c in tau_grid:
        residual = np.angle(np.exp(1j * (phase + omega * tau_c)))  # wrap to [-π,π]
        weighted_inliers = np.sum((np.abs(residual) < inlier_tol) * mag)
        if weighted_inliers > best_score:
            best_score = weighted_inliers
            best_tau = tau_c
    # Refine: weighted least-squares on inliers
    residual = np.angle(np.exp(1j * (phase + omega * best_tau)))
    inliers = np.abs(residual) < inlier_tol
    if inliers.sum() > 5:
        # Fit τ minimizing sum w_i * (phi_i + omega_i * τ + 2π k_i)^2
        # with k_i chosen to be optimal (already wrapped)
        w = mag[inliers]
        omega_in = omega[inliers]
        phi_in = phase[inliers] + omega_in * best_tau  # use wrapped
        phi_in = np.angle(np.exp(1j * phi_in))
        # phase = -omega·delta_τ → delta_τ = -phi / omega
        delta_tau = -np.sum(w * phi_in * omega_in) / np.sum(w * omega_in ** 2)
        best_tau += delta_tau
    return doa_from_tau(best_tau)


def h40_with_nlms(chans, sr, sig_type, band=(500, 5000)):
    """RANSAC phase-fit on NLMS residuals."""
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    e_L = nlms(chans_b["ldv"], chans_b["mic_l"])
    e_R = nlms(chans_b["ldv"], chans_b["mic_r"])
    n = max(len(e_L), len(e_R))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xl = rfft(e_L, n_fft); Xr = rfft(e_R, n_fft)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    G = Xr * np.conj(Xl)
    in_band = (f >= band[0]) & (f <= band[1])
    f_b = f[in_band]
    phase = np.angle(G[in_band])
    mag = np.abs(G[in_band])
    tau_grid = np.linspace(-PHYS_MAX, PHYS_MAX, 401)
    omega = 2 * np.pi * f_b
    best_tau = 0.0; best_score = -1
    for tau_c in tau_grid:
        residual = np.angle(np.exp(1j * (phase + omega * tau_c)))
        weighted_inliers = np.sum((np.abs(residual) < 0.3) * mag)
        if weighted_inliers > best_score:
            best_score = weighted_inliers
            best_tau = tau_c
    return doa_from_tau(best_tau)


# ============================================================================
# H41 — Bispectrum-based TDoA (4th-order cumulant rejects Gaussian)
# ============================================================================

def h41_bispectrum_tdoa(chans, sr, sig_type, band=(500, 4000)):
    """Compute fourth-order cumulant cross-correlation that's zero for Gaussian
    interference (room reverb approximation) and non-zero for non-Gaussian source.

    Approximation: use higher-power phase coherence:
      G_4(τ) = IFFT( (Xl·Xl·conj(Xr)·conj(Xr)) / |...|² )
    Theoretically picks 2τ instead of τ (frequency doubling). Resolution × 2.
    """
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    n = max(len(chans_b["mic_l"]), len(chans_b["mic_r"]))
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    Xl = rfft(chans_b["mic_l"], n_fft)
    Xr = rfft(chans_b["mic_r"], n_fft)
    # 4th order: |X|² has 2nd-order nonlinearity → enhances peaks
    # G_4 = X_L · X_L · conj(X_R) · conj(X_R)
    G = (Xl ** 2) * (np.conj(Xr) ** 2)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    G = G * ((f >= band[0]) & (f <= band[1]))
    eps = 1e-12
    r = np.fft.fftshift(irfft(G / (np.abs(G) + eps), n_fft))
    max_n = int(np.ceil(2 * PHYS_MAX * sr))  # 2τ range
    mid = len(r) // 2
    R = np.abs(r[mid - max_n: mid + max_n + 1])
    lags = (np.arange(len(R)) - max_n) / sr / 2  # divide by 2: 2τ → τ
    pk = int(np.argmax(R))
    tau = float(lags[pk])
    return doa_from_tau(tau)


def h41b_squared_envelope_gcc(chans, sr, sig_type, band=(500, 4000)):
    """Cross-correlate squared envelopes — 4th-order in original signal,
    rejects Gaussian interference."""
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    env_l = np.abs(hilbert(chans_b["mic_l"])) ** 2
    env_r = np.abs(hilbert(chans_b["mic_r"])) ** 2
    env_l = env_l - np.mean(env_l)
    env_r = env_r - np.mean(env_r)
    return doa_from_tau(gcc_phat(env_l, env_r, sr, band))


# ============================================================================
# H42 — RIR deconvolution + early-window (chirp only)
# ============================================================================

def h42_rir_early(chans, sr, sig_type, band=(500, 6000), early_ms=2):
    """For chirp: deconvolve mic with synthetic chirp template to get RIR.
    Window first 'early_ms' to keep direct + early reflections only.
    Then mic-mic GCC on early-windowed signals."""
    if sig_type != "chirp":
        return None
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    # Detect chirp params
    inst_phase = np.unwrap(np.angle(hilbert(chans_b["mic_l"])))
    inst_freq = np.gradient(inst_phase) * sr / (2 * np.pi)
    # Crude chirp model: linear sweep from f1 to f0 over duration T
    n = len(chans_b["mic_l"])
    duration = n / sr
    f_start = float(np.median(inst_freq[:int(0.05 * n)]))
    f_end = float(np.median(inst_freq[-int(0.05 * n):]))
    if abs(f_start - f_end) < 100 or f_start < 200 or f_end < 200:
        return None  # not a chirp
    # Synthetic template
    t = np.arange(n) / sr
    template = np.sin(2 * np.pi * (f_start * t +
                                   (f_end - f_start) / (2 * duration) * t ** 2))
    template = bp(template, sr, *band)
    # Deconvolve: H = X / S (with PHAT-style regularization)
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    T = rfft(template, n_fft)
    Xl = rfft(chans_b["mic_l"], n_fft)
    Xr = rfft(chans_b["mic_r"], n_fft)
    eps = 1e-3 * np.abs(T).max()
    Hl = Xl / (T + eps * np.exp(1j * np.angle(T)))
    Hr = Xr / (T + eps * np.exp(1j * np.angle(T)))
    # Inverse FFT to get RIRs
    rir_l = irfft(Hl, n_fft)
    rir_r = irfft(Hr, n_fft)
    # Early window
    n_early = int(early_ms / 1000 * sr)
    rir_l_early = rir_l[:n_early]
    rir_r_early = rir_r[:n_early]
    # GCC on early RIRs
    return doa_from_tau(gcc_phat(rir_l_early, rir_r_early, sr, band))


def h42b_rir_with_nlms(chans, sr, sig_type, band=(500, 6000), early_ms=3):
    """RIR-early + NLMS LDV-subtract first."""
    if sig_type != "chirp":
        return None
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    e_L = nlms(chans_b["ldv"], chans_b["mic_l"])
    e_R = nlms(chans_b["ldv"], chans_b["mic_r"])
    inst_phase = np.unwrap(np.angle(hilbert(e_L)))
    inst_freq = np.gradient(inst_phase) * sr / (2 * np.pi)
    n = len(e_L)
    duration = n / sr
    f_start = float(np.median(inst_freq[:int(0.05 * n)]))
    f_end = float(np.median(inst_freq[-int(0.05 * n):]))
    if abs(f_start - f_end) < 100 or f_start < 200:
        return None
    t = np.arange(n) / sr
    template = np.sin(2 * np.pi * (f_start * t +
                                   (f_end - f_start) / (2 * duration) * t ** 2))
    template = bp(template, sr, *band)
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    T = rfft(template, n_fft)
    El = rfft(e_L, n_fft); Er = rfft(e_R, n_fft)
    eps = 1e-3 * np.abs(T).max()
    rir_l = irfft(El / (T + eps), n_fft)[:int(early_ms / 1000 * sr)]
    rir_r = irfft(Er / (T + eps), n_fft)[:int(early_ms / 1000 * sr)]
    return doa_from_tau(gcc_phat(rir_l, rir_r, sr, band))


STRATS = [
    ("H39_per_frame_max_abs_80ms", lambda c, sr, s: h39_per_frame_max_abs(c, sr, s, 80, 40)),
    ("H39_per_frame_max_abs_50ms", lambda c, sr, s: h39_per_frame_max_abs(c, sr, s, 50, 25)),
    ("H39_per_frame_max_abs_120ms", lambda c, sr, s: h39_per_frame_max_abs(c, sr, s, 120, 60)),
    ("H39b_trimmed_mean_80ms", lambda c, sr, s: h39b_per_frame_trimmed_mean(c, sr, s, 80, 40, 0.15)),
    ("H40_ransac_phase_500_5k", lambda c, sr, s: h40_ransac_phase(c, sr, s, (500, 5000))),
    ("H40_ransac_phase_300_4k", lambda c, sr, s: h40_ransac_phase(c, sr, s, (300, 4000))),
    ("H40_ransac_with_nlms_500_5k", lambda c, sr, s: h40_with_nlms(c, sr, s, (500, 5000))),
    ("H40_ransac_with_nlms_300_4k", lambda c, sr, s: h40_with_nlms(c, sr, s, (300, 4000))),
    ("H41_bispectrum_500_4k", lambda c, sr, s: h41_bispectrum_tdoa(c, sr, s, (500, 4000))),
    ("H41_bispectrum_1k_5k", lambda c, sr, s: h41_bispectrum_tdoa(c, sr, s, (1000, 5000))),
    ("H41b_squared_env_500_4k", lambda c, sr, s: h41b_squared_envelope_gcc(c, sr, s, (500, 4000))),
    ("H42_rir_early_2ms", lambda c, sr, s: h42_rir_early(c, sr, s, (500, 6000), 2)),
    ("H42_rir_early_5ms", lambda c, sr, s: h42_rir_early(c, sr, s, (500, 6000), 5)),
    ("H42b_rir_nlms_3ms", lambda c, sr, s: h42b_rir_with_nlms(c, sr, s, (500, 6000), 3)),
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
                        print(f"    x={pos}: NA")
                        continue
                    print(f"    x={pos}: true={r['true']:+6.2f}° "
                          f"est={r['est']:+6.2f}° err={r['err']:5.2f}°")

    out = OUT_DIR / "H_round8.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
