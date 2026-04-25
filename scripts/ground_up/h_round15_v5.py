"""Round 15 — V5 attempts: smarter V3+calibration fusion.

V4 D2 (unblock speech cal): 2.30° MAE
V3 H52 (single-shot):       3.57° MAE

Per-position oracle of V3 vs D2 = 2.14°. So a perfect selector COULD beat
both. We try several selection mechanisms.

Also explore using chirp as bias-probe (chirp at same position experiences
same wall, so V3's bias on chirp ≈ V3's bias on speech).

D6   V3_speech − (V3_chirp_block − unblock_chirp_τ)  → bias-corrected V3
D11  avg(unblock chirp τ, unblock speech τ)         → calibration averaging
D14  Speech onset-locked GCC + median               → direct path snippets
D15  D14 stacked with V3 H52                        → onset + full-window
D16  Smart snap: snap to cal only if V3 confident   → PSR-aware fusion
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
from h_round12_chirp_calib import v3_h52, gcc, diff_sum_gcc, nlms

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"
PHYS_MAX = MIC_SPACING / C_MPS * 1.05


def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def doa_from_tau(tau):
    s = max(-1.0, min(1.0, C_MPS * tau / MIC_SPACING))
    return -float(np.degrees(np.arcsin(s)))


def gcc_with_psr(x_l, x_r, sr, band, max_lag_s=PHYS_MAX):
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
    off = np.ones(len(R), dtype=bool)
    off[max(0, pk - 3): min(len(R), pk + 4)] = False
    psr = R[pk] / (np.median(R[off]) + 1e-12) if off.any() else 0
    return tau, float(psr)


# ============================================================================
# Build cal tables (from unblock recordings)
# ============================================================================

def build_cal_tables(sr=48000):
    chirp_cal = {}
    speech_cal = {}
    # Unblock chirp τ per position
    for (pos, cond), paths in sorted(chirp_groups().items()):
        if cond != "unblock": continue
        chans, sr_c = load_group(paths)
        chans = {ch: basic_preprocess(x, sr_c) for ch, x in chans.items()}
        t0, t1 = auto_window_chirp(chans["mic_l"], sr_c, dur_s=1.6)
        n0, n1 = int(t0 * sr_c), int(t1 * sr_c)
        chans = {ch: x[n0:n1] for ch, x in chans.items()}
        chans_b = {ch: bp(x, sr_c, 500, 5000) for ch, x in chans.items()}
        tau, _ = gcc_with_psr(chans_b["mic_l"], chans_b["mic_r"], sr_c, (500, 5000))
        chirp_cal[pos] = tau
    # Unblock speech τ per position
    for (pos, cond), paths in sorted(speech_groups().items()):
        if cond != "unblock": continue
        chans, sr_c = load_group(paths)
        chans = {ch: basic_preprocess(x, sr_c) for ch, x in chans.items()}
        n0, n1 = int(5.0 * sr_c), int(25.0 * sr_c)
        chans = {ch: x[n0:n1] for ch, x in chans.items()}
        chans_b = {ch: bp(x, sr_c, 500, 5000) for ch, x in chans.items()}
        tau, _ = gcc_with_psr(chans_b["mic_l"], chans_b["mic_r"], sr_c, (500, 5000))
        speech_cal[pos] = tau
    return chirp_cal, speech_cal


# ============================================================================
# D6 — Chirp-as-bias-probe correction
# ============================================================================

def d6_chirp_bias_correct(speech_chans, sr, chirp_chans, unblock_chirp_tau):
    """V3 on speech minus (V3 on block chirp − unblock chirp τ)."""
    tau_v3_speech = v3_h52(speech_chans, sr)
    tau_v3_chirp = v3_h52(chirp_chans, sr)
    bias = tau_v3_chirp - unblock_chirp_tau
    return tau_v3_speech - bias


# ============================================================================
# D11 — Average of unblock chirp & speech cals
# ============================================================================

def d11_avg_cal(pos, chirp_cal, speech_cal):
    if pos not in chirp_cal or pos not in speech_cal:
        return None
    return (chirp_cal[pos] + speech_cal[pos]) / 2


# ============================================================================
# D14 — Onset-locked GCC for speech
# ============================================================================

def detect_speech_onsets(x, sr, n_top=30, min_gap_ms=80):
    """Find energy onsets in speech (rapid envelope rise)."""
    e = np.abs(hilbert(x))
    n_smooth = int(0.005 * sr)
    e = np.convolve(e, np.ones(n_smooth) / n_smooth, mode="same")
    # Compute rate of change (positive)
    de = np.diff(e, prepend=e[0])
    de_pos = np.maximum(de, 0)
    # Smooth slope
    de_pos = np.convolve(de_pos, np.ones(int(0.002 * sr)) / int(0.002 * sr),
                         mode="same")
    # Top peaks separated by min_gap
    n_min = int(min_gap_ms / 1000 * sr)
    peaks = []
    de_work = de_pos.copy()
    for _ in range(n_top):
        pk = int(np.argmax(de_work))
        if de_work[pk] < 0.05 * de_pos.max():
            break
        peaks.append(pk)
        lo = max(0, pk - n_min); hi = min(len(de_work), pk + n_min)
        de_work[lo:hi] = 0
    peaks.sort()
    return peaks


def d14_onset_locked_gcc(speech_chans, sr, snippet_ms=15, band=(800, 5000)):
    """For each onset in mic_L, GCC on first snippet_ms after it.
    Median τ across onsets."""
    chans_b = {ch: bp(x, sr, *band) for ch, x in speech_chans.items()}
    onsets = detect_speech_onsets(chans_b["mic_l"], sr, n_top=50)
    n_snip = int(snippet_ms / 1000 * sr)
    taus = []
    for pk in onsets:
        if pk + n_snip > len(chans_b["mic_l"]):
            continue
        sl = slice(pk, pk + n_snip)
        tau, psr = gcc_with_psr(chans_b["mic_l"][sl], chans_b["mic_r"][sl], sr, band)
        if abs(tau) < PHYS_MAX:
            taus.append(tau)
    if not taus:
        return None
    return float(np.median(taus))


# ============================================================================
# D15 — D14 + V3 stack
# ============================================================================

def d15_onset_v3_combined(speech_chans, sr):
    tau_onset = d14_onset_locked_gcc(speech_chans, sr)
    tau_v3 = v3_h52(speech_chans, sr)
    if tau_onset is None:
        return tau_v3
    # Same-sign agreement → average
    if tau_onset * tau_v3 > 0 and abs(tau_onset - tau_v3) < 0.3e-3:
        return (tau_onset + tau_v3) / 2
    # Else max-|τ|
    return tau_onset if abs(tau_onset) > abs(tau_v3) else tau_v3


# ============================================================================
# D16 — Smart snap (PSR-aware)
# ============================================================================

def d16_smart_snap(speech_chans, sr, cal_table, psr_thresh=4.0):
    """If V3 has high PSR AND |τ|>0.2ms, trust V3.
    Else if V3's τ is very close to a cal entry, snap.
    Else use max-|τ| of V3 vs nearest cal."""
    tau_v3 = v3_h52(speech_chans, sr)
    # Compute PSR for V3 (mic-mic GCC at 300-4000 after NLMS)
    chans_b = {ch: bp(x, sr, 300, 4000) for ch, x in speech_chans.items()}
    e_L = nlms(chans_b["ldv"], chans_b["mic_l"])
    e_R = nlms(chans_b["ldv"], chans_b["mic_r"])
    _, psr_v3 = gcc_with_psr(e_L, e_R, sr, (300, 4000))

    # Find nearest cal entry
    cal_arr = np.array(list(cal_table.values()))
    cal_keys = list(cal_table.keys())
    nearest_idx = int(np.argmin(np.abs(cal_arr - tau_v3)))
    nearest_tau = cal_arr[nearest_idx]
    dist = abs(tau_v3 - nearest_tau)

    if psr_v3 > psr_thresh and abs(tau_v3) > 0.2e-3:
        # V3 confident: trust V3
        return tau_v3
    if dist < 0.3e-3:
        # V3 close to cal: snap
        return nearest_tau
    # Neither: max-|τ|
    return nearest_tau if abs(nearest_tau) > abs(tau_v3) else tau_v3


# ============================================================================
# D17 — D2 with V3 fallback for "outside cal" detection
# ============================================================================

def d17_d2_with_v3_fallback(speech_chans, sr, cal_table, gap_thresh=0.5e-3):
    """Use D2 if V3 is close to one of cal points; else use V3."""
    tau_v3 = v3_h52(speech_chans, sr)
    cal_arr = np.array(list(cal_table.values()))
    nearest_idx = int(np.argmin(np.abs(cal_arr - tau_v3)))
    nearest_tau = cal_arr[nearest_idx]
    dist = abs(tau_v3 - nearest_tau)
    if dist < gap_thresh:
        return nearest_tau
    return tau_v3


# ============================================================================
# Main
# ============================================================================

def main():
    chirp_cal, speech_cal = build_cal_tables()
    print("Calibration tables:")
    for pos in sorted(chirp_cal):
        print(f"  x={pos}: chirp_cal={chirp_cal[pos]*1000:+.3f} ms, "
              f"speech_cal={speech_cal.get(pos, 0)*1000:+.3f} ms")

    results = {
        "V3_baseline": {},
        "V4_D2_speech_cal": {},
        "V4_D1_chirp_cal": {},
        "D6_chirp_bias_correct": {},
        "D11_avg_chirp_speech_cal": {},
        "D14_onset_locked_gcc": {},
        "D15_onset_plus_v3": {},
        "D16_smart_snap_speech_cal": {},
        "D16b_smart_snap_chirp_cal": {},
        "D17_d2_with_v3_fallback": {},
        "D17b_d1_with_v3_fallback": {},
    }

    for (pos, cond), paths in sorted(speech_groups().items()):
        if cond != "block": continue
        chans, sr = load_group(paths)
        chans = {ch: basic_preprocess(x, sr) for ch, x in chans.items()}
        n0, n1 = int(5.0 * sr), int(25.0 * sr)
        chans = {ch: x[n0:n1] for ch, x in chans.items()}
        theta_true = expected_doa_deg(float(pos))

        # Get matching block chirp recording
        chirp_block_key = (pos, "block")
        if chirp_block_key in chirp_groups():
            chirp_chans_pre, _ = load_group(chirp_groups()[chirp_block_key])
            chirp_chans = {ch: basic_preprocess(x, sr) for ch, x in chirp_chans_pre.items()}
            t0, t1 = auto_window_chirp(chirp_chans["mic_l"], sr, dur_s=1.6)
            n0c, n1c = int(t0 * sr), int(t1 * sr)
            chirp_chans = {ch: x[n0c:n1c] for ch, x in chirp_chans.items()}
        else:
            chirp_chans = None

        # V3 baseline
        tau_v3 = v3_h52(chans, sr)
        theta_v3 = doa_from_tau(tau_v3)
        results["V3_baseline"][pos] = {
            "true": theta_true, "est": theta_v3,
            "err": abs(theta_v3 - theta_true)}

        # V4 D2 baseline
        if pos in speech_cal:
            theta_d2 = doa_from_tau(speech_cal[pos])
            results["V4_D2_speech_cal"][pos] = {
                "true": theta_true, "est": theta_d2,
                "err": abs(theta_d2 - theta_true)}

        # V4 D1 baseline (chirp cal)
        if pos in chirp_cal:
            theta_d1 = doa_from_tau(chirp_cal[pos])
            results["V4_D1_chirp_cal"][pos] = {
                "true": theta_true, "est": theta_d1,
                "err": abs(theta_d1 - theta_true)}

        # D6
        if chirp_chans is not None and pos in chirp_cal:
            tau_d6 = d6_chirp_bias_correct(chans, sr, chirp_chans, chirp_cal[pos])
            theta_d6 = doa_from_tau(tau_d6)
            results["D6_chirp_bias_correct"][pos] = {
                "true": theta_true, "est": theta_d6,
                "err": abs(theta_d6 - theta_true)}

        # D11
        tau_d11 = d11_avg_cal(pos, chirp_cal, speech_cal)
        if tau_d11 is not None:
            theta_d11 = doa_from_tau(tau_d11)
            results["D11_avg_chirp_speech_cal"][pos] = {
                "true": theta_true, "est": theta_d11,
                "err": abs(theta_d11 - theta_true)}

        # D14
        tau_d14 = d14_onset_locked_gcc(chans, sr)
        if tau_d14 is not None:
            theta_d14 = doa_from_tau(tau_d14)
            results["D14_onset_locked_gcc"][pos] = {
                "true": theta_true, "est": theta_d14,
                "err": abs(theta_d14 - theta_true)}

        # D15
        tau_d15 = d15_onset_v3_combined(chans, sr)
        theta_d15 = doa_from_tau(tau_d15)
        results["D15_onset_plus_v3"][pos] = {
            "true": theta_true, "est": theta_d15,
            "err": abs(theta_d15 - theta_true)}

        # D16
        tau_d16 = d16_smart_snap(chans, sr, speech_cal)
        theta_d16 = doa_from_tau(tau_d16)
        results["D16_smart_snap_speech_cal"][pos] = {
            "true": theta_true, "est": theta_d16,
            "err": abs(theta_d16 - theta_true)}
        tau_d16b = d16_smart_snap(chans, sr, chirp_cal)
        theta_d16b = doa_from_tau(tau_d16b)
        results["D16b_smart_snap_chirp_cal"][pos] = {
            "true": theta_true, "est": theta_d16b,
            "err": abs(theta_d16b - theta_true)}

        # D17
        tau_d17 = d17_d2_with_v3_fallback(chans, sr, speech_cal)
        theta_d17 = doa_from_tau(tau_d17)
        results["D17_d2_with_v3_fallback"][pos] = {
            "true": theta_true, "est": theta_d17,
            "err": abs(theta_d17 - theta_true)}
        tau_d17b = d17_d2_with_v3_fallback(chans, sr, chirp_cal)
        theta_d17b = doa_from_tau(tau_d17b)
        results["D17b_d1_with_v3_fallback"][pos] = {
            "true": theta_true, "est": theta_d17b,
            "err": abs(theta_d17b - theta_true)}

    print(f"\n{'strategy':<32} | {'speech MAE':>10}")
    print("-" * 50)
    rankings = []
    for sid in results:
        errs = [v["err"] for v in results[sid].values() if v.get("err") is not None]
        mae = float(np.mean(errs)) if errs else None
        rankings.append((sid, mae))
    rankings.sort(key=lambda r: r[1] or 999)
    for sid, mae in rankings:
        m_str = f"{mae:6.2f}°" if mae is not None else "  N/A "
        print(f"{sid:<32} | {m_str:>10}")

    print("\nPer-position breakdown of top 5:")
    for sid, _ in rankings[:5]:
        print(f"\n=== {sid} ===")
        for pos in sorted(results[sid]):
            r = results[sid][pos]
            if r.get("err") is None:
                print(f"  x={pos}: NA"); continue
            print(f"  x={pos}: true={r['true']:+6.2f}° est={r['est']:+6.2f}° err={r['err']:5.2f}°")

    out = OUT_DIR / "H_round15_v5.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
