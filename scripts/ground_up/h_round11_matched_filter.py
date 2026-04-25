"""Round 11 — chirp matched filter (with CORRECT upsweep direction!).

Previous H42 used downsweep template (6kHz→500Hz) which was wrong. Actual
chirp is upsweep ~500Hz→~7000Hz over ~1.5s, repeated every 2s, 6 bursts.

With correct template:
  - Match filter on each channel → precise arrival time per burst
  - 6 bursts per recording → coherent average → √6 SNR boost
  - Mic-mic τ_LR from arrival difference (no GCC needed)
  - Each burst is broadband 500-7000 Hz (Cramer-Rao τ precision << 0.01 ms)

For chirp +x where mic SNR is poor, matched filter SNR boost ~30 dB
(time-bandwidth product B·T ~ 6500·1.5 = 9750 ~ 40 dB processing gain)
should rescue the lock-to-zero problem.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy import signal as sp

from _loader import chirp_groups, load_group
from _pigs import (MIC_L, MIC_R, LDV, C_MPS, MIC_SPACING,
                   preprocess as basic_preprocess)
from _geometry import expected_doa_deg, expected_tdoa_ms, REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"


def doa_from_tau(tau):
    s = max(-1.0, min(1.0, C_MPS * tau / MIC_SPACING))
    return -float(np.degrees(np.arcsin(s)))


def synth_upchirp(sr, f0=500.0, f1=7000.0, dur=1.5):
    """Synthesize upsweep linear chirp from f0 to f1 over dur seconds."""
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t ** 2))


def matched_filter(x, template, sr, max_search_s=None):
    """Cross-correlate x with template, return peak time and full response."""
    n_x = len(x); n_t = len(template)
    n_full = n_x + n_t - 1
    n_fft = 1 << int(np.ceil(np.log2(n_full)))
    X = np.fft.rfft(x, n_fft)
    T = np.fft.rfft(template[::-1], n_fft)  # time-reversed → correlation
    R = np.fft.irfft(X * T, n_fft)
    # Result valid range: 0 to n_x + n_t - 1
    R = R[:n_full]
    # The peak at index k corresponds to template aligned with x at offset k - (n_t - 1)
    # i.e., template start at sample (k - n_t + 1) of x
    return R


def find_burst_starts_via_matched_filter(R, sr, n_bursts_expected=6,
                                         min_interval_s=1.0):
    """Find peaks in matched-filter output."""
    R_abs = np.abs(R)
    # Smooth a bit to reject narrow noise spikes
    R_smooth = R_abs
    # Find top peaks separated by min_interval_s
    n_min = int(min_interval_s * sr)
    peaks = []
    R_work = R_smooth.copy()
    for _ in range(n_bursts_expected):
        pk = int(np.argmax(R_work))
        peaks.append(pk)
        # Suppress around this peak
        lo = max(0, pk - n_min); hi = min(len(R_work), pk + n_min)
        R_work[lo:hi] = 0
    peaks.sort()
    return peaks


def chirp_matched_filter_doa(chans, sr, template, max_lag_s=0.005):
    """For each channel, find chirp arrival times via matched filter,
    then compute mic-mic τ from differences across multiple bursts.
    Return median τ over bursts.
    """
    n_t = len(template)
    R_l = matched_filter(chans["mic_l"], template, sr)
    R_r = matched_filter(chans["mic_r"], template, sr)
    # Find burst starts in mic_L
    peaks_l = find_burst_starts_via_matched_filter(R_l, sr)
    # For each peak in mic_L, find the local max in mic_R within ±5ms
    n_search = int(max_lag_s * sr) + 50  # cushion
    taus = []
    for pk_l in peaks_l:
        lo = max(0, pk_l - n_search); hi = min(len(R_r), pk_l + n_search)
        if hi <= lo:
            continue
        local_R_r = np.abs(R_r[lo: hi])
        pk_r_local = int(np.argmax(local_R_r))
        pk_r = lo + pk_r_local
        # Sub-sample peak via parabolic interp
        if 1 <= pk_r_local < len(local_R_r) - 1:
            ym, y0, yp = local_R_r[pk_r_local - 1], local_R_r[pk_r_local], local_R_r[pk_r_local + 1]
            d = ym - 2 * y0 + yp
            offset = 0.5 * (ym - yp) / d if abs(d) > 1e-12 else 0
            pk_r_sub = pk_r + offset
        else:
            pk_r_sub = pk_r
        # Also sub-sample mic_L peak
        local_R_l = np.abs(R_l[max(0, pk_l - 5): min(len(R_l), pk_l + 5)])
        if len(local_R_l) >= 3:
            mid = len(local_R_l) // 2
            ym, y0, yp = local_R_l[mid - 1], local_R_l[mid], local_R_l[mid + 1]
            d = ym - 2 * y0 + yp
            offset_l = 0.5 * (ym - yp) / d if abs(d) > 1e-12 else 0
            pk_l_sub = pk_l + offset_l
        else:
            pk_l_sub = pk_l
        tau = (pk_r_sub - pk_l_sub) / sr
        if abs(tau) < 0.01:  # 10ms physical max with cushion
            taus.append(tau)
    if not taus:
        return None, []
    # Robust: median over bursts
    return float(np.median(taus)), taus


def chirp_matched_filter_VL_VR(chans, sr, template):
    """Cross-modal: match-filter LDV with template, mic_L, mic_R with template.
    Return τ_VL, τ_VR, τ_LR computed from differences across 6 bursts."""
    R_v = matched_filter(chans["ldv"], template, sr)
    R_l = matched_filter(chans["mic_l"], template, sr)
    R_r = matched_filter(chans["mic_r"], template, sr)
    peaks_l = find_burst_starts_via_matched_filter(R_l, sr)
    n_search = int(0.012 * sr) + 50

    def find_local_peak_subsample(R, pk_center, search):
        lo = max(0, pk_center - search); hi = min(len(R), pk_center + search)
        if hi <= lo:
            return None
        local = np.abs(R[lo: hi])
        pk = int(np.argmax(local))
        if 1 <= pk < len(local) - 1:
            ym, y0, yp = local[pk - 1], local[pk], local[pk + 1]
            d = ym - 2 * y0 + yp
            offset = 0.5 * (ym - yp) / d if abs(d) > 1e-12 else 0
        else:
            offset = 0
        return lo + pk + offset

    taus_vl = []; taus_vr = []; taus_lr = []
    for pk_l in peaks_l:
        l_sub = find_local_peak_subsample(R_l, pk_l, 5)
        v_sub = find_local_peak_subsample(R_v, pk_l, n_search)
        r_sub = find_local_peak_subsample(R_r, pk_l, n_search)
        if l_sub is None or v_sub is None or r_sub is None:
            continue
        tau_vl = (l_sub - v_sub) / sr
        tau_vr = (r_sub - v_sub) / sr
        tau_lr = (r_sub - l_sub) / sr
        if abs(tau_lr) < 0.01:  # physically sensible
            taus_vl.append(tau_vl)
            taus_vr.append(tau_vr)
            taus_lr.append(tau_lr)
    if not taus_lr:
        return None, None, None, []
    return (float(np.median(taus_vl)), float(np.median(taus_vr)),
            float(np.median(taus_lr)), taus_lr)


def run_matched_filter_strats():
    """Test different chirp templates."""
    templates = [
        ("MF_500_7k_1.5s", lambda sr: synth_upchirp(sr, 500, 7000, 1.5)),
        ("MF_500_6k_1.0s", lambda sr: synth_upchirp(sr, 500, 6000, 1.0)),
        ("MF_400_8k_1.5s", lambda sr: synth_upchirp(sr, 400, 8000, 1.5)),
        ("MF_300_5k_1.5s", lambda sr: synth_upchirp(sr, 300, 5000, 1.5)),
        ("MF_500_7k_1.0s", lambda sr: synth_upchirp(sr, 500, 7000, 1.0)),
    ]
    results = {}
    for sid, template_fn in templates:
        groups = chirp_groups()
        for (pos, cond), paths in sorted(groups.items()):
            if cond != "block":
                continue
            chans, sr = load_group(paths)
            chans = {ch: basic_preprocess(x, sr) for ch, x in chans.items()}
            template = template_fn(sr)
            tau_lr, all_taus = chirp_matched_filter_doa(chans, sr, template)
            theta_true = expected_doa_deg(float(pos))
            theta_est = doa_from_tau(tau_lr) if tau_lr is not None else None
            err = abs(theta_est - theta_true) if theta_est is not None else None
            results.setdefault(sid, {})[pos] = {
                "true": theta_true, "est": theta_est, "err": err,
                "tau_ms": tau_lr * 1000 if tau_lr else None,
                "n_bursts": len(all_taus),
                "all_taus_ms": [t * 1000 for t in all_taus],
            }
    return results


def main():
    results = run_matched_filter_strats()
    print(f"\n{'strategy':<24} | {'chirp MAE':>10}")
    print("-" * 40)
    rankings = []
    for sid in results:
        errs = [v["err"] for v in results[sid].values() if v["err"] is not None]
        mae = float(np.mean(errs)) if errs else None
        rankings.append((sid, mae))
    rankings.sort(key=lambda r: r[1] or 999)
    for sid, mae in rankings:
        m_str = f"{mae:6.2f}°" if mae is not None else "  N/A "
        print(f"{sid:<24} | {m_str:>10}")

    print("\nPer-position breakdown of best:")
    best = rankings[0][0]
    print(f"\n=== {best} ===")
    for pos in sorted(results[best]):
        r = results[best][pos]
        if r["err"] is None:
            print(f"  x={pos}: NA")
            continue
        all_taus = r["all_taus_ms"][:6]
        taus_str = ", ".join(f"{t:+.3f}" for t in all_taus)
        print(f"  x={pos}: tau={r['tau_ms']:+.3f}ms (n={r['n_bursts']}: {taus_str}) | "
              f"true={r['true']:+6.2f}° est={r['est']:+6.2f}° err={r['err']:5.2f}°")

    out = OUT_DIR / "H_round11_matched_filter.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
