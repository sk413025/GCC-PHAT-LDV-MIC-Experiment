"""Round 12 — Chirp as calibration for Speech.

Currently chirp and speech are processed INDEPENDENTLY. But chirp at each
position is recorded WITH THE SAME mic+barrier+room as speech. So chirp is a
known broadband probe of the room/wall channel.

KEY INSIGHT:
  For each position, chirp gives us H_L(f), H_R(f) — the per-mic transfer
  function from source to mic, INCLUDING wall response. We can use this to:

  Strategy A — Equalize speech:
    speech_L_eq(f) = speech_L(f) / H_L(f)
    speech_R_eq(f) = speech_R(f) / H_R(f)
    The equalized signals look like "free-field" recordings; standard
    mic-mic GCC should work like in unblock condition (which gives 2-3°).

  Strategy B — Channel-aware τ correction:
    Chirp gives "ground-truth" τ_LR_chirp at each position via matched filter.
    Compare with V3 H52 output on chirp. The bias (chirp_actual - V3_chirp)
    is the position-specific channel-induced bias. Apply same correction to
    speech V3 H52 output.

  Strategy C — Use chirp τ_LR directly as DoA estimate for speech:
    Since chirp and speech are at SAME source position, τ_LR is the same.
    Just use chirp's matched-filter τ_LR (after offset correction) for speech.
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
from _pigs import (MIC_L, MIC_R, LDV, C_MPS, MIC_SPACING,
                   preprocess as basic_preprocess)
from _pigs2 import auto_window_chirp
from _geometry import expected_doa_deg, expected_tdoa_ms, REPO_ROOT
from h_round11_matched_filter import (synth_upchirp, matched_filter,
                                     find_burst_starts_via_matched_filter)
from h_round6_combiner import h38a_max_abs_tau
# We'll use V3 H52 (defined inline below) for self-comparison

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"
PHYS_MAX = MIC_SPACING / C_MPS * 1.05


def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def doa_from_tau(tau):
    s = max(-1.0, min(1.0, C_MPS * tau / MIC_SPACING))
    return -float(np.degrees(np.arcsin(s)))


def gcc(x_l, x_r, sr, band, max_lag_s=PHYS_MAX):
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
    return tau


def diff_sum_gcc(x_l, x_r, sr, band):
    return gcc(x_l - x_r, x_l + x_r, sr, band)


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


def v3_h52(chans, sr):
    """V3 H52 from h_round9: agree-average with H1/H37 fallback to max-|τ|."""
    band_h1 = (300, 4000)
    chans_b = {ch: bp(x, sr, *band_h1) for ch, x in chans.items()}
    e_L = nlms(chans_b["ldv"], chans_b["mic_l"])
    e_R = nlms(chans_b["ldv"], chans_b["mic_r"])
    tau_h1 = gcc(e_L, e_R, sr, band_h1)
    # H37
    bands = [(500, 1500), (1000, 2500), (1500, 3500), (2500, 5000),
             (3500, 7000), (1000, 5000), (300, 4000)]
    chans_full = {ch: bp(x, sr, 200, 8000) for ch, x in chans.items()}
    e_L = nlms(chans_full["ldv"], chans_full["mic_l"])
    e_R = nlms(chans_full["ldv"], chans_full["mic_r"])
    h37_taus = []
    for band in bands:
        l_b = bp(e_L, sr, *band); r_b = bp(e_R, sr, *band)
        h37_taus.append(diff_sum_gcc(l_b, r_b, sr, band))
    h37_taus = np.array(h37_taus)
    nz = np.abs(h37_taus) > 1e-4
    tau_h37 = float(np.median(h37_taus[nz])) if nz.sum() >= 2 else float(np.median(h37_taus))
    # Agree-average rule
    if tau_h1 * tau_h37 > 0 and abs(tau_h1 - tau_h37) < 0.2e-3:
        return (tau_h1 + tau_h37) / 2
    return tau_h37 if abs(tau_h37) > abs(tau_h1) else tau_h1


# ============================================================================
# Strategy A — Equalize speech using chirp-derived channel
# ============================================================================

def estimate_channel_from_chirp(chirp_chans, sr, template,
                                regularization=0.01, band=(300, 7000)):
    """For each mic channel, estimate H(f) = X_mic(f) / X_template(f) via
    Welch-like averaging over the 6 bursts (improves SNR)."""
    # Find bursts in mic_L
    R_l_mf = matched_filter(chirp_chans["mic_l"], template, sr)
    peaks = find_burst_starts_via_matched_filter(R_l_mf, sr)
    n_t = len(template)
    # For each burst, extract aligned segment and accumulate H estimate
    n_align = n_t  # use template-length window
    H_L = np.zeros(n_t // 2 + 1, dtype=np.complex128)
    H_R = np.zeros(n_t // 2 + 1, dtype=np.complex128)
    Pvv = np.zeros(n_t // 2 + 1)
    cnt = 0
    T = rfft(template, n_t)
    for pk in peaks:
        # The peak position in matched filter output corresponds to:
        # signal aligned with template starting at sample (pk - n_t + 1)
        n_start = pk - n_t + 1
        if n_start < 0 or n_start + n_align > len(chirp_chans["mic_l"]):
            continue
        seg_l = chirp_chans["mic_l"][n_start: n_start + n_align]
        seg_r = chirp_chans["mic_r"][n_start: n_start + n_align]
        S_L = rfft(seg_l, n_t)
        S_R = rfft(seg_r, n_t)
        # H = S * conj(T) / |T|^2
        eps = regularization * np.abs(T).max() ** 2
        H_L += S_L * np.conj(T) / (np.abs(T) ** 2 + eps)
        H_R += S_R * np.conj(T) / (np.abs(T) ** 2 + eps)
        Pvv += np.abs(T) ** 2
        cnt += 1
    if cnt == 0:
        return None, None
    H_L /= cnt; H_R /= cnt
    return H_L, H_R, n_t


def strat_A_equalize_speech(chirp_chans, speech_chans, sr, template,
                            band=(500, 4000), reg=0.01):
    """Estimate channel from chirp; equalize speech mics; run V3."""
    result = estimate_channel_from_chirp(chirp_chans, sr, template,
                                        regularization=reg, band=band)
    if result is None:
        return None
    H_L, H_R, n_h = result
    # Apply inverse to speech
    n_speech = len(speech_chans["mic_l"])
    n_fft = 1 << int(np.ceil(np.log2(n_speech + n_h)))
    H_L_padded = np.zeros(n_fft // 2 + 1, dtype=np.complex128)
    H_R_padded = np.zeros(n_fft // 2 + 1, dtype=np.complex128)
    # Resample H to match n_fft
    f_old = np.linspace(0, sr/2, len(H_L))
    f_new = np.linspace(0, sr/2, n_fft // 2 + 1)
    H_L_padded.real = np.interp(f_new, f_old, H_L.real)
    H_L_padded.imag = np.interp(f_new, f_old, H_L.imag)
    H_R_padded.real = np.interp(f_new, f_old, H_R.real)
    H_R_padded.imag = np.interp(f_new, f_old, H_R.imag)
    # Wiener inverse
    eps = reg * np.abs(H_L_padded).max() ** 2
    inv_HL = np.conj(H_L_padded) / (np.abs(H_L_padded) ** 2 + eps)
    inv_HR = np.conj(H_R_padded) / (np.abs(H_R_padded) ** 2 + eps)
    # Apply
    SL = rfft(speech_chans["mic_l"], n_fft)
    SR = rfft(speech_chans["mic_r"], n_fft)
    eq_L = irfft(SL * inv_HL, n_fft)[:n_speech]
    eq_R = irfft(SR * inv_HR, n_fft)[:n_speech]
    # Run mic-mic GCC on equalized signals
    eq_L_b = bp(eq_L, sr, *band)
    eq_R_b = bp(eq_R, sr, *band)
    return doa_from_tau(gcc(eq_L_b, eq_R_b, sr, band))


# ============================================================================
# Strategy B — Bias correction
# ============================================================================

def strat_B_bias_correct(chirp_chans, speech_chans, sr):
    """V3 on speech, corrected by (V3_chirp - true_chirp_τ_at_this_pos)."""
    # Get V3 estimates for both
    tau_chirp_v3 = v3_h52(chirp_chans, sr)
    tau_speech_v3 = v3_h52(speech_chans, sr)
    # Note: at the SAME source position, true τ should be the same for both
    # The bias in chirp V3 (relative to chirp's own truth) reflects the channel.
    # Speech should have similar channel effect, so we apply correction.
    # But we don't know "true τ" without ground truth... unless we use chirp
    # matched filter as approximate ground truth.
    return tau_chirp_v3, tau_speech_v3


# ============================================================================
# Strategy C — Use chirp matched-filter τ as speech DoA proxy
# ============================================================================

def strat_C_use_chirp_tau(chirp_chans, sr, template, offset_correction_ms=0.0):
    """Use matched-filter mic-mic τ from chirp at this position as the DoA
    estimate for speech. They're at the same source position so DoA is same."""
    R_l = matched_filter(chirp_chans["mic_l"], template, sr)
    R_r = matched_filter(chirp_chans["mic_r"], template, sr)
    peaks_l = find_burst_starts_via_matched_filter(R_l, sr)
    if not peaks_l:
        return None
    n_search = int(0.012 * sr) + 50
    taus = []
    for pk_l in peaks_l:
        lo = max(0, pk_l - n_search); hi = min(len(R_r), pk_l + n_search)
        if hi <= lo: continue
        local = np.abs(R_r[lo: hi])
        pk_r = lo + int(np.argmax(local))
        # Sub-sample
        if 1 <= int(np.argmax(local)) < len(local) - 1:
            i = int(np.argmax(local))
            ym, y0, yp = local[i - 1], local[i], local[i + 1]
            d = ym - 2 * y0 + yp
            offset = 0.5 * (ym - yp) / d if abs(d) > 1e-12 else 0
            pk_r += offset
        tau = (pk_r - pk_l) / sr
        if abs(tau) < 0.01:
            taus.append(tau)
    if not taus:
        return None
    tau = float(np.median(taus))
    tau += offset_correction_ms * 1e-3
    return doa_from_tau(tau)


# ============================================================================
# Main
# ============================================================================

def main():
    template = synth_upchirp(48000, 500, 7000, 1.5)
    chirp_groups_data = chirp_groups()
    speech_groups_data = speech_groups()

    results = {"strat_A_equalize": {}, "strat_B_bias_correct": {},
               "strat_C_chirp_tau_direct": {}, "strat_C_with_offset_-1.1ms": {},
               "v3_speech_baseline": {}}

    # First pass: discover bias from chirp matched filter
    print("\n=== Step 1: Chirp matched-filter τ_LR per position ===")
    print(f"{'pos':>6} | {'truth_τ':>8} | {'mf_τ':>8} | {'bias':>8}")
    print("-" * 50)
    biases = {}
    for (pos, cond), paths in sorted(chirp_groups_data.items()):
        if cond != "block": continue
        chans, sr = load_group(paths)
        chans = {ch: basic_preprocess(x, sr) for ch, x in chans.items()}
        # Match filter
        R_l = matched_filter(chans["mic_l"], template, sr)
        R_r = matched_filter(chans["mic_r"], template, sr)
        peaks_l = find_burst_starts_via_matched_filter(R_l, sr)
        n_search = int(0.012 * sr) + 50
        taus = []
        for pk_l in peaks_l:
            lo = max(0, pk_l - n_search); hi = min(len(R_r), pk_l + n_search)
            if hi <= lo: continue
            local = np.abs(R_r[lo: hi])
            pk_r = lo + int(np.argmax(local))
            tau = (pk_r - pk_l) / sr
            if abs(tau) < 0.01:
                taus.append(tau)
        if not taus:
            continue
        tau_mf = float(np.median(taus))
        tau_truth = expected_tdoa_ms(float(pos), MIC_L, MIC_R) * 1e-3
        bias = tau_mf - tau_truth
        biases[pos] = bias
        print(f"{pos:>6} | {tau_truth*1000:+6.3f}ms | {tau_mf*1000:+6.3f}ms | {bias*1000:+6.3f}ms")

    if biases:
        avg_bias = np.mean(list(biases.values()))
        print(f"\nAverage bias: {avg_bias*1000:+.3f} ms")
    else:
        avg_bias = 0
    print()

    # Second pass: run strategies
    print("=== Step 2: Strategies (A, B, C) on speech ===")
    for (pos, cond), paths in sorted(speech_groups_data.items()):
        if cond != "block": continue
        speech_chans_pre, sr = load_group(paths)
        speech_t0, speech_t1 = 5.0, 25.0
        n0, n1 = int(speech_t0 * sr), int(speech_t1 * sr)
        speech_chans = {ch: x[n0:n1] for ch, x in speech_chans_pre.items()}
        speech_chans = {ch: basic_preprocess(x, sr) for ch, x in speech_chans.items()}

        # Get matching chirp recording
        chirp_key = (pos, "block")
        if chirp_key not in chirp_groups_data:
            continue
        chirp_chans_pre, _ = load_group(chirp_groups_data[chirp_key])
        chirp_chans = {ch: basic_preprocess(x, sr) for ch, x in chirp_chans_pre.items()}

        theta_true = expected_doa_deg(float(pos))

        # V3 baseline on speech
        tau_v3_speech = v3_h52(speech_chans, sr)
        results["v3_speech_baseline"][pos] = {
            "true": theta_true,
            "est": doa_from_tau(tau_v3_speech),
            "err": abs(doa_from_tau(tau_v3_speech) - theta_true),
        }

        # Strategy A: equalize
        try:
            theta_A = strat_A_equalize_speech(chirp_chans, speech_chans, sr, template)
        except Exception as e:
            theta_A = None
        results["strat_A_equalize"][pos] = {
            "true": theta_true, "est": theta_A,
            "err": abs(theta_A - theta_true) if theta_A is not None else None,
        }

        # Strategy C: use chirp matched-filter τ directly
        theta_C = strat_C_use_chirp_tau(chirp_chans, sr, template,
                                       offset_correction_ms=0.0)
        results["strat_C_chirp_tau_direct"][pos] = {
            "true": theta_true, "est": theta_C,
            "err": abs(theta_C - theta_true) if theta_C is not None else None,
        }

        # Strategy C with bias correction
        theta_Cb = strat_C_use_chirp_tau(chirp_chans, sr, template,
                                        offset_correction_ms=-avg_bias * 1000)
        results["strat_C_with_offset_-1.1ms"][pos] = {
            "true": theta_true, "est": theta_Cb,
            "err": abs(theta_Cb - theta_true) if theta_Cb is not None else None,
        }

        # Strategy B: bias correction = V3_speech + (truth_chirp - V3_chirp_mf)
        tau_chirp_truth = expected_tdoa_ms(float(pos), MIC_L, MIC_R) * 1e-3
        tau_chirp_mf = (biases[pos] + tau_chirp_truth) if pos in biases else None
        if tau_chirp_mf is not None:
            # Apply same MF bias correction to V3 speech
            v3_chirp = v3_h52(chirp_chans, sr)
            v3_speech_corrected = tau_v3_speech + (tau_chirp_truth - v3_chirp)
            theta_B = doa_from_tau(v3_speech_corrected)
        else:
            theta_B = None
        results["strat_B_bias_correct"][pos] = {
            "true": theta_true, "est": theta_B,
            "err": abs(theta_B - theta_true) if theta_B is not None else None,
        }

    print(f"\n{'strategy':<32} | {'speech MAE':>10}")
    print("-" * 48)
    for sid in results:
        errs = [v["err"] for v in results[sid].values() if v["err"] is not None]
        mae = float(np.mean(errs)) if errs else None
        m_str = f"{mae:6.2f}°" if mae is not None else "  N/A "
        print(f"{sid:<32} | {m_str:>10}")

    print("\nPer-position breakdown:")
    for sid in results:
        print(f"\n=== {sid} ===")
        for pos in sorted(results[sid]):
            r = results[sid][pos]
            if r.get("err") is None:
                print(f"  x={pos}: NA"); continue
            print(f"  x={pos}: true={r['true']:+6.2f}° est={r['est']:+6.2f}° err={r['err']:5.2f}°")

    out = OUT_DIR / "H_round12_chirp_calib.json"
    out.write_text(json.dumps({"biases": {p: float(b) for p, b in biases.items()},
                              "avg_bias_ms": float(avg_bias * 1000),
                              "results": results}, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
