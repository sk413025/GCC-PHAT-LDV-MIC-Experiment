"""Phase B.2 — synthetic symbol/sign verification.

Place a known source, simulate ideal free-space delays at LDV/MIC_L/MIC_R, run
PI-GS, verify recovered position & DoA match. This nails down the sign convention
before touching real data.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from _pigs import (
    cross_phat, estimate_doa_pigs, estimate_doa_micmic,
    expected_tau_VM, expected_tau_LR,
    MIC_L, MIC_R, LDV, C_MPS,
)


def synth_chirp(sr=48000, dur=2.0, f0=500, f1=6000):
    t = np.arange(int(sr * dur)) / sr
    # Linear sweep
    return 0.5 * np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t ** 2))


def delay_signal(s, sr, delay_s):
    n_delay = int(round(delay_s * sr))
    if n_delay >= 0:
        out = np.zeros(len(s) + n_delay)
        out[n_delay:n_delay + len(s)] = s
    else:
        out = np.zeros(len(s) - n_delay)
        out[:len(s) + n_delay] = s[-n_delay:]  # n_delay negative
    return out


def simulate(source_pos, sr=48000, dur=2.0, snr_db=40):
    s = synth_chirp(sr=sr, dur=dur)
    d_v = np.hypot(source_pos[0] - LDV[0], source_pos[1] - LDV[1])
    d_l = np.hypot(source_pos[0] - MIC_L[0], source_pos[1] - MIC_L[1])
    d_r = np.hypot(source_pos[0] - MIC_R[0], source_pos[1] - MIC_R[1])
    t_v, t_l, t_r = d_v / C_MPS, d_l / C_MPS, d_r / C_MPS

    # Pad and apply per-receiver delay (use minimum delay as zero reference)
    t_min = min(t_v, t_l, t_r)
    n_pad = int(np.ceil(max(t_v, t_l, t_r) * sr)) + 100
    n_total = len(s) + n_pad
    base = np.zeros(n_total)
    base[:len(s)] = s

    def shift(sig, dt):
        n = int(round(dt * sr))
        out = np.zeros_like(sig)
        if n >= 0:
            out[n:] = sig[:n_total - n]
        else:
            out[:n] = sig[-n:]
        return out

    x_v = shift(base, t_v - t_min)
    x_l = shift(base, t_l - t_min)
    x_r = shift(base, t_r - t_min)

    # Add noise
    sig_p = np.mean(s ** 2)
    n_p = sig_p / (10 ** (snr_db / 10))
    rng = np.random.default_rng(42)
    x_v = x_v + rng.normal(scale=np.sqrt(n_p), size=x_v.shape)
    x_l = x_l + rng.normal(scale=np.sqrt(n_p), size=x_l.shape)
    x_r = x_r + rng.normal(scale=np.sqrt(n_p), size=x_r.shape)
    return x_l, x_r, x_v, sr


def main():
    print("=" * 72)
    print("Phase B.2 symbol & geometry sanity (synthetic)")
    print("=" * 72)
    print(f"\nLDV at {LDV}, MIC_L at {MIC_L}, MIC_R at {MIC_R}, c={C_MPS}\n")

    test_positions = [(-0.8, 0.0), (-0.4, 0.0), (0.0, 0.0), (0.4, 0.0), (0.8, 0.0)]
    print(f"{'src_x':>6} | {'τ_VL_exp':>10} {'τ_VL_meas':>10} | "
          f"{'τ_VR_exp':>10} {'τ_VR_meas':>10} | "
          f"{'θ_exp':>7} {'θ_pigs':>7} {'p_hat':>14}")
    print("-" * 100)

    fails = 0
    for src in test_positions:
        x_l, x_r, x_v, sr = simulate(src, snr_db=40)
        # Check R_VL peak
        lags_vl, R_vl = cross_phat(x_v, x_l, sr, max_lag_s=0.01)
        peak_vl = lags_vl[int(np.argmax(np.abs(R_vl)))]
        tau_vl_exp = expected_tau_VM(src, MIC_L)

        lags_vr, R_vr = cross_phat(x_v, x_r, sr, max_lag_s=0.01)
        peak_vr = lags_vr[int(np.argmax(np.abs(R_vr)))]
        tau_vr_exp = expected_tau_VM(src, MIC_R)

        out = estimate_doa_pigs(x_l, x_r, x_v, sr,
                               grid_kwargs=dict(x_lo=-1.5, x_hi=1.5, x_step=0.005,
                                                y_lo=-0.05, y_hi=0.5, y_step=0.05))

        # Plane-wave expected angle
        from _geometry import expected_doa_deg
        theta_exp = expected_doa_deg(src[0])
        print(f"{src[0]:+6.2f} | {tau_vl_exp*1000:+8.3f}ms {peak_vl*1000:+8.3f}ms | "
              f"{tau_vr_exp*1000:+8.3f}ms {peak_vr*1000:+8.3f}ms | "
              f"{theta_exp:+6.2f}° {out['theta_deg']:+6.2f}° "
              f"({out['p_hat'][0]:+.2f},{out['p_hat'][1]:+.2f})")

        # Verify within tolerance
        if abs(peak_vl - tau_vl_exp) > 0.0005:
            print("  !! R_VL peak mismatch")
            fails += 1
        if abs(peak_vr - tau_vr_exp) > 0.0005:
            print("  !! R_VR peak mismatch")
            fails += 1
        if abs(out["theta_deg"] - theta_exp) > 1.0:
            print("  !! DoA mismatch")
            fails += 1

    print("\nMic-only baseline:")
    print(f"{'src_x':>6} | {'τ_LR_exp':>10} {'τ_LR_meas':>10} | "
          f"{'θ_exp':>7} {'θ_meas':>7}")
    print("-" * 60)
    for src in test_positions:
        x_l, x_r, x_v, sr = simulate(src, snr_db=40)
        theta_meas, tau_meas, _, _ = estimate_doa_micmic(x_l, x_r, sr)
        from _geometry import expected_doa_deg
        theta_exp = expected_doa_deg(src[0])
        tau_exp = expected_tau_LR(src)
        print(f"{src[0]:+6.2f} | {tau_exp*1000:+8.3f}ms {tau_meas*1000:+8.3f}ms | "
              f"{theta_exp:+6.2f}° {theta_meas:+6.2f}°")
        if abs(tau_meas - tau_exp) > 0.0005:
            print("  !! mic-mic τ mismatch")
            fails += 1

    print()
    if fails == 0:
        print("OK — all symbol/sign conventions verified.")
    else:
        print(f"FAIL — {fails} sanity assertions broke. Inspect signs.")
    return fails


if __name__ == "__main__":
    sys.exit(main())
