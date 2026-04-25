"""Round 13 — UNBLOCK as calibration (cleaner version of round 12).

Round 12 used BLOCK chirp matched-filter τ as calibration, but block chirp
itself has wall-multipath-induced bias.

UNBLOCK condition has NO BARRIER, so mic-mic GCC gives near-perfect τ_LR
(verified in Phase A: ~2.16° MAE).

Insight: if we record unblock at each position once (calibration phase),
then deploy with block (operation phase), unblock gives the geometric
ground-truth τ for each position.

Strategies tested:
  D1  Use unblock CHIRP τ directly as block speech DoA
  D2  Use unblock SPEECH τ directly as block speech DoA (literal cheat)
  D3  Block speech V3 + bias correction from (unblock_chirp_τ − V3_chirp_τ)
  D4  Block speech V3 + scale correction from unblock-block ratio
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
from _pigs import (MIC_L, MIC_R, C_MPS, MIC_SPACING,
                   preprocess as basic_preprocess)
from _pigs2 import auto_window_chirp
from _geometry import expected_doa_deg, REPO_ROOT
from h_round12_chirp_calib import v3_h52, gcc

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "strategies"


def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def doa_from_tau(tau):
    s = max(-1.0, min(1.0, C_MPS * tau / MIC_SPACING))
    return -float(np.degrees(np.arcsin(s)))


def get_unblock_tau_micmic(chans, sr, band=(500, 5000)):
    """Mic-mic GCC on unblock; band 500-5000 was shown to give 2.16° MAE."""
    chans_b = {ch: bp(x, sr, *band) for ch, x in chans.items()}
    return gcc(chans_b["mic_l"], chans_b["mic_r"], sr, band)


def main():
    chirp_g = chirp_groups()
    speech_g = speech_groups()

    print("\n=== Phase 1: Compute unblock τ_LR for each position (truth proxy) ===\n")
    print(f"{'pos':>5} | {'unblock_chirp_τ':>16} | {'unblock_speech_τ':>17} | {'truth_geom_τ':>13}")
    print("-" * 60)
    unblock_chirp_taus = {}
    unblock_speech_taus = {}
    truth_taus = {}
    for (pos, cond), paths in sorted(chirp_g.items()):
        if cond != "unblock": continue
        chans, sr = load_group(paths)
        chans = {ch: basic_preprocess(x, sr) for ch, x in chans.items()}
        # Auto-window chirp burst
        if "mic_l" in chans:
            t0, t1 = auto_window_chirp(chans["mic_l"], sr, dur_s=1.6)
            n0, n1 = int(t0 * sr), int(t1 * sr)
            chans = {ch: x[n0:n1] for ch, x in chans.items()}
        tau = get_unblock_tau_micmic(chans, sr)
        unblock_chirp_taus[pos] = tau

    for (pos, cond), paths in sorted(speech_g.items()):
        if cond != "unblock": continue
        chans, sr = load_group(paths)
        chans = {ch: basic_preprocess(x, sr) for ch, x in chans.items()}
        n0, n1 = int(5.0 * sr), int(25.0 * sr)
        chans = {ch: x[n0:n1] for ch, x in chans.items()}
        tau = get_unblock_tau_micmic(chans, sr)
        unblock_speech_taus[pos] = tau

    for pos in sorted(unblock_chirp_taus):
        from _geometry import expected_tdoa_ms
        tau_truth = expected_tdoa_ms(float(pos), MIC_L, MIC_R) * 1e-3
        truth_taus[pos] = tau_truth
        uc = unblock_chirp_taus[pos]
        us = unblock_speech_taus.get(pos, float('nan'))
        print(f"{pos:>5} | {uc*1000:+8.3f} ms     | {us*1000:+9.3f} ms       | {tau_truth*1000:+8.3f} ms")

    # Phase 2: Apply unblock-derived τ as DoA estimates for BLOCK speech
    print("\n=== Phase 2: Test calibration strategies on block speech ===\n")
    results = {
        "D1_unblock_chirp_tau": {},
        "D2_unblock_speech_tau": {},
        "D3_v3_block_speech": {},
        "D4_v3_with_chirp_bias_corr": {},
        "D5_v3_with_speech_bias_corr": {},
    }
    for (pos, cond), paths in sorted(speech_g.items()):
        if cond != "block": continue
        chans, sr = load_group(paths)
        chans = {ch: basic_preprocess(x, sr) for ch, x in chans.items()}
        n0, n1 = int(5.0 * sr), int(25.0 * sr)
        chans = {ch: x[n0:n1] for ch, x in chans.items()}
        theta_true = expected_doa_deg(float(pos))

        # D1: Use unblock chirp τ directly
        if pos in unblock_chirp_taus:
            theta_D1 = doa_from_tau(unblock_chirp_taus[pos])
            results["D1_unblock_chirp_tau"][pos] = {
                "true": theta_true, "est": theta_D1,
                "err": abs(theta_D1 - theta_true)}

        # D2: Use unblock speech τ directly
        if pos in unblock_speech_taus:
            theta_D2 = doa_from_tau(unblock_speech_taus[pos])
            results["D2_unblock_speech_tau"][pos] = {
                "true": theta_true, "est": theta_D2,
                "err": abs(theta_D2 - theta_true)}

        # D3: V3 baseline
        tau_v3 = v3_h52(chans, sr)
        theta_D3 = doa_from_tau(tau_v3)
        results["D3_v3_block_speech"][pos] = {
            "true": theta_true, "est": theta_D3,
            "err": abs(theta_D3 - theta_true)}

        # D4: V3 corrected by chirp bias (V3_block_chirp - unblock_chirp)
        block_chirp_key = (pos, "block")
        if block_chirp_key in chirp_g and pos in unblock_chirp_taus:
            chirp_chans, _ = load_group(chirp_g[block_chirp_key])
            chirp_chans = {ch: basic_preprocess(x, sr) for ch, x in chirp_chans.items()}
            t0, t1 = auto_window_chirp(chirp_chans["mic_l"], sr, dur_s=1.6)
            n0, n1 = int(t0 * sr), int(t1 * sr)
            chirp_chans = {ch: x[n0:n1] for ch, x in chirp_chans.items()}
            tau_v3_chirp_block = v3_h52(chirp_chans, sr)
            bias = tau_v3_chirp_block - unblock_chirp_taus[pos]
            tau_D4 = tau_v3 - bias
            theta_D4 = doa_from_tau(tau_D4)
            results["D4_v3_with_chirp_bias_corr"][pos] = {
                "true": theta_true, "est": theta_D4,
                "err": abs(theta_D4 - theta_true),
                "bias_ms": bias * 1000}

        # D5: V3 corrected by speech bias (V3_block_speech - unblock_speech)
        if pos in unblock_speech_taus:
            bias = tau_v3 - unblock_speech_taus[pos]
            # NB: this requires KNOWING speech truth; only useful as oracle
            # tau_D5 = tau_v3 - bias = unblock_speech_taus[pos]
            # → same as D2!  Skip
            pass

    print(f"{'strategy':<32} | {'speech MAE':>10}")
    print("-" * 48)
    for sid in results:
        errs = [v["err"] for v in results[sid].values() if v.get("err") is not None]
        mae = float(np.mean(errs)) if errs else None
        m_str = f"{mae:6.2f}°" if mae is not None else "  N/A "
        print(f"{sid:<32} | {m_str:>10}")

    print("\nPer-position breakdown:")
    for sid in results:
        if not results[sid]:
            continue
        print(f"\n=== {sid} ===")
        for pos in sorted(results[sid]):
            r = results[sid][pos]
            if r.get("err") is None:
                print(f"  x={pos}: NA"); continue
            extra = f" bias={r.get('bias_ms', 0):.2f}ms" if 'bias_ms' in r else ""
            print(f"  x={pos}: true={r['true']:+6.2f}° est={r['est']:+6.2f}° err={r['err']:5.2f}°{extra}")

    out = OUT_DIR / "H_round13_unblock_calib.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
