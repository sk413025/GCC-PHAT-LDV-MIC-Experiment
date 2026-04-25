"""Phase B.3 — diagnostic: plot R_VL and R_VR for real recordings against expected τ.

If the GCC peaks fall ANYWHERE NEAR the expected τ_VL/τ_VR for the true source
position, PI-GS has a chance. If they're elsewhere, the cross-modal alignment
assumption is broken and we need different strategies.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from _loader import chirp_groups, speech_groups, load_group
from _pigs import cross_phat, expected_tau_VM, MIC_L, MIC_R, preprocess
from _geometry import REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "audit"


def bp(x, sr, lo, hi, order=4):
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return filtfilt(b, a, x)


def plot_for_groups(groups, signal_type, t_window, band_hz):
    fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
    rows = sorted({k[0] for k in groups})
    for r, pos in enumerate(rows):
        key = (pos, "block")
        if key not in groups:
            continue
        chans, sr = load_group(groups[key])
        n0 = int(t_window[0] * sr); n1 = int(t_window[1] * sr)
        chans = {ch: x[n0:n1] for ch, x in chans.items()}
        chans = {ch: preprocess(x, sr) for ch, x in chans.items()}
        if band_hz is not None:
            chans = {ch: bp(x, sr, *band_hz) for ch, x in chans.items()}

        x_l, x_r, x_v = chans["mic_l"], chans["mic_r"], chans["ldv"]
        for c, mic_label, mic_pos, x_m in [(0, "MIC_L", MIC_L, x_l), (1, "MIC_R", MIC_R, x_r)]:
            lags, R = cross_phat(x_v, x_m, sr, max_lag_s=0.012, band_hz=band_hz)
            ax = axes[r][c]
            ax.plot(lags * 1000, np.abs(R) / (np.abs(R).max() + 1e-12), lw=0.8)
            tau_exp = expected_tau_VM((float(pos), 0.0), mic_pos)
            ax.axvline(tau_exp * 1000, color="red", linestyle="--",
                       label=f"τ_exp={tau_exp*1000:+.2f}ms")
            ax.set_title(f"R_V-{mic_label}  x={pos} m", fontsize=9)
            ax.set_xlabel("lag (ms)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    fig.suptitle(f"{signal_type} — cross-modal GCC-PHAT magnitude (band={band_hz})")
    fig.tight_layout()
    out = OUT_DIR / f"diag_R_{signal_type}_{band_hz}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Wrote {out.name}")


def main():
    plot_for_groups(chirp_groups(), "chirp", (0.0, 2.0), (500, 2000))
    plot_for_groups(speech_groups(), "speech", (5.0, 25.0), (500, 2000))
    plot_for_groups(chirp_groups(), "chirp", (0.0, 2.0), None)
    plot_for_groups(speech_groups(), "speech", (5.0, 25.0), None)


if __name__ == "__main__":
    main()
