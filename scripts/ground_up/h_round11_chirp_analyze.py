"""Round 11 step 1 — reverse-engineer chirp parameters from data.

Examine LDV (strongest signal) at each position to find:
  - Sweep direction (up/down)
  - Start / end frequencies
  - Duration of each burst
  - Inter-burst spacing
  - Number of bursts per recording
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy import signal as sp
import matplotlib.pyplot as plt

from _loader import chirp_groups, load_group
from _geometry import REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "audit"


def main():
    groups = chirp_groups()
    print(f"{'pos':>5} | {'burst_starts':<35} | {'period':>8} | {'duration':>9} | {'f_sweep':>15}")
    print("-" * 90)

    for (pos, cond), paths in sorted(groups.items()):
        if cond != "block":
            continue
        chans, sr = load_group(paths)
        x = chans["ldv"]  # strongest channel for chirp detection
        n = len(x)

        # Energy envelope to find bursts
        e = np.abs(hilbert(x))
        n_smooth = int(0.005 * sr)
        e = np.convolve(e, np.ones(n_smooth) / n_smooth, mode="same")
        thresh = 0.2 * e.max()
        above = e > thresh

        # Find rising edges
        rising = np.where(np.diff(above.astype(int)) > 0)[0]
        falling = np.where(np.diff(above.astype(int)) < 0)[0]

        # Filter very short bursts (< 200ms)
        burst_starts = []
        burst_ends = []
        for r in rising:
            # find next falling after r
            f_after = falling[falling > r]
            if len(f_after) == 0:
                continue
            f_idx = f_after[0]
            if (f_idx - r) / sr > 0.2:  # > 200ms
                burst_starts.append(r / sr)
                burst_ends.append(f_idx / sr)

        starts_str = ",".join(f"{s:.2f}" for s in burst_starts[:6])
        if len(burst_starts) >= 2:
            period = burst_starts[1] - burst_starts[0]
            duration = burst_ends[0] - burst_starts[0]
        else:
            period = float("nan")
            duration = burst_ends[0] - burst_starts[0] if burst_starts else float("nan")

        # Estimate frequency sweep within first burst
        if burst_starts:
            n0 = int(burst_starts[0] * sr)
            n1 = int((burst_starts[0] + duration) * sr)
            burst_x = x[n0:n1]
            # bandpass to remove DC and high-freq noise
            burst_x = filtfilt(*butter(4, [50/(sr/2), 8000/(sr/2)], btype="band"), burst_x)
            # Instantaneous frequency from analytic signal
            burst_a = hilbert(burst_x)
            inst_phase = np.unwrap(np.angle(burst_a))
            inst_freq = np.gradient(inst_phase) * sr / (2 * np.pi)
            # Filter out outliers
            inst_freq_smooth = np.convolve(np.abs(inst_freq),
                                          np.ones(int(0.01*sr))/int(0.01*sr),
                                          mode="same")
            f_first = float(np.median(inst_freq_smooth[:int(0.05 * len(burst_x))]))
            f_last = float(np.median(inst_freq_smooth[-int(0.05 * len(burst_x)):]))
            f_sweep = f"{f_first:.0f}→{f_last:.0f} Hz"
        else:
            f_sweep = "N/A"

        print(f"{pos:>5} | {starts_str:<35} | {period:>6.3f}s | {duration:>7.3f}s | {f_sweep:>15}")

    # Plot one position's spectrogram with detected bursts overlaid
    for target in [("+0.0", "block"), ("+0.4", "block"), ("-0.4", "block")]:
        if target not in groups:
            continue
        chans, sr = load_group(groups[target])
        x = chans["ldv"]
        f, t, Sxx = sp.spectrogram(x, fs=sr, nperseg=2048, noverlap=1536)
        fig, ax = plt.subplots(figsize=(14, 4))
        db = 10 * np.log10(Sxx + 1e-12)
        ax.pcolormesh(t, f, db, cmap="magma", vmin=np.percentile(db, 30),
                     vmax=np.percentile(db, 99), shading="auto")
        ax.set_ylim(0, 8000)
        ax.set_title(f"LDV spectrogram at x={target[0]} m, {target[1]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        out = OUT_DIR / f"chirp_param_inspect_{target[0]}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
