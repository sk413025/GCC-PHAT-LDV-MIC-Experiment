"""Phase A.3 — high-resolution spectrogram of a few representative recordings.

Goal: see what's actually IN the time-frequency plane, since the envelope plot
suggested multiple repetitions (paper claims chirp 0-2s 3x, speech 3-8s).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _geometry import load_table1_files, REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "audit"

INSPECT = [("+0.0", "block"), ("+0.0", "unblock"), ("-0.4", "block")]


def main():
    groups = load_table1_files()
    for pos, cond in INSPECT:
        key = (pos, cond)
        paths = groups.get(key)
        if not paths:
            continue
        chs = ["mic_l", "mic_r"] + (["ldv"] if "ldv" in paths else [])
        fig, axes = plt.subplots(len(chs), 1, figsize=(14, 3 * len(chs)), sharex=True)
        if len(chs) == 1:
            axes = [axes]
        for ax, ch in zip(axes, chs):
            x, sr = sf.read(str(paths[ch]))
            if x.ndim > 1:
                x = x.mean(axis=1)
            x = x.astype(np.float64)
            f, t, Sxx = signal.spectrogram(x, fs=sr, window="hann", nperseg=2048,
                                           noverlap=1024, scaling="density")
            db = 10 * np.log10(Sxx + 1e-12)
            vmin = np.percentile(db, 30)
            vmax = np.percentile(db, 99)
            im = ax.pcolormesh(t, f, db, cmap="magma", vmin=vmin, vmax=vmax, shading="auto")
            ax.set_ylim(0, 8000)
            ax.set_ylabel(f"{ch.upper()}\nHz")
            fig.colorbar(im, ax=ax, label="dB")
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Spectrogram — x={pos} m, {cond}")
        fig.tight_layout()
        out = OUT_DIR / f"spec_{pos}_{cond}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
