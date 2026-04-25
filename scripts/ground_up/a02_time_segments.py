"""Phase A.2 — verify chirp/speech time-segment structure.

Paper says chirp at t=0-2s, speech at t=3-8s. Let's verify by computing per-channel
short-term energy envelopes.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _geometry import load_table1_files, REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "audit"


def envelope(x, sr, win_ms=20):
    n = int(sr * win_ms / 1000)
    e = np.sqrt(np.convolve(x ** 2, np.ones(n) / n, mode="same"))
    return e


def main():
    groups = load_table1_files()
    fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
    rows = sorted({k[0] for k in groups})
    for r, pos in enumerate(rows):
        for c, cond in enumerate(("block", "unblock")):
            key = (pos, cond)
            if key not in groups:
                continue
            ax = axes[r][c]
            paths = groups[key]
            for ch, color in (("mic_l", "tab:blue"), ("mic_r", "tab:orange"), ("ldv", "tab:green")):
                if ch not in paths:
                    continue
                x, sr = sf.read(str(paths[ch]))
                if x.ndim > 1:
                    x = x.mean(axis=1)
                t = np.arange(len(x)) / sr
                e = envelope(x.astype(np.float64), sr) / (np.max(np.abs(x)) + 1e-12)
                ax.plot(t, e, label=ch.upper(), alpha=0.8, color=color, lw=0.8)
            ax.set_title(f"x={pos} m, {cond}", fontsize=9)
            ax.axvspan(0, 2, alpha=0.1, color="red", label="chirp window" if (r == 0 and c == 0) else None)
            ax.axvspan(3, 8, alpha=0.1, color="green", label="speech window" if (r == 0 and c == 0) else None)
            ax.set_xlim(0, 14)
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("Normalized envelopes — chirp (0-2s) and speech (3-8s) windows", fontsize=11)
    fig.tight_layout()
    out_png = OUT_DIR / "envelopes.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
