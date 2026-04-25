"""Phase A.4 — inspect 0224 speech dataset at /home/sbplab/jiawei/speech/.

Key discovery: 0223 dataset under worktree contains ONLY chirps. The actual speech
recordings live OUTSIDE the worktree at /home/sbplab/jiawei/speech/ with 0224 prefix.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _geometry import REPO_ROOT

SPEECH_ROOT = Path("/home/sbplab/jiawei/speech")
OUT_DIR = REPO_ROOT / "results" / "ground_up" / "audit"


def find_speech_groups():
    groups = {}
    for sub in sorted(SPEECH_ROOT.iterdir()):
        if not sub.is_dir():
            continue
        # name like "block-2(high)"
        cond = "block" if sub.name.startswith("block-") else (
            "unblock" if sub.name.startswith("unblock-") else None)
        if cond is None:
            continue
        for f in sub.glob("*.wav"):
            name = f.name
            i = name.index("boy(") + 4
            j = name.index("m)", i)
            pos = name[i:j]
            if "-MIC-LEFT-" in name:
                ch = "mic_l"
            elif "-MIC-RIGHT-" in name:
                ch = "mic_r"
            elif "-LDV-" in name:
                ch = "ldv"
            else:
                continue
            groups.setdefault((pos, cond), {})[ch] = f
    return groups


def main():
    groups = find_speech_groups()
    print(f"Speech groups found: {len(groups)}")
    for k in sorted(groups):
        print(f"  {k}: channels={sorted(groups[k].keys())}")

    # Spectrogram of one speech block recording
    target = ("+0.0", "block")
    paths = groups[target]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, ch in zip(axes, ("mic_l", "mic_r", "ldv")):
        x, sr = sf.read(str(paths[ch]))
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x.astype(np.float64)
        f, t, Sxx = signal.spectrogram(x, fs=sr, window="hann", nperseg=2048,
                                       noverlap=1536, scaling="density")
        db = 10 * np.log10(Sxx + 1e-12)
        vmin = np.percentile(db, 30); vmax = np.percentile(db, 99)
        im = ax.pcolormesh(t, f, db, cmap="magma", vmin=vmin, vmax=vmax, shading="auto")
        ax.set_ylim(0, 4000)
        ax.set_ylabel(f"{ch.upper()} (Hz)")
        fig.colorbar(im, ax=ax)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Speech recording — x={target[0]} m, {target[1]}")
    fig.tight_layout()
    out = OUT_DIR / f"speech_spec_{target[0]}_{target[1]}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Wrote {out.name}")

    # Also envelope summary across all 5 positions
    fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
    rows = sorted({k[0] for k in groups})
    for r, pos in enumerate(rows):
        for c, cond in enumerate(("block", "unblock")):
            key = (pos, cond)
            if key not in groups:
                continue
            ax = axes[r][c]
            for ch, color in (("mic_l", "tab:blue"), ("mic_r", "tab:orange"), ("ldv", "tab:green")):
                if ch not in groups[key]:
                    continue
                x, sr = sf.read(str(groups[key][ch]))
                if x.ndim > 1:
                    x = x.mean(axis=1)
                x = x.astype(np.float64)
                tt = np.arange(len(x)) / sr
                n = int(sr * 0.02)
                e = np.sqrt(np.convolve(x ** 2, np.ones(n) / n, mode="same"))
                e /= np.max(np.abs(x)) + 1e-12
                ax.plot(tt, e, label=ch.upper(), alpha=0.8, color=color, lw=0.8)
            ax.set_title(f"speech x={pos} m, {cond}", fontsize=9)
            ax.set_xlim(0, 30)
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Speech (0224) — normalized envelopes")
    fig.tight_layout()
    out2 = OUT_DIR / "speech_envelopes.png"
    fig.savefig(out2, dpi=110)
    plt.close(fig)
    print(f"Wrote {out2.name}")


if __name__ == "__main__":
    main()
