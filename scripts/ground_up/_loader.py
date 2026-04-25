"""Unified loader for chirp (0223) and speech (0224) datasets.

Returns dict keyed by (pos_str, cond) -> {"mic_l", "mic_r", "ldv": Path or None}.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf

CHIRP_ROOT = Path("/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation/dataset/0223")
SPEECH_ROOT = Path("/home/sbplab/jiawei/speech")


def _channel(name: str) -> str | None:
    if "-MIC-LEFT-" in name:
        return "mic_l"
    if "-MIC-RIGHT-" in name:
        return "mic_r"
    if "-LDV-" in name:
        return "ldv"
    return None


def _condition(name: str) -> str | None:
    if name.endswith("-block.wav"):
        return "block"
    if name.endswith("-unblock.wav"):
        return "unblock"
    return None


def _position(name: str) -> str | None:
    try:
        i = name.index("boy(") + 4
        j = name.index("m)", i)
        return name[i:j]
    except ValueError:
        return None


def discover_groups(root: Path):
    groups = {}
    for sub in sorted(root.rglob("*.wav")):
        name = sub.name
        pos, cond, ch = _position(name), _condition(name), _channel(name)
        if pos is None or cond is None or ch is None:
            continue
        # Skip duplicates: prefer the first encountered
        groups.setdefault((pos, cond), {})
        if ch not in groups[(pos, cond)]:
            groups[(pos, cond)][ch] = sub
    return groups


def chirp_groups():
    return discover_groups(CHIRP_ROOT)


def speech_groups():
    return discover_groups(SPEECH_ROOT)


def load_mono(path: Path, target_sr: int | None = None):
    """Read mono float64. If target_sr given, polyphase resample."""
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    if target_sr is not None and sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g)
        sr = target_sr
    return x, sr


def load_group(paths: dict, target_sr: int | None = None):
    """Return dict ch -> (x, sr). Channels missing are absent."""
    out = {}
    for ch, p in paths.items():
        x, sr = load_mono(p, target_sr)
        out[ch] = (x, sr)
    # Verify same sr across channels
    srs = {v[1] for v in out.values()}
    if len(srs) != 1:
        raise ValueError(f"sample rate mismatch in {paths}: {srs}")
    sr = srs.pop()
    # Trim to common length
    n = min(len(v[0]) for v in out.values())
    return {ch: x[:n] for ch, (x, _) in out.items()}, sr


if __name__ == "__main__":
    cg = chirp_groups()
    sg = speech_groups()
    print(f"chirp groups: {len(cg)}; speech groups: {len(sg)}")
    for k in sorted(cg):
        print(f"  CHIRP  {k}: {sorted(cg[k].keys())}")
    for k in sorted(sg):
        print(f"  SPEECH {k}: {sorted(sg[k].keys())}")
