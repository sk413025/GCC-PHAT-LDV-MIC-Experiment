"""Geometry constants extracted from paper main.tex (Section 2/3) and asset manifest.

Coordinate system: x parallel to barrier, y perpendicular (source side y<0, receiver y>0).
All distances in metres.
"""
import json
from pathlib import Path

C_MPS = 343.0
MIC_L = (-0.7, 2.0)
MIC_R = (+0.7, 2.0)
LDV = (0.0, 0.25)
SPEAKER_Y = 0.0
SPEAKER_XS = (-0.8, -0.4, 0.0, +0.4, +0.8)

# Speaker-id -> position label, derived from manifest filename pattern.
# Each (position, condition) appears in exactly one subdir; the manifest
# enumerates every WAV path and we just parse it.
REPO_ROOT = Path("/home/sbplab/jiawei/data-worktrees/exp-tdoa-cross-correlation")
DATASET_ROOT = REPO_ROOT / "dataset" / "0223"
MANIFEST_PATH = REPO_ROOT / "paper" / "repro_asset_manifest.json"


def euclid(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def load_table1_files():
    """Return list of dicts grouping wavs per (speaker_x, condition).

    Output: {(x_str, condition): {"mic_l": Path, "mic_r": Path, "ldv": Path or None}}
    condition ∈ {"block", "unblock"}.
    """
    manifest = json.loads(MANIFEST_PATH.read_text())
    paths = manifest["assets"]["table1"]["data_files"]
    groups = {}
    for rel in paths:
        full = DATASET_ROOT / rel
        # Filename pattern: 0223-{CHANNEL}-40-boy({POS}m)-{ID}-{block|unblock}.wav
        name = full.name
        cond = "block" if name.endswith("-block.wav") else "unblock"
        # Extract channel
        if "-MIC-LEFT-" in name:
            ch = "mic_l"
        elif "-MIC-RIGHT-" in name:
            ch = "mic_r"
        elif "-LDV-" in name:
            ch = "ldv"
        else:
            raise ValueError(f"Unknown channel in {name}")
        # Extract position string between "boy(" and "m)"
        i = name.index("boy(") + 4
        j = name.index("m)", i)
        pos_str = name[i:j]
        key = (pos_str, cond)
        groups.setdefault(key, {})[ch] = full
    return groups


def expected_tdoa_ms(speaker_x, mic_a, mic_b):
    """Free-space TDoA for source at (x, 0) between two mics.

    Returns t_b - t_a in milliseconds. Positive means signal hits mic_b later.
    """
    src = (speaker_x, SPEAKER_Y)
    da = euclid(src, mic_a)
    db = euclid(src, mic_b)
    return (db - da) / C_MPS * 1000.0


MIC_SPACING = abs(MIC_R[0] - MIC_L[0])  # 1.4 m


def tau_to_doa_deg(tau_ms):
    """Plane-wave (far-field) DoA from inter-mic delay.

    theta = arcsin(c * tau / d_mic). This is the convention paper uses.
    Positive tau (mic_R later than mic_L) means source on +x side, returns positive.
    """
    import math
    s = C_MPS * (tau_ms * 1e-3) / MIC_SPACING
    s = max(-1.0, min(1.0, s))
    # By convention, DoA sign matches speaker_x sign:
    # source at +0.8m -> mic_L sees later -> tau_LR = t_R - t_L < 0
    # So plane-wave DoA = -arcsin(c*tau_LR/d) to keep sign consistent with x.
    # We'll define the "ground truth tau" as t_R - t_L; positive => source on -x side.
    # Then theta_paper = arcsin(-c*tau/d) so source at +0.8 maps to +21°.
    return -math.degrees(math.asin(s))


def expected_doa_deg(speaker_x):
    """Plane-wave DoA matching paper's arcsin convention (±20.8° at ±0.8 m)."""
    tau = expected_tdoa_ms(speaker_x, MIC_L, MIC_R)
    return tau_to_doa_deg(tau)


if __name__ == "__main__":
    groups = load_table1_files()
    print(f"Found {len(groups)} (position, condition) groups:")
    for key in sorted(groups):
        chans = sorted(groups[key].keys())
        print(f"  {key}: channels={chans}")
    print()
    print("Expected TDoA (mic_R - mic_L) in ms and DoA in deg:")
    for x in SPEAKER_XS:
        tdoa = expected_tdoa_ms(x, MIC_L, MIC_R)
        doa = expected_doa_deg(x)
        print(f"  x={x:+.1f}m: TDoA={tdoa:+.4f} ms, DoA={doa:+.2f} deg")
