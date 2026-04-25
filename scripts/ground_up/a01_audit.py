"""Phase A — signal forensics on the 25 table1 WAVs.

For each (position, condition) group:
  - PSD overlay (LDV vs Mic_L vs Mic_R)
  - LDV-MicL and LDV-MicR magnitude-squared coherence
  - Detect dominant tonal peaks (likely AC hum + speaker harmonics)
  - Estimate noise floor band
  - Check clipping / DC offset / level statistics

Output:
  results/ground_up/audit/{pos}_{cond}.png
  results/ground_up/audit/audit_summary.md
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _geometry import load_table1_files, REPO_ROOT, expected_tdoa_ms, MIC_L, MIC_R, LDV

OUT_DIR = REPO_ROOT / "results" / "ground_up" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_mono(path: Path):
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float64), sr


def stat_dict(x: np.ndarray, sr: int) -> dict:
    abs_max = float(np.max(np.abs(x))) if x.size else 0.0
    return {
        "n_samples": int(x.size),
        "duration_s": float(x.size / sr),
        "sample_rate": int(sr),
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "abs_max": abs_max,
        "dc_offset": float(np.mean(x)),
        "clip_ratio": float(np.mean(np.abs(x) >= 0.999 * abs_max if abs_max > 0 else 0)),
    }


def welch_psd(x: np.ndarray, sr: int, nperseg=8192):
    f, p = signal.welch(x, fs=sr, nperseg=min(nperseg, len(x)), noverlap=nperseg // 2,
                        window="hann", scaling="density")
    return f, p


def find_tonal_peaks(f, p, n_peaks=8, min_freq=20.0, max_freq=8000.0):
    """Return (freq_hz, power_db) of top-n tonal peaks above local median."""
    mask = (f >= min_freq) & (f <= max_freq)
    fp, pp = f[mask], p[mask]
    pdb = 10.0 * np.log10(pp + 1e-30)
    # Local prominence: subtract running median over ~50 Hz window
    df = fp[1] - fp[0]
    win = max(5, int(50.0 / df))
    if win % 2 == 0:
        win += 1
    from scipy.ndimage import median_filter
    floor = median_filter(pdb, size=win)
    prom = pdb - floor
    # Peak picking
    peaks, _ = signal.find_peaks(prom, prominence=6.0, distance=int(20.0 / df))
    if len(peaks) == 0:
        return []
    order = np.argsort(prom[peaks])[::-1][:n_peaks]
    return [(float(fp[peaks[i]]), float(pdb[peaks[i]]), float(prom[peaks[i]])) for i in order]


def coherence(x, y, sr, nperseg=8192):
    f, c = signal.coherence(x, y, fs=sr, nperseg=min(nperseg, min(len(x), len(y))),
                            noverlap=nperseg // 2, window="hann")
    return f, c


def usable_band(f, c, threshold=0.3):
    """Return (lo, hi) Hz where coherence stays above threshold over largest contiguous span."""
    above = c >= threshold
    if not above.any():
        return None
    # Find longest run
    runs = []
    start = None
    for i, a in enumerate(above):
        if a and start is None:
            start = i
        elif not a and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(above)))
    if not runs:
        return None
    best = max(runs, key=lambda r: r[1] - r[0])
    return float(f[best[0]]), float(f[best[1] - 1])


def audit_group(key, paths, summary_lines):
    pos_str, cond = key
    chans = {}
    sr_set = set()
    for ch in ("mic_l", "mic_r", "ldv"):
        if ch in paths:
            x, sr = load_mono(paths[ch])
            chans[ch] = x
            sr_set.add(sr)
    if len(sr_set) > 1:
        raise ValueError(f"sample rate mismatch in {key}: {sr_set}")
    sr = next(iter(sr_set))

    # Stats
    stats = {ch: stat_dict(x, sr) for ch, x in chans.items()}

    # Trim to common length
    n = min(len(v) for v in chans.values())
    chans = {k: v[:n] for k, v in chans.items()}

    # PSDs
    psds = {ch: welch_psd(x, sr) for ch, x in chans.items()}
    peaks = {ch: find_tonal_peaks(*psds[ch]) for ch in chans}

    # Coherence (only if LDV present)
    coh_band = {}
    cohs = {}
    if "ldv" in chans:
        for ch in ("mic_l", "mic_r"):
            if ch in chans:
                f, c = coherence(chans["ldv"], chans[ch], sr)
                cohs[ch] = (f, c)
                coh_band[ch] = usable_band(f, c, threshold=0.3)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax = axes[0]
    colors = {"mic_l": "tab:blue", "mic_r": "tab:orange", "ldv": "tab:green"}
    for ch, (f, p) in psds.items():
        ax.semilogy(f, p, label=ch.upper(), alpha=0.8, color=colors[ch])
    ax.set_xlim(0, 8000)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(f"PSD — speaker x={pos_str} m, {cond}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for ch, (f, c) in cohs.items():
        ax.plot(f, c, label=f"LDV vs {ch.upper()}", alpha=0.8, color=colors[ch])
    ax.set_xlim(0, 8000)
    ax.set_ylim(0, 1)
    ax.axhline(0.3, color="red", linestyle="--", alpha=0.5, label="coh threshold 0.3")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("MS Coherence")
    ax.set_title(f"LDV-Mic coherence — speaker x={pos_str} m, {cond}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = OUT_DIR / f"{pos_str}_{cond}.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    # Summary block
    summary_lines.append(f"\n## x={pos_str} m, {cond}\n")
    summary_lines.append(f"- Sample rate: {sr} Hz; duration: {stats[next(iter(stats))]['duration_s']:.2f} s")
    summary_lines.append(f"- Expected geometric TDoA(R-L): {expected_tdoa_ms(float(pos_str), MIC_L, MIC_R):+.4f} ms")
    for ch in ("mic_l", "mic_r", "ldv"):
        if ch in stats:
            s = stats[ch]
            summary_lines.append(f"- {ch.upper()}: rms={s['rms']:.4f}, "
                                 f"abs_max={s['abs_max']:.4f}, "
                                 f"DC={s['dc_offset']:+.4e}, clip_ratio={s['clip_ratio']:.4f}")
    summary_lines.append("- Top tonal peaks (Hz @ dB prominence):")
    for ch in ("mic_l", "mic_r", "ldv"):
        if ch in peaks and peaks[ch]:
            top = ", ".join(f"{f:.1f}Hz(+{p:.1f}dB)" for f, _, p in peaks[ch][:5])
            summary_lines.append(f"  - {ch.upper()}: {top}")
    if coh_band:
        summary_lines.append("- Useable LDV-Mic coherence band (γ²>0.3):")
        for ch, band in coh_band.items():
            if band:
                summary_lines.append(f"  - LDV-{ch.upper()}: {band[0]:.0f}–{band[1]:.0f} Hz")
            else:
                summary_lines.append(f"  - LDV-{ch.upper()}: NONE (no band crosses threshold)")

    return {"stats": stats, "peaks": peaks, "coh_band": coh_band}


def main():
    groups = load_table1_files()
    summary_lines = ["# Audit Summary — Table 1 raw WAVs",
                     "",
                     f"Generated by `{Path(__file__).name}`. 25 files, 5 positions × 2 conditions.",
                     ""]
    audit_data = {}
    for key in sorted(groups.keys()):
        paths = groups[key]
        if "mic_l" not in paths or "mic_r" not in paths:
            continue
        # Only audit blocked groups (all three channels) and unblocked (mic-only)
        info = audit_group(key, paths, summary_lines)
        audit_data[f"{key[0]}_{key[1]}"] = {
            "stats": {ch: info["stats"][ch] for ch in info["stats"]},
            "tonal_peaks": {ch: info["peaks"][ch] for ch in info["peaks"]},
            "coh_band": {ch: info["coh_band"][ch] for ch in info["coh_band"]},
        }
    # Write summary
    (OUT_DIR / "audit_summary.md").write_text("\n".join(summary_lines))
    (OUT_DIR / "audit_data.json").write_text(json.dumps(audit_data, indent=2, default=str))
    print(f"Wrote {len(audit_data)} groups to {OUT_DIR}")


if __name__ == "__main__":
    main()
