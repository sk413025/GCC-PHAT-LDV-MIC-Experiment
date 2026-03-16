#!/usr/bin/env python3
"""
Generate the paper's spatial-score figure data from a representative blocked trial.

The output is a PGFPlots-ready table with one row per X candidate and two curves:
- MicMic: unguided MIC-L/MIC-R GCC-PHAT sampled at the predicted mic-pair delay
- PIGS:   LDV-anchored PI-GS score sampled at the predicted LDV-Mic delays
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

from paper_repro_helpers import build_file_manifest, git_state, sync_outputs, write_json

C_MPS = 343.0
MIC_Y_M = 2.0
BOARD_Y_M = 0.25
MIC_HALF_SPACING_M = 0.7


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate the paper's spatial-score plot data.")
    ap.add_argument("--data_root", type=str, default="", help="Optional dataset root used for relative-path metadata.")
    ap.add_argument("--ldv_wav", type=str, required=True)
    ap.add_argument("--micl_wav", type=str, required=True)
    ap.add_argument("--micr_wav", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--sync_dir", type=str, default="")
    ap.add_argument("--segment_center_sec", type=float, default=1.30)
    ap.add_argument("--segment_duration_sec", type=float, default=0.50)
    ap.add_argument("--bandpass_low_hz", type=float, default=500.0)
    ap.add_argument("--bandpass_high_hz", type=float, default=2000.0)
    ap.add_argument("--x_min_m", type=float, default=-1.0)
    ap.add_argument("--x_max_m", type=float, default=1.0)
    ap.add_argument("--x_step_m", type=float, default=0.02)
    ap.add_argument("--source_y_m", type=float, default=0.0)
    return ap.parse_args()


def _read_wav(path: Path) -> tuple[int, np.ndarray]:
    fs, data = wavfile.read(path)
    if data.ndim != 1:
        raise ValueError(f"Expected mono WAV, got shape={data.shape} for {path}")
    x = data.astype(np.float64)
    x -= float(np.mean(x))
    return int(fs), x


def _slice(x: np.ndarray, fs: int, center_sec: float, duration_sec: float) -> np.ndarray:
    half = int(round(0.5 * duration_sec * fs))
    center = int(round(center_sec * fs))
    lo = center - half
    hi = lo + int(round(duration_sec * fs))
    if lo < 0 or hi > len(x):
        raise ValueError(f"Requested segment exceeds signal length: center={center_sec}, duration={duration_sec}")
    seg = x[lo:hi].copy()
    seg -= float(np.mean(seg))
    return seg


def _bandpass(x: np.ndarray, fs: int, low_hz: float, high_hz: float) -> np.ndarray:
    nyq = 0.5 * float(fs)
    low = float(low_hz) / nyq
    high = float(high_hz) / nyq
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, x)


def _gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(sig1) + len(sig2) - 1
    nfft = 2 ** int(np.ceil(np.log2(n)))
    x1 = np.fft.rfft(sig1, n=nfft)
    x2 = np.fft.rfft(sig2, n=nfft)
    cps = x1 * np.conj(x2)
    cps /= np.maximum(np.abs(cps), 1e-15)
    gcc = np.fft.irfft(cps, n=nfft)
    gcc = np.concatenate([gcc[nfft // 2 :], gcc[: nfft // 2]])
    lags = np.arange(-nfft // 2, nfft // 2) / fs
    return np.abs(gcc).astype(np.float64), lags.astype(np.float64)


def _sample_curve(abs_cc: np.ndarray, lags_sec: np.ndarray, tau_sec: float) -> float:
    pos = np.interp(float(tau_sec), lags_sec, np.arange(len(lags_sec), dtype=np.float64))
    if pos <= 0 or pos >= len(abs_cc) - 1:
        return 0.0
    i0 = int(np.floor(pos))
    frac = pos - i0
    return float(abs_cc[i0] * (1.0 - frac) + abs_cc[i0 + 1] * frac)


def _tau_mic_pair(x_m: float, source_y_m: float) -> float:
    d_l = np.hypot(x_m + MIC_HALF_SPACING_M, MIC_Y_M - source_y_m)
    d_r = np.hypot(x_m - MIC_HALF_SPACING_M, MIC_Y_M - source_y_m)
    return float((d_l - d_r) / C_MPS)


def _tau_ldv_mic(x_m: float, mic_sign: float) -> float:
    d_vm = np.hypot(x_m - mic_sign * MIC_HALF_SPACING_M, MIC_Y_M - BOARD_Y_M)
    return float(-d_vm / C_MPS)


def _smooth(y: np.ndarray, width: int = 5) -> np.ndarray:
    if width <= 1:
        return y
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(y, kernel, mode="same")


def main() -> None:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "results" / f"spatial_score_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=False)

    ldv_path = Path(args.ldv_wav).expanduser().resolve()
    micl_path = Path(args.micl_wav).expanduser().resolve()
    micr_path = Path(args.micr_wav).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    for path in (ldv_path, micl_path, micr_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing WAV: {path}")

    fs_ldv, ldv = _read_wav(ldv_path)
    fs_l, micl = _read_wav(micl_path)
    fs_r, micr = _read_wav(micr_path)
    if not (fs_ldv == fs_l == fs_r):
        raise ValueError(f"Sample rate mismatch: {[fs_ldv, fs_l, fs_r]}")

    ldv_seg = _bandpass(_slice(ldv, fs_ldv, args.segment_center_sec, args.segment_duration_sec), fs_ldv, args.bandpass_low_hz, args.bandpass_high_hz)
    micl_seg = _bandpass(_slice(micl, fs_l, args.segment_center_sec, args.segment_duration_sec), fs_l, args.bandpass_low_hz, args.bandpass_high_hz)
    micr_seg = _bandpass(_slice(micr, fs_r, args.segment_center_sec, args.segment_duration_sec), fs_r, args.bandpass_low_hz, args.bandpass_high_hz)

    abs_lr, lags_lr = _gcc_phat(micl_seg, micr_seg, fs_l)
    abs_vl, lags_vl = _gcc_phat(ldv_seg, micl_seg, fs_l)
    abs_vr, lags_vr = _gcc_phat(ldv_seg, micr_seg, fs_l)

    x_values = np.arange(args.x_min_m, args.x_max_m + 0.5 * args.x_step_m, args.x_step_m, dtype=np.float64)
    mic_curve = []
    pigs_curve = []
    for x_m in x_values:
        tau_lr = _tau_mic_pair(float(x_m), float(args.source_y_m))
        tau_vl = _tau_ldv_mic(float(x_m), -1.0)
        tau_vr = _tau_ldv_mic(float(x_m), +1.0)
        mic_curve.append(_sample_curve(abs_lr, lags_lr, tau_lr))
        pigs_curve.append(_sample_curve(abs_vl, lags_vl, tau_vl) + _sample_curve(abs_vr, lags_vr, tau_vr))

    mic_curve = _smooth(np.asarray(mic_curve, dtype=np.float64), width=5)
    pigs_curve = _smooth(np.asarray(pigs_curve, dtype=np.float64), width=5)
    mic_curve /= max(float(np.max(mic_curve)), 1e-12)
    pigs_curve /= max(float(np.max(pigs_curve)), 1e-12)

    dat_path = out_dir / "spatial_score_curves.dat"
    with dat_path.open("w", encoding="utf-8") as f:
        f.write("X MicMic PIGS\n")
        for x_m, mic_y, pigs_y in zip(x_values, mic_curve, pigs_curve):
            f.write(f"{x_m:.3f} {mic_y:.6f} {pigs_y:.6f}\n")

    peak_idx_mic = int(np.argmax(mic_curve))
    peak_idx_pigs = int(np.argmax(pigs_curve))
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "segment": {
            "center_sec": float(args.segment_center_sec),
            "duration_sec": float(args.segment_duration_sec),
        },
        "bandpass": {
            "low_hz": float(args.bandpass_low_hz),
            "high_hz": float(args.bandpass_high_hz),
        },
        "grid": {
            "x_min_m": float(args.x_min_m),
            "x_max_m": float(args.x_max_m),
            "x_step_m": float(args.x_step_m),
            "source_y_m": float(args.source_y_m),
            "board_y_m": BOARD_Y_M,
        },
        "peaks": {
            "mic_mic_x_m": float(x_values[peak_idx_mic]),
            "pi_gs_x_m": float(x_values[peak_idx_pigs]),
        },
        "input_files": build_file_manifest([ldv_path, micl_path, micr_path], root=data_root),
        "git": git_state(repo_root),
    }
    write_json(out_dir / "spatial_score_meta.json", meta)

    if args.sync_dir:
        sync_dir = Path(args.sync_dir)
        if not sync_dir.is_absolute():
            sync_dir = repo_root / sync_dir
        copied = sync_outputs(
            {
                "spatial_score_curves.dat": dat_path,
                "spatial_score_meta.json": out_dir / "spatial_score_meta.json",
            },
            sync_dir,
        )
        write_json(out_dir / "sync_manifest.json", {"sync_dir": str(sync_dir), "copied": copied})

    print(f"Wrote: {dat_path}")
    print(f"Wrote: {out_dir / 'spatial_score_meta.json'}")


if __name__ == "__main__":
    main()
