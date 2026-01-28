#!/usr/bin/env python3
"""
Run beamforming and MUSIC on simulated array WAVs.

Outputs per-file DoA estimates and summary statistics.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile


def next_pow2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def load_multich_wav(path: Path, target_fs: int) -> np.ndarray:
    fs, data = wavfile.read(path)
    if fs != target_fs:
        raise ValueError(f"Sampling rate mismatch: {fs} != {target_fs}")
    if data.ndim == 1:
        raise ValueError(f"Expected multi-channel audio, got mono: {path}")
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    return data


def build_linear_positions(num_mics: int, spacing_m: float) -> np.ndarray:
    center = (num_mics - 1) / 2.0
    return (np.arange(num_mics, dtype=np.float64) - center) * spacing_m


def compute_stft_multich(
    data: np.ndarray,
    fs: int,
    n_fft: int,
    hop_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_mics = data.shape[1]
    f, t, z = signal.stft(
        data[:, 0], fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length,
        window="hann", boundary=None, padded=False
    )
    stft = np.zeros((num_mics, len(f), len(t)), dtype=np.complex64)
    stft[0] = z
    for m in range(1, num_mics):
        _, _, z_m = signal.stft(
            data[:, m], fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length,
            window="hann", boundary=None, padded=False
        )
        stft[m] = z_m
    return f, t, stft


def steering_vectors(
    freqs_hz: np.ndarray,
    angles_deg: np.ndarray,
    positions_m: np.ndarray,
    c: float,
) -> np.ndarray:
    # freqs: (F,), angles: (A,), positions: (M,)
    theta = np.deg2rad(angles_deg)
    delays = positions_m[None, :] * np.sin(theta)[:, None] / c  # (A, M)
    phase = -2.0 * np.pi * freqs_hz[:, None, None] * delays[None, :, :]  # (F, A, M)
    return np.exp(1j * phase).astype(np.complex64)


def beamform_ds_power(stft: np.ndarray, steer: np.ndarray) -> np.ndarray:
    # stft: (M, F, T), steer: (F, A, M)
    mics, num_freqs, num_frames = stft.shape
    angles = steer.shape[1]
    power = np.zeros(angles, dtype=np.float64)
    for fi in range(num_freqs):
        A = steer[fi]  # (A, M)
        Y = A.conj() @ stft[:, fi, :]  # (A, T)
        power += np.mean(np.abs(Y) ** 2, axis=1)
    return power


def music_spectrum(
    stft: np.ndarray,
    steer: np.ndarray,
    num_sources: int,
    eps: float = 1e-12,
) -> np.ndarray:
    mics, num_freqs, num_frames = stft.shape
    angles = steer.shape[1]
    if num_sources >= mics:
        raise ValueError("num_sources must be less than num_mics")
    spectrum = np.zeros(angles, dtype=np.float64)
    for fi in range(num_freqs):
        X = stft[:, fi, :]
        R = (X @ X.conj().T) / float(max(1, num_frames))
        eigvals, eigvecs = np.linalg.eigh(R)
        order = np.argsort(eigvals)
        En = eigvecs[:, order[: mics - num_sources]]
        A = steer[fi]  # (A, M)
        proj = A @ En  # (A, M-k)
        denom = np.sum(np.abs(proj) ** 2, axis=1)
        spectrum += 1.0 / (denom + eps)
    return spectrum


def gcc_phat_tau(x: np.ndarray, y: np.ndarray, fs: int, max_tau_s: float) -> Optional[float]:
    if x.size == 0 or y.size == 0:
        return None
    nfft = next_pow2(2 * x.size)
    X = np.fft.rfft(x, n=nfft)
    Y = np.fft.rfft(y, n=nfft)
    R = X.conj() * Y
    R = R / (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R, n=nfft)
    cc = np.concatenate([cc[-(nfft // 2):], cc[: nfft // 2]])
    lags = np.arange(-(nfft // 2), nfft // 2, dtype=np.int64)
    max_samp = int(max_tau_s * fs)
    mask = np.abs(lags) <= max_samp
    if not np.any(mask):
        return None
    cc_sel = cc[mask]
    lags_sel = lags[mask]
    peak_idx = int(np.argmax(np.abs(cc_sel)))
    tau_samples = float(lags_sel[peak_idx])
    return tau_samples / float(fs)


def angle_from_tau(tau_s: float, baseline_m: float, c: float) -> Optional[float]:
    if baseline_m <= 0:
        return None
    val = tau_s * c / baseline_m
    val = max(-1.0, min(1.0, val))
    return math.degrees(math.asin(val))


def summarize_errors(errors: List[float]) -> Dict[str, Optional[float]]:
    if not errors:
        return {
            "num_defined": 0,
            "mean_deg": None,
            "median_deg": None,
            "p90_deg": None,
        }
    arr = np.array(errors, dtype=np.float64)
    return {
        "num_defined": int(arr.size),
        "mean_deg": float(np.mean(arr)),
        "median_deg": float(np.median(arr)),
        "p90_deg": float(np.percentile(arr, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Beamforming/MUSIC on array WAVs")
    parser.add_argument("--array_root", type=str, required=True, help="Root with manifest.json and wavs/")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--fs", type=int, default=16000, help="Sampling rate")
    parser.add_argument("--num_mics", type=int, default=4, help="Number of microphones")
    parser.add_argument("--spacing_m", type=float, default=0.035, help="Mic spacing in meters")
    parser.add_argument("--c", type=float, default=343.0, help="Speed of sound (m/s)")
    parser.add_argument("--n_fft", type=int, default=2048, help="FFT size")
    parser.add_argument("--hop_length", type=int, default=160, help="Hop length")
    parser.add_argument("--freq_min", type=float, default=300.0, help="Min frequency")
    parser.add_argument("--freq_max", type=float, default=3000.0, help="Max frequency")
    parser.add_argument("--angle_min", type=float, default=-90.0, help="Min scan angle")
    parser.add_argument("--angle_max", type=float, default=90.0, help="Max scan angle")
    parser.add_argument("--angle_step", type=float, default=2.0, help="Angle step")
    parser.add_argument("--num_sources", type=int, default=1, help="Number of sources for MUSIC")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of files")

    args = parser.parse_args()

    array_root = Path(args.array_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = array_root / "manifest.json"
    manifest = None
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    wav_dir = array_root / "wavs"
    if not wav_dir.exists():
        wav_dir = array_root

    wav_files = sorted(wav_dir.glob("*.wav"))
    if args.max_files is not None:
        wav_files = wav_files[: args.max_files]

    if not wav_files:
        raise SystemExit("No WAV files found in array_root")

    angles = np.arange(args.angle_min, args.angle_max + 1e-6, args.angle_step, dtype=np.float64)
    positions = build_linear_positions(args.num_mics, args.spacing_m)

    detailed = []
    ds_errors = []
    music_errors = []
    gcc_errors = []

    for wav_path in wav_files:
        data = load_multich_wav(wav_path, args.fs)
        if data.shape[1] != args.num_mics:
            raise ValueError(f"Expected {args.num_mics} channels, got {data.shape[1]} in {wav_path}")

        freqs, _, stft = compute_stft_multich(data, args.fs, args.n_fft, args.hop_length)
        freq_mask = (freqs >= args.freq_min) & (freqs <= args.freq_max)
        freqs_sel = freqs[freq_mask]
        stft_sel = stft[:, freq_mask, :]

        steer = steering_vectors(freqs_sel, angles, positions, args.c)
        ds_power = beamform_ds_power(stft_sel, steer)
        music_power = music_spectrum(stft_sel, steer, args.num_sources)

        ds_idx = int(np.argmax(ds_power))
        music_idx = int(np.argmax(music_power))
        ds_theta = float(angles[ds_idx])
        music_theta = float(angles[music_idx])

        item = {
            "array_wav": str(wav_path),
            "ds": {
                "theta_hat_deg": ds_theta,
                "peak_value": float(ds_power[ds_idx]),
            },
            "music": {
                "theta_hat_deg": music_theta,
                "peak_value": float(music_power[music_idx]),
            },
        }

        angle_gt = None
        if manifest is not None:
            for it in manifest.get("items", []):
                if Path(it.get("array_wav", "")).name == wav_path.name:
                    angle_gt = float(it.get("angle_deg"))
                    item["angle_gt_deg"] = angle_gt
                    break

        if angle_gt is not None:
            ds_errors.append(abs(ds_theta - angle_gt))
            music_errors.append(abs(music_theta - angle_gt))

        baseline = args.spacing_m * (args.num_mics - 1)
        tau = gcc_phat_tau(data[:, 0], data[:, -1], args.fs, max_tau_s=baseline / args.c)
        if tau is not None:
            gcc_theta = angle_from_tau(tau, baseline, args.c)
            item["gcc_phat"] = {
                "tau_ms": float(tau * 1000.0),
                "theta_hat_deg": gcc_theta,
            }
            if angle_gt is not None and gcc_theta is not None:
                gcc_errors.append(abs(gcc_theta - angle_gt))

        detailed.append(item)

    summary = {
        "created": datetime.now().isoformat(),
        "num_files": len(detailed),
        "config": {
            "fs": args.fs,
            "num_mics": args.num_mics,
            "spacing_m": args.spacing_m,
            "c": args.c,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "freq_min": args.freq_min,
            "freq_max": args.freq_max,
            "angle_min": args.angle_min,
            "angle_max": args.angle_max,
            "angle_step": args.angle_step,
            "num_sources": args.num_sources,
        },
        "ds": summarize_errors(ds_errors),
        "music": summarize_errors(music_errors),
        "gcc_phat": summarize_errors(gcc_errors),
    }

    with open(out_dir / "detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved detailed results to {out_dir / 'detailed_results.json'}")
    print(f"Saved summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
