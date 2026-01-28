#!/usr/bin/env python3
"""
Simulate a linear microphone array from mono WAV files.

Generates multi-channel WAVs by applying fractional delays corresponding to
plane-wave arrival from a specified DOA angle.
"""

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from scipy.io import wavfile


def parse_angle_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    if x.size == 0:
        return x
    n = np.arange(x.size, dtype=np.float64)
    t = n - delay_samples
    return np.interp(t, n, x, left=0.0, right=0.0).astype(np.float32)


def build_linear_positions(num_mics: int, spacing_m: float) -> np.ndarray:
    center = (num_mics - 1) / 2.0
    return (np.arange(num_mics, dtype=np.float64) - center) * spacing_m


def load_mono_wav(path: Path, target_fs: int) -> np.ndarray:
    fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    if fs != target_fs:
        raise ValueError(f"Sampling rate mismatch: {fs} != {target_fs}")
    return data


def write_multich_wav(path: Path, fs: int, data: np.ndarray) -> None:
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        data = data / peak
    data = np.clip(data * 0.98, -1.0, 1.0)
    wavfile.write(path, fs, (data * 32767.0).astype(np.int16))


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate array WAVs from mono inputs")
    parser.add_argument("--in_dir", type=str, required=True, help="Input directory with mono WAVs")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--fs", type=int, default=16000, help="Sampling rate")
    parser.add_argument("--num_mics", type=int, default=4, help="Number of microphones")
    parser.add_argument("--spacing_m", type=float, default=0.035, help="Mic spacing in meters")
    parser.add_argument("--c", type=float, default=343.0, help="Speed of sound (m/s)")
    parser.add_argument("--angle_mode", type=str, default="random", choices=["random", "grid"], help="Angle assignment")
    parser.add_argument("--angle_min", type=float, default=-60.0, help="Min DOA angle (deg)")
    parser.add_argument("--angle_max", type=float, default=60.0, help="Max DOA angle (deg)")
    parser.add_argument("--angle_step", type=float, default=5.0, help="Angle step for grid (deg)")
    parser.add_argument("--angles", type=str, default=None, help="Comma-separated angles (deg)")
    parser.add_argument("--snr_db", type=float, default=None, help="Additive noise SNR in dB")
    parser.add_argument("--gain_jitter_db", type=float, default=0.0, help="Per-channel gain jitter (dB)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of files")

    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.wav"))
    if args.max_files is not None:
        files = files[: args.max_files]

    if not files:
        raise SystemExit("No WAV files found in input directory")

    rng = np.random.default_rng(args.seed)

    if args.angles:
        angle_list = parse_angle_list(args.angles)
    elif args.angle_mode == "grid":
        angle_list = list(np.arange(args.angle_min, args.angle_max + 1e-6, args.angle_step))
    else:
        angle_list = []

    positions = build_linear_positions(args.num_mics, args.spacing_m)

    manifest = {
        "created": datetime.now().isoformat(),
        "config": {
            "fs": args.fs,
            "num_mics": args.num_mics,
            "spacing_m": args.spacing_m,
            "c": args.c,
            "angle_mode": args.angle_mode,
            "angle_min": args.angle_min,
            "angle_max": args.angle_max,
            "angle_step": args.angle_step,
            "angles": angle_list if args.angles else None,
            "snr_db": args.snr_db,
            "gain_jitter_db": args.gain_jitter_db,
            "seed": args.seed,
        },
        "items": [],
    }

    for idx, src_path in enumerate(files):
        if args.angle_mode == "random" and not args.angles:
            angle_deg = float(rng.uniform(args.angle_min, args.angle_max))
        else:
            angle_deg = float(angle_list[idx % len(angle_list)])

        signal = load_mono_wav(src_path, args.fs)
        theta = math.radians(angle_deg)
        delays_sec = positions * math.sin(theta) / args.c
        delays_samples = delays_sec * args.fs

        channels = []
        for m in range(args.num_mics):
            y = fractional_delay(signal, delays_samples[m])
            if args.gain_jitter_db > 0:
                gain = 10 ** (rng.normal(0.0, args.gain_jitter_db) / 20.0)
                y = y * gain
            if args.snr_db is not None:
                sig_power = float(np.mean(y ** 2)) if y.size else 0.0
                noise_power = sig_power / (10 ** (args.snr_db / 10.0)) if sig_power > 0 else 0.0
                noise = rng.normal(0.0, math.sqrt(noise_power), size=y.shape).astype(np.float32)
                y = y + noise
            channels.append(y)

        multich = np.stack(channels, axis=1)
        out_name = src_path.stem + f"_array_M{args.num_mics}.wav"
        out_path = wav_dir / out_name
        write_multich_wav(out_path, args.fs, multich)

        manifest["items"].append({
            "source_wav": str(src_path),
            "array_wav": str(out_path),
            "angle_deg": angle_deg,
            "delays_sec": delays_sec.tolist(),
            "delays_samples": delays_samples.tolist(),
        })

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(manifest['items'])} files to {wav_dir}")
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
