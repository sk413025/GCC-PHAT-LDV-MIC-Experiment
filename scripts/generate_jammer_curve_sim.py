#!/usr/bin/env python3
"""
Generate a *simulated* jammer resilience curve by mixing an interferer into the
microphone channels of a blocked target trial, then evaluating DoA estimation
error vs. SJR.

This script is intended to produce a stable, low-cognitive-load Figure 4 curve:
- Mic–Mic baseline: unguided GCC-PHAT peak on the mic pair
- PI-GS (LDV anchor): WLS solve using *only* LDV–Mic constraints (tau2, tau3)

Important:
- No guided peak search is used (no ground-truth leakage).
- SJR is imposed by scaling jammer power relative to target power on the
  microphone channels (per-window, to reduce speech amplitude variance).
- Optional small jammer leakage can be injected into the LDV channel.

Outputs:
- <out_dir>/jammer_resilience_curve_sim.dat  (PGFPlots-ready)
- <out_dir>/jammer_resilience_curve_sim_meta.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.fft import irfft, rfft, rfftfreq

from paper_repro_helpers import build_file_manifest, git_state, sync_outputs, write_json

try:
    import multi_sensor_fusion_doa as msnf
except ModuleNotFoundError:  # pragma: no cover
    # Allows execution via `python -m scripts.generate_jammer_curve_sim` from repo root.
    from scripts import multi_sensor_fusion_doa as msnf


@dataclass(frozen=True)
class WindowSpec:
    start_sample: int
    n_samples: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulate jammer resilience by mixing jammer into mic channels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Target trial (blocked)
    p.add_argument("--data_root", type=str, default="", help="Optional dataset root used for relative-path metadata.")
    p.add_argument("--target_ldv_wav", type=str, required=True)
    p.add_argument("--target_micl_wav", type=str, required=True)
    p.add_argument("--target_micr_wav", type=str, required=True)

    # Jammer trial (unblocked / in-room interference proxy)
    p.add_argument("--jammer_micl_wav", type=str, required=True)
    p.add_argument("--jammer_micr_wav", type=str, required=True)

    # Geometry / ground truth
    p.add_argument(
        "--speaker_key",
        type=str,
        default="19",
        help="Key in msnf.GEOMETRY['speakers'] used as target ground truth.",
    )
    p.add_argument("--speed_of_sound", type=float, default=343.0)
    p.add_argument("--mic_spacing", type=float, default=1.4)

    # Segment + windowing
    p.add_argument("--t0_sec", type=float, default=3.0)
    p.add_argument("--t1_sec", type=float, default=8.0)
    p.add_argument("--window_sec", type=float, default=0.5)
    p.add_argument("--num_windows", type=int, default=25)
    p.add_argument(
        "--jammer_time_shift_sec",
        type=float,
        default=0.0,
        help="Time-shift applied to jammer window extraction (wrap-around) to reduce unintended waveform coherence.",
    )
    p.add_argument(
        "--window_schedule",
        type=str,
        default="linspace",
        choices=("linspace",),
        help="Deterministic schedule for window start times within [t0, t1-window].",
    )

    # SJR sweep
    p.add_argument("--sjr_min_db", type=float, default=-40.0)
    p.add_argument("--sjr_max_db", type=float, default=20.0)
    p.add_argument("--sjr_step_db", type=float, default=2.0)

    # GCC configuration
    p.add_argument("--stft_n_fft", type=int, default=1024)
    p.add_argument("--stft_hop", type=int, default=256)
    p.add_argument("--gcc_bandpass_low_hz", type=float, default=500.0)
    p.add_argument("--gcc_bandpass_high_hz", type=float, default=2000.0)
    p.add_argument("--gcc_max_lag_ms", type=float, default=10.0)
    p.add_argument("--psr_exclude_samples", type=int, default=50)

    # Jammer injection model
    p.add_argument(
        "--ldv_jammer_leak_db",
        type=float,
        default=-120.0,
        help="Relative level (dB) of jammer leakage into LDV vs jammer-in-mic scale. "
        "Use a very negative value to approximate 'no jammer in LDV'.",
    )

    # Output
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory. If empty, uses results/jammer_curve_sim_<timestamp>.",
    )
    p.add_argument(
        "--sync_dir",
        type=str,
        default="",
        help="Optional fixed output directory to keep paper-facing generated assets in sync.",
    )

    return p.parse_args()


def _ensure_same_fs(fs_list: list[int]) -> int:
    fs0 = fs_list[0]
    if any(fs != fs0 for fs in fs_list):
        raise ValueError(f"All WAVs must have the same sample rate, got {fs_list}")
    return fs0


def _make_windows(
    *,
    t0_sec: float,
    t1_sec: float,
    window_sec: float,
    num_windows: int,
    fs: int,
) -> list[WindowSpec]:
    if window_sec <= 0:
        raise ValueError(f"window_sec must be > 0, got {window_sec}")
    if t1_sec <= t0_sec:
        raise ValueError(f"t1_sec must be > t0_sec, got t0={t0_sec}, t1={t1_sec}")
    if num_windows <= 0:
        raise ValueError(f"num_windows must be > 0, got {num_windows}")

    start_min = int(round(t0_sec * fs))
    start_max = int(round((t1_sec - window_sec) * fs))
    if start_max < start_min:
        raise ValueError(
            "Segment too short for the requested window_sec. "
            f"t0_sec={t0_sec}, t1_sec={t1_sec}, window_sec={window_sec}"
        )
    n_samples = int(round(window_sec * fs))

    if num_windows == 1:
        starts = np.array([int(round(0.5 * (start_min + start_max)))], dtype=int)
    else:
        starts = np.linspace(start_min, start_max, num_windows)
        starts = np.round(starts).astype(int)

    return [WindowSpec(start_sample=int(s), n_samples=n_samples) for s in starts]


def _power(x: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    return float(np.mean(x * x))


def _scale_for_sjr_db(
    *,
    target_micl: np.ndarray,
    target_micr: np.ndarray,
    jammer_micl: np.ndarray,
    jammer_micr: np.ndarray,
    sjr_db: float,
) -> float:
    p_target = 0.5 * (_power(target_micl) + _power(target_micr))
    p_jammer = 0.5 * (_power(jammer_micl) + _power(jammer_micr))
    if p_target <= 0.0 or not np.isfinite(p_target):
        raise ValueError(f"Invalid target power: {p_target}")
    if p_jammer <= 0.0 or not np.isfinite(p_jammer):
        raise ValueError(f"Invalid jammer power: {p_jammer}")
    # SJR = 10*log10(P_target / (scale^2 * P_jammer))
    # => scale = sqrt(P_target / (P_jammer * 10^(SJR/10)))
    scale = np.sqrt(p_target / (p_jammer * (10.0 ** (sjr_db / 10.0))))
    return float(scale)


def _stft_rfft(
    x: np.ndarray, *, n_fft: int, hop: int
) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    if n_fft <= 0 or hop <= 0:
        raise ValueError(f"Invalid STFT params: n_fft={n_fft}, hop={hop}")
    if len(x) < n_fft:
        raise ValueError(f"Signal shorter than n_fft: len={len(x)}, n_fft={n_fft}")

    n_frames = 1 + (len(x) - n_fft) // hop
    frame_stride = x.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, n_fft),
        strides=(hop * frame_stride, frame_stride),
        writeable=False,
    )
    window = np.hanning(n_fft).astype(np.float64, copy=False)
    X = rfft(frames * window[None, :], n=n_fft, axis=1)
    return X.T  # (n_freq, n_frames)


def _gcc_phat_stft_abs_cc(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fs: int,
    n_fft: int,
    hop: int,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
) -> np.ndarray:
    """
    GCC-PHAT using STFT time-aggregation, returning |cc| over circular lags.

    Output length is n_fft and corresponds to lag samples:
      lag_samples = np.arange(-n_fft//2, n_fft//2)
    """
    X = _stft_rfft(x, n_fft=n_fft, hop=hop)
    Y = _stft_rfft(y, n_fft=n_fft, hop=hop)
    n_freq = X.shape[0]

    Sxy = np.mean(X * np.conj(Y), axis=1)

    if bandpass_low_hz > 0 and bandpass_high_hz > bandpass_low_hz:
        freqs = rfftfreq(n_fft, d=1.0 / float(fs))
        mask = (freqs >= float(bandpass_low_hz)) & (freqs <= float(bandpass_high_hz))
        Sxy = Sxy * mask[:n_freq]

    Sxy = Sxy / (np.abs(Sxy) + 1e-10)
    cc = np.real(irfft(Sxy, n=n_fft))

    half = n_fft // 2
    cc = np.concatenate((cc[-half:], cc[:half]))
    return np.abs(cc).astype(np.float64, copy=False)


def _peak_tau_psr_from_abs_cc(
    abs_cc: np.ndarray,
    *,
    fs: int,
    max_lag_ms: float,
    psr_exclude_samples: int,
) -> tuple[float, float]:
    n = int(abs_cc.shape[0])
    half = n // 2
    max_shift = int(round(float(max_lag_ms) / 1000.0 * float(fs)))
    max_shift = int(np.clip(max_shift, 1, half - 1))

    lo = half - max_shift
    hi = half + max_shift + 1
    view = abs_cc[lo:hi]
    peak_rel = int(np.argmax(view))
    peak_idx = int(lo + peak_rel)

    # Parabolic interpolation.
    if 0 < peak_idx < n - 1:
        y0 = abs_cc[peak_idx - 1]
        y1 = abs_cc[peak_idx]
        y2 = abs_cc[peak_idx + 1]
        denom = y0 - 2.0 * y1 + y2
        shift = (0.5 * (y0 - y2) / denom) if abs(denom) > 1e-12 else 0.0
    else:
        shift = 0.0

    tau_sec = ((peak_idx - half) + float(shift)) / float(fs)

    excl = int(psr_exclude_samples)
    mask = np.ones_like(view, dtype=bool)
    peak_in_view = peak_idx - lo
    mlo = max(0, peak_in_view - excl)
    mhi = min(view.shape[0], peak_in_view + excl + 1)
    mask[mlo:mhi] = False
    sidelobe_max = float(view[mask].max()) if np.any(mask) else 0.0
    peak_val = float(abs_cc[peak_idx])
    psr_db = float(20.0 * np.log10(peak_val / (sidelobe_max + 1e-10)))
    return float(tau_sec), float(psr_db)


def _sample_abs_cc(abs_cc: np.ndarray, *, tau_sec: float, fs: int) -> float:
    n = int(abs_cc.shape[0])
    half = n // 2
    pos = float(tau_sec) * float(fs) + float(half)
    if pos < 0.0 or pos >= float(n - 1):
        return 0.0
    i0 = int(np.floor(pos))
    frac = pos - float(i0)
    return float(abs_cc[i0] * (1.0 - frac) + abs_cc[i0 + 1] * frac)


def main() -> None:
    args = _parse_args()

    fs_ldv, tgt_ldv = msnf.load_wav(args.target_ldv_wav)
    fs_l, tgt_micl = msnf.load_wav(args.target_micl_wav)
    fs_r, tgt_micr = msnf.load_wav(args.target_micr_wav)
    fs_jl, jam_micl = msnf.load_wav(args.jammer_micl_wav)
    fs_jr, jam_micr = msnf.load_wav(args.jammer_micr_wav)
    fs = _ensure_same_fs([fs_ldv, fs_l, fs_r, fs_jl, fs_jr])

    if args.speaker_key not in msnf.GEOMETRY["speakers"]:
        raise ValueError(f"Unknown speaker_key: {args.speaker_key}")
    gt = msnf.compute_all_ground_truths(
        args.speaker_key, c=float(args.speed_of_sound), d=float(args.mic_spacing)
    )
    theta_true = float(gt["theta_true_deg"])

    windows = _make_windows(
        t0_sec=float(args.t0_sec),
        t1_sec=float(args.t1_sec),
        window_sec=float(args.window_sec),
        num_windows=int(args.num_windows),
        fs=fs,
    )

    # Validate that all signals are long enough for the max window end.
    max_end = max(w.start_sample + w.n_samples for w in windows)
    min_target_len = min(len(tgt_ldv), len(tgt_micl), len(tgt_micr))
    min_jammer_len = min(len(jam_micl), len(jam_micr))
    if max_end > min_target_len:
        raise ValueError(
            "Target WAVs shorter than requested segment/window. "
            f"need_end={max_end}, min_len={min_target_len}"
        )
    if max_end > min_jammer_len:
        raise ValueError(
            "Jammer WAVs shorter than requested segment/window. "
            f"need_end={max_end}, min_len={min_jammer_len}"
        )
    jam_shift = int(round(float(args.jammer_time_shift_sec) * float(fs)))
    jam_period = int(min_jammer_len)
    if jam_period <= 0:
        raise ValueError("Invalid jammer length")

    sjr_values = np.arange(
        float(args.sjr_min_db),
        float(args.sjr_max_db) + 0.5 * float(args.sjr_step_db),
        float(args.sjr_step_db),
        dtype=np.float64,
    )

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("results")
        / f"jammer_curve_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dat = out_dir / "jammer_resilience_curve_sim.dat"
    out_meta = out_dir / "jammer_resilience_curve_sim_meta.json"

    records: list[dict] = []
    with out_dat.open("w", encoding="utf-8") as f:
        f.write("SJR_dB\tMic-Mic_MAE\tPI-GS_MAE\n")

        for sjr_db in sjr_values:
            mic_errs: list[float] = []
            pigs_errs: list[float] = []

            for w in windows:
                s = w.start_sample
                e = s + w.n_samples

                tgt_l = tgt_micl[s:e]
                tgt_r = tgt_micr[s:e]
                tgt_v = tgt_ldv[s:e]
                if jam_shift == 0:
                    js = s
                else:
                    # Wrap-around to ensure a valid window in the jammer recordings.
                    js = (s + jam_shift) % max(1, jam_period - w.n_samples + 1)
                je = js + w.n_samples
                jam_l = jam_micl[js:je]
                jam_r = jam_micr[js:je]

                scale = _scale_for_sjr_db(
                    target_micl=tgt_l,
                    target_micr=tgt_r,
                    jammer_micl=jam_l,
                    jammer_micr=jam_r,
                    sjr_db=float(sjr_db),
                )

                mix_l = tgt_l + scale * jam_l
                mix_r = tgt_r + scale * jam_r

                leak_linear = 10.0 ** (float(args.ldv_jammer_leak_db) / 20.0)
                if leak_linear > 0.0 and np.isfinite(leak_linear):
                    jam_mono = 0.5 * (jam_l + jam_r)
                    mix_v = tgt_v + (scale * leak_linear) * jam_mono
                else:
                    mix_v = tgt_v

                # Mic–Mic (unguided)
                abs_lr = _gcc_phat_stft_abs_cc(
                    mix_l,
                    mix_r,
                    fs=fs,
                    n_fft=int(args.stft_n_fft),
                    hop=int(args.stft_hop),
                    bandpass_low_hz=float(args.gcc_bandpass_low_hz),
                    bandpass_high_hz=float(args.gcc_bandpass_high_hz),
                )
                tau1_sec, _psr1 = _peak_tau_psr_from_abs_cc(
                    abs_lr,
                    fs=fs,
                    max_lag_ms=float(args.gcc_max_lag_ms),
                    psr_exclude_samples=int(args.psr_exclude_samples),
                )
                theta_mic = msnf.tau_to_doa(
                    tau1_sec * 1000.0,
                    c=float(args.speed_of_sound),
                    d=float(args.mic_spacing),
                )
                mic_errs.append(abs(float(theta_mic) - theta_true))

                # PI-GS objective: score(x) = |R_VL(tau_VL(x))| + |R_VR(tau_VR(x))|
                abs_vl = _gcc_phat_stft_abs_cc(
                    mix_v,
                    mix_l,
                    fs=fs,
                    n_fft=int(args.stft_n_fft),
                    hop=int(args.stft_hop),
                    bandpass_low_hz=float(args.gcc_bandpass_low_hz),
                    bandpass_high_hz=float(args.gcc_bandpass_high_hz),
                )
                abs_vr = _gcc_phat_stft_abs_cc(
                    mix_v,
                    mix_r,
                    fs=fs,
                    n_fft=int(args.stft_n_fft),
                    hop=int(args.stft_hop),
                    bandpass_low_hz=float(args.gcc_bandpass_low_hz),
                    bandpass_high_hz=float(args.gcc_bandpass_high_hz),
                )

                x_candidates = np.linspace(-0.8, 0.8, 161, dtype=np.float64)
                best_score = -np.inf
                best_x = float(x_candidates[0])
                for x_s in x_candidates.tolist():
                    tau_vl = -msnf.tau2_model(float(x_s), c=float(args.speed_of_sound))
                    tau_vr = -msnf.tau3_model(float(x_s), c=float(args.speed_of_sound))
                    score = _sample_abs_cc(abs_vl, tau_sec=float(tau_vl), fs=fs) + _sample_abs_cc(
                        abs_vr, tau_sec=float(tau_vr), fs=fs
                    )
                    if score > best_score:
                        best_score = float(score)
                        best_x = float(x_s)

                theta_hat = msnf.xs_to_theta(
                    best_x, c=float(args.speed_of_sound), d=float(args.mic_spacing)
                )
                pigs_errs.append(abs(float(theta_hat) - theta_true))

            mic_mae = float(np.mean(mic_errs)) if mic_errs else float("nan")
            pigs_mae = float(np.mean(pigs_errs)) if pigs_errs else float("nan")

            f.write(f"{float(sjr_db):.1f}\t{mic_mae:.3f}\t{pigs_mae:.3f}\n")
            f.flush()

            records.append(
                {
                    "SJR_dB": float(sjr_db),
                    "Mic-Mic_MAE": mic_mae,
                    "PI-GS_MAE": pigs_mae,
                }
            )

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fs": int(fs),
        "speaker_key": str(args.speaker_key),
        "theta_true_deg": theta_true,
        "paths": {
            "target_ldv_wav": str(args.target_ldv_wav),
            "target_micl_wav": str(args.target_micl_wav),
            "target_micr_wav": str(args.target_micr_wav),
            "jammer_micl_wav": str(args.jammer_micl_wav),
            "jammer_micr_wav": str(args.jammer_micr_wav),
        },
        "segment": {
            "t0_sec": float(args.t0_sec),
            "t1_sec": float(args.t1_sec),
            "window_sec": float(args.window_sec),
            "num_windows": int(args.num_windows),
            "jammer_time_shift_sec": float(args.jammer_time_shift_sec),
            "windows": [asdict(w) for w in windows],
        },
        "sjr_sweep": {
            "sjr_min_db": float(args.sjr_min_db),
            "sjr_max_db": float(args.sjr_max_db),
            "sjr_step_db": float(args.sjr_step_db),
        },
        "gcc": {
            "bandpass_low_hz": float(args.gcc_bandpass_low_hz),
            "bandpass_high_hz": float(args.gcc_bandpass_high_hz),
            "mic_mic_max_lag_ms": float(args.gcc_max_lag_ms),
            "psr_exclude_samples": int(args.psr_exclude_samples),
            "stft_n_fft": int(args.stft_n_fft),
            "stft_hop": int(args.stft_hop),
        },
        "jammer_model": {
            "ldv_jammer_leak_db": float(args.ldv_jammer_leak_db),
        },
        "output": {
            "dat": str(out_dat),
            "meta_json": str(out_meta),
        },
    }
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    meta["input_files"] = build_file_manifest(
        [
            Path(args.target_ldv_wav).expanduser().resolve(),
            Path(args.target_micl_wav).expanduser().resolve(),
            Path(args.target_micr_wav).expanduser().resolve(),
            Path(args.jammer_micl_wav).expanduser().resolve(),
            Path(args.jammer_micr_wav).expanduser().resolve(),
        ],
        root=data_root,
    )
    meta["git"] = git_state(Path(__file__).resolve().parent.parent)
    write_json(out_meta, meta)

    if args.sync_dir:
        repo_root = Path(__file__).resolve().parent.parent
        sync_dir = Path(args.sync_dir)
        if not sync_dir.is_absolute():
            sync_dir = repo_root / sync_dir
        copied = sync_outputs(
            {
                "jammer_resilience_curve_sim.dat": out_dat,
                "jammer_resilience_curve_sim_meta.json": out_meta,
            },
            sync_dir,
        )
        write_json(out_dir / "sync_manifest.json", {"sync_dir": str(sync_dir), "copied": copied})

    print(f"Wrote: {out_dat}")
    print(f"Wrote: {out_meta}")


if __name__ == "__main__":
    main()
