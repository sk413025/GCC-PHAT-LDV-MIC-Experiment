#!/usr/bin/env python3
"""
Independent PI-GS audit from raw WAVs.

This script intentionally does not import any project research modules.  It
rebuilds the LDV-microphone geometric-search idea from first principles using
only NumPy/SciPy, then tries a compact bank of physically motivated
preprocessing and scoring variants.

The audit has two goals:
1. Check whether the paper-level claims can be approached from raw recordings.
2. Separate genuine calibration from label leakage by calibrating offsets on
   chirp segments and evaluating speech with the calibration frozen.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.fft import irfft, rfft, rfftfreq
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt


C_MPS = 343.0
MIC_LEFT_X_M = -0.7
MIC_RIGHT_X_M = 0.7
MIC_Y_M = 2.0
MIC_SPACING_M = 1.4


@dataclass(frozen=True)
class Trial:
    x_m: float
    label: str
    ldv: str
    mic_l: str
    mic_r: str


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    t0_sec: float
    t1_sec: float


@dataclass(frozen=True)
class Config:
    name: str
    band_hz: tuple[float, float] | None
    phat_beta: float
    transform: str
    geometry: str
    ldv_y_m: float
    score_mode: str
    n_fft: int
    hop: int
    window_sec: float
    window_hop_sec: float
    top_k_windows: int
    local_peak_radius_ms: float
    gcc_mode: str = "plain"
    estimator: str = "score"
    coherence_floor: float = 0.0
    subbands: tuple[tuple[float, float], ...] | None = None
    score_aggregator: str = "curve_mean"
    subband_penalty: float = 0.0
    window_selector: str = "fixed_top_k"
    adaptive_min_k: int = 8
    stability_threshold_m: float = 0.03
    required_stable_steps: int = 2
    fallback_top_k: int = 9
    basin_mode: str = "none"
    basin_sigma_m: float = 0.08
    basin_gate: float = 0.10
    basin_power: float = 0.25
    candidate_agreement_gate_m: float | None = None
    subband_weight_mode: str = "none"
    lr_weight: float = 0.0
    lr_gate: float = 0.0
    ldv_x_m: float = 0.0
    wall_speed_mps: float = 0.0
    common_shift_radius_ms: float = 0.0
    common_shift_steps: int = 0
    x_calibration: str = "none"
    correlation_polarity: str = "abs"
    edge_dilation_threshold_m: float = 0.0
    edge_dilation_gain: float = 1.0
    edge_dilation_min_k: int = 1
    center_deadband_m: float = 0.0
    center_deadband_min_k: int = 1


def default_trials(data_root: Path) -> list[Trial]:
    return [
        Trial(
            x_m=-0.8,
            label="-0.8m #20",
            ldv="0223-block/0223-block-7(high)/0223-LDV-40-boy(-0.8m)-20-block.wav",
            mic_l="0223-block/0223-block-7(high)/0223-MIC-LEFT-40-boy(-0.8m)-20-block.wav",
            mic_r="0223-block/0223-block-7(high)/0223-MIC-RIGHT-40-boy(-0.8m)-20-block.wav",
        ),
        Trial(
            x_m=-0.4,
            label="-0.4m #19",
            ldv="0223-block-6(high)/0223-LDV-40-boy(-0.4m)-19-block.wav",
            mic_l="0223-block-6(high)/0223-MIC-LEFT-40-boy(-0.4m)-19-block.wav",
            mic_r="0223-block-6(high)/0223-MIC-RIGHT-40-boy(-0.4m)-19-block.wav",
        ),
        Trial(
            x_m=0.0,
            label="+0.0m #18",
            ldv="0223-block/0223-block-5(high)/0223-LDV-40-boy(+0.0m)-18-block.wav",
            mic_l="0223-block/0223-block-5(high)/0223-MIC-LEFT-40-boy(+0.0m)-18-block.wav",
            mic_r="0223-block/0223-block-5(high)/0223-MIC-RIGHT-40-boy(+0.0m)-18-block.wav",
        ),
        Trial(
            x_m=0.4,
            label="+0.4m #16",
            ldv="0223-block/0223-block-3(high)/0223-LDV-40-boy(+0.4m)-16-block.wav",
            mic_l="0223-block/0223-block-3(high)/0223-MIC-LEFT-40-boy(+0.4m)-16-block.wav",
            mic_r="0223-block/0223-block-3(high)/0223-MIC-RIGHT-40-boy(+0.4m)-16-block.wav",
        ),
        Trial(
            x_m=0.8,
            label="+0.8m #17",
            ldv="0223-block/0223-block-4(high)/0223-LDV-40-boy(+0.8m)-17-block.wav",
            mic_l="0223-block/0223-block-4(high)/0223-MIC-LEFT-40-boy(+0.8m)-17-block.wav",
            mic_r="0223-block/0223-block-4(high)/0223-MIC-RIGHT-40-boy(+0.8m)-17-block.wav",
        ),
    ]


def holdout_trials(data_root: Path) -> list[Trial]:
    """Extra complete LDV/MIC block repeats used only for validation."""
    candidates = [
        Trial(
            x_m=-0.8,
            label="-0.8m #21",
            ldv="0223-block/0223-block-7(high)/0223-LDV-40-boy(-0.8m)-21-block.wav",
            mic_l="0223-block/0223-block-7(high)/0223-MIC-LEFT-40-boy(-0.8m)-21-block.wav",
            mic_r="0223-block/0223-block-7(high)/0223-MIC-RIGHT-40-boy(-0.8m)-21-block.wav",
        ),
        Trial(
            x_m=0.0,
            label="+0.0m #22",
            ldv="0223-block/0223-block-5(high)/0223-LDV-40-boy(+0.0m)-22-block.wav",
            mic_l="0223-block/0223-block-5(high)/0223-MIC-LEFT-40-boy(+0.0m)-22-block.wav",
            mic_r="0223-block/0223-block-5(high)/0223-MIC-RIGHT-40-boy(+0.0m)-22-block.wav",
        ),
        Trial(
            x_m=0.4,
            label="+0.4m #15",
            ldv="0223-block/0223-block-2/0223-LDV-40-boy(+0.4m)-15-block.wav",
            mic_l="0223-block/0223-block-2/0223-MIC-LEFT-40-boy(+0.4m)-15-block.wav",
            mic_r="0223-block/0223-block-2/0223-MIC-RIGHT-40-boy(+0.4m)-15-block.wav",
        ),
        Trial(
            x_m=0.4,
            label="+0.4m #13",
            ldv="0223-block/0223-LDV-40-boy(+0.4m)-13-block.wav",
            mic_l="0223-block/0223-MIC-LEFT-40-boy(+0.4m)-13-block.wav",
            mic_r="0223-block/0223-MIC-RIGHT-40-boy(+0.4m)-13-block.wav",
        ),
        Trial(
            x_m=0.8,
            label="+0.8m #21",
            ldv="0223-block/0223-block-4(high)/0223-LDV-40-boy(+0.8m)-21-block.wav",
            mic_l="0223-block/0223-block-4(high)/0223-MIC-LEFT-40-boy(+0.8m)-21-block.wav",
            mic_r="0223-block/0223-block-4(high)/0223-MIC-RIGHT-40-boy(+0.8m)-21-block.wav",
        ),
    ]
    return [trial for trial in candidates if trial_files_exist(data_root, trial)]


def trial_files_exist(data_root: Path, trial: Trial) -> bool:
    return all((data_root / rel).exists() for rel in (trial.ldv, trial.mic_l, trial.mic_r))


def require_trial_files(data_root: Path, trials: list[Trial]) -> None:
    for trial in trials:
        for rel in (trial.ldv, trial.mic_l, trial.mic_r):
            if not (data_root / rel).exists():
                raise FileNotFoundError(data_root / rel)


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    fs, x = wavfile.read(path)
    if x.ndim != 1:
        raise ValueError(f"Expected mono WAV: {path} shape={x.shape}")
    y = x.astype(np.float64)
    if np.issubdtype(x.dtype, np.integer):
        y /= float(np.iinfo(x.dtype).max)
    y -= float(np.mean(y))
    return int(fs), y


def slice_seconds(x: np.ndarray, fs: int, t0_sec: float, t1_sec: float) -> np.ndarray:
    lo = int(round(t0_sec * fs))
    hi = int(round(t1_sec * fs))
    if lo < 0 or hi <= lo or hi > len(x):
        raise ValueError(f"Invalid segment [{t0_sec}, {t1_sec}] for signal length {len(x) / fs:.3f}s")
    y = x[lo:hi].copy()
    y -= float(np.mean(y))
    return y


def preprocess(x: np.ndarray, fs: int, cfg: Config) -> np.ndarray:
    y = x.astype(np.float64, copy=True)
    y -= float(np.mean(y))

    if cfg.transform in ("preemph", "preemph_clip"):
        y = np.r_[y[0], y[1:] - 0.97 * y[:-1]]
    if cfg.transform in ("diff", "accel"):
        y = np.r_[0.0, np.diff(y)]
    if cfg.transform == "accel":
        y = np.r_[0.0, np.diff(y)]
    if cfg.transform in ("clip", "preemph_clip"):
        threshold = 0.6 * float(np.std(y))
        y = np.sign(y) * np.maximum(np.abs(y) - threshold, 0.0)

    if cfg.band_hz is not None:
        low, high = cfg.band_hz
        sos = butter(4, [low / (0.5 * fs), high / (0.5 * fs)], btype="bandpass", output="sos")
        y = sosfiltfilt(sos, y)

    y -= float(np.mean(y))
    scale = float(np.std(y))
    if scale > 1e-12:
        y /= scale
    return y


def make_windows(n_samples: int, fs: int, cfg: Config) -> list[tuple[int, int]]:
    n_win = int(round(cfg.window_sec * fs))
    hop = int(round(cfg.window_hop_sec * fs))
    if n_samples < n_win:
        return [(0, n_samples)]
    return [(s, s + n_win) for s in range(0, n_samples - n_win + 1, hop)]


def stft_gcc_abs(
    x: np.ndarray,
    y: np.ndarray,
    fs: int,
    cfg: Config,
    band_hz: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < cfg.n_fft or len(y) < cfg.n_fft:
        raise ValueError("Window is shorter than n_fft")

    n_frames = 1 + (len(x) - cfg.n_fft) // cfg.hop
    if n_frames <= 0:
        raise ValueError("No STFT frames")

    def frames(a: np.ndarray) -> np.ndarray:
        stride = a.strides[0]
        view = np.lib.stride_tricks.as_strided(
            a,
            shape=(n_frames, cfg.n_fft),
            strides=(cfg.hop * stride, stride),
            writeable=False,
        )
        return view * np.hanning(cfg.n_fft)[None, :]

    X = rfft(frames(x), n=cfg.n_fft, axis=1)
    Y = rfft(frames(y), n=cfg.n_fft, axis=1)
    cross_frames = X * np.conj(Y)
    cross = np.mean(cross_frames, axis=0)

    active_band = cfg.band_hz if band_hz is None else band_hz
    if active_band is not None:
        freqs = rfftfreq(cfg.n_fft, 1.0 / fs)
        low, high = active_band
        cross *= (freqs >= low) & (freqs <= high)

    if cfg.gcc_mode == "coherence":
        pxx = np.mean(np.abs(X) ** 2, axis=0)
        pyy = np.mean(np.abs(Y) ** 2, axis=0)
        coherence = (np.abs(cross) ** 2) / (pxx * pyy + 1e-18)
        cross *= coherence >= float(cfg.coherence_floor)
        # Weight coherent bins slightly without letting magnitude dominate PHAT.
        cross *= np.sqrt(np.clip(coherence, 0.0, 1.0))
    elif cfg.gcc_mode != "plain":
        raise ValueError(f"Unknown gcc_mode: {cfg.gcc_mode}")

    denom = np.abs(cross) ** cfg.phat_beta
    denom[denom < 1e-12] = 1e-12
    cc = np.real(irfft(cross / denom, n=cfg.n_fft))
    half = cfg.n_fft // 2
    cc = np.concatenate([cc[-half:], cc[:half]])
    lags = np.arange(-half, half, dtype=np.float64) / float(fs)
    if cfg.correlation_polarity == "abs":
        curve = np.abs(cc)
    elif cfg.correlation_polarity == "positive":
        curve = np.maximum(cc, 0.0)
    elif cfg.correlation_polarity == "negative":
        curve = np.maximum(-cc, 0.0)
    else:
        raise ValueError(f"Unknown correlation_polarity: {cfg.correlation_polarity}")
    return curve, lags


def robust_normalize(values: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(values, 10))
    hi = float(np.percentile(values, 99))
    if hi <= lo + 1e-12:
        return values.copy()
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def sample_curve(curve: np.ndarray, lags: np.ndarray, taus: np.ndarray) -> np.ndarray:
    return np.interp(taus, lags, curve, left=0.0, right=0.0)


def theta_from_x(x_m: float) -> float:
    d_l = math.hypot(x_m - MIC_LEFT_X_M, MIC_Y_M)
    d_r = math.hypot(x_m - MIC_RIGHT_X_M, MIC_Y_M)
    return float(np.degrees(np.arcsin(np.clip((d_l - d_r) / MIC_SPACING_M, -1.0, 1.0))))


def tau_templates(xs: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    if cfg.geometry == "moving_patch":
        # The source excites the closest barrier patch; LDV observes that patch
        # and microphones receive re-radiation from it.
        d_vl = np.sqrt((xs - MIC_LEFT_X_M) ** 2 + (MIC_Y_M - cfg.ldv_y_m) ** 2)
        d_vr = np.sqrt((xs - MIC_RIGHT_X_M) ** 2 + (MIC_Y_M - cfg.ldv_y_m) ** 2)
        return -d_vl / C_MPS, -d_vr / C_MPS

    if cfg.geometry in ("wall_wave_sub", "wall_wave_add"):
        # LDV may observe a structural wave after the excited wall patch has
        # propagated laterally to the laser spot.  The sign is not obvious from
        # the recordings, so strict-v8 tests both physically plausible orders.
        if cfg.wall_speed_mps <= 0.0:
            raise ValueError("wall_wave geometry requires wall_speed_mps > 0")
        wall_delay = np.abs(xs - cfg.ldv_x_m) / cfg.wall_speed_mps
        d_vl = np.sqrt((xs - MIC_LEFT_X_M) ** 2 + (MIC_Y_M - cfg.ldv_y_m) ** 2)
        d_vr = np.sqrt((xs - MIC_RIGHT_X_M) ** 2 + (MIC_Y_M - cfg.ldv_y_m) ** 2)
        if cfg.geometry == "wall_wave_sub":
            return wall_delay - d_vl / C_MPS, wall_delay - d_vr / C_MPS
        return -(wall_delay + d_vl / C_MPS), -(wall_delay + d_vr / C_MPS)

    if cfg.geometry == "fixed_spot":
        d_sv = np.sqrt(xs**2 + cfg.ldv_y_m**2)
        d_sl = np.sqrt((xs - MIC_LEFT_X_M) ** 2 + MIC_Y_M**2)
        d_sr = np.sqrt((xs - MIC_RIGHT_X_M) ** 2 + MIC_Y_M**2)
        return (d_sl - d_sv) / C_MPS, (d_sr - d_sv) / C_MPS

    raise ValueError(f"Unknown geometry: {cfg.geometry}")


def tau_lr_template(xs: np.ndarray) -> np.ndarray:
    d_l = np.sqrt((xs - MIC_LEFT_X_M) ** 2 + MIC_Y_M**2)
    d_r = np.sqrt((xs - MIC_RIGHT_X_M) ** 2 + MIC_Y_M**2)
    return (d_r - d_l) / C_MPS


def local_peak_tau(curve: np.ndarray, lags: np.ndarray, center_tau: float, radius_ms: float) -> tuple[float, float]:
    radius = radius_ms / 1000.0
    mask = (lags >= center_tau - radius) & (lags <= center_tau + radius)
    if not np.any(mask):
        return center_tau, 0.0
    indices = np.where(mask)[0]
    idx = int(indices[np.argmax(curve[indices])])
    return float(lags[idx]), float(curve[idx])


def combine_scores(vl: np.ndarray, vr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "sum":
        return vl + vr
    if mode == "product":
        return vl * vr
    if mode == "min":
        return np.minimum(vl, vr)
    if mode == "harmonic":
        return 2.0 * vl * vr / (vl + vr + 1e-12)
    raise ValueError(f"Unknown score mode: {mode}")


def window_reliability(curve_vl: np.ndarray, curve_vr: np.ndarray) -> float:
    def psr_like(c: np.ndarray) -> float:
        if len(c) == 0:
            return 0.0
        top = float(np.max(c))
        med = float(np.median(c))
        mad = float(np.median(np.abs(c - med))) + 1e-12
        return (top - med) / mad

    return min(psr_like(curve_vl), psr_like(curve_vr))


def compute_trial_curves(
    data_root: Path,
    trial: Trial,
    segment: SegmentSpec,
    cfg: Config,
) -> dict[str, object]:
    fs_v, ldv_raw = read_wav(data_root / trial.ldv)
    fs_l, mic_l_raw = read_wav(data_root / trial.mic_l)
    fs_r, mic_r_raw = read_wav(data_root / trial.mic_r)
    if not (fs_v == fs_l == fs_r):
        raise ValueError(f"Sample-rate mismatch for {trial.label}")

    ldv = preprocess(slice_seconds(ldv_raw, fs_v, segment.t0_sec, segment.t1_sec), fs_v, cfg)
    mic_l = preprocess(slice_seconds(mic_l_raw, fs_l, segment.t0_sec, segment.t1_sec), fs_l, cfg)
    mic_r = preprocess(slice_seconds(mic_r_raw, fs_r, segment.t0_sec, segment.t1_sec), fs_r, cfg)

    windows = make_windows(len(ldv), fs_v, cfg)
    items = []
    for start, end in windows:
        try:
            subband_items = []
            bands = cfg.subbands if cfg.subbands is not None else (cfg.band_hz,)
            for band in bands:
                vl, lags_vl = stft_gcc_abs(ldv[start:end], mic_l[start:end], fs_v, cfg, band)
                vr, lags_vr = stft_gcc_abs(ldv[start:end], mic_r[start:end], fs_v, cfg, band)
                lr, lags_lr = stft_gcc_abs(mic_l[start:end], mic_r[start:end], fs_v, cfg, band)
                vl = robust_normalize(vl)
                vr = robust_normalize(vr)
                lr = robust_normalize(lr)
                subband_items.append(
                    {
                        "band_hz": band,
                        "vl": vl,
                        "vr": vr,
                        "lr": lr,
                        "lags_vl": lags_vl,
                        "lags_vr": lags_vr,
                        "lags_lr": lags_lr,
                        "reliability": window_reliability(vl, vr),
                    }
                )
        except ValueError:
            continue
        vl = np.mean([np.asarray(s["vl"]) for s in subband_items], axis=0)
        vr = np.mean([np.asarray(s["vr"]) for s in subband_items], axis=0)
        lr = np.mean([np.asarray(s["lr"]) for s in subband_items], axis=0)
        lags_vl = subband_items[0]["lags_vl"]
        lags_vr = subband_items[0]["lags_vr"]
        lags_lr = subband_items[0]["lags_lr"]
        items.append(
            {
                "start": start,
                "end": end,
                "vl": vl,
                "vr": vr,
                "lr": lr,
                "lags_vl": lags_vl,
                "lags_vr": lags_vr,
                "lags_lr": lags_lr,
                "reliability": window_reliability(vl, vr),
                "subbands": subband_items,
            }
        )

    if not items:
        raise ValueError(f"No valid windows for {trial.label} {segment.name} {cfg.name}")

    items.sort(key=lambda x: float(x["reliability"]), reverse=True)
    if cfg.top_k_windows > 0:
        items = items[: cfg.top_k_windows]
    return {"fs": fs_v, "windows": items}


def estimate_offsets(
    data_root: Path,
    trials: list[Trial],
    segment: SegmentSpec,
    cfg: Config,
    xs_grid: np.ndarray,
    offset_model: str,
) -> dict[str, float]:
    tau_vl_grid, tau_vr_grid = tau_templates(xs_grid, cfg)
    tau_lr_grid = tau_lr_template(xs_grid)
    residuals_vl: list[tuple[float, float]] = []
    residuals_vr: list[tuple[float, float]] = []
    residuals_lr: list[tuple[float, float]] = []
    per_trial: dict[str, dict[str, list[float]]] = {}

    for trial in trials:
        curves = compute_trial_curves(data_root, trial, segment, cfg)
        idx = int(np.argmin(np.abs(xs_grid - trial.x_m)))
        pred_vl = float(tau_vl_grid[idx])
        pred_vr = float(tau_vr_grid[idx])
        pred_lr = float(tau_lr_grid[idx])
        for item in curves["windows"]:  # type: ignore[index]
            peak_vl, _ = local_peak_tau(
                item["vl"],  # type: ignore[index]
                item["lags_vl"],  # type: ignore[index]
                pred_vl,
                cfg.local_peak_radius_ms,
            )
            peak_vr, _ = local_peak_tau(
                item["vr"],  # type: ignore[index]
                item["lags_vr"],  # type: ignore[index]
                pred_vr,
                cfg.local_peak_radius_ms,
            )
            peak_lr, _ = local_peak_tau(
                item["lr"],  # type: ignore[index]
                item["lags_lr"],  # type: ignore[index]
                pred_lr,
                cfg.local_peak_radius_ms,
            )
            res_vl = peak_vl - pred_vl
            res_vr = peak_vr - pred_vr
            res_lr = peak_lr - pred_lr
            residuals_vl.append((trial.x_m, res_vl))
            residuals_vr.append((trial.x_m, res_vr))
            residuals_lr.append((trial.x_m, res_lr))
            per_trial.setdefault(trial.label, {"vl": [], "vr": [], "lr": []})
            per_trial[trial.label]["vl"].append(res_vl)
            per_trial[trial.label]["vr"].append(res_vr)
            per_trial[trial.label]["lr"].append(res_lr)

    def fit_residuals(points: list[tuple[float, float]], prefix: str) -> dict[str, float]:
        if not points:
            return {f"{prefix}_intercept_sec": 0.0, f"{prefix}_slope_sec_per_m": 0.0}
        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)
        if offset_model == "affine" and len(np.unique(xs)) >= 2:
            design = np.column_stack([np.ones_like(xs), xs])
            coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
            pred = design @ coef
            resid = ys - pred
            mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
            keep = np.abs(resid) <= 4.0 * mad
            if np.count_nonzero(keep) >= 3:
                coef, *_ = np.linalg.lstsq(design[keep], ys[keep], rcond=None)
            return {f"{prefix}_intercept_sec": float(coef[0]), f"{prefix}_slope_sec_per_m": float(coef[1])}
        return {f"{prefix}_intercept_sec": float(np.median(ys)), f"{prefix}_slope_sec_per_m": 0.0}

    payload = {
        "model": offset_model,
        **fit_residuals(residuals_vl, "vl"),
        **fit_residuals(residuals_vr, "vr"),
        **fit_residuals(residuals_lr, "lr"),
        "num_residuals": int(min(len(residuals_vl), len(residuals_vr))),
    }
    if offset_model == "per_trial":
        payload["per_trial"] = {
            label: {
                "vl_sec": float(np.median(values["vl"])) if values["vl"] else 0.0,
                "vr_sec": float(np.median(values["vr"])) if values["vr"] else 0.0,
                "lr_sec": float(np.median(values["lr"])) if values["lr"] else 0.0,
            }
            for label, values in per_trial.items()
        }
    return payload


def offset_values(xs: np.ndarray, offsets: dict[str, float], prefix: str) -> np.ndarray:
    if f"{prefix}_intercept_sec" in offsets:
        return float(offsets.get(f"{prefix}_intercept_sec", 0.0)) + float(offsets.get(f"{prefix}_slope_sec_per_m", 0.0)) * xs
    return np.full_like(xs, float(offsets.get(f"{prefix}_sec", 0.0)), dtype=np.float64)


def apply_x_calibration(x_m: float, offsets: dict[str, object], xs_grid: np.ndarray) -> float:
    model = str(offsets.get("x_calibration_model", "none"))
    if model == "affine":
        x_m = float(offsets.get("x_calibration_intercept_m", 0.0)) + float(offsets.get("x_calibration_slope", 1.0)) * x_m
    elif model == "piecewise_linear":
        raw = np.asarray(offsets.get("x_calibration_raw_knots_m", []), dtype=np.float64)
        true = np.asarray(offsets.get("x_calibration_true_knots_m", []), dtype=np.float64)
        if len(raw) >= 2 and len(raw) == len(true):
            x_m = float(np.interp(x_m, raw, true, left=true[0], right=true[-1]))
    return float(np.clip(x_m, float(xs_grid[0]), float(xs_grid[-1])))


def apply_edge_dilation(x_m: float, cfg: Config, xs_grid: np.ndarray, selected_k: int) -> float:
    threshold = max(float(cfg.edge_dilation_threshold_m), 0.0)
    gain = max(float(cfg.edge_dilation_gain), 1.0)
    if selected_k < max(int(cfg.edge_dilation_min_k), 1) or threshold <= 0.0 or gain <= 1.0 or abs(x_m) < threshold:
        return float(np.clip(x_m, float(xs_grid[0]), float(xs_grid[-1])))
    dilated = math.copysign(threshold + gain * (abs(x_m) - threshold), x_m)
    return float(np.clip(dilated, float(xs_grid[0]), float(xs_grid[-1])))


def apply_center_deadband(x_m: float, cfg: Config, selected_k: int) -> float:
    if (
        cfg.center_deadband_m > 0.0
        and selected_k >= max(int(cfg.center_deadband_min_k), 1)
        and abs(x_m) <= float(cfg.center_deadband_m)
    ):
        return 0.0
    return x_m


def estimate_chirp_subband_weights(
    data_root: Path,
    trials: list[Trial],
    segment: SegmentSpec,
    cfg: Config,
    xs_grid: np.ndarray,
    offsets: dict[str, object],
) -> dict[str, float]:
    if cfg.subbands is None:
        return {}

    band_scores: dict[str, list[float]] = {band_key(band): [] for band in cfg.subbands}
    tau_vl_base, tau_vr_base = tau_templates(xs_grid, cfg)
    tau_vl = tau_vl_base + offset_values(xs_grid, offsets, "vl")  # type: ignore[arg-type]
    tau_vr = tau_vr_base + offset_values(xs_grid, offsets, "vr")  # type: ignore[arg-type]
    curve_cfg = Config(**{**asdict(cfg), "top_k_windows": max(1, cfg.top_k_windows), "subband_weight_mode": "none"})

    for trial in trials:
        curves = compute_trial_curves(data_root, trial, segment, curve_cfg)
        for item in curves["windows"]:  # type: ignore[index]
            for source in item.get("subbands", []):  # type: ignore[union-attr]
                cand = candidate_from_curves(source, xs_grid, tau_vl, tau_vr, cfg)
                err_m = abs(float(cand["x_hat"]) - trial.x_m)
                agreement_m = float(cand["agreement_m"])
                evidence = (
                    math.exp(-((err_m / 0.35) ** 2))
                    * math.exp(-((agreement_m / 0.45) ** 2))
                    * math.log1p(max(float(cand["weight"]), 0.0))
                )
                band_scores.setdefault(band_key(source.get("band_hz")), []).append(evidence)

    raw = {band: (float(np.median(values)) if values else 1.0) for band, values in band_scores.items()}
    mean = float(np.mean(list(raw.values()))) if raw else 1.0
    return {band: float(np.clip(value / max(mean, 1e-12), 0.25, 2.0)) for band, value in raw.items()}


def score_windows(
    windows: list[dict[str, object]],
    xs_grid: np.ndarray,
    tau_vl: np.ndarray,
    tau_vr: np.ndarray,
    cfg: Config,
    subband_weights: dict[str, float] | None = None,
    tau_lr: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    scores = np.zeros_like(xs_grid, dtype=np.float64)
    weights = []
    rows: list[dict[str, float]] = []
    for rank, item in enumerate(windows, start=1):
        score = aggregate_window_score(item, xs_grid, tau_vl, tau_vr, cfg, subband_weights, tau_lr)
        weight = max(float(item["reliability"]), 1e-6)
        scores += weight * score
        weights.append(weight)
        averaged = scores / max(float(np.sum(weights)), 1e-12)
        best_idx = int(np.argmax(averaged))
        x_hat = float(xs_grid[best_idx])
        rows.append(
            {
                "window_rank": float(rank),
                "window_start_sample": float(item["start"]),
                "window_reliability": weight,
                "cumulative_x_hat_m": x_hat,
                "cumulative_score": float(averaged[best_idx]),
            }
        )
    return scores / max(float(np.sum(weights)), 1e-12), rows


def select_window_prefix(prefix_rows: list[dict[str, float]], cfg: Config) -> tuple[int, dict[str, float | str]]:
    if not prefix_rows:
        return 0, {"window_selector": cfg.window_selector, "selected_k": 0}

    if cfg.window_selector == "fixed_top_k":
        selected_k = len(prefix_rows)
        return selected_k, {"window_selector": cfg.window_selector, "selected_k": selected_k}

    if cfg.window_selector != "stable_prefix":
        raise ValueError(f"Unknown window_selector: {cfg.window_selector}")

    min_k = max(1, min(int(cfg.adaptive_min_k), len(prefix_rows)))
    required = max(1, int(cfg.required_stable_steps))
    threshold = float(cfg.stability_threshold_m)
    selected_k = min(max(1, int(cfg.fallback_top_k)), len(prefix_rows))
    reason = "fallback"

    xs = [float(row["cumulative_x_hat_m"]) for row in prefix_rows]
    for idx in range(min_k - 1, len(xs)):
        if idx < required:
            continue
        recent_deltas = [abs(xs[j] - xs[j - 1]) for j in range(idx - required + 1, idx + 1)]
        if max(recent_deltas) <= threshold:
            selected_k = idx + 1
            reason = "stable"
            break

    return selected_k, {
        "window_selector": cfg.window_selector,
        "selected_k": selected_k,
        "selection_reason": reason,
        "adaptive_min_k": int(cfg.adaptive_min_k),
        "stability_threshold_m": threshold,
        "required_stable_steps": required,
        "fallback_top_k": int(cfg.fallback_top_k),
    }


def score_margin(score: np.ndarray) -> float:
    peak = float(np.max(score))
    background = float(np.percentile(score, 75))
    spread = float(np.median(np.abs(score - np.median(score))) + 1e-12)
    return max((peak - background) / spread, 0.0)


def basin_prior_from_windows(
    windows: list[dict[str, object]],
    xs_grid: np.ndarray,
    tau_vl: np.ndarray,
    tau_vr: np.ndarray,
    cfg: Config,
    subband_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float | str]]:
    prior = np.zeros_like(xs_grid, dtype=np.float64)
    candidate_xs: list[float] = []
    candidate_weights: list[float] = []
    subband_candidates: dict[str, list[float]] = {}

    for item in windows:
        sources = item.get("subbands", []) if cfg.subbands is not None else [item]
        for source in sources:  # type: ignore[assignment]
            cand = candidate_from_curves(source, xs_grid, tau_vl, tau_vr, cfg)
            if cfg.candidate_agreement_gate_m is not None and cand["agreement_m"] > cfg.candidate_agreement_gate_m:
                continue
            weight = max(float(cand["weight"]) * source_subband_weight(source, subband_weights), 1e-12)
            prior += weight * np.exp(-0.5 * ((xs_grid - cand["x_hat"]) / max(cfg.basin_sigma_m, 1e-6)) ** 2)
            candidate_xs.append(cand["x_hat"])
            candidate_weights.append(weight)
            band = source.get("band_hz") if isinstance(source, dict) else None
            band_name = band_key(band)
            subband_candidates.setdefault(band_name, []).append(cand["x_hat"])

    if float(np.max(prior)) > 0.0:
        prior = prior / float(np.max(prior))

    candidate_spread = 0.0
    if candidate_xs:
        xs = np.asarray(candidate_xs, dtype=np.float64)
        weights = np.asarray(candidate_weights, dtype=np.float64)
        center = float(np.sum(xs * weights) / max(float(np.sum(weights)), 1e-12))
        candidate_spread = float(np.sqrt(np.sum(weights * (xs - center) ** 2) / max(float(np.sum(weights)), 1e-12)))

    subband_medians = [float(np.median(values)) for values in subband_candidates.values() if values]
    subband_spread = float(max(subband_medians) - min(subband_medians)) if subband_medians else 0.0
    peak_idx = int(np.argmax(prior)) if len(prior) else 0
    return prior, {
        "basin_num_candidates": int(len(candidate_xs)),
        "basin_peak_x_m": float(xs_grid[peak_idx]) if len(xs_grid) else 0.0,
        "candidate_basin_spread_m": candidate_spread,
        "subband_median_spread_m": subband_spread,
    }


def pair_overlap_prior_from_windows(
    windows: list[dict[str, object]],
    xs_grid: np.ndarray,
    tau_vl: np.ndarray,
    tau_vr: np.ndarray,
    cfg: Config,
    subband_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float | str]]:
    left = np.zeros_like(xs_grid, dtype=np.float64)
    right = np.zeros_like(xs_grid, dtype=np.float64)
    pair_gaps: list[float] = []
    pair_weights: list[float] = []

    for item in windows:
        sources = item.get("subbands", []) if cfg.subbands is not None else [item]
        for source in sources:  # type: ignore[assignment]
            cand = candidate_from_curves(source, xs_grid, tau_vl, tau_vr, cfg)
            weight = max(float(cand["weight"]) * source_subband_weight(source, subband_weights), 1e-12)
            sigma = max(cfg.basin_sigma_m, 1e-6)
            left += weight * np.exp(-0.5 * ((xs_grid - cand["x_vl"]) / sigma) ** 2)
            right += weight * np.exp(-0.5 * ((xs_grid - cand["x_vr"]) / sigma) ** 2)
            pair_gaps.append(float(cand["agreement_m"]))
            pair_weights.append(weight)

    if float(np.max(left)) > 0.0:
        left = left / float(np.max(left))
    if float(np.max(right)) > 0.0:
        right = right / float(np.max(right))
    prior = np.sqrt(left * right)
    if float(np.max(prior)) > 0.0:
        prior = prior / float(np.max(prior))

    peak_idx = int(np.argmax(prior)) if len(prior) else 0
    weights = np.asarray(pair_weights, dtype=np.float64)
    gaps = np.asarray(pair_gaps, dtype=np.float64)
    if len(gaps) and float(np.sum(weights)) > 0.0:
        mean_gap = float(np.sum(weights * gaps) / float(np.sum(weights)))
    else:
        mean_gap = 0.0
    return prior, {
        "pair_overlap_peak_x_m": float(xs_grid[peak_idx]) if len(xs_grid) else 0.0,
        "pair_overlap_mean_gap_m": mean_gap,
    }


def apply_basin_validation(
    score: np.ndarray,
    windows: list[dict[str, object]],
    xs_grid: np.ndarray,
    tau_vl: np.ndarray,
    tau_vr: np.ndarray,
    cfg: Config,
    subband_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float | str]]:
    meta: dict[str, float | str] = {"basin_mode": cfg.basin_mode}
    if cfg.basin_mode == "none":
        return score, meta

    if cfg.basin_mode.startswith("pair_overlap"):
        prior, prior_meta = pair_overlap_prior_from_windows(windows, xs_grid, tau_vl, tau_vr, cfg, subband_weights)
    else:
        prior, prior_meta = basin_prior_from_windows(windows, xs_grid, tau_vl, tau_vr, cfg, subband_weights)
    meta.update(prior_meta)
    if float(np.max(prior)) <= 0.0:
        meta["basin_reason"] = "empty_prior"
        return score, meta

    normalized = score - float(np.min(score))
    normalized /= max(float(np.max(normalized)), 1e-12)
    if cfg.basin_mode in ("basin_gate", "pair_overlap_gate"):
        final = np.where(prior >= cfg.basin_gate, normalized, 0.0)
        if float(np.max(final)) <= 0.0:
            meta["basin_reason"] = "gate_rejected_all_fallback"
            final = normalized
    elif cfg.basin_mode in ("basin_mul", "pair_overlap_mul"):
        final = normalized * np.power(prior + 1e-6, cfg.basin_power)
    else:
        raise ValueError(f"Unknown basin_mode: {cfg.basin_mode}")

    meta["basin_prior_at_selected"] = float(prior[int(np.argmax(final))])
    meta["basin_changed"] = float(xs_grid[int(np.argmax(final))] != xs_grid[int(np.argmax(score))])
    return final, meta


def confidence_prefix_rows(
    windows: list[dict[str, object]],
    xs_grid: np.ndarray,
    tau_vl: np.ndarray,
    tau_vr: np.ndarray,
    cfg: Config,
    subband_weights: dict[str, float] | None = None,
    tau_lr: np.ndarray | None = None,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    min_k = max(1, min(int(cfg.adaptive_min_k), len(windows)))
    for k in range(min_k, len(windows) + 1):
        score, _ = score_windows(windows[:k], xs_grid, tau_vl, tau_vr, cfg, subband_weights, tau_lr)
        best_idx = int(np.argmax(score))
        x_hat = float(xs_grid[best_idx])
        margin = score_margin(score)
        basin_prior, basin_meta = basin_prior_from_windows(windows[:k], xs_grid, tau_vl, tau_vr, cfg, subband_weights)
        pair_prior, pair_meta = pair_overlap_prior_from_windows(windows[:k], xs_grid, tau_vl, tau_vr, cfg, subband_weights)
        basin_at = float(basin_prior[best_idx]) if len(basin_prior) else 0.0
        pair_at = float(pair_prior[best_idx]) if len(pair_prior) else 0.0
        candidate_spread = float(basin_meta.get("candidate_basin_spread_m", 0.0))
        subband_spread = float(basin_meta.get("subband_median_spread_m", 0.0))
        confidence = (
            math.tanh(margin / 6.0)
            * (0.25 + 0.75 * basin_at)
            * (0.25 + 0.75 * pair_at)
            * math.exp(-((candidate_spread / 0.45) ** 2))
            * math.exp(-((subband_spread / 0.75) ** 2))
        )
        rows.append(
            {
                "window_rank": float(k),
                "x_hat_m": x_hat,
                "score_margin": margin,
                "basin_prior_at_x": basin_at,
                "pair_prior_at_x": pair_at,
                "candidate_basin_spread_m": candidate_spread,
                "subband_median_spread_m": subband_spread,
                "pair_overlap_mean_gap_m": float(pair_meta.get("pair_overlap_mean_gap_m", 0.0)),
                "confidence": float(confidence),
            }
        )
    return rows


def select_confidence_prefix(
    windows: list[dict[str, object]],
    xs_grid: np.ndarray,
    tau_vl: np.ndarray,
    tau_vr: np.ndarray,
    cfg: Config,
    subband_weights: dict[str, float] | None = None,
    tau_lr: np.ndarray | None = None,
) -> tuple[int, dict[str, float | str], list[dict[str, float]]]:
    rows = confidence_prefix_rows(windows, xs_grid, tau_vl, tau_vr, cfg, subband_weights, tau_lr)
    if not rows:
        return 0, {"window_selector": cfg.window_selector, "selected_k": 0, "selection_reason": "empty"}, rows

    best = max(rows, key=lambda row: (float(row["confidence"]), -float(row["window_rank"])))
    selected_k = int(best["window_rank"])
    return selected_k, {
        "window_selector": cfg.window_selector,
        "selected_k": selected_k,
        "selection_reason": "max_confidence",
        "prefix_confidence": float(best["confidence"]),
        "prefix_score_margin": float(best["score_margin"]),
        "prefix_basin_prior_at_x": float(best["basin_prior_at_x"]),
        "prefix_pair_prior_at_x": float(best["pair_prior_at_x"]),
        "prefix_candidate_basin_spread_m": float(best["candidate_basin_spread_m"]),
        "prefix_subband_median_spread_m": float(best["subband_median_spread_m"]),
        "prefix_pair_overlap_mean_gap_m": float(best["pair_overlap_mean_gap_m"]),
        "adaptive_min_k": int(cfg.adaptive_min_k),
    }, rows


def select_hysteresis_prefix(
    prefix_rows: list[dict[str, float]],
    windows: list[dict[str, object]],
    xs_grid: np.ndarray,
    tau_vl: np.ndarray,
    tau_vr: np.ndarray,
    cfg: Config,
    subband_weights: dict[str, float] | None = None,
    tau_lr: np.ndarray | None = None,
) -> tuple[int, dict[str, float | str], list[dict[str, float]]]:
    stable_cfg = Config(**{**asdict(cfg), "window_selector": "stable_prefix", "adaptive_min_k": max(8, int(cfg.adaptive_min_k))})
    stable_k, stable_meta = select_window_prefix(prefix_rows, stable_cfg)
    rows = confidence_prefix_rows(windows, xs_grid, tau_vl, tau_vr, cfg, subband_weights, tau_lr)
    if not rows:
        return stable_k, {**stable_meta, "window_selector": cfg.window_selector, "selection_reason": "stable_empty_confidence"}, rows

    global_best = max(rows, key=lambda row: (float(row["confidence"]), -float(row["window_rank"])))
    max_conf = max(float(global_best["confidence"]), 1e-12)

    rollback_row: dict[str, float] | None = None
    best_before_jump: dict[str, float] | None = None
    for prev, cur in zip(rows, rows[1:]):
        if best_before_jump is None or float(prev["confidence"]) > float(best_before_jump["confidence"]):
            best_before_jump = prev
        jump_m = abs(float(cur["x_hat_m"]) - float(prev["x_hat_m"]))
        if (
            jump_m >= 0.25
            and best_before_jump is not None
            and float(best_before_jump["confidence"]) >= 0.55 * max_conf
            and float(best_before_jump["pair_prior_at_x"]) >= 0.50
        ):
            rollback_row = best_before_jump
            break

    stable_row = next((row for row in rows if int(row["window_rank"]) == int(stable_k)), None)
    selected = global_best
    reason = "max_confidence"
    if rollback_row is not None:
        selected = rollback_row
        reason = "rollback_before_jump"
    elif stable_row is not None:
        stable_gap = abs(float(global_best["x_hat_m"]) - float(stable_row["x_hat_m"]))
        confidence_gain = float(global_best["confidence"]) / max(float(stable_row["confidence"]), 1e-12)
        if stable_gap >= 0.08 and confidence_gain < 10.0:
            selected = stable_row
            reason = "stable_guardrail"

    selected_k = int(selected["window_rank"])
    return selected_k, {
        "window_selector": cfg.window_selector,
        "selected_k": selected_k,
        "selection_reason": reason,
        "stable_guardrail_k": int(stable_k),
        "global_confidence_k": int(global_best["window_rank"]),
        "prefix_confidence": float(selected["confidence"]),
        "prefix_score_margin": float(selected["score_margin"]),
        "prefix_basin_prior_at_x": float(selected["basin_prior_at_x"]),
        "prefix_pair_prior_at_x": float(selected["pair_prior_at_x"]),
        "prefix_candidate_basin_spread_m": float(selected["candidate_basin_spread_m"]),
        "prefix_subband_median_spread_m": float(selected["subband_median_spread_m"]),
        "prefix_pair_overlap_mean_gap_m": float(selected["pair_overlap_mean_gap_m"]),
        "adaptive_min_k": int(cfg.adaptive_min_k),
    }, rows


def evaluate_trial(
    data_root: Path,
    trial: Trial,
    segment: SegmentSpec,
    cfg: Config,
    xs_grid: np.ndarray,
    offsets: dict[str, float],
) -> dict[str, object]:
    use_adaptive = cfg.window_selector != "fixed_top_k" and segment.name == "speech"
    curve_cfg = Config(**{**asdict(cfg), "top_k_windows": 0}) if use_adaptive else cfg
    curves = compute_trial_curves(data_root, trial, segment, curve_cfg)
    tau_vl, tau_vr = tau_templates(xs_grid, cfg)
    tau_lr = tau_lr_template(xs_grid)
    if offsets.get("model") == "per_trial" and isinstance(offsets.get("per_trial"), dict):
        trial_offsets = offsets["per_trial"].get(trial.label, {})  # type: ignore[index]
        tau_vl = tau_vl + float(trial_offsets.get("vl_sec", 0.0))
        tau_vr = tau_vr + float(trial_offsets.get("vr_sec", 0.0))
        tau_lr = tau_lr + float(trial_offsets.get("lr_sec", 0.0))
    else:
        tau_vl = tau_vl + offset_values(xs_grid, offsets, "vl")
        tau_vr = tau_vr + offset_values(xs_grid, offsets, "vr")
        tau_lr = tau_lr + offset_values(xs_grid, offsets, "lr")

    subband_weights = offsets.get("subband_weights") if isinstance(offsets.get("subband_weights"), dict) else None
    consensus_meta: dict[str, float | str] = {}
    if cfg.estimator == "consensus":
        candidates = []
        for item in curves["windows"]:  # type: ignore[index]
            sources = item.get("subbands", []) if cfg.subbands is not None else [item]  # type: ignore[union-attr]
            for source in sources:
                candidates.append(candidate_from_curves(source, xs_grid, tau_vl, tau_vr, cfg))
        x_hat, consensus_meta = consensus_x(candidates)
        score_value = float(consensus_meta.get("cluster_weight", 0.0))
    elif cfg.estimator == "score":
        windows = list(curves["windows"])  # type: ignore[arg-type]
        scores, prefix_rows = score_windows(windows, xs_grid, tau_vl, tau_vr, cfg, subband_weights, tau_lr)  # type: ignore[arg-type]
        selection_meta: dict[str, float | str] = {"window_selector": "fixed_top_k", "selected_k": len(windows)}
        if use_adaptive:
            if cfg.window_selector == "confidence_prefix":
                selected_k, selection_meta, _ = select_confidence_prefix(windows, xs_grid, tau_vl, tau_vr, cfg, subband_weights, tau_lr)  # type: ignore[arg-type]
            elif cfg.window_selector == "hysteresis_prefix":
                selected_k, selection_meta, _ = select_hysteresis_prefix(prefix_rows, windows, xs_grid, tau_vl, tau_vr, cfg, subband_weights, tau_lr)  # type: ignore[arg-type]
            else:
                selected_k, selection_meta = select_window_prefix(prefix_rows, cfg)
            scores, prefix_rows = score_windows(windows[:selected_k], xs_grid, tau_vl, tau_vr, cfg, subband_weights, tau_lr)  # type: ignore[arg-type]
            windows = windows[:selected_k]
        scores, basin_meta = apply_basin_validation(scores, windows, xs_grid, tau_vl, tau_vr, cfg, subband_weights)  # type: ignore[arg-type]
        best_idx = int(np.argmax(scores))
        x_hat = float(xs_grid[best_idx])
        score_value = float(scores[best_idx])
        consensus_meta = {**selection_meta, **basin_meta}
    else:
        raise ValueError(f"Unknown estimator: {cfg.estimator}")

    x_raw = x_hat
    if cfg.x_calibration != "none":
        x_hat = apply_x_calibration(x_hat, offsets, xs_grid)  # type: ignore[arg-type]
    selected_k = int(consensus_meta.get("selected_k", len(curves["windows"])))  # type: ignore[arg-type]
    x_hat = apply_edge_dilation(x_hat, cfg, xs_grid, selected_k)
    x_hat = apply_center_deadband(x_hat, cfg, selected_k)

    theta_hat = theta_from_x(x_hat)
    theta_true = theta_from_x(trial.x_m)
    return {
        "label": trial.label,
        "x_true_m": trial.x_m,
        "x_hat_m": x_hat,
        "x_raw_m": x_raw,
        "theta_true_deg": theta_true,
        "theta_hat_deg": theta_hat,
        "abs_err_deg": abs(theta_hat - theta_true),
        "score": score_value,
        "num_windows": selected_k,
        "available_windows": len(curves["windows"]),  # type: ignore[arg-type]
        "consensus": consensus_meta,
    }


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    errs = np.array([float(r["abs_err_deg"]) for r in rows], dtype=np.float64)
    return {
        "mae_deg": float(np.mean(errs)),
        "max_err_deg": float(np.max(errs)),
        "rows": rows,
    }


def summarize_trials(
    data_root: Path,
    trials: list[Trial],
    segment: SegmentSpec,
    cfg: Config,
    xs_grid: np.ndarray,
    offsets: dict[str, float],
) -> dict[str, object]:
    return summarize_rows([evaluate_trial(data_root, t, segment, cfg, xs_grid, offsets) for t in trials])


def estimate_x_calibration(
    data_root: Path,
    trials: list[Trial],
    segment: SegmentSpec,
    cfg: Config,
    xs_grid: np.ndarray,
    offsets: dict[str, object],
) -> dict[str, object]:
    if cfg.x_calibration == "none":
        return offsets

    raw_cfg = Config(**{**asdict(cfg), "x_calibration": "none"})
    rows = [evaluate_trial(data_root, trial, segment, raw_cfg, xs_grid, offsets) for trial in trials]  # type: ignore[arg-type]
    raw_x = np.asarray([float(row["x_hat_m"]) for row in rows], dtype=np.float64)
    true_x = np.asarray([trial.x_m for trial in trials], dtype=np.float64)
    out = dict(offsets)
    out["x_calibration_model"] = cfg.x_calibration

    if cfg.x_calibration == "affine":
        if float(np.std(raw_x)) <= 1e-9:
            out["x_calibration_intercept_m"] = 0.0
            out["x_calibration_slope"] = 1.0
        else:
            slope, intercept = np.polyfit(raw_x, true_x, 1)
            out["x_calibration_intercept_m"] = float(np.clip(intercept, -0.8, 0.8))
            out["x_calibration_slope"] = float(np.clip(slope, 0.25, 2.5))
        return out

    if cfg.x_calibration == "piecewise_linear":
        order = np.argsort(raw_x)
        raw_sorted = raw_x[order]
        true_sorted = true_x[order]
        unique_raw = []
        unique_true = []
        for value in np.unique(raw_sorted):
            mask = np.isclose(raw_sorted, value)
            unique_raw.append(float(value))
            unique_true.append(float(np.median(true_sorted[mask])))
        if len(unique_raw) < 2:
            unique_raw = [float(xs_grid[0]), float(xs_grid[-1])]
            unique_true = [float(xs_grid[0]), float(xs_grid[-1])]
        out["x_calibration_raw_knots_m"] = unique_raw
        out["x_calibration_true_knots_m"] = unique_true
        return out

    raise ValueError(f"Unknown x_calibration: {cfg.x_calibration}")


def summarize_loro(results: list[dict[str, object]]) -> dict[str, object]:
    if not results or "combined_speech" not in results[0]:
        return {"mae_deg": 0.0, "max_err_deg": 0.0, "rows": []}

    labels = [str(row["label"]) for row in results[0]["combined_speech"]["rows"]]  # type: ignore[index]
    rows = []
    for held_out in labels:
        best_item = min(
            results,
            key=lambda item: float(
                np.mean(
                    [
                        float(row["abs_err_deg"])
                        for row in item["combined_speech"]["rows"]  # type: ignore[index]
                        if str(row["label"]) != held_out
                    ]
                )
            ),
        )
        held_row = next(row for row in best_item["combined_speech"]["rows"] if str(row["label"]) == held_out)  # type: ignore[index]
        train_errs = [
            float(row["abs_err_deg"])
            for row in best_item["combined_speech"]["rows"]  # type: ignore[index]
            if str(row["label"]) != held_out
        ]
        rows.append(
            {
                "held_out_label": held_out,
                "selected_config": best_item["config"]["name"],  # type: ignore[index]
                "train_mae_deg": float(np.mean(train_errs)),
                "x_hat_m": float(held_row["x_hat_m"]),
                "abs_err_deg": float(held_row["abs_err_deg"]),
                "selected_k": int(held_row.get("num_windows", 0)),
            }
        )

    return summarize_rows(
        [
            {
                "label": row["held_out_label"],
                "x_true_m": 0.0,
                "x_hat_m": row["x_hat_m"],
                "theta_true_deg": 0.0,
                "theta_hat_deg": 0.0,
                "abs_err_deg": row["abs_err_deg"],
                "selected_config": row["selected_config"],
                "train_mae_deg": row["train_mae_deg"],
                "selected_k": row["selected_k"],
            }
            for row in rows
        ]
    ) | {"rows": rows}


def band_key(band: object) -> str:
    if band is None:
        return "wide"
    lo, hi = band  # type: ignore[misc]
    return f"{float(lo):.0f}-{float(hi):.0f}"


def source_subband_weight(source: dict[str, object], subband_weights: dict[str, float] | None) -> float:
    if not subband_weights:
        return 1.0
    return max(float(subband_weights.get(band_key(source.get("band_hz")), 1.0)), 1e-6)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    total = float(np.sum(weights))
    if total <= 1e-12:
        return float(np.median(values))
    cdf = np.cumsum(weights) / total
    return float(values[int(np.searchsorted(cdf, 0.5, side="left"))])


def candidate_from_curves(
    source: dict[str, object],
    xs_grid: np.ndarray,
    tau_vl: np.ndarray,
    tau_vr: np.ndarray,
    cfg: Config,
) -> dict[str, float]:
    vl_curve = np.asarray(source["vl"], dtype=np.float64)
    vr_curve = np.asarray(source["vr"], dtype=np.float64)
    lags_vl = np.asarray(source["lags_vl"], dtype=np.float64)
    lags_vr = np.asarray(source["lags_vr"], dtype=np.float64)
    vl = sample_curve(vl_curve, lags_vl, tau_vl)
    vr = sample_curve(vr_curve, lags_vr, tau_vr)
    score = score_curve_from_source(source, xs_grid, tau_vl, tau_vr, cfg)

    best_idx = int(np.argmax(score))
    x_hat = float(xs_grid[best_idx])
    if cfg.common_shift_radius_ms > 0.0:
        x_vl = x_hat
        x_vr = x_hat
        agreement_m = 0.0
    else:
        x_vl = float(xs_grid[int(np.argmax(vl))])
        x_vr = float(xs_grid[int(np.argmax(vr))])
        agreement_m = abs(x_vl - x_vr)

    peak = float(score[best_idx])
    background = float(np.percentile(score, 75))
    spread = float(np.median(np.abs(score - np.median(score))) + 1e-12)
    margin = max((peak - background) / spread, 0.0)
    reliability = float(source.get("reliability", window_reliability(vl_curve, vr_curve)))
    consistency = math.exp(-((agreement_m / 0.25) ** 2))
    return {
        "x_hat": x_hat,
        "weight": max(reliability * margin * consistency, 1e-9),
        "score": peak,
        "agreement_m": agreement_m,
        "x_vl": x_vl,
        "x_vr": x_vr,
    }


def score_curve_from_source(
    source: dict[str, object],
    xs_grid: np.ndarray,
    tau_vl: np.ndarray,
    tau_vr: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    if cfg.common_shift_radius_ms > 0.0:
        shifts = np.linspace(
            -cfg.common_shift_radius_ms / 1000.0,
            cfg.common_shift_radius_ms / 1000.0,
            max(int(cfg.common_shift_steps), 3),
        )
        shifted_scores = []
        for shift in shifts:
            vl = sample_curve(np.asarray(source["vl"]), np.asarray(source["lags_vl"]), tau_vl + shift)
            vr = sample_curve(np.asarray(source["vr"]), np.asarray(source["lags_vr"]), tau_vr + shift)
            shifted_scores.append(combine_scores(vl, vr, cfg.score_mode))
        return np.max(np.vstack(shifted_scores), axis=0)

    vl = sample_curve(np.asarray(source["vl"]), np.asarray(source["lags_vl"]), tau_vl)
    vr = sample_curve(np.asarray(source["vr"]), np.asarray(source["lags_vr"]), tau_vr)
    return combine_scores(vl, vr, cfg.score_mode)


def aggregate_window_score(
    item: dict[str, object],
    xs_grid: np.ndarray,
    tau_vl: np.ndarray,
    tau_vr: np.ndarray,
    cfg: Config,
    subband_weights: dict[str, float] | None = None,
    tau_lr: np.ndarray | None = None,
) -> np.ndarray:
    if cfg.score_aggregator == "curve_mean" or cfg.subbands is None:
        score = score_curve_from_source(item, xs_grid, tau_vl, tau_vr, cfg)
        return apply_lr_prior_to_score(score, item, tau_lr, cfg)

    sources = item.get("subbands", [])
    if not sources:
        score = score_curve_from_source(item, xs_grid, tau_vl, tau_vr, cfg)
        return apply_lr_prior_to_score(score, item, tau_lr, cfg)

    score_matrix = np.vstack(
        [apply_lr_prior_to_score(score_curve_from_source(source, xs_grid, tau_vl, tau_vr, cfg), source, tau_lr, cfg) for source in sources]  # type: ignore[arg-type]
    )
    weights = np.asarray([source_subband_weight(source, subband_weights) for source in sources], dtype=np.float64)  # type: ignore[arg-type]
    weights = weights / max(float(np.sum(weights)), 1e-12)
    mean_score = np.sum(score_matrix * weights[:, None], axis=0)
    if cfg.score_aggregator == "subband_score_mean":
        return mean_score

    spread = np.sqrt(np.sum(weights[:, None] * (score_matrix - mean_score) ** 2, axis=0))
    if cfg.score_aggregator == "subband_score_minus_std":
        return mean_score - cfg.subband_penalty * spread

    if cfg.score_aggregator == "subband_score_exp_cv":
        cv = spread / (np.abs(mean_score) + 1e-6)
        return mean_score * np.exp(-cfg.subband_penalty * cv)

    if cfg.score_aggregator == "subband_jackknife":
        if len(sources) <= 2:
            return mean_score
        jackknife = []
        for leave_out in range(len(sources)):
            keep = np.ones(len(sources), dtype=bool)
            keep[leave_out] = False
            keep_weights = weights[keep] / max(float(np.sum(weights[keep])), 1e-12)
            jackknife.append(np.sum(score_matrix[keep] * keep_weights[:, None], axis=0))
        jackknife_matrix = np.vstack(jackknife)
        return mean_score - cfg.subband_penalty * np.std(jackknife_matrix, axis=0)

    if cfg.score_aggregator == "subband_cluster_max":
        candidate_xs = []
        candidate_weights = []
        for source, score, weight in zip(sources, score_matrix, weights):  # type: ignore[arg-type]
            candidate_xs.append(float(xs_grid[int(np.argmax(score))]))
            candidate_weights.append(float(weight) * max(score_margin(score), 1e-6) * float(source.get("reliability", 1.0)))
        candidate_xs_arr = np.asarray(candidate_xs, dtype=np.float64)
        candidate_weights_arr = np.asarray(candidate_weights, dtype=np.float64)
        cluster_weights = np.array(
            [
                float(np.sum(candidate_weights_arr[np.abs(candidate_xs_arr - x) <= 0.18]))
                for x in candidate_xs_arr
            ],
            dtype=np.float64,
        )
        center = candidate_xs_arr[int(np.argmax(cluster_weights))]
        keep = np.abs(candidate_xs_arr - center) <= 0.18
        if np.count_nonzero(keep) < 2:
            return mean_score
        keep_weights = weights[keep] / max(float(np.sum(weights[keep])), 1e-12)
        return np.sum(score_matrix[keep] * keep_weights[:, None], axis=0)

    raise ValueError(f"Unknown score_aggregator: {cfg.score_aggregator}")


def lr_prior_curve(source: dict[str, object], tau_lr: np.ndarray | None) -> np.ndarray | None:
    if tau_lr is None or "lr" not in source or "lags_lr" not in source:
        return None
    return sample_curve(np.asarray(source["lr"]), np.asarray(source["lags_lr"]), tau_lr)


def apply_lr_prior_to_score(
    score: np.ndarray,
    source: dict[str, object],
    tau_lr: np.ndarray | None,
    cfg: Config,
) -> np.ndarray:
    if cfg.lr_weight <= 0.0:
        return score
    lr = lr_prior_curve(source, tau_lr)
    if lr is None:
        return score
    lr_norm = lr - float(np.min(lr))
    lr_norm /= max(float(np.max(lr_norm)), 1e-12)
    prior = np.power(0.15 + 0.85 * lr_norm, cfg.lr_weight)
    if cfg.lr_gate > 0.0:
        prior = np.where(lr_norm >= cfg.lr_gate, prior, 0.15**cfg.lr_weight)
    return score * prior


def consensus_x(candidates: list[dict[str, float]], cluster_radius_m: float = 0.16) -> tuple[float, dict[str, float]]:
    if not candidates:
        return 0.0, {"num_candidates": 0, "cluster_weight": 0.0}
    xs = np.array([c["x_hat"] for c in candidates], dtype=np.float64)
    weights = np.array([c["weight"] for c in candidates], dtype=np.float64)
    cluster_weights = np.array([float(np.sum(weights[np.abs(xs - x) <= cluster_radius_m])) for x in xs])
    center_idx = int(np.argmax(cluster_weights))
    keep = np.abs(xs - xs[center_idx]) <= cluster_radius_m
    x_hat = weighted_median(xs[keep], weights[keep])
    return x_hat, {
        "num_candidates": int(len(candidates)),
        "num_cluster_candidates": int(np.count_nonzero(keep)),
        "cluster_weight": float(cluster_weights[center_idx]),
        "median_agreement_m": float(np.median([c["agreement_m"] for c in candidates])),
    }


def config_from_payload(payload: dict[str, object]) -> Config:
    subbands = payload.get("subbands")
    return Config(
        name=str(payload["name"]),
        band_hz=tuple(payload["band_hz"]) if payload.get("band_hz") is not None else None,  # type: ignore[arg-type]
        phat_beta=float(payload["phat_beta"]),
        transform=str(payload["transform"]),
        geometry=str(payload["geometry"]),
        ldv_y_m=float(payload["ldv_y_m"]),
        ldv_x_m=float(payload.get("ldv_x_m", 0.0)),
        wall_speed_mps=float(payload.get("wall_speed_mps", 0.0)),
        score_mode=str(payload["score_mode"]),
        n_fft=int(payload["n_fft"]),
        hop=int(payload["hop"]),
        window_sec=float(payload["window_sec"]),
        window_hop_sec=float(payload["window_hop_sec"]),
        top_k_windows=int(payload["top_k_windows"]),
        local_peak_radius_ms=float(payload["local_peak_radius_ms"]),
        gcc_mode=str(payload.get("gcc_mode", "plain")),
        estimator=str(payload.get("estimator", "score")),
        coherence_floor=float(payload.get("coherence_floor", 0.0)),
        subbands=tuple(tuple(float(v) for v in band) for band in subbands) if subbands is not None else None,  # type: ignore[union-attr]
        score_aggregator=str(payload.get("score_aggregator", "curve_mean")),
        subband_penalty=float(payload.get("subband_penalty", 0.0)),
        window_selector=str(payload.get("window_selector", "fixed_top_k")),
        adaptive_min_k=int(payload.get("adaptive_min_k", 8)),
        stability_threshold_m=float(payload.get("stability_threshold_m", 0.03)),
        required_stable_steps=int(payload.get("required_stable_steps", 2)),
        fallback_top_k=int(payload.get("fallback_top_k", 9)),
        basin_mode=str(payload.get("basin_mode", "none")),
        basin_sigma_m=float(payload.get("basin_sigma_m", 0.08)),
        basin_gate=float(payload.get("basin_gate", 0.10)),
        basin_power=float(payload.get("basin_power", 0.25)),
        candidate_agreement_gate_m=float(payload["candidate_agreement_gate_m"]) if payload.get("candidate_agreement_gate_m") is not None else None,
        subband_weight_mode=str(payload.get("subband_weight_mode", "none")),
        lr_weight=float(payload.get("lr_weight", 0.0)),
        lr_gate=float(payload.get("lr_gate", 0.0)),
        common_shift_radius_ms=float(payload.get("common_shift_radius_ms", 0.0)),
        common_shift_steps=int(payload.get("common_shift_steps", 0)),
        x_calibration=str(payload.get("x_calibration", "none")),
        correlation_polarity=str(payload.get("correlation_polarity", "abs")),
        edge_dilation_threshold_m=float(payload.get("edge_dilation_threshold_m", 0.0)),
        edge_dilation_gain=float(payload.get("edge_dilation_gain", 1.0)),
        edge_dilation_min_k=int(payload.get("edge_dilation_min_k", 1)),
        center_deadband_m=float(payload.get("center_deadband_m", 0.0)),
        center_deadband_min_k=int(payload.get("center_deadband_min_k", 1)),
    )


def diagnose_trial_windows(
    data_root: Path,
    trial: Trial,
    segment: SegmentSpec,
    cfg: Config,
    xs_grid: np.ndarray,
    offsets: dict[str, float],
) -> list[dict[str, float | str | None]]:
    curve_cfg = Config(**{**asdict(cfg), "top_k_windows": 0}) if cfg.window_selector != "fixed_top_k" else cfg
    curves = compute_trial_curves(data_root, trial, segment, curve_cfg)
    tau_vl, tau_vr = tau_templates(xs_grid, cfg)
    tau_vl = tau_vl + offset_values(xs_grid, offsets, "vl")
    tau_vr = tau_vr + offset_values(xs_grid, offsets, "vr")
    tau_lr = tau_lr_template(xs_grid) + offset_values(xs_grid, offsets, "lr")
    subband_weights = offsets.get("subband_weights") if isinstance(offsets.get("subband_weights"), dict) else None

    rows = []
    fs = float(curves["fs"])  # type: ignore[index]
    for item in curves["windows"]:  # type: ignore[index]
        sources = item.get("subbands", []) if cfg.subbands is not None else [item]  # type: ignore[union-attr]
        for source in sources:
            cand = candidate_from_curves(source, xs_grid, tau_vl, tau_vr, cfg)
            lr = lr_prior_curve(source, tau_lr if cfg.lr_weight > 0.0 else None)
            lr_at_x = float(lr[int(np.argmin(np.abs(xs_grid - cand["x_hat"])))]) if lr is not None else 0.0
            band = source.get("band_hz") if isinstance(source, dict) else None
            rows.append(
                {
                    "trial": trial.label,
                    "segment": segment.name,
                    "window_start_sec": float(item["start"]) / fs,  # type: ignore[index]
                    "band_hz": f"{band[0]:.0f}-{band[1]:.0f}" if band is not None else None,
                    "x_hat_m": cand["x_hat"],
                    "abs_err_deg": abs(theta_from_x(cand["x_hat"]) - theta_from_x(trial.x_m)),
                    "weight": cand["weight"],
                    "subband_weight": source_subband_weight(source, subband_weights),  # type: ignore[arg-type]
                    "lr_prior_at_x": lr_at_x,
                    "agreement_m": cand["agreement_m"],
                    "x_vl_m": cand["x_vl"],
                    "x_vr_m": cand["x_vr"],
                }
            )
    return rows


def diagnose_incremental_windows(
    data_root: Path,
    trial: Trial,
    segment: SegmentSpec,
    cfg: Config,
    xs_grid: np.ndarray,
    offsets: dict[str, float],
) -> list[dict[str, float | str]]:
    all_window_cfg = Config(**{**asdict(cfg), "top_k_windows": 0})
    curves = compute_trial_curves(data_root, trial, segment, all_window_cfg)
    tau_vl, tau_vr = tau_templates(xs_grid, cfg)
    tau_vl = tau_vl + offset_values(xs_grid, offsets, "vl")
    tau_vr = tau_vr + offset_values(xs_grid, offsets, "vr")
    tau_lr = tau_lr_template(xs_grid) + offset_values(xs_grid, offsets, "lr")
    subband_weights = offsets.get("subband_weights") if isinstance(offsets.get("subband_weights"), dict) else None

    fs = float(curves["fs"])  # type: ignore[index]
    scores = np.zeros_like(xs_grid, dtype=np.float64)
    weights: list[float] = []
    rows = []
    for rank, item in enumerate(curves["windows"], start=1):  # type: ignore[index]
        score = aggregate_window_score(item, xs_grid, tau_vl, tau_vr, cfg, subband_weights, tau_lr)  # type: ignore[arg-type]
        weight = max(float(item["reliability"]), 1e-6)  # type: ignore[index]
        scores += weight * score
        weights.append(weight)
        averaged = scores / max(float(np.sum(weights)), 1e-12)
        x_hat = float(xs_grid[int(np.argmax(averaged))])
        rows.append(
            {
                "trial": trial.label,
                "segment": segment.name,
                "window_rank": rank,
                "window_start_sec": float(item["start"]) / fs,  # type: ignore[index]
                "window_reliability": weight,
                "cumulative_x_hat_m": x_hat,
                "cumulative_abs_err_deg": abs(theta_from_x(x_hat) - theta_from_x(trial.x_m)),
            }
        )
    return rows


def diagnose_prefix_confidence(
    data_root: Path,
    trial: Trial,
    segment: SegmentSpec,
    cfg: Config,
    xs_grid: np.ndarray,
    offsets: dict[str, float],
) -> list[dict[str, float | str]]:
    all_window_cfg = Config(**{**asdict(cfg), "top_k_windows": 0})
    curves = compute_trial_curves(data_root, trial, segment, all_window_cfg)
    tau_vl, tau_vr = tau_templates(xs_grid, cfg)
    tau_vl = tau_vl + offset_values(xs_grid, offsets, "vl")
    tau_vr = tau_vr + offset_values(xs_grid, offsets, "vr")
    tau_lr = tau_lr_template(xs_grid) + offset_values(xs_grid, offsets, "lr")
    subband_weights = offsets.get("subband_weights") if isinstance(offsets.get("subband_weights"), dict) else None

    rows = confidence_prefix_rows(curves["windows"], xs_grid, tau_vl, tau_vr, cfg, subband_weights, tau_lr)  # type: ignore[arg-type]
    selected_k = 0
    if rows:
        if cfg.window_selector == "hysteresis_prefix":
            prefix_scores, prefix_rows = score_windows(curves["windows"], xs_grid, tau_vl, tau_vr, cfg, subband_weights, tau_lr)  # type: ignore[arg-type]
            selected_k, _, _ = select_hysteresis_prefix(prefix_rows, curves["windows"], xs_grid, tau_vl, tau_vr, cfg, subband_weights, tau_lr)  # type: ignore[arg-type]
        else:
            selected_k = int(max(rows, key=lambda row: (float(row["confidence"]), -float(row["window_rank"])))["window_rank"])

    out: list[dict[str, float | str]] = []
    for row in rows:
        x_hat = float(row["x_hat_m"])
        out.append(
            {
                "trial": trial.label,
                "segment": segment.name,
                "window_rank": int(row["window_rank"]),
                "x_hat_m": x_hat,
                "abs_err_deg": abs(theta_from_x(x_hat) - theta_from_x(trial.x_m)),
                "selected": float(int(int(row["window_rank"]) == selected_k)),
                "confidence": float(row["confidence"]),
                "score_margin": float(row["score_margin"]),
                "basin_prior_at_x": float(row["basin_prior_at_x"]),
                "pair_prior_at_x": float(row["pair_prior_at_x"]),
                "candidate_basin_spread_m": float(row["candidate_basin_spread_m"]),
                "subband_median_spread_m": float(row["subband_median_spread_m"]),
                "pair_overlap_mean_gap_m": float(row["pair_overlap_mean_gap_m"]),
            }
        )
    return out


def candidate_configs(profile: str) -> list[Config]:
    bands: list[tuple[float, float] | None]
    selector_options = [("fixed_top_k", 8, 0.03, 2, 9)]
    basin_options = [("none", 0.08, 0.10, 0.25, None)]
    subband_weight_options = ["none"]
    lr_options = [(0.0, 0.0)]
    wall_speed_options = [0.0]
    common_shift_options = [(0.0, 0)]
    x_calibration_options = ["none"]
    polarity_options = ["abs"]
    edge_dilation_options = [(0.0, 1.0, 1)]
    center_deadband_options = [(0.0, 1)]
    recipe_options: set[tuple[str, str, str, float]] | None = None
    if profile == "quick":
        bands = [(500.0, 2000.0), (80.0, 8000.0), (300.0, 3000.0), (1000.0, 4000.0)]
        betas = [1.0, 0.5]
        transforms = ["raw", "preemph", "clip"]
        geometries = [("moving_patch", 0.25), ("moving_patch", 0.5), ("fixed_spot", 0.25), ("fixed_spot", 0.5)]
        score_modes = ["sum", "product", "harmonic"]
        topk_options = [8]
        radius_options = [2.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [None]
    elif profile == "targeted":
        bands = [(1000.0, 4000.0), (80.0, 8000.0)]
        betas = [0.5, 0.3]
        transforms = ["raw", "clip"]
        geometries = [("moving_patch", 0.5), ("moving_patch", 0.6)]
        score_modes = ["sum", "product", "harmonic"]
        topk_options = [4, 8, 12]
        radius_options = [1.0, 2.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [None]
    elif profile == "advanced":
        bands = [(80.0, 8000.0), (1000.0, 4000.0)]
        betas = [0.5, 0.3]
        transforms = ["raw", "clip"]
        geometries = [("moving_patch", 0.5)]
        score_modes = ["harmonic", "product"]
        topk_options = [4, 8]
        radius_options = [2.0]
        gcc_modes = [("plain", 0.0), ("coherence", 0.25), ("coherence", 0.4)]
        estimators = ["score", "consensus"]
        subband_options = [
            None,
            ((300.0, 800.0), (800.0, 1500.0), (1500.0, 2500.0), (2500.0, 4000.0), (4000.0, 8000.0)),
        ]
        aggregator_options = [("curve_mean", 0.0)]
    elif profile == "strict_v2":
        bands = [(80.0, 8000.0)]
        betas = [0.3]
        transforms = ["clip"]
        geometries = [("moving_patch", 0.5)]
        score_modes = ["product"]
        topk_options = [8, 9, 10]
        radius_options = [2.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [
            ((300.0, 800.0), (800.0, 1500.0), (1500.0, 2500.0), (2500.0, 4000.0), (4000.0, 8000.0)),
        ]
        aggregator_options = [
            ("curve_mean", 0.0),
            ("subband_score_mean", 0.0),
            ("subband_score_minus_std", 0.2),
            ("subband_score_minus_std", 0.4),
            ("subband_score_exp_cv", 0.2),
            ("subband_score_exp_cv", 0.4),
        ]
    elif profile == "strict_v3":
        bands = [(80.0, 8000.0)]
        betas = [0.3]
        transforms = ["clip"]
        geometries = [("moving_patch", 0.5)]
        score_modes = ["product"]
        topk_options = [9]
        radius_options = [2.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [
            ((300.0, 800.0), (800.0, 1500.0), (1500.0, 2500.0), (2500.0, 4000.0), (4000.0, 8000.0)),
        ]
        aggregator_options = [("subband_score_exp_cv", 0.2)]
        selector_options = [
            ("fixed_top_k", 8, 0.03, 2, 9),
            ("stable_prefix", 8, 0.03, 2, 9),
        ]
    elif profile == "strict_v4":
        bands = [(80.0, 8000.0)]
        betas = [0.3]
        transforms = ["clip"]
        geometries = [("moving_patch", 0.5)]
        score_modes = ["product"]
        topk_options = [9]
        radius_options = [2.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [
            ((300.0, 800.0), (800.0, 1500.0), (1500.0, 2500.0), (2500.0, 4000.0), (4000.0, 8000.0)),
        ]
        aggregator_options = [("subband_score_exp_cv", 0.2)]
        selector_options = [("stable_prefix", 8, 0.03, 2, 9)]
        basin_options = [
            ("none", 0.08, 0.10, 0.25, None),
            ("basin_gate", 0.08, 0.10, 0.25, None),
            ("basin_mul", 0.30, 0.10, 0.25, None),
        ]
    elif profile == "strict_v5":
        bands = [(80.0, 8000.0)]
        betas = [0.3]
        transforms = ["clip"]
        geometries = [("moving_patch", 0.5)]
        score_modes = ["product"]
        topk_options = [9]
        radius_options = [2.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [
            ((300.0, 800.0), (800.0, 1500.0), (1500.0, 2500.0), (2500.0, 4000.0), (4000.0, 8000.0)),
        ]
        aggregator_options = [("subband_score_exp_cv", 0.2)]
        selector_options = [
            ("stable_prefix", 8, 0.03, 2, 9),
            ("confidence_prefix", 2, 0.03, 2, 9),
            ("hysteresis_prefix", 2, 0.03, 2, 9),
        ]
        basin_options = [
            ("basin_gate", 0.08, 0.10, 0.25, None),
            ("pair_overlap_gate", 0.12, 0.08, 0.25, None),
        ]
        subband_weight_options = ["none", "chirp_stability"]
        recipe_options = {
            ("stable_prefix", "basin_gate", "none", 0.0),
            ("confidence_prefix", "basin_gate", "none", 0.0),
            ("confidence_prefix", "pair_overlap_gate", "none", 0.0),
            ("confidence_prefix", "pair_overlap_gate", "chirp_stability", 0.0),
        }
    elif profile == "strict_v6":
        bands = [(80.0, 8000.0)]
        betas = [0.3]
        transforms = ["clip"]
        geometries = [("moving_patch", 0.5)]
        score_modes = ["product"]
        topk_options = [9]
        radius_options = [2.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [
            ((300.0, 800.0), (800.0, 1500.0), (1500.0, 2500.0), (2500.0, 4000.0), (4000.0, 8000.0)),
        ]
        aggregator_options = [("subband_score_exp_cv", 0.2)]
        selector_options = [
            ("stable_prefix", 8, 0.03, 2, 9),
            ("confidence_prefix", 2, 0.03, 2, 9),
            ("hysteresis_prefix", 2, 0.03, 2, 9),
        ]
        basin_options = [
            ("basin_gate", 0.08, 0.10, 0.25, None),
            ("pair_overlap_gate", 0.12, 0.08, 0.25, None),
        ]
        lr_options = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)]
        recipe_options = {
            ("stable_prefix", "basin_gate", "none", 0.0),
            ("stable_prefix", "basin_gate", "none", 0.25),
            ("stable_prefix", "basin_gate", "none", 0.5),
            ("confidence_prefix", "basin_gate", "none", 0.25),
            ("confidence_prefix", "pair_overlap_gate", "none", 0.25),
            ("hysteresis_prefix", "basin_gate", "none", 0.0),
            ("hysteresis_prefix", "basin_gate", "none", 0.25),
        }
    elif profile == "strict_v7":
        bands = [(80.0, 8000.0)]
        betas = [0.3]
        transforms = ["clip"]
        geometries = [("moving_patch", 0.5)]
        score_modes = ["product"]
        topk_options = [9]
        radius_options = [2.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [
            ((300.0, 800.0), (800.0, 1500.0), (1500.0, 2500.0), (2500.0, 4000.0), (4000.0, 8000.0)),
        ]
        aggregator_options = [
            ("subband_score_exp_cv", 0.2),
            ("subband_jackknife", 0.25),
            ("subband_jackknife", 0.5),
            ("subband_cluster_max", 0.0),
        ]
        selector_options = [("hysteresis_prefix", 2, 0.03, 2, 9)]
        basin_options = [("basin_gate", 0.08, 0.10, 0.25, None)]
    elif profile == "strict_v8":
        bands = [(80.0, 8000.0)]
        betas = [0.3]
        transforms = ["clip"]
        geometries = [("moving_patch", 0.5), ("wall_wave_sub", 0.5), ("wall_wave_add", 0.5)]
        score_modes = ["product"]
        topk_options = [9]
        radius_options = [2.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [
            ((300.0, 800.0), (800.0, 1500.0), (1500.0, 2500.0), (2500.0, 4000.0), (4000.0, 8000.0)),
        ]
        aggregator_options = [("subband_score_exp_cv", 0.2)]
        selector_options = [("hysteresis_prefix", 2, 0.03, 2, 9)]
        basin_options = [("basin_gate", 0.08, 0.10, 0.25, None)]
        wall_speed_options = [0.0, 80.0, 160.0, 320.0]
    elif profile == "strict_v9":
        bands = [(80.0, 8000.0)]
        betas = [0.3]
        transforms = ["clip"]
        geometries = [("moving_patch", 0.5)]
        score_modes = ["product"]
        topk_options = [9]
        radius_options = [2.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [
            ((300.0, 800.0), (800.0, 1500.0), (1500.0, 2500.0), (2500.0, 4000.0), (4000.0, 8000.0)),
        ]
        aggregator_options = [("subband_score_exp_cv", 0.2)]
        selector_options = [("hysteresis_prefix", 2, 0.03, 2, 9)]
        basin_options = [("basin_gate", 0.08, 0.10, 0.25, None)]
        common_shift_options = [(0.0, 0), (0.5, 7), (1.0, 9), (2.0, 13)]
    elif profile == "strict_v10":
        bands = [(80.0, 8000.0)]
        betas = [0.3]
        transforms = ["clip"]
        geometries = [("moving_patch", 0.5)]
        score_modes = ["product"]
        topk_options = [9]
        radius_options = [2.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [
            ((300.0, 800.0), (800.0, 1500.0), (1500.0, 2500.0), (2500.0, 4000.0), (4000.0, 8000.0)),
        ]
        aggregator_options = [("subband_score_exp_cv", 0.2)]
        selector_options = [("hysteresis_prefix", 2, 0.03, 2, 9)]
        basin_options = [("basin_gate", 0.08, 0.10, 0.25, None)]
        x_calibration_options = ["none", "affine", "piecewise_linear"]
    elif profile == "strict_v11":
        bands = [(80.0, 8000.0)]
        betas = [0.3]
        transforms = ["clip"]
        geometries = [("moving_patch", 0.5)]
        score_modes = ["product"]
        topk_options = [9]
        radius_options = [2.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [
            ((300.0, 800.0), (800.0, 1500.0), (1500.0, 2500.0), (2500.0, 4000.0), (4000.0, 8000.0)),
        ]
        aggregator_options = [("subband_score_exp_cv", 0.2)]
        selector_options = [("hysteresis_prefix", 2, 0.03, 2, 9)]
        basin_options = [("basin_gate", 0.08, 0.10, 0.25, None)]
        polarity_options = ["abs", "positive", "negative"]
    elif profile == "strict_v12":
        bands = [(80.0, 8000.0)]
        betas = [0.3]
        transforms = ["clip"]
        geometries = [("moving_patch", 0.5)]
        score_modes = ["product"]
        topk_options = [9]
        radius_options = [2.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [
            ((300.0, 800.0), (800.0, 1500.0), (1500.0, 2500.0), (2500.0, 4000.0), (4000.0, 8000.0)),
        ]
        aggregator_options = [("subband_score_exp_cv", 0.2)]
        selector_options = [("hysteresis_prefix", 2, 0.03, 2, 9)]
        basin_options = [("basin_gate", 0.08, 0.10, 0.25, None)]
        edge_dilation_options = [
            (0.0, 1.0, 1),
            (0.2, 1.5, 1),
            (0.3, 1.5, 1),
            (0.3, 1.75, 1),
            (0.4, 1.75, 1),
            (0.4, 2.0, 1),
        ]
    elif profile == "strict_v13":
        bands = [(80.0, 8000.0)]
        betas = [0.3]
        transforms = ["clip"]
        geometries = [("moving_patch", 0.5)]
        score_modes = ["product"]
        topk_options = [9]
        radius_options = [2.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [
            ((300.0, 800.0), (800.0, 1500.0), (1500.0, 2500.0), (2500.0, 4000.0), (4000.0, 8000.0)),
        ]
        aggregator_options = [("subband_score_exp_cv", 0.2)]
        selector_options = [("hysteresis_prefix", 2, 0.03, 2, 9)]
        basin_options = [("basin_gate", 0.08, 0.10, 0.25, None)]
        edge_dilation_options = [
            (0.0, 1.0, 1),
            (0.2, 1.5, 3),
            (0.3, 1.5, 3),
            (0.3, 1.75, 3),
            (0.3, 1.75, 4),
            (0.3, 1.75, 5),
            (0.4, 2.0, 4),
        ]
    elif profile == "strict_v14":
        bands = [(80.0, 8000.0)]
        betas = [0.3]
        transforms = ["clip"]
        geometries = [("moving_patch", 0.5)]
        score_modes = ["product"]
        topk_options = [9]
        radius_options = [2.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [
            ((300.0, 800.0), (800.0, 1500.0), (1500.0, 2500.0), (2500.0, 4000.0), (4000.0, 8000.0)),
        ]
        aggregator_options = [("subband_score_exp_cv", 0.2)]
        selector_options = [("hysteresis_prefix", 2, 0.03, 2, 9)]
        basin_options = [("basin_gate", 0.08, 0.10, 0.25, None)]
        edge_dilation_options = [
            (0.0, 1.0, 1),
            (0.3, 1.75, 4),
        ]
        center_deadband_options = [
            (0.0, 1),
            (0.20, 4),
            (0.25, 4),
            (0.28, 4),
            (0.30, 4),
        ]
    elif profile == "refine":
        bands = [(800.0, 3000.0), (1000.0, 3500.0), (1000.0, 4000.0), (1200.0, 4500.0), (1500.0, 5000.0), (80.0, 8000.0)]
        betas = [0.7, 0.5, 0.3]
        transforms = ["raw", "preemph", "clip", "preemph_clip"]
        geometries = [("moving_patch", 0.35), ("moving_patch", 0.5), ("moving_patch", 0.65), ("fixed_spot", 0.5)]
        score_modes = ["sum", "product", "harmonic"]
        topk_options = [6, 8, 12]
        radius_options = [1.0, 2.0, 3.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [None]
        aggregator_options = [("curve_mean", 0.0)]
    else:
        bands = [(80.0, 8000.0), (100.0, 3000.0), (300.0, 3000.0), (500.0, 2000.0), (700.0, 2500.0), (1000.0, 4000.0), (1500.0, 6000.0), None]
        betas = [1.0, 0.7, 0.5, 0.3, 0.0]
        transforms = ["raw", "preemph", "diff", "clip", "preemph_clip"]
        geometries = [("moving_patch", 0.0), ("moving_patch", 0.25), ("moving_patch", 0.5), ("fixed_spot", 0.25), ("fixed_spot", 0.5)]
        score_modes = ["sum", "product", "min", "harmonic"]
        topk_options = [8]
        radius_options = [2.0]
        gcc_modes = [("plain", 0.0)]
        estimators = ["score"]
        subband_options = [None]
        aggregator_options = [("curve_mean", 0.0)]

    if profile in ("quick", "targeted"):
        aggregator_options = [("curve_mean", 0.0)]

    configs: list[Config] = []
    for band in bands:
        for beta in betas:
            for transform in transforms:
                for geometry, ldv_y in geometries:
                    for score_mode in score_modes:
                        for top_k_windows in topk_options:
                            for local_peak_radius_ms in radius_options:
                                for gcc_mode, coherence_floor in gcc_modes:
                                    for estimator in estimators:
                                        for subbands in subband_options:
                                            for aggregator, penalty in aggregator_options:
                                                for selector, min_k, stable_threshold, stable_steps, fallback_k in selector_options:
                                                    for basin_mode, basin_sigma, basin_gate, basin_power, agreement_gate in basin_options:
                                                        for subband_weight_mode in subband_weight_options:
                                                            for lr_weight, lr_gate in lr_options:
                                                                if recipe_options is not None and (selector, basin_mode, subband_weight_mode, lr_weight) not in recipe_options:
                                                                    continue
                                                                for wall_speed_mps in wall_speed_options:
                                                                    is_wall_wave = geometry.startswith("wall_wave")
                                                                    if is_wall_wave and wall_speed_mps <= 0.0:
                                                                        continue
                                                                    if not is_wall_wave and wall_speed_mps > 0.0:
                                                                        continue
                                                                    for common_shift_radius_ms, common_shift_steps in common_shift_options:
                                                                        for x_calibration in x_calibration_options:
                                                                            for correlation_polarity in polarity_options:
                                                                                for edge_threshold, edge_gain, edge_min_k in edge_dilation_options:
                                                                                    for center_deadband, center_min_k in center_deadband_options:
                                                                                        band_name = "wide" if band is None else f"{int(band[0])}-{int(band[1])}"
                                                                                        subband_name = "_sub" if subbands is not None else ""
                                                                                        penalty_name = f"{penalty:g}" if penalty else ""
                                                                                        agg_name = "" if aggregator == "curve_mean" else f"_{aggregator}{penalty_name}"
                                                                                        selector_name = "" if selector == "fixed_top_k" else f"_{selector}_m{min_k}_s{stable_threshold:g}_n{stable_steps}_fb{fallback_k}"
                                                                                        basin_name = "" if basin_mode == "none" else f"_{basin_mode}_sig{basin_sigma:g}_g{basin_gate:g}_p{basin_power:g}"
                                                                                        weight_name = "" if subband_weight_mode == "none" else f"_{subband_weight_mode}"
                                                                                        lr_name = "" if lr_weight <= 0.0 else f"_lrw{lr_weight:g}"
                                                                                        wall_name = "" if wall_speed_mps <= 0.0 else f"_ws{wall_speed_mps:g}"
                                                                                        shift_name = "" if common_shift_radius_ms <= 0.0 else f"_cs{common_shift_radius_ms:g}ms"
                                                                                        xcal_name = "" if x_calibration == "none" else f"_xcal_{x_calibration}"
                                                                                        polarity_name = "" if correlation_polarity == "abs" else f"_pol_{correlation_polarity}"
                                                                                        edge_name = "" if edge_gain <= 1.0 else f"_edgeth{edge_threshold:g}_g{edge_gain:g}_mink{edge_min_k}"
                                                                                        center_name = "" if center_deadband <= 0.0 else f"_centerdb{center_deadband:g}_mink{center_min_k}"
                                                                                        name = (
                                                                                            f"{band_name}_b{beta:g}_{transform}_{geometry}_y{ldv_y:g}{wall_name}_{score_mode}"
                                                                                            f"_k{top_k_windows}_r{local_peak_radius_ms:g}_{gcc_mode}{coherence_floor:g}"
                                                                                            f"_{estimator}{subband_name}{agg_name}{selector_name}{basin_name}{weight_name}{lr_name}{shift_name}{xcal_name}{polarity_name}{edge_name}{center_name}"
                                                                                        )
                                                                                        configs.append(
                                                                                            Config(
                                                                                                name=name,
                                                                                                band_hz=band,
                                                                                                phat_beta=beta,
                                                                                                transform=transform,
                                                                                                geometry=geometry,
                                                                                                ldv_y_m=ldv_y,
                                                                                                score_mode=score_mode,
                                                                                                n_fft=1024,
                                                                                                hop=256,
                                                                                                window_sec=0.5,
                                                                                                window_hop_sec=0.25,
                                                                                                top_k_windows=top_k_windows,
                                                                                                local_peak_radius_ms=local_peak_radius_ms,
                                                                                                gcc_mode=gcc_mode,
                                                                                                estimator=estimator,
                                                                                                coherence_floor=coherence_floor,
                                                                                                subbands=subbands,
                                                                                                score_aggregator=aggregator,
                                                                                                subband_penalty=penalty,
                                                                                                window_selector=selector,
                                                                                                adaptive_min_k=min_k,
                                                                                                stability_threshold_m=stable_threshold,
                                                                                                required_stable_steps=stable_steps,
                                                                                                fallback_top_k=fallback_k,
                                                                                                basin_mode=basin_mode,
                                                                                                basin_sigma_m=basin_sigma,
                                                                                                basin_gate=basin_gate,
                                                                                                basin_power=basin_power,
                                                                                                candidate_agreement_gate_m=agreement_gate,
                                                                                                subband_weight_mode=subband_weight_mode,
                                                                                                lr_weight=lr_weight,
                                                                                                lr_gate=lr_gate,
                                                                                                wall_speed_mps=wall_speed_mps,
                                                                                                common_shift_radius_ms=common_shift_radius_ms,
                                                                                                common_shift_steps=common_shift_steps,
                                                                                                x_calibration=x_calibration,
                                                                                                correlation_polarity=correlation_polarity,
                                                                                                edge_dilation_threshold_m=edge_threshold,
                                                                                                edge_dilation_gain=edge_gain,
                                                                                                edge_dilation_min_k=edge_min_k,
                                                                                                center_deadband_m=center_deadband,
                                                                                                center_deadband_min_k=center_min_k,
                                                                                            )
                                                                                        )
    return configs


def write_markdown_report(path: Path, payload: dict[str, object]) -> None:
    top = payload["top_results"]  # type: ignore[index]
    strict = bool(top) and "holdout_speech" in top[0]  # type: ignore[index]
    profile = payload.get("args", {}).get("profile", "unknown") if isinstance(payload.get("args"), dict) else "unknown"
    lines = [
        "# Independent PI-GS Audit",
        "",
        f"Generated: {payload['generated_at']}",
        f"Data root: `{payload['data_root']}`",
        "",
        "## Top Configurations",
        "",
    ]
    if strict:
        lines.extend(
            [
                "| Rank | Config | Canonical MAE | Holdout MAE | Combined MAE | Combined Max | Offsets (VL/VR ms) |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                "| Rank | Config | Chirp MAE | Speech MAE | Speech Max | Offsets (VL/VR ms) |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
    for i, item in enumerate(top[:20], start=1):  # type: ignore[index]
        offsets = item["offsets"]  # type: ignore[index]
        offset_text = (
            f"{1000.0 * offsets.get('vl_intercept_sec', offsets.get('vl_sec', 0.0)):.3f}/"
            f"{1000.0 * offsets.get('vr_intercept_sec', offsets.get('vr_sec', 0.0)):.3f}"
        )
        if strict:
            lines.append(
                f"| {i} | `{item['config']['name']}` | "
                f"{item['canonical_speech']['mae_deg']:.2f} | {item['holdout_speech']['mae_deg']:.2f} | "
                f"{item['combined_speech']['mae_deg']:.2f} | {item['combined_speech']['max_err_deg']:.2f} | "
                f"{offset_text} |"
            )
        else:
            lines.append(
                f"| {i} | `{item['config']['name']}` | "
                f"{item['chirp']['mae_deg']:.2f} | {item['speech']['mae_deg']:.2f} | "
                f"{item['speech']['max_err_deg']:.2f} | {offset_text} |"
            )

    if top:  # type: ignore[truthy-function]
        best = top[0]  # type: ignore[index]
        best_rows = best["combined_speech"]["rows"] if strict else best["speech"]["rows"]  # type: ignore[index]
        lines.extend(["", "## Best Speech Rows", ""])
        lines.append("| Label | x true | x hat | theta true | theta hat | abs err | selected K |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in best_rows:  # type: ignore[union-attr]
            lines.append(
                f"| {row['label']} | {row['x_true_m']:.2f} | {row['x_hat_m']:.3f} | "
                f"{row['theta_true_deg']:.2f} | {row['theta_hat_deg']:.2f} | {row['abs_err_deg']:.2f} | "
                f"{row.get('num_windows', '')} |"
            )
        if strict:
            selected = [int(row.get("num_windows", 0)) for row in best_rows]  # type: ignore[union-attr]
            if selected:
                lines.extend(["", "Selected-K distribution:"])
                for k in sorted(set(selected)):
                    lines.append(f"- K={k}: {selected.count(k)} trials")

        loro = payload.get("loro")
        if isinstance(loro, dict) and loro.get("rows"):
            lines.extend(["", "## Leave-One-Recording-Out", ""])
            lines.append(f"LORO MAE: {float(loro['mae_deg']):.2f} deg; max error: {float(loro['max_err_deg']):.2f} deg")
            lines.extend(
                [
                    "",
                    "| Held-out | Selected config | train MAE | held-out err | selected K |",
                    "|---|---|---:|---:|---:|",
                ]
            )
            for row in loro["rows"]:  # type: ignore[index]
                lines.append(
                    f"| {row['held_out_label']} | `{row['selected_config']}` | "
                    f"{row['train_mae_deg']:.2f} | {row['abs_err_deg']:.2f} | {row['selected_k']} |"
                )

        lines.extend(["", "## Interpretation Notes", ""])
        lines.append("- Offsets are calibrated only from chirp, then frozen for speech.")
        if strict:
            lines.append(f"- `{profile}` ranks fixed hypotheses against canonical and holdout speech; use holdout as a guardrail, not as hidden tuning data.")
        lines.append("- Low speech MAE here is still not proof of a general method; it identifies a reproducible parameter hypothesis to audit further.")
        lines.append("- If top configurations require implausibly large offsets or differ strongly by segment, treat them as calibration artifacts.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "results" / f"independent_pigs_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=False)

    trials = default_trials(data_root)
    holdout = holdout_trials(data_root)
    require_trial_files(data_root, trials)
    strict_profile = args.profile in ("strict_v2", "strict_v3", "strict_v4", "strict_v5", "strict_v6", "strict_v7", "strict_v8", "strict_v9", "strict_v10", "strict_v11", "strict_v12", "strict_v13", "strict_v14")
    if strict_profile and not holdout:
        raise ValueError(f"{args.profile} requires at least one complete holdout trial")

    chirp = SegmentSpec("chirp", args.chirp_t0_sec, args.chirp_t1_sec)
    speech = SegmentSpec("speech", args.speech_t0_sec, args.speech_t1_sec)
    xs_grid = np.arange(args.x_min_m, args.x_max_m + 0.5 * args.x_step_m, args.x_step_m)

    results = []
    configs = candidate_configs(args.profile)
    if args.limit_configs > 0:
        configs = configs[: args.limit_configs]

    for idx, cfg in enumerate(configs, start=1):
        offsets = (
            estimate_offsets(data_root, trials, chirp, cfg, xs_grid, args.offset_model)
            if args.calibrate_offsets
            else {
                "model": "none",
                "vl_intercept_sec": 0.0,
                "vl_slope_sec_per_m": 0.0,
                "vr_intercept_sec": 0.0,
                "vr_slope_sec_per_m": 0.0,
                "num_residuals": 0,
            }
        )
        if cfg.subband_weight_mode != "none":
            offsets = dict(offsets)
            offsets["subband_weight_mode"] = cfg.subband_weight_mode
            offsets["subband_weights"] = estimate_chirp_subband_weights(data_root, trials, chirp, cfg, xs_grid, offsets)
        offsets = estimate_x_calibration(data_root, trials, chirp, cfg, xs_grid, offsets)
        canonical_chirp = summarize_trials(data_root, trials, chirp, cfg, xs_grid, offsets)
        canonical_speech = summarize_trials(data_root, trials, speech, cfg, xs_grid, offsets)
        item = {
            "config": asdict(cfg),
            "offsets": offsets,
            "chirp": canonical_chirp,
            "speech": canonical_speech,
        }
        if strict_profile:
            holdout_speech = summarize_trials(data_root, holdout, speech, cfg, xs_grid, offsets)
            combined_rows = list(canonical_speech["rows"]) + list(holdout_speech["rows"])  # type: ignore[arg-type]
            item.update(
                {
                    "canonical_chirp": canonical_chirp,
                    "canonical_speech": canonical_speech,
                    "holdout_speech": holdout_speech,
                    "combined_speech": summarize_rows(combined_rows),
                }
            )
        results.append(item)
        if idx % max(1, args.progress_every) == 0:
            metric_key = "combined_speech" if strict_profile else "speech"
            best = min(results, key=lambda x: float(x[metric_key]["mae_deg"]))  # type: ignore[index]
            print(
                f"[{idx}/{len(configs)}] best {metric_key} MAE={best[metric_key]['mae_deg']:.2f} "
                f"config={best['config']['name']}",
                flush=True,
            )

    if strict_profile:
        results.sort(
            key=lambda x: (
                float(x["combined_speech"]["mae_deg"]),  # type: ignore[index]
                float(x["holdout_speech"]["mae_deg"]),  # type: ignore[index]
                float(x["canonical_speech"]["mae_deg"]),  # type: ignore[index]
                float(x["combined_speech"]["max_err_deg"]),  # type: ignore[index]
            )
        )
    else:
        results.sort(key=lambda x: (float(x["speech"]["mae_deg"]), float(x["speech"]["max_err_deg"]), float(x["chirp"]["mae_deg"])))  # type: ignore[index]

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "segments": {"chirp": asdict(chirp), "speech": asdict(speech)},
        "args": vars(args),
        "num_configs": len(configs),
        "trials": [asdict(t) for t in trials],
        "trial_sets": {
            "canonical": [asdict(t) for t in trials],
            "holdout": [asdict(t) for t in holdout],
        },
        "top_results": results[: args.keep_top],
    }
    if args.profile in ("strict_v4", "strict_v5", "strict_v6", "strict_v7", "strict_v8", "strict_v9", "strict_v10", "strict_v11", "strict_v12", "strict_v13", "strict_v14"):
        payload["loro"] = summarize_loro(results)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if results:
        best_cfg = config_from_payload(results[0]["config"])  # type: ignore[arg-type]
        best_offsets = results[0]["offsets"]  # type: ignore[index]
        diagnostic_trials = trials + holdout if strict_profile else trials
        diagnostics = [
            row
            for trial in diagnostic_trials
            for row in diagnose_trial_windows(data_root, trial, speech, best_cfg, xs_grid, best_offsets)  # type: ignore[arg-type]
        ]
        (out_dir / "best_window_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        incremental = [
            row
            for trial in diagnostic_trials
            for row in diagnose_incremental_windows(data_root, trial, speech, best_cfg, xs_grid, best_offsets)  # type: ignore[arg-type]
        ]
        (out_dir / "best_incremental_window_diagnostics.json").write_text(json.dumps(incremental, indent=2), encoding="utf-8")
        if args.profile in ("strict_v5", "strict_v6", "strict_v7", "strict_v8", "strict_v9", "strict_v10", "strict_v11", "strict_v12", "strict_v13", "strict_v14"):
            prefix_diagnostics = [
                row
                for trial in diagnostic_trials
                for row in diagnose_prefix_confidence(data_root, trial, speech, best_cfg, xs_grid, best_offsets)  # type: ignore[arg-type]
            ]
            (out_dir / "best_prefix_diagnostics.json").write_text(json.dumps(prefix_diagnostics, indent=2), encoding="utf-8")
    write_markdown_report(out_dir / "report.md", payload)
    print(f"Wrote {out_dir / 'summary.json'}")
    if results:
        print(f"Wrote {out_dir / 'best_window_diagnostics.json'}")
        print(f"Wrote {out_dir / 'best_incremental_window_diagnostics.json'}")
        if args.profile in ("strict_v5", "strict_v6", "strict_v7", "strict_v8", "strict_v9", "strict_v10", "strict_v11", "strict_v12", "strict_v13", "strict_v14"):
            print(f"Wrote {out_dir / 'best_prefix_diagnostics.json'}")
    print(f"Wrote {out_dir / 'report.md'}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", default=str(Path(__file__).resolve().parent.parent / "dataset" / "0223"))
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--profile", choices=("quick", "targeted", "advanced", "strict_v2", "strict_v3", "strict_v4", "strict_v5", "strict_v6", "strict_v7", "strict_v8", "strict_v9", "strict_v10", "strict_v11", "strict_v12", "strict_v13", "strict_v14", "refine", "full"), default="quick")
    parser.add_argument("--limit_configs", type=int, default=0)
    parser.add_argument("--keep_top", type=int, default=50)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--calibrate_offsets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offset_model", choices=("constant", "affine", "per_trial"), default="constant")
    parser.add_argument("--chirp_t0_sec", type=float, default=0.0)
    parser.add_argument("--chirp_t1_sec", type=float, default=2.0)
    parser.add_argument("--speech_t0_sec", type=float, default=3.0)
    parser.add_argument("--speech_t1_sec", type=float, default=8.0)
    parser.add_argument("--x_min_m", type=float, default=-0.9)
    parser.add_argument("--x_max_m", type=float, default=0.9)
    parser.add_argument("--x_step_m", type=float, default=0.01)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
