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
    return np.abs(cc), lags


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

    if cfg.geometry == "fixed_spot":
        d_sv = np.sqrt(xs**2 + cfg.ldv_y_m**2)
        d_sl = np.sqrt((xs - MIC_LEFT_X_M) ** 2 + MIC_Y_M**2)
        d_sr = np.sqrt((xs - MIC_RIGHT_X_M) ** 2 + MIC_Y_M**2)
        return (d_sl - d_sv) / C_MPS, (d_sr - d_sv) / C_MPS

    raise ValueError(f"Unknown geometry: {cfg.geometry}")


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
                vl = robust_normalize(vl)
                vr = robust_normalize(vr)
                subband_items.append(
                    {
                        "band_hz": band,
                        "vl": vl,
                        "vr": vr,
                        "lags_vl": lags_vl,
                        "lags_vr": lags_vr,
                        "reliability": window_reliability(vl, vr),
                    }
                )
        except ValueError:
            continue
        vl = np.mean([np.asarray(s["vl"]) for s in subband_items], axis=0)
        vr = np.mean([np.asarray(s["vr"]) for s in subband_items], axis=0)
        lags_vl = subband_items[0]["lags_vl"]
        lags_vr = subband_items[0]["lags_vr"]
        items.append(
            {
                "start": start,
                "end": end,
                "vl": vl,
                "vr": vr,
                "lags_vl": lags_vl,
                "lags_vr": lags_vr,
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
    residuals_vl: list[tuple[float, float]] = []
    residuals_vr: list[tuple[float, float]] = []
    per_trial: dict[str, dict[str, list[float]]] = {}

    for trial in trials:
        curves = compute_trial_curves(data_root, trial, segment, cfg)
        idx = int(np.argmin(np.abs(xs_grid - trial.x_m)))
        pred_vl = float(tau_vl_grid[idx])
        pred_vr = float(tau_vr_grid[idx])
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
            res_vl = peak_vl - pred_vl
            res_vr = peak_vr - pred_vr
            residuals_vl.append((trial.x_m, res_vl))
            residuals_vr.append((trial.x_m, res_vr))
            per_trial.setdefault(trial.label, {"vl": [], "vr": []})
            per_trial[trial.label]["vl"].append(res_vl)
            per_trial[trial.label]["vr"].append(res_vr)

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
        "num_residuals": int(min(len(residuals_vl), len(residuals_vr))),
    }
    if offset_model == "per_trial":
        payload["per_trial"] = {
            label: {
                "vl_sec": float(np.median(values["vl"])) if values["vl"] else 0.0,
                "vr_sec": float(np.median(values["vr"])) if values["vr"] else 0.0,
            }
            for label, values in per_trial.items()
        }
    return payload


def offset_values(xs: np.ndarray, offsets: dict[str, float], prefix: str) -> np.ndarray:
    if f"{prefix}_intercept_sec" in offsets:
        return float(offsets.get(f"{prefix}_intercept_sec", 0.0)) + float(offsets.get(f"{prefix}_slope_sec_per_m", 0.0)) * xs
    return np.full_like(xs, float(offsets.get(f"{prefix}_sec", 0.0)), dtype=np.float64)


def score_windows(
    windows: list[dict[str, object]],
    xs_grid: np.ndarray,
    tau_vl: np.ndarray,
    tau_vr: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    scores = np.zeros_like(xs_grid, dtype=np.float64)
    weights = []
    rows: list[dict[str, float]] = []
    for rank, item in enumerate(windows, start=1):
        score = aggregate_window_score(item, xs_grid, tau_vl, tau_vr, cfg)
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
    if offsets.get("model") == "per_trial" and isinstance(offsets.get("per_trial"), dict):
        trial_offsets = offsets["per_trial"].get(trial.label, {})  # type: ignore[index]
        tau_vl = tau_vl + float(trial_offsets.get("vl_sec", 0.0))
        tau_vr = tau_vr + float(trial_offsets.get("vr_sec", 0.0))
    else:
        tau_vl = tau_vl + offset_values(xs_grid, offsets, "vl")
        tau_vr = tau_vr + offset_values(xs_grid, offsets, "vr")

    consensus_meta: dict[str, float] = {}
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
        scores, prefix_rows = score_windows(windows, xs_grid, tau_vl, tau_vr, cfg)
        selection_meta: dict[str, float | str] = {"window_selector": "fixed_top_k", "selected_k": len(windows)}
        if use_adaptive:
            selected_k, selection_meta = select_window_prefix(prefix_rows, cfg)
            scores, prefix_rows = score_windows(windows[:selected_k], xs_grid, tau_vl, tau_vr, cfg)
        best_idx = int(np.argmax(scores))
        x_hat = float(xs_grid[best_idx])
        score_value = float(scores[best_idx])
        consensus_meta = selection_meta
    else:
        raise ValueError(f"Unknown estimator: {cfg.estimator}")

    theta_hat = theta_from_x(x_hat)
    theta_true = theta_from_x(trial.x_m)
    return {
        "label": trial.label,
        "x_true_m": trial.x_m,
        "x_hat_m": x_hat,
        "theta_true_deg": theta_true,
        "theta_hat_deg": theta_hat,
        "abs_err_deg": abs(theta_hat - theta_true),
        "score": score_value,
        "num_windows": int(consensus_meta.get("selected_k", len(curves["windows"]))),  # type: ignore[arg-type]
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
    score = combine_scores(vl, vr, cfg.score_mode)

    best_idx = int(np.argmax(score))
    x_hat = float(xs_grid[best_idx])
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
    vl = sample_curve(np.asarray(source["vl"]), np.asarray(source["lags_vl"]), tau_vl)
    vr = sample_curve(np.asarray(source["vr"]), np.asarray(source["lags_vr"]), tau_vr)
    return combine_scores(vl, vr, cfg.score_mode)


def aggregate_window_score(
    item: dict[str, object],
    xs_grid: np.ndarray,
    tau_vl: np.ndarray,
    tau_vr: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    if cfg.score_aggregator == "curve_mean" or cfg.subbands is None:
        return score_curve_from_source(item, xs_grid, tau_vl, tau_vr, cfg)

    sources = item.get("subbands", [])
    if not sources:
        return score_curve_from_source(item, xs_grid, tau_vl, tau_vr, cfg)

    score_matrix = np.vstack(
        [score_curve_from_source(source, xs_grid, tau_vl, tau_vr, cfg) for source in sources]  # type: ignore[arg-type]
    )
    mean_score = np.mean(score_matrix, axis=0)
    if cfg.score_aggregator == "subband_score_mean":
        return mean_score

    spread = np.std(score_matrix, axis=0)
    if cfg.score_aggregator == "subband_score_minus_std":
        return mean_score - cfg.subband_penalty * spread

    if cfg.score_aggregator == "subband_score_exp_cv":
        cv = spread / (np.abs(mean_score) + 1e-6)
        return mean_score * np.exp(-cfg.subband_penalty * cv)

    raise ValueError(f"Unknown score_aggregator: {cfg.score_aggregator}")


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

    rows = []
    fs = float(curves["fs"])  # type: ignore[index]
    for item in curves["windows"]:  # type: ignore[index]
        sources = item.get("subbands", []) if cfg.subbands is not None else [item]  # type: ignore[union-attr]
        for source in sources:
            cand = candidate_from_curves(source, xs_grid, tau_vl, tau_vr, cfg)
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

    fs = float(curves["fs"])  # type: ignore[index]
    scores = np.zeros_like(xs_grid, dtype=np.float64)
    weights: list[float] = []
    rows = []
    for rank, item in enumerate(curves["windows"], start=1):  # type: ignore[index]
        score = aggregate_window_score(item, xs_grid, tau_vl, tau_vr, cfg)  # type: ignore[arg-type]
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


def candidate_configs(profile: str) -> list[Config]:
    bands: list[tuple[float, float] | None]
    selector_options = [("fixed_top_k", 8, 0.03, 2, 9)]
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
                                                    band_name = "wide" if band is None else f"{int(band[0])}-{int(band[1])}"
                                                    subband_name = "_sub" if subbands is not None else ""
                                                    penalty_name = f"{penalty:g}" if penalty else ""
                                                    agg_name = "" if aggregator == "curve_mean" else f"_{aggregator}{penalty_name}"
                                                    selector_name = "" if selector == "fixed_top_k" else f"_{selector}_m{min_k}_s{stable_threshold:g}_n{stable_steps}_fb{fallback_k}"
                                                    name = (
                                                        f"{band_name}_b{beta:g}_{transform}_{geometry}_y{ldv_y:g}_{score_mode}"
                                                        f"_k{top_k_windows}_r{local_peak_radius_ms:g}_{gcc_mode}{coherence_floor:g}"
                                                        f"_{estimator}{subband_name}{agg_name}{selector_name}"
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
    strict_profile = args.profile in ("strict_v2", "strict_v3")
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
    write_markdown_report(out_dir / "report.md", payload)
    print(f"Wrote {out_dir / 'summary.json'}")
    if results:
        print(f"Wrote {out_dir / 'best_window_diagnostics.json'}")
        print(f"Wrote {out_dir / 'best_incremental_window_diagnostics.json'}")
    print(f"Wrote {out_dir / 'report.md'}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", default=str(Path(__file__).resolve().parent.parent / "dataset" / "0223"))
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--profile", choices=("quick", "targeted", "advanced", "strict_v2", "strict_v3", "refine", "full"), default="quick")
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
