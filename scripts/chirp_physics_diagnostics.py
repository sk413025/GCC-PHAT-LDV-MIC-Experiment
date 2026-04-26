#!/usr/bin/env python3
"""
Chirp-first physical diagnostics from raw WAV files.

This script is intentionally separate from the main PI-GS audit.  It treats the
chirp as a known excitation and asks which pieces of physically constrained
signal processing actually preserve source position:

* mic-mic TDOA from ordinary cross-correlation/GCC;
* mic-mic TDOA after synthetic chirp matched filtering;
* LDV-mic geometric scoring after synthetic chirp matched filtering.

The goal is not to tune the speech result directly.  It is to make the chirp
failure modes visible: center collapse, edge flips, non-monotone estimates,
and dependence on unknown common electronic/structural delays.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.fft import irfft, rfft, rfftfreq
from scipy.io import wavfile
from scipy.signal import butter, correlate, correlation_lags, fftconvolve, hilbert, sosfiltfilt


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
    split: str


@dataclass(frozen=True)
class PairCondition:
    family: str
    band_hz: tuple[float, float]
    window_sec: tuple[float, float]
    transform: str
    phat_beta: float
    lag_sign: int
    chirp_template: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class LdvMicCondition:
    family: str
    band_hz: tuple[float, float]
    window_sec: tuple[float, float]
    transform: str
    ldv_y_m: float
    delay_sign: int
    common_shift_ms: float
    score_mode: str
    chirp_template: tuple[float, float, float]


def default_trials(data_root: Path) -> list[Trial]:
    candidates = [
        Trial(
            -0.8,
            "-0.8m #20",
            "0223-block/0223-block-7(high)/0223-LDV-40-boy(-0.8m)-20-block.wav",
            "0223-block/0223-block-7(high)/0223-MIC-LEFT-40-boy(-0.8m)-20-block.wav",
            "0223-block/0223-block-7(high)/0223-MIC-RIGHT-40-boy(-0.8m)-20-block.wav",
            "canonical",
        ),
        Trial(
            -0.4,
            "-0.4m #19",
            "0223-block-6(high)/0223-LDV-40-boy(-0.4m)-19-block.wav",
            "0223-block-6(high)/0223-MIC-LEFT-40-boy(-0.4m)-19-block.wav",
            "0223-block-6(high)/0223-MIC-RIGHT-40-boy(-0.4m)-19-block.wav",
            "canonical",
        ),
        Trial(
            0.0,
            "+0.0m #18",
            "0223-block/0223-block-5(high)/0223-LDV-40-boy(+0.0m)-18-block.wav",
            "0223-block/0223-block-5(high)/0223-MIC-LEFT-40-boy(+0.0m)-18-block.wav",
            "0223-block/0223-block-5(high)/0223-MIC-RIGHT-40-boy(+0.0m)-18-block.wav",
            "canonical",
        ),
        Trial(
            0.4,
            "+0.4m #16",
            "0223-block/0223-block-3(high)/0223-LDV-40-boy(+0.4m)-16-block.wav",
            "0223-block/0223-block-3(high)/0223-MIC-LEFT-40-boy(+0.4m)-16-block.wav",
            "0223-block/0223-block-3(high)/0223-MIC-RIGHT-40-boy(+0.4m)-16-block.wav",
            "canonical",
        ),
        Trial(
            0.8,
            "+0.8m #17",
            "0223-block/0223-block-4(high)/0223-LDV-40-boy(+0.8m)-17-block.wav",
            "0223-block/0223-block-4(high)/0223-MIC-LEFT-40-boy(+0.8m)-17-block.wav",
            "0223-block/0223-block-4(high)/0223-MIC-RIGHT-40-boy(+0.8m)-17-block.wav",
            "canonical",
        ),
        Trial(
            -0.8,
            "-0.8m #21",
            "0223-block/0223-block-7(high)/0223-LDV-40-boy(-0.8m)-21-block.wav",
            "0223-block/0223-block-7(high)/0223-MIC-LEFT-40-boy(-0.8m)-21-block.wav",
            "0223-block/0223-block-7(high)/0223-MIC-RIGHT-40-boy(-0.8m)-21-block.wav",
            "holdout",
        ),
        Trial(
            0.0,
            "+0.0m #22",
            "0223-block/0223-block-5(high)/0223-LDV-40-boy(+0.0m)-22-block.wav",
            "0223-block/0223-block-5(high)/0223-MIC-LEFT-40-boy(+0.0m)-22-block.wav",
            "0223-block/0223-block-5(high)/0223-MIC-RIGHT-40-boy(+0.0m)-22-block.wav",
            "holdout",
        ),
        Trial(
            0.4,
            "+0.4m #15",
            "0223-block/0223-block-2/0223-LDV-40-boy(+0.4m)-15-block.wav",
            "0223-block/0223-block-2/0223-MIC-LEFT-40-boy(+0.4m)-15-block.wav",
            "0223-block/0223-block-2/0223-MIC-RIGHT-40-boy(+0.4m)-15-block.wav",
            "holdout",
        ),
        Trial(
            0.4,
            "+0.4m #13",
            "0223-block/0223-LDV-40-boy(+0.4m)-13-block.wav",
            "0223-block/0223-MIC-LEFT-40-boy(+0.4m)-13-block.wav",
            "0223-block/0223-MIC-RIGHT-40-boy(+0.4m)-13-block.wav",
            "holdout",
        ),
        Trial(
            0.8,
            "+0.8m #21",
            "0223-block/0223-block-4(high)/0223-LDV-40-boy(+0.8m)-21-block.wav",
            "0223-block/0223-block-4(high)/0223-MIC-LEFT-40-boy(+0.8m)-21-block.wav",
            "0223-block/0223-block-4(high)/0223-MIC-RIGHT-40-boy(+0.8m)-21-block.wav",
            "holdout",
        ),
    ]
    return [trial for trial in candidates if trial_files_exist(data_root, trial)]


def trial_files_exist(data_root: Path, trial: Trial) -> bool:
    return all((data_root / rel).exists() for rel in (trial.ldv, trial.mic_l, trial.mic_r))


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    fs, data = wavfile.read(path)
    if data.ndim != 1:
        raise ValueError(f"Expected mono WAV: {path}")
    x = data.astype(np.float64)
    if np.issubdtype(data.dtype, np.integer):
        x /= float(np.iinfo(data.dtype).max)
    x -= float(np.mean(x))
    return int(fs), x


def slice_seconds(x: np.ndarray, fs: int, window_sec: tuple[float, float]) -> np.ndarray:
    lo = int(round(window_sec[0] * fs))
    hi = int(round(window_sec[1] * fs))
    if lo < 0 or hi <= lo or hi > len(x):
        raise ValueError(f"Invalid window {window_sec} for {len(x) / fs:.3f}s signal")
    y = x[lo:hi].copy()
    y -= float(np.mean(y))
    return y


def preprocess(x: np.ndarray, fs: int, band_hz: tuple[float, float], transform: str) -> np.ndarray:
    y = x.astype(np.float64, copy=True)
    y -= float(np.mean(y))
    if transform == "diff":
        y = np.r_[0.0, np.diff(y)]
    elif transform == "clip":
        threshold = 0.6 * float(np.std(y))
        y = np.sign(y) * np.maximum(np.abs(y) - threshold, 0.0)
    elif transform != "plain":
        raise ValueError(f"Unknown transform: {transform}")

    low, high = band_hz
    sos = butter(4, [low / (0.5 * fs), high / (0.5 * fs)], btype="bandpass", output="sos")
    y = sosfiltfilt(sos, y)
    y -= float(np.mean(y))
    scale = float(np.std(y))
    if scale > 1e-12:
        y /= scale
    return y


def synth_chirp(fs: int, f0_hz: float, f1_hz: float, duration_sec: float) -> np.ndarray:
    n = int(round(duration_sec * fs))
    t = np.arange(n, dtype=np.float64) / float(fs)
    slope = (f1_hz - f0_hz) / duration_sec
    phase = 2.0 * np.pi * (f0_hz * t + 0.5 * slope * t * t)
    y = np.sin(phase) * np.hanning(n)
    y -= float(np.mean(y))
    y /= max(float(np.std(y)), 1e-12)
    return y


def matched_filter_envelope(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    corr = fftconvolve(x, template[::-1], mode="same")
    env = np.abs(hilbert(corr))
    env -= float(np.mean(env))
    env /= max(float(np.std(env)), 1e-12)
    return env


def gcc_curve(
    a: np.ndarray,
    b: np.ndarray,
    fs: int,
    phat_beta: float,
    band_hz: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = 1
    while n < len(a) + len(b):
        n *= 2
    a_win = a * np.hanning(len(a))
    b_win = b * np.hanning(len(b))
    cross = rfft(b_win, n=n) * np.conj(rfft(a_win, n=n))
    if band_hz is not None:
        freqs = rfftfreq(n, 1.0 / fs)
        low, high = band_hz
        cross *= (freqs >= low) & (freqs <= high)
    if phat_beta > 0.0:
        denom = np.abs(cross) ** phat_beta
        cross = cross / np.maximum(denom, 1e-12)
    cc = np.real(irfft(cross, n=n))
    half = n // 2
    cc = np.concatenate([cc[-half:], cc[:half]])
    lags = np.arange(-half, half, dtype=np.float64) / float(fs)
    return cc, lags


def plain_corr_curve(a: np.ndarray, b: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    cc = correlate(b, a, mode="full", method="fft")
    lags = correlation_lags(len(b), len(a), mode="full").astype(np.float64) / float(fs)
    return cc, lags


def pick_lag(cc: np.ndarray, lags: np.ndarray, max_lag_ms: float) -> tuple[float, float]:
    radius_sec = max_lag_ms / 1000.0
    mask = np.abs(lags) <= radius_sec
    if not np.any(mask):
        return 0.0, 0.0
    indices = np.where(mask)[0]
    idx = int(indices[np.argmax(np.abs(cc[indices]))])
    return float(lags[idx]), float(abs(cc[idx]))


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


def tau_lr_template(xs: np.ndarray) -> np.ndarray:
    d_l = np.sqrt((xs - MIC_LEFT_X_M) ** 2 + MIC_Y_M**2)
    d_r = np.sqrt((xs - MIC_RIGHT_X_M) ** 2 + MIC_Y_M**2)
    return (d_r - d_l) / C_MPS


def tau_ldv_mic_templates(xs: np.ndarray, ldv_y_m: float) -> tuple[np.ndarray, np.ndarray]:
    d_l = np.sqrt((xs - MIC_LEFT_X_M) ** 2 + (MIC_Y_M - ldv_y_m) ** 2)
    d_r = np.sqrt((xs - MIC_RIGHT_X_M) ** 2 + (MIC_Y_M - ldv_y_m) ** 2)
    return d_l / C_MPS, d_r / C_MPS


def x_from_lr_tau(tau_sec: float, xs_grid: np.ndarray) -> float:
    taus = tau_lr_template(xs_grid)
    return float(xs_grid[int(np.argmin(np.abs(taus - tau_sec)))])


def summarize_estimates(rows: list[dict[str, object]]) -> dict[str, object]:
    errs = np.asarray([float(row["abs_err_deg"]) for row in rows], dtype=np.float64)
    x_hats = [float(row["x_hat_m"]) for row in rows]
    monotone_pairs = sum(1 for left, right in zip(x_hats, x_hats[1:]) if left <= right + 1e-12)
    edge_hits = sum(1 for x in x_hats if abs(x) >= 1.19)
    return {
        "mae_deg": float(np.mean(errs)),
        "max_err_deg": float(np.max(errs)),
        "monotone_pairs": int(monotone_pairs),
        "num_pairs": max(len(x_hats) - 1, 0),
        "edge_hits": int(edge_hits),
        "x_hat_sequence_m": x_hats,
        "rows": rows,
    }


def apply_center_edge_prior(x_m: float, deadband_m: float, edge_threshold_m: float, edge_gain: float) -> float:
    x = float(x_m)
    if abs(x) <= deadband_m:
        x = 0.0
    if abs(x) > edge_threshold_m and edge_gain > 1.0:
        x = math.copysign(min(1.2, edge_threshold_m + edge_gain * (abs(x) - edge_threshold_m)), x)
    return x


def evaluate_pair_condition(
    signals: dict[str, dict[str, object]],
    trials: list[Trial],
    condition: PairCondition,
    xs_grid: np.ndarray,
    cache: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, object]:
    if cache is None:
        cache = {}
    rows: list[dict[str, object]] = []
    for trial in trials:
        item = signals[trial.label]
        fs = int(item["fs"])
        key = (
            "pair",
            trial.label,
            condition.family,
            tuple(condition.band_hz),
            tuple(condition.window_sec),
            condition.transform,
            condition.phat_beta,
            None if condition.chirp_template is None else tuple(condition.chirp_template),
        )
        if key in cache:
            cc, lags = cache[key]
        else:
            left = slice_seconds(item["mic_l"], fs, condition.window_sec)  # type: ignore[arg-type]
            right = slice_seconds(item["mic_r"], fs, condition.window_sec)  # type: ignore[arg-type]
            left = preprocess(left, fs, condition.band_hz, condition.transform)
            right = preprocess(right, fs, condition.band_hz, condition.transform)
            if condition.family == "mic_mf_env":
                if condition.chirp_template is None:
                    raise ValueError("Matched-filter condition requires chirp_template")
                template = synth_chirp(fs, *condition.chirp_template)
                left = matched_filter_envelope(left, template)
                right = matched_filter_envelope(right, template)
                cc, lags = plain_corr_curve(left, right, fs)
            elif condition.family == "mic_gcc":
                cc, lags = gcc_curve(left, right, fs, condition.phat_beta, None)
            else:
                raise ValueError(f"Unknown pair family: {condition.family}")
            cache[key] = (cc, lags)
        tau, evidence = pick_lag(cc, lags, max_lag_ms=5.0)
        x_hat = x_from_lr_tau(condition.lag_sign * tau, xs_grid)
        theta_hat = theta_from_x(x_hat)
        theta_true = theta_from_x(trial.x_m)
        rows.append(
            {
                "label": trial.label,
                "split": trial.split,
                "x_true_m": trial.x_m,
                "x_hat_m": x_hat,
                "tau_ms": float(condition.lag_sign * tau * 1000.0),
                "theta_true_deg": theta_true,
                "theta_hat_deg": theta_hat,
                "abs_err_deg": abs(theta_hat - theta_true),
                "evidence": evidence,
            }
        )
    summary = summarize_estimates(rows)
    return {"condition": asdict(condition), **summary}


def evaluate_ldv_mic_condition(
    signals: dict[str, dict[str, object]],
    trials: list[Trial],
    condition: LdvMicCondition,
    xs_grid: np.ndarray,
    cache: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> dict[str, object]:
    if cache is None:
        cache = {}
    tau_l_base, tau_r_base = tau_ldv_mic_templates(xs_grid, condition.ldv_y_m)
    shift_sec = condition.common_shift_ms / 1000.0
    tau_l = condition.delay_sign * tau_l_base + shift_sec
    tau_r = condition.delay_sign * tau_r_base + shift_sec
    rows: list[dict[str, object]] = []

    for trial in trials:
        item = signals[trial.label]
        fs = int(item["fs"])
        key = (
            "ldv_mic",
            trial.label,
            tuple(condition.band_hz),
            tuple(condition.window_sec),
            condition.transform,
            tuple(condition.chirp_template),
        )
        if key in cache:
            score_l, lags_l, score_r, lags_r = cache[key]
        else:
            ldv = slice_seconds(item["ldv"], fs, condition.window_sec)  # type: ignore[arg-type]
            left = slice_seconds(item["mic_l"], fs, condition.window_sec)  # type: ignore[arg-type]
            right = slice_seconds(item["mic_r"], fs, condition.window_sec)  # type: ignore[arg-type]
            ldv = preprocess(ldv, fs, condition.band_hz, condition.transform)
            left = preprocess(left, fs, condition.band_hz, condition.transform)
            right = preprocess(right, fs, condition.band_hz, condition.transform)
            template = synth_chirp(fs, *condition.chirp_template)
            ldv = matched_filter_envelope(ldv, template)
            left = matched_filter_envelope(left, template)
            right = matched_filter_envelope(right, template)
            cc_l, lags_l = plain_corr_curve(ldv, left, fs)
            cc_r, lags_r = plain_corr_curve(ldv, right, fs)
            score_l = robust_normalize(np.abs(cc_l))
            score_r = robust_normalize(np.abs(cc_r))
            cache[key] = (score_l, lags_l, score_r, lags_r)
        sampled_l = sample_curve(score_l, lags_l, tau_l)
        sampled_r = sample_curve(score_r, lags_r, tau_r)
        if condition.score_mode == "product":
            score = sampled_l * sampled_r
        elif condition.score_mode == "min":
            score = np.minimum(sampled_l, sampled_r)
        elif condition.score_mode == "sum":
            score = sampled_l + sampled_r
        else:
            raise ValueError(f"Unknown score mode: {condition.score_mode}")
        best_idx = int(np.argmax(score))
        x_hat = float(xs_grid[best_idx])
        theta_hat = theta_from_x(x_hat)
        theta_true = theta_from_x(trial.x_m)
        rows.append(
            {
                "label": trial.label,
                "split": trial.split,
                "x_true_m": trial.x_m,
                "x_hat_m": x_hat,
                "theta_true_deg": theta_true,
                "theta_hat_deg": theta_hat,
                "abs_err_deg": abs(theta_hat - theta_true),
                "score": float(score[best_idx]),
                "score_l": float(sampled_l[best_idx]),
                "score_r": float(sampled_r[best_idx]),
            }
        )
    summary = summarize_estimates(rows)
    return {"condition": asdict(condition), **summary}


def load_signals(data_root: Path, trials: Iterable[Trial]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for trial in trials:
        fs_v, ldv = read_wav(data_root / trial.ldv)
        fs_l, mic_l = read_wav(data_root / trial.mic_l)
        fs_r, mic_r = read_wav(data_root / trial.mic_r)
        if not (fs_v == fs_l == fs_r):
            raise ValueError(f"Sample-rate mismatch for {trial.label}")
        out[trial.label] = {"fs": fs_v, "ldv": ldv, "mic_l": mic_l, "mic_r": mic_r}
    return out


def build_pair_conditions() -> list[PairCondition]:
    bands = [(80.0, 800.0), (300.0, 1500.0), (1500.0, 4000.0), (4000.0, 8000.0), (80.0, 8000.0)]
    windows = [(0.0, 0.5), (0.25, 0.75), (0.5, 1.0), (0.75, 1.25), (1.0, 1.5), (1.25, 1.75), (0.0, 1.5), (0.0, 2.0)]
    templates = [(300.0, 8000.0, 1.5), (500.0, 7000.0, 1.5)]
    conditions: list[PairCondition] = []
    for band in bands:
        for window in windows:
            for transform in ("plain", "diff", "clip"):
                for lag_sign in (1, -1):
                    conditions.append(PairCondition("mic_gcc", band, window, transform, 0.0, lag_sign))
                    conditions.append(PairCondition("mic_gcc", band, window, transform, 1.0, lag_sign))
                for template in templates:
                    for lag_sign in (1, -1):
                        conditions.append(PairCondition("mic_mf_env", band, window, transform, 0.0, lag_sign, template))
    return conditions


def build_ldv_mic_conditions() -> list[LdvMicCondition]:
    bands = [(80.0, 800.0), (300.0, 1500.0), (1500.0, 4000.0), (4000.0, 8000.0), (80.0, 8000.0)]
    windows = [(0.0, 0.5), (0.25, 0.75), (0.5, 1.0), (0.75, 1.25), (1.0, 1.5), (1.25, 1.75), (0.0, 1.5), (0.0, 2.0)]
    templates = [(300.0, 8000.0, 1.5), (500.0, 7000.0, 1.5)]
    shifts = [-2.0, -1.0, 0.0, 1.0, 2.0]
    conditions: list[LdvMicCondition] = []
    for band in bands:
        for window in windows:
            for transform in ("plain", "diff", "clip"):
                for template in templates:
                    for delay_sign in (1, -1):
                        for shift in shifts:
                            conditions.append(
                                LdvMicCondition(
                                    "ldv_mic_mf_env",
                                    band,
                                    window,
                                    transform,
                                    0.5,
                                    delay_sign,
                                    shift,
                                    "product",
                                    template,
                                )
                            )
    return conditions


def build_expanded_ldv_mic_conditions() -> list[LdvMicCondition]:
    """Denser LDV-mic grid for chirp-only physical forensics.

    The first chirp pass showed that an effective LDV reference closer to the
    excited wall patch can outperform the nominal 0.5 m geometry on repeated
    recordings.  This grid keeps the search physically narrow: positive
    acoustic delay, one plausible synthetic chirp, and a small common-delay
    range.
    """

    bands = [(80.0, 800.0), (300.0, 1500.0), (1500.0, 4000.0), (4000.0, 8000.0), (80.0, 8000.0)]
    windows = [(0.0, 0.5), (0.125, 0.625), (0.25, 0.75), (0.375, 0.875), (0.5, 1.0), (0.75, 1.25), (1.0, 1.5), (1.25, 1.75)]
    ldv_ys = [round(v, 1) for v in np.arange(0.0, 0.81, 0.1)]
    shifts = [round(v, 2) for v in np.arange(0.5, 3.01, 0.25)]
    conditions: list[LdvMicCondition] = []
    for band in bands:
        for window in windows:
            for transform in ("diff", "plain"):
                for ldv_y in ldv_ys:
                    for shift in shifts:
                        for score_mode in ("product", "sum"):
                            conditions.append(
                                LdvMicCondition(
                                    "ldv_mic_mf_env",
                                    band,
                                    window,
                                    transform,
                                    ldv_y,
                                    1,
                                    shift,
                                    score_mode,
                                    (500.0, 7000.0, 1.5),
                                )
                            )
    return conditions


def split_summary(results: list[dict[str, object]], split: str) -> list[dict[str, object]]:
    out = []
    for result in results:
        rows = [row for row in result["rows"] if row["split"] == split]  # type: ignore[index]
        if rows:
            out.append({"condition": result["condition"], **summarize_estimates(rows)})  # type: ignore[arg-type]
    out.sort(key=lambda row: (float(row["mae_deg"]), float(row["max_err_deg"])))
    return out


def sliding_windows(total_sec: float = 2.0, window_sec: float = 0.5, hop_sec: float = 0.125) -> list[tuple[float, float]]:
    windows = []
    start = 0.0
    while start + window_sec <= total_sec + 1e-9:
        windows.append((round(start, 6), round(start + window_sec, 6)))
        start += hop_sec
    return windows


def sliding_oracle_report(
    signals: dict[str, dict[str, object]],
    trials: list[Trial],
    xs_grid: np.ndarray,
    cache: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    """Diagnostic-only report: choose each trial's window using its label.

    This is intentionally marked oracle because it uses the known x position to
    pick a window.  A strong oracle result says the chirp contains useful
    information somewhere; it does not say we have solved blind window
    selection.
    """

    base_conditions = [
        LdvMicCondition(
            "ldv_mic_mf_env",
            (1500.0, 4000.0),
            (0.25, 0.75),
            "diff",
            0.3,
            1,
            1.5,
            "product",
            (500.0, 7000.0, 1.5),
        ),
        LdvMicCondition(
            "ldv_mic_mf_env",
            (1500.0, 4000.0),
            (0.25, 0.75),
            "diff",
            0.5,
            1,
            2.0,
            "product",
            (500.0, 7000.0, 1.5),
        ),
    ]
    reports = []
    for base in base_conditions:
        chosen_rows = []
        for trial in trials:
            candidates = []
            for window in sliding_windows():
                cond = LdvMicCondition(**{**asdict(base), "window_sec": window})
                result = evaluate_ldv_mic_condition(signals, [trial], cond, xs_grid, cache)
                row = dict(result["rows"][0])  # type: ignore[index]
                row["window_sec"] = window
                candidates.append(row)
            chosen_rows.append(min(candidates, key=lambda row: float(row["abs_err_deg"])))
        reports.append({"base_condition": asdict(base), **summarize_estimates(chosen_rows)})
    reports.sort(key=lambda row: (float(row["mae_deg"]), float(row["max_err_deg"])))
    return reports


def fixed_regime_fusion_report(
    signals: dict[str, dict[str, object]],
    trials: list[Trial],
    xs_grid: np.ndarray,
    cache: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    """Fuse three physically distinct chirp regimes.

    This is label-free at the row level: every trial uses the same fixed
    regimes and weights.  It is still a diagnostic hypothesis because the
    regimes were chosen after inspecting this dataset's failure modes.
    """

    regimes = [
        (
            "high_res_early",
            LdvMicCondition(
                "ldv_mic_mf_env",
                (1500.0, 4000.0),
                (0.25, 0.75),
                "diff",
                0.3,
                1,
                1.5,
                "product",
                (500.0, 7000.0, 1.5),
            ),
        ),
        (
            "low_freq_stable",
            LdvMicCondition(
                "ldv_mic_mf_env",
                (80.0, 800.0),
                (0.75, 1.25),
                "diff",
                0.8,
                1,
                1.25,
                "product",
                (500.0, 7000.0, 1.5),
            ),
        ),
        (
            "mid_late_repeat_stable",
            LdvMicCondition(
                "ldv_mic_mf_env",
                (1500.0, 4000.0),
                (0.375, 0.875),
                "diff",
                0.1,
                1,
                1.0,
                "sum",
                (500.0, 7000.0, 1.5),
            ),
        ),
    ]
    regime_results = {
        name: evaluate_ldv_mic_condition(signals, trials, cond, xs_grid, cache)
        for name, cond in regimes
    }

    def build_rows(deadband_m: float, edge_threshold_m: float, edge_gain: float) -> list[dict[str, object]]:
        rows = []
        for row_idx, trial in enumerate(trials):
            components = []
            x_values = []
            for name, _ in regimes:
                source = regime_results[name]["rows"][row_idx]  # type: ignore[index]
                x_value = float(source["x_hat_m"])
                x_values.append(x_value)
                components.append({"name": name, "x_hat_m": x_value, "score": float(source.get("score", 0.0))})
            x_raw = float(np.mean(x_values))
            x_hat = apply_center_edge_prior(x_raw, deadband_m, edge_threshold_m, edge_gain)
            theta_hat = theta_from_x(x_hat)
            theta_true = theta_from_x(trial.x_m)
            rows.append(
                {
                    "label": trial.label,
                    "split": trial.split,
                    "x_true_m": trial.x_m,
                    "x_hat_m": x_hat,
                    "x_raw_m": x_raw,
                    "theta_true_deg": theta_true,
                    "theta_hat_deg": theta_hat,
                    "abs_err_deg": abs(theta_hat - theta_true),
                    "components": components,
                }
            )
        return rows

    reports = [
        {
            "fusion": "fixed_equal_three_regime",
            "weights": {name: 1.0 / len(regimes) for name, _ in regimes},
            "postprocess": {"center_deadband_m": 0.0, "edge_threshold_m": 99.0, "edge_gain": 1.0},
            **summarize_estimates(build_rows(0.0, 99.0, 1.0)),
        },
        {
            "fusion": "fixed_equal_three_regime_center_edge_tuned",
            "weights": {name: 1.0 / len(regimes) for name, _ in regimes},
            "postprocess": {"center_deadband_m": 0.275, "edge_threshold_m": 0.2, "edge_gain": 1.1},
            **summarize_estimates(build_rows(0.275, 0.2, 1.1)),
        },
    ]
    reports.sort(key=lambda row: (float(row["mae_deg"]), float(row["max_err_deg"])))
    return reports


def topk_ensemble_report(results: list[dict[str, object]], top_n: int) -> list[dict[str, object]]:
    """Dataset-ranked ensemble upper-bound diagnostic.

    The condition ranking uses the current evaluated rows, so this should be
    read as an in-dataset diagnostic rather than an external validation result.
    """

    if not results:
        return []

    labels = [str(row["label"]) for row in results[0]["rows"]]  # type: ignore[index]
    rows_by_label = {str(row["label"]): row for row in results[0]["rows"]}  # type: ignore[index]
    reports = []
    for k in (3, 5, 8, 10, 15):
        if k > len(results):
            continue
        selected = results[:k]
        for mode in ("mean", "weighted_mean", "median"):
            rows_raw = []
            for label in labels:
                x_values = []
                weights = []
                components = []
                for result in selected:
                    row = next(r for r in result["rows"] if str(r["label"]) == label)  # type: ignore[index]
                    x_value = float(row["x_hat_m"])
                    weight = 1.0 / max(float(result["mae_deg"]), 1e-9) ** 2
                    x_values.append(x_value)
                    weights.append(weight)
                    components.append({"x_hat_m": x_value, "condition_mae_deg": float(result["mae_deg"])})
                if mode == "mean":
                    x_hat = float(np.mean(x_values))
                elif mode == "weighted_mean":
                    x_hat = float(np.average(x_values, weights=weights))
                else:
                    x_hat = float(np.median(x_values))
                base = rows_by_label[label]
                theta_hat = theta_from_x(x_hat)
                theta_true = theta_from_x(float(base["x_true_m"]))
                rows_raw.append(
                    {
                        "label": label,
                        "split": base["split"],
                        "x_true_m": float(base["x_true_m"]),
                        "x_hat_m": x_hat,
                        "x_raw_m": x_hat,
                        "theta_true_deg": theta_true,
                        "theta_hat_deg": theta_hat,
                        "abs_err_deg": abs(theta_hat - theta_true),
                        "components": components,
                    }
                )
            variants = [
                ("none", {"center_deadband_m": 0.0, "edge_threshold_m": 99.0, "edge_gain": 1.0}),
                ("center_edge_tuned", {"center_deadband_m": 0.15, "edge_threshold_m": 0.2, "edge_gain": 1.35}),
            ]
            for variant_name, post in variants:
                rows = []
                for row in rows_raw:
                    x_raw = float(row["x_raw_m"])
                    x_hat = apply_center_edge_prior(
                        x_raw,
                        float(post["center_deadband_m"]),
                        float(post["edge_threshold_m"]),
                        float(post["edge_gain"]),
                    )
                    theta_hat = theta_from_x(x_hat)
                    rows.append(
                        {
                            **row,
                            "x_hat_m": x_hat,
                            "theta_hat_deg": theta_hat,
                            "abs_err_deg": abs(theta_hat - float(row["theta_true_deg"])),
                        }
                    )
                reports.append(
                    {
                        "fusion": "dataset_ranked_topk",
                        "top_k": k,
                        "mode": mode,
                        "postprocess_name": variant_name,
                        "postprocess": post,
                        **summarize_estimates(rows),
                    }
                )
    reports.sort(key=lambda row: (float(row["mae_deg"]), float(row["max_err_deg"])))
    return reports[:top_n]


def canonical_grid_calibration_report(source_reports: list[dict[str, object]], top_n: int) -> list[dict[str, object]]:
    """Calibrate fused chirp estimates with known canonical grid points.

    This is appropriate only for the diagnostic question "can chirp act as a
    calibration signal for known source positions?"  It is not a blind
    continuous-localization metric because it uses canonical labels to define a
    monotone x-axis map and can optionally snap to the known discrete grid.
    """

    reports = []
    for source in source_reports[: min(len(source_reports), top_n)]:
        rows = list(source["rows"])  # type: ignore[arg-type]
        canonical = [row for row in rows if row["split"] == "canonical"]
        if len(canonical) < 2:
            continue
        raw_c = np.asarray([float(row.get("x_raw_m", row["x_hat_m"])) for row in canonical], dtype=np.float64)
        true_c = np.asarray([float(row["x_true_m"]) for row in canonical], dtype=np.float64)
        order = np.argsort(raw_c)
        raw_knots = raw_c[order]
        # Use the known left-to-right calibration grid as the monotone target.
        true_knots = np.sort(true_c)
        grid = np.unique(true_knots)

        for mode in ("piecewise", "piecewise_grid_snap"):
            out_rows = []
            for row in rows:
                raw_x = float(row.get("x_raw_m", row["x_hat_m"]))
                x_hat = float(np.interp(raw_x, raw_knots, true_knots, left=true_knots[0], right=true_knots[-1]))
                if mode == "piecewise_grid_snap":
                    x_hat = float(grid[int(np.argmin(np.abs(grid - x_hat)))])
                theta_hat = theta_from_x(x_hat)
                theta_true = theta_from_x(float(row["x_true_m"]))
                out_rows.append(
                    {
                        "label": row["label"],
                        "split": row["split"],
                        "x_true_m": float(row["x_true_m"]),
                        "x_hat_m": x_hat,
                        "x_raw_m": raw_x,
                        "theta_true_deg": theta_true,
                        "theta_hat_deg": theta_hat,
                        "abs_err_deg": abs(theta_hat - theta_true),
                    }
                )
            reports.append(
                {
                    "calibration": "canonical_grid",
                    "mode": mode,
                    "source_fusion": source.get("fusion", "unknown"),
                    "source_top_k": source.get("top_k"),
                    "source_mode": source.get("mode"),
                    "source_postprocess_name": source.get("postprocess_name"),
                    "raw_knots_m": raw_knots.tolist(),
                    "true_knots_m": true_knots.tolist(),
                    **summarize_estimates(out_rows),
                }
            )

    reports.sort(key=lambda row: (float(row["mae_deg"]), float(row["max_err_deg"])))
    return reports[:top_n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset/0223"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/chirp_physics_diagnostics"))
    parser.add_argument("--x-min", type=float, default=-1.2)
    parser.add_argument("--x-max", type=float, default=1.2)
    parser.add_argument("--x-step", type=float, default=0.001)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--all-trials-sweep",
        action="store_true",
        help="Also rank every condition directly on canonical+holdout trials.",
    )
    parser.add_argument(
        "--expanded-ldv-grid",
        action="store_true",
        help="Run a denser chirp-specific LDV geometry grid over all trials.",
    )
    args = parser.parse_args()

    trials = default_trials(args.data_root)
    if not trials:
        raise SystemExit(f"No trials found under {args.data_root}")
    canonical = [trial for trial in trials if trial.split == "canonical"]
    xs_grid = np.arange(args.x_min, args.x_max + 0.5 * args.x_step, args.x_step)
    signals = load_signals(args.data_root, trials)

    pair_cache: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] = {}
    ldv_cache: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    pair_conditions = build_pair_conditions()
    ldv_conditions = build_ldv_mic_conditions()

    pair_results = [evaluate_pair_condition(signals, canonical, cond, xs_grid, pair_cache) for cond in pair_conditions]
    pair_results.sort(key=lambda row: (float(row["mae_deg"]), float(row["max_err_deg"])))
    ldv_results = [evaluate_ldv_mic_condition(signals, canonical, cond, xs_grid, ldv_cache) for cond in ldv_conditions]
    ldv_results.sort(key=lambda row: (float(row["mae_deg"]), float(row["max_err_deg"])))

    best_conditions = []
    for result in pair_results[: args.top_n]:
        cond = PairCondition(**result["condition"])  # type: ignore[arg-type]
        best_conditions.append(evaluate_pair_condition(signals, trials, cond, xs_grid, pair_cache))
    for result in ldv_results[: args.top_n]:
        cond = LdvMicCondition(**result["condition"])  # type: ignore[arg-type]
        best_conditions.append(evaluate_ldv_mic_condition(signals, trials, cond, xs_grid, ldv_cache))
    best_conditions.sort(key=lambda row: (float(row["mae_deg"]), float(row["max_err_deg"])))

    all_trials_top: list[dict[str, object]] = []
    if args.all_trials_sweep:
        all_pair = [evaluate_pair_condition(signals, trials, cond, xs_grid, pair_cache) for cond in pair_conditions]
        all_ldv = [evaluate_ldv_mic_condition(signals, trials, cond, xs_grid, ldv_cache) for cond in ldv_conditions]
        all_trials_top = sorted(all_pair + all_ldv, key=lambda row: (float(row["mae_deg"]), float(row["max_err_deg"])))[: args.top_n]

    expanded_ldv_top: list[dict[str, object]] = []
    if args.expanded_ldv_grid:
        expanded_results = [
            evaluate_ldv_mic_condition(signals, trials, cond, xs_grid, ldv_cache)
            for cond in build_expanded_ldv_mic_conditions()
        ]
        expanded_ldv_top = sorted(expanded_results, key=lambda row: (float(row["mae_deg"]), float(row["max_err_deg"])))[: args.top_n]

    fixed_fusion = fixed_regime_fusion_report(signals, trials, xs_grid, ldv_cache)
    topk_fusion = topk_ensemble_report(expanded_ldv_top, args.top_n)
    calibration_sources = topk_fusion + fixed_fusion

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "num_trials": len(trials),
        "num_canonical_trials": len(canonical),
        "x_grid": {"min": args.x_min, "max": args.x_max, "step": args.x_step},
        "pair_canonical_top": pair_results[: args.top_n],
        "ldv_mic_canonical_top": ldv_results[: args.top_n],
        "best_conditions_all_trials": best_conditions[: args.top_n],
        "holdout_replay_top": split_summary(best_conditions, "holdout")[: args.top_n],
        "all_trials_direct_top": all_trials_top,
        "expanded_ldv_all_trials_top": expanded_ldv_top,
        "fixed_regime_fusion": fixed_fusion,
        "expanded_topk_ensemble": topk_fusion,
        "canonical_grid_calibration": canonical_grid_calibration_report(calibration_sources, args.top_n),
        "sliding_window_oracle": sliding_oracle_report(signals, trials, xs_grid, ldv_cache),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {args.output_dir / 'summary.json'}")
    print("Top canonical mic/mic condition:")
    print(json.dumps(pair_results[0], indent=2)[:3000])
    print("Top canonical LDV/mic condition:")
    print(json.dumps(ldv_results[0], indent=2)[:3000])
    print("Top replay on all trials:")
    print(json.dumps(best_conditions[0], indent=2)[:3000])


if __name__ == "__main__":
    main()
