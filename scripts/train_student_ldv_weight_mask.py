#!/usr/bin/env python
"""
Train and evaluate a simple student that predicts LDV-informed MIC-MIC GCC weights.

Training data
-------------
Input is the teacher dataset produced by:
  scripts/ldv_informed_gcc_micmic.py

The dataset contains:
- X: (N, B) LDV log-PSD per band (B=64 across 500–2000 Hz)
- Y: (N, B) teacher band weights in [0, 1]
- speaker_id: (N,) string
- center_sec: (N,) float
- band_edges_hz: (B+1,) float

Student model (plan-locked)
---------------------------
Per-band ridge regression with intercept:
  w_hat = clip(a_b * x_b + c_b, 0, 1)
Ridge lambda is fixed: 1e-2, applied to slope only.

Evaluation split (plan-locked)
------------------------------
Per speaker, time-based:
- Train windows: center_sec in [100, 450]
- Test windows : center_sec in (450, 600]

Evaluation metric (plan-locked)
-------------------------------
Compare baseline MIC-MIC vs teacher vs student on test windows:
- Primary truth: chirp reference (tau_ref_ms/theta_ref_deg) from truth_ref_root.
- Success: student achieves >= 80% of teacher p95 improvement (pooled),
  and student median does not worsen by >5% vs baseline.

Example
-------
python -u scripts/train_student_ldv_weight_mask.py \\
  --teacher_dataset results/ldv_informed_micmic_20260213_120000/teacher_dataset.npz \\
  --data_root /path/to/GCC-PHAT-LDV-MIC-Experiment \\
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \\
  --out_dir results/ldv_weight_student_20260213_121000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Plan-locked constants
# ─────────────────────────────────────────────────────────────────────

FS_EXPECTED = 48_000
WINDOW_SEC = 5.0
BAND_HZ = (500.0, 2000.0)

GCC_MAX_LAG_MS = 10.0
GCC_GUIDED_RADIUS_MS = 0.3
PSR_EXCLUDE_SAMPLES = 50
EPS = 1e-12

RIDGE_LAMBDA = 1e-2

TRAIN_MAX_CENTER_SEC = 450.0  # inclusive


GEOMETRY = {
    "mic_left": (-0.7, 2.0),
    "mic_right": (0.7, 2.0),
    "speakers": {
        "18": (0.8, 0.0),
        "19": (0.4, 0.0),
        "20": (0.0, 0.0),
        "21": (-0.4, 0.0),
        "22": (-0.8, 0.0),
    },
}


# ─────────────────────────────────────────────────────────────────────
# Reproducibility helpers
# ─────────────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head_and_dirty() -> tuple[str | None, bool | None]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() != ""
        return head, dirty
    except Exception:
        return None, None


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def configure_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def write_code_state(out_dir: Path, script_path: Path) -> None:
    head, dirty = git_head_and_dirty()
    payload = {
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "git_head": head,
        "dirty": dirty,
        "timestamp": datetime.now().isoformat(),
    }
    write_json(out_dir / "code_state.json", payload)


# ─────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────


def load_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(str(path))
    if data.ndim != 1:
        raise ValueError(f"Expected mono WAV: {path} (shape={data.shape})")
    if data.dtype == np.int16:
        data = (data.astype(np.float32) / 32768.0).astype(np.float32, copy=False)
    elif data.dtype == np.int32:
        data = (data.astype(np.float32) / 2147483648.0).astype(np.float32, copy=False)
    elif data.dtype == np.float32:
        pass
    elif data.dtype == np.float64:
        data = data.astype(np.float32)
    else:
        data = data.astype(np.float32)
    return int(sr), data.astype(np.float32, copy=False)


def list_mic_files(data_root: Path, speaker: str) -> tuple[Path, Path]:
    sp_dir = data_root / speaker
    micl_files = sorted(sp_dir.glob("*LEFT*.wav"))
    micr_files = sorted(sp_dir.glob("*RIGHT*.wav"))
    if not micl_files or not micr_files:
        raise FileNotFoundError(f"Missing MIC WAVs in {sp_dir} (LEFT={len(micl_files)}, RIGHT={len(micr_files)})")
    return micl_files[0], micr_files[0]


def extract_centered_window(
    signal: np.ndarray, *, fs: int, center_sec: float, window_sec: float
) -> np.ndarray:
    win_samples = int(round(float(window_sec) * float(fs)))
    center_samp = int(round(float(center_sec) * float(fs)))
    start = int(center_samp - win_samples // 2)
    end = int(start + win_samples)
    if start < 0 or end > len(signal):
        raise ValueError(
            f"Window out of bounds: center_sec={center_sec}, start={start}, end={end}, len={len(signal)}"
        )
    return signal[start:end]


# ─────────────────────────────────────────────────────────────────────
# Truth + conversions
# ─────────────────────────────────────────────────────────────────────


def compute_geometry_truth(speaker_id: str, *, c: float = 343.0, d: float = 1.4) -> dict[str, float]:
    key = speaker_id.split("-")[0]
    if key not in GEOMETRY["speakers"]:
        raise ValueError(f"Unknown speaker key: {key}")
    x_s, y_s = GEOMETRY["speakers"][key]
    x_l, y_l = GEOMETRY["mic_left"]
    x_r, y_r = GEOMETRY["mic_right"]
    d_l = float(np.hypot(x_s - x_l, y_s - y_l))
    d_r = float(np.hypot(x_s - x_r, y_s - y_r))
    tau_true = (d_l - d_r) / float(c)
    sin_theta = float(np.clip(tau_true * float(c) / float(d), -1.0, 1.0))
    theta_true = float(np.degrees(np.arcsin(sin_theta)))
    return {"tau_true_ms": float(tau_true * 1000.0), "theta_true_deg": theta_true}


def tau_to_theta_deg(tau_sec: float, *, c: float = 343.0, d: float = 1.4) -> float:
    sin_theta = float(np.clip(float(tau_sec) * float(c) / float(d), -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_theta)))


def load_truth_reference(summary_path: Path) -> dict[str, float]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    ref = payload.get("truth_reference", None)
    if not isinstance(ref, dict):
        raise ValueError(f"Missing truth_reference in {summary_path}")
    if "tau_ref_ms" not in ref or "theta_ref_deg" not in ref:
        raise ValueError(f"truth_reference missing tau_ref_ms/theta_ref_deg in {summary_path}")
    return {
        "tau_ref_ms": float(ref["tau_ref_ms"]),
        "theta_ref_deg": float(ref["theta_ref_deg"]),
        "label": str(ref.get("label", "unknown")),
    }


# ─────────────────────────────────────────────────────────────────────
# GCC with band weights
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GCCResult:
    tau_sec: float
    psr_db: float
    theta_deg: float
    theta_error_ref_deg: float
    theta_error_geo_deg: float


def band_mask(freqs_hz: np.ndarray, band_hz: tuple[float, float]) -> np.ndarray:
    lo, hi = float(band_hz[0]), float(band_hz[1])
    return (freqs_hz >= lo) & (freqs_hz <= hi)


def fft_band_weight_from_bands(
    freqs_fft_hz: np.ndarray, band_edges_hz: np.ndarray, w_bands: np.ndarray
) -> np.ndarray:
    if w_bands.ndim != 1:
        raise ValueError("w_bands must be 1D")
    if band_edges_hz.ndim != 1 or band_edges_hz.size != w_bands.size + 1:
        raise ValueError("band_edges_hz must have size B+1")
    idx = np.searchsorted(band_edges_hz, freqs_fft_hz, side="right") - 1
    idx = np.clip(idx, 0, w_bands.size - 1)
    w = w_bands[idx]
    # Zero outside overall band to avoid edge leakage.
    w = np.where(band_mask(freqs_fft_hz, (float(band_edges_hz[0]), float(band_edges_hz[-1]))), w, 0.0)
    return w.astype(np.float64, copy=False)


def estimate_tau_from_weighted_phat(
    R_w: np.ndarray,
    *,
    fs: int,
    n_fft: int,
    max_tau_sec: float,
    guided_tau_sec: float,
    guided_radius_sec: float,
) -> tuple[float, float]:
    cc = np.fft.irfft(R_w, n_fft)
    cc = np.real(cc)

    max_shift = int(round(float(max_tau_sec) * float(fs)))
    if max_shift <= 0:
        raise ValueError("max_shift must be > 0")

    cc_win = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    abs_cc = np.abs(cc_win)

    guided_center = int(round(float(guided_tau_sec) * float(fs))) + max_shift
    guided_radius = int(round(float(guided_radius_sec) * float(fs)))
    lo = max(0, guided_center - guided_radius)
    hi = min(len(abs_cc) - 1, guided_center + guided_radius)
    peak_idx = int(np.argmax(abs_cc[lo : hi + 1])) + lo

    shift = 0.0
    if 0 < peak_idx < len(abs_cc) - 1:
        y0 = abs_cc[peak_idx - 1]
        y1 = abs_cc[peak_idx]
        y2 = abs_cc[peak_idx + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(float(denom)) > 1e-12:
            shift = float(0.5 * (y0 - y2) / denom)

    tau_sec = ((peak_idx - max_shift) + shift) / float(fs)

    mask = np.ones_like(abs_cc, dtype=bool)
    exc = int(PSR_EXCLUDE_SAMPLES)
    lo_e = max(0, peak_idx - exc)
    hi_e = min(len(abs_cc), peak_idx + exc + 1)
    mask[lo_e:hi_e] = False
    sidelobe_max = float(abs_cc[mask].max()) if np.any(mask) else 0.0
    peak_val = float(abs_cc[peak_idx])
    psr_db = 20.0 * float(np.log10(peak_val / (sidelobe_max + 1e-10)))
    return float(tau_sec), float(psr_db)


def gcc_micmic_weighted(
    micl: np.ndarray,
    micr: np.ndarray,
    *,
    fs: int,
    guided_tau_sec: float,
    guided_radius_sec: float,
    max_tau_sec: float,
    weight_fft: np.ndarray,
    theta_ref_deg: float,
    theta_true_deg: float,
) -> GCCResult:
    if micl.shape != micr.shape:
        raise ValueError("micl/micr shape mismatch")
    n_fft = int(micl.size + micr.size)
    X = np.fft.rfft(micl.astype(np.float64, copy=False), n_fft)
    Y = np.fft.rfft(micr.astype(np.float64, copy=False), n_fft)
    R = X * np.conj(Y)
    R_phat = R / (np.abs(R) + EPS)
    if weight_fft.shape != R_phat.shape:
        raise ValueError("weight_fft shape mismatch")
    tau_sec, psr_db = estimate_tau_from_weighted_phat(
        (R_phat * weight_fft).astype(np.complex128, copy=False),
        fs=fs,
        n_fft=n_fft,
        max_tau_sec=max_tau_sec,
        guided_tau_sec=guided_tau_sec,
        guided_radius_sec=guided_radius_sec,
    )
    theta = tau_to_theta_deg(tau_sec)
    return GCCResult(
        tau_sec=float(tau_sec),
        psr_db=float(psr_db),
        theta_deg=float(theta),
        theta_error_ref_deg=float(abs(theta - float(theta_ref_deg))),
        theta_error_geo_deg=float(abs(theta - float(theta_true_deg))),
    )


# ─────────────────────────────────────────────────────────────────────
# Model fit
# ─────────────────────────────────────────────────────────────────────


def fit_univariate_ridge_with_intercept(x: np.ndarray, y: np.ndarray, *, lam: float) -> tuple[float, float]:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1D with same size")
    # Design matrix: [x, 1]
    sxx = float(np.dot(x, x))
    sx1 = float(np.sum(x))
    s11 = float(x.size)
    sxy = float(np.dot(x, y))
    s1y = float(np.sum(y))
    # Ridge on slope only.
    a11 = sxx + float(lam)
    a12 = sx1
    a21 = sx1
    a22 = s11
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-18:
        # Degenerate: fall back to constant predictor = mean(y).
        return 0.0, float(np.mean(y)) if y.size else 0.0
    a = (a22 * sxy - a12 * s1y) / det
    c = (-a21 * sxy + a11 * s1y) / det
    return float(a), float(c)


def summarize_errors(errors: np.ndarray) -> dict[str, float]:
    if errors.size == 0:
        return {"count": 0}
    return {
        "count": int(errors.size),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
    }


def fail_rate(errors: np.ndarray, *, threshold_deg: float) -> float:
    if errors.size == 0:
        return float("nan")
    return float(np.mean(errors > float(threshold_deg)))


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate student LDV weight mask")
    parser.add_argument("--teacher_dataset", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--truth_ref_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    configure_logging(out_dir)
    write_code_state(out_dir, Path(__file__))

    teacher_dataset = Path(args.teacher_dataset)
    data_root = Path(args.data_root)
    truth_ref_root = Path(args.truth_ref_root)

    payload = np.load(str(teacher_dataset), allow_pickle=False)
    X = payload["X"].astype(np.float64, copy=False)
    Y = payload["Y"].astype(np.float64, copy=False)
    speaker_id = payload["speaker_id"].astype(str)
    center_sec = payload["center_sec"].astype(np.float64, copy=False)
    band_edges_hz = payload["band_edges_hz"].astype(np.float64, copy=False)

    if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape:
        raise ValueError(f"Invalid X/Y shapes: X={X.shape}, Y={Y.shape}")
    n_samples, n_bands = X.shape
    if band_edges_hz.size != n_bands + 1:
        raise ValueError("band_edges_hz size mismatch")

    # Train/test split
    train_mask = center_sec <= float(TRAIN_MAX_CENTER_SEC)
    test_mask = center_sec > float(TRAIN_MAX_CENTER_SEC)
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("Empty train or test split; check center_sec range in teacher_dataset")

    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test, Y_test = X[test_mask], Y[test_mask]
    spk_test = speaker_id[test_mask]
    center_test = center_sec[test_mask]

    logger.info("Dataset: N=%d, bands=%d", n_samples, n_bands)
    logger.info("Train: %d samples, Test: %d samples", int(X_train.shape[0]), int(X_test.shape[0]))

    slopes = np.zeros((n_bands,), dtype=np.float64)
    intercepts = np.zeros((n_bands,), dtype=np.float64)
    for b in range(n_bands):
        a, c = fit_univariate_ridge_with_intercept(X_train[:, b], Y_train[:, b], lam=float(RIDGE_LAMBDA))
        slopes[b] = a
        intercepts[b] = c

    model_path = out_dir / "student_model_linear_bandmask.npz"
    model_meta = {
        "generated": datetime.now().isoformat(),
        "teacher_dataset": str(teacher_dataset),
        "ridge_lambda": float(RIDGE_LAMBDA),
        "n_bands": int(n_bands),
        "band_edges_hz": band_edges_hz.tolist(),
        "train_max_center_sec": float(TRAIN_MAX_CENTER_SEC),
    }
    np.savez_compressed(
        model_path,
        slopes=slopes.astype(np.float64),
        intercepts=intercepts.astype(np.float64),
        band_edges_hz=band_edges_hz.astype(np.float64),
        metadata_json=json.dumps(model_meta),
    )
    logger.info("Saved model: %s", model_path)

    # Evaluation (test windows)
    # Cache MIC WAVs per speaker.
    mic_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    truth_cache: dict[str, dict[str, float]] = {}
    geom_cache: dict[str, dict[str, float]] = {}

    base_err_ref: list[float] = []
    teach_err_ref: list[float] = []
    stud_err_ref: list[float] = []
    base_err_geo: list[float] = []
    teach_err_geo: list[float] = []
    stud_err_geo: list[float] = []

    per_window_path = out_dir / "test_windows.jsonl"
    with per_window_path.open("w", encoding="utf-8") as f:
        for i in range(int(X_test.shape[0])):
            spk = str(spk_test[i])
            csec = float(center_test[i])

            if spk not in mic_cache:
                micl_path, micr_path = list_mic_files(data_root, spk)
                sr_l, micl = load_wav_mono(micl_path)
                sr_r, micr = load_wav_mono(micr_path)
                if not (sr_l == sr_r == FS_EXPECTED):
                    raise ValueError(f"Sample rate mismatch for {spk}: {sr_l}, {sr_r}, expected={FS_EXPECTED}")
                mic_cache[spk] = (micl, micr)

            if spk not in truth_cache:
                truth_cache[spk] = load_truth_reference(truth_ref_root / spk / "summary.json")
                geom_cache[spk] = compute_geometry_truth(spk)

            micl, micr = mic_cache[spk]
            seg_l = extract_centered_window(micl, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)
            seg_r = extract_centered_window(micr, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)

            truth = truth_cache[spk]
            geom = geom_cache[spk]
            guided_tau_sec = float(truth["tau_ref_ms"]) / 1000.0
            guided_radius_sec = float(GCC_GUIDED_RADIUS_MS) / 1000.0
            max_tau_sec = float(GCC_MAX_LAG_MS) / 1000.0

            n_fft = int(seg_l.size + seg_r.size)
            freqs_fft = np.fft.rfftfreq(n_fft, d=1.0 / float(FS_EXPECTED)).astype(np.float64, copy=False)
            band_fft = band_mask(freqs_fft, BAND_HZ).astype(np.float64)

            # Baseline: uniform in band.
            base = gcc_micmic_weighted(
                seg_l,
                seg_r,
                fs=FS_EXPECTED,
                guided_tau_sec=guided_tau_sec,
                guided_radius_sec=guided_radius_sec,
                max_tau_sec=max_tau_sec,
                weight_fft=band_fft,
                theta_ref_deg=float(truth["theta_ref_deg"]),
                theta_true_deg=float(geom["theta_true_deg"]),
            )

            # Teacher: band weights from dataset Y_test.
            w_teacher_b = Y_test[i, :].astype(np.float64, copy=False)
            w_teacher_fft = fft_band_weight_from_bands(freqs_fft, band_edges_hz, w_teacher_b) * band_fft
            teacher = gcc_micmic_weighted(
                seg_l,
                seg_r,
                fs=FS_EXPECTED,
                guided_tau_sec=guided_tau_sec,
                guided_radius_sec=guided_radius_sec,
                max_tau_sec=max_tau_sec,
                weight_fft=w_teacher_fft,
                theta_ref_deg=float(truth["theta_ref_deg"]),
                theta_true_deg=float(geom["theta_true_deg"]),
            )

            # Student: predicted band weights from X_test.
            w_hat_b = np.clip(slopes * X_test[i, :] + intercepts, 0.0, 1.0)
            w_hat_fft = fft_band_weight_from_bands(freqs_fft, band_edges_hz, w_hat_b) * band_fft
            student = gcc_micmic_weighted(
                seg_l,
                seg_r,
                fs=FS_EXPECTED,
                guided_tau_sec=guided_tau_sec,
                guided_radius_sec=guided_radius_sec,
                max_tau_sec=max_tau_sec,
                weight_fft=w_hat_fft,
                theta_ref_deg=float(truth["theta_ref_deg"]),
                theta_true_deg=float(geom["theta_true_deg"]),
            )

            base_err_ref.append(float(base.theta_error_ref_deg))
            teach_err_ref.append(float(teacher.theta_error_ref_deg))
            stud_err_ref.append(float(student.theta_error_ref_deg))
            base_err_geo.append(float(base.theta_error_geo_deg))
            teach_err_geo.append(float(teacher.theta_error_geo_deg))
            stud_err_geo.append(float(student.theta_error_geo_deg))

            f.write(
                json.dumps(
                    {
                        "speaker_id": spk,
                        "center_sec": csec,
                        "truth_reference": truth,
                        "geometry_truth": geom,
                        "baseline": base.__dict__,
                        "teacher": teacher.__dict__,
                        "student": student.__dict__,
                    }
                )
                + "\n"
            )

    base_err_ref_a = np.asarray(base_err_ref, dtype=np.float64)
    teach_err_ref_a = np.asarray(teach_err_ref, dtype=np.float64)
    stud_err_ref_a = np.asarray(stud_err_ref, dtype=np.float64)
    base_err_geo_a = np.asarray(base_err_geo, dtype=np.float64)
    teach_err_geo_a = np.asarray(teach_err_geo, dtype=np.float64)
    stud_err_geo_a = np.asarray(stud_err_geo, dtype=np.float64)

    base_stats = summarize_errors(base_err_ref_a)
    teach_stats = summarize_errors(teach_err_ref_a)
    stud_stats = summarize_errors(stud_err_ref_a)

    base_p95 = float(base_stats.get("p95", np.nan))
    teach_p95 = float(teach_stats.get("p95", np.nan))
    stud_p95 = float(stud_stats.get("p95", np.nan))

    teach_impr = (base_p95 - teach_p95) / max(base_p95, 1e-9)
    stud_impr = (base_p95 - stud_p95) / max(base_p95, 1e-9)
    frac_of_teacher = float(stud_impr / teach_impr) if teach_impr > 0 else float("nan")

    median_guard = (float(stud_stats.get("median", np.nan)) - float(base_stats.get("median", np.nan))) / max(
        float(base_stats.get("median", np.nan)), 1e-9
    )

    acceptance = {
        "teacher_p95_improvement_positive": bool(teach_impr > 0),
        "student_reaches_80pct_teacher_p95_improvement": bool(teach_impr > 0 and stud_impr >= 0.8 * teach_impr),
        "student_median_not_worse_frac_le_0p05": bool(median_guard <= 0.05),
        "overall_pass": bool(teach_impr > 0 and stud_impr >= 0.8 * teach_impr and median_guard <= 0.05),
        "computed": {
            "teacher_p95_improvement_frac": float(teach_impr),
            "student_p95_improvement_frac": float(stud_impr),
            "student_fraction_of_teacher": float(frac_of_teacher),
            "student_median_worsening_frac": float(median_guard),
        },
    }

    summary = {
        "generated": datetime.now().isoformat(),
        "teacher_dataset": str(teacher_dataset),
        "model_path": str(model_path),
        "eval_split": {"train_max_center_sec": float(TRAIN_MAX_CENTER_SEC)},
        "acceptance": acceptance,
        "test_metrics_vs_ref": {
            "baseline": {"theta_error_ref_deg": base_stats, "fail_rate_gt5deg": fail_rate(base_err_ref_a, threshold_deg=5.0)},
            "teacher": {"theta_error_ref_deg": teach_stats, "fail_rate_gt5deg": fail_rate(teach_err_ref_a, threshold_deg=5.0)},
            "student": {"theta_error_ref_deg": stud_stats, "fail_rate_gt5deg": fail_rate(stud_err_ref_a, threshold_deg=5.0)},
        },
        "test_metrics_vs_geo": {
            "baseline": {"theta_error_geo_deg": summarize_errors(base_err_geo_a), "fail_rate_gt5deg": fail_rate(base_err_geo_a, threshold_deg=5.0)},
            "teacher": {"theta_error_geo_deg": summarize_errors(teach_err_geo_a), "fail_rate_gt5deg": fail_rate(teach_err_geo_a, threshold_deg=5.0)},
            "student": {"theta_error_geo_deg": summarize_errors(stud_err_geo_a), "fail_rate_gt5deg": fail_rate(stud_err_geo_a, threshold_deg=5.0)},
        },
    }
    write_json(out_dir / "summary.json", summary)

    manifest = {
        "generated": datetime.now().isoformat(),
        "teacher_dataset": {"path": str(teacher_dataset), "sha256": sha256_file(teacher_dataset)},
        "data_root": str(data_root),
        "truth_ref_root": str(truth_ref_root),
    }
    write_json(out_dir / "manifest.json", manifest)

    report_lines: list[str] = []
    report_lines.append("# Student LDV Weight Mask Report")
    report_lines.append("")
    report_lines.append(f"Generated: {summary['generated']}")
    report_lines.append(f"Run dir: {out_dir}")
    report_lines.append("")
    report_lines.append("## Acceptance (pooled test windows)")
    report_lines.append("")
    report_lines.append(f"- teacher p95 improvement frac: {acceptance['computed']['teacher_p95_improvement_frac']:.3f}")
    report_lines.append(f"- student p95 improvement frac: {acceptance['computed']['student_p95_improvement_frac']:.3f}")
    report_lines.append(f"- student / teacher: {acceptance['computed']['student_fraction_of_teacher']:.3f}")
    report_lines.append(f"- student median worsening frac: {acceptance['computed']['student_median_worsening_frac']:.3f}")
    report_lines.append(f"- OVERALL: {'PASS' if acceptance['overall_pass'] else 'FAIL'}")
    report_lines.append("")
    report_lines.append("## Test Metrics (vs chirp reference)")
    report_lines.append("")
    report_lines.append("| Method | count | median | p90 | p95 | fail_rate(>5°) |")
    report_lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    report_lines.append(
        f"| baseline | {base_stats.get('count', 0)} | {base_stats.get('median', float('nan')):.3f} | {base_stats.get('p90', float('nan')):.3f} | {base_stats.get('p95', float('nan')):.3f} | {summary['test_metrics_vs_ref']['baseline']['fail_rate_gt5deg']:.3f} |"
    )
    report_lines.append(
        f"| teacher | {teach_stats.get('count', 0)} | {teach_stats.get('median', float('nan')):.3f} | {teach_stats.get('p90', float('nan')):.3f} | {teach_stats.get('p95', float('nan')):.3f} | {summary['test_metrics_vs_ref']['teacher']['fail_rate_gt5deg']:.3f} |"
    )
    report_lines.append(
        f"| student | {stud_stats.get('count', 0)} | {stud_stats.get('median', float('nan')):.3f} | {stud_stats.get('p90', float('nan')):.3f} | {stud_stats.get('p95', float('nan')):.3f} | {summary['test_metrics_vs_ref']['student']['fail_rate_gt5deg']:.3f} |"
    )
    report_lines.append("")
    report_lines.append("## Test Metrics (vs geometry truth)")
    report_lines.append("")
    base_geo_stats = summary["test_metrics_vs_geo"]["baseline"]["theta_error_geo_deg"]
    teach_geo_stats = summary["test_metrics_vs_geo"]["teacher"]["theta_error_geo_deg"]
    stud_geo_stats = summary["test_metrics_vs_geo"]["student"]["theta_error_geo_deg"]
    report_lines.append("| Method | count | median | p90 | p95 | fail_rate(>5°) |")
    report_lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    report_lines.append(
        f"| baseline | {base_geo_stats.get('count', 0)} | {base_geo_stats.get('median', float('nan')):.3f} | {base_geo_stats.get('p90', float('nan')):.3f} | {base_geo_stats.get('p95', float('nan')):.3f} | {summary['test_metrics_vs_geo']['baseline']['fail_rate_gt5deg']:.3f} |"
    )
    report_lines.append(
        f"| teacher | {teach_geo_stats.get('count', 0)} | {teach_geo_stats.get('median', float('nan')):.3f} | {teach_geo_stats.get('p90', float('nan')):.3f} | {teach_geo_stats.get('p95', float('nan')):.3f} | {summary['test_metrics_vs_geo']['teacher']['fail_rate_gt5deg']:.3f} |"
    )
    report_lines.append(
        f"| student | {stud_geo_stats.get('count', 0)} | {stud_geo_stats.get('median', float('nan')):.3f} | {stud_geo_stats.get('p90', float('nan')):.3f} | {stud_geo_stats.get('p95', float('nan')):.3f} | {summary['test_metrics_vs_geo']['student']['fail_rate_gt5deg']:.3f} |"
    )
    (out_dir / "eval_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    logger.info("Done. Results: %s", out_dir)
    logger.info("Acceptance overall_pass=%s", acceptance["overall_pass"])


if __name__ == "__main__":
    main()
