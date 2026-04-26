#!/usr/bin/env python
"""
LDV-informed GCC-PHAT for MIC-MIC speech DoA (teacher).

Goal
----
Improve MIC-MIC delay estimation robustness on speech by using LDV as a
source-reference channel to build a frequency-domain weight W(f).

Key design points (plan-locked)
-------------------------------
- Evaluate fixed 5.0 s windows with centers 100..600 s (step 1 s).
- Keep only "speech-active" windows: RMS(MicL) >= 50th percentile.
- Estimate noise PSD from the lowest-energy 1% windows by RMS(MicL).
- Build per-window LDV-informed Wiener-like weight:
    S_hat(f) = max(P_ldv(f) - N_ldv(f), 0)
    W(f)     = S_hat(f) / (S_hat(f) + N_mic_avg(f) + eps)
  Smooth W(f) with Savitzky–Golay (31 bins, poly=3) and clip to [0, 1].
- Use weighted GCC in 500–2000 Hz, guided around chirp truth tau_ref_ms
  with radius 0.3 ms.

Outputs (under --out_dir)
-------------------------
- run.log, run_config.json, code_state.json, manifest.json
- per_speaker/<speaker>/windows.jsonl
- per_speaker/<speaker>/summary.json
- teacher_dataset.npz (X=LDV log-PSD bands, Y=teacher band weights)
- grid_report.md and summary.json

Example
-------
python -u scripts/ldv_informed_gcc_micmic.py \\
  --data_root /path/to/GCC-PHAT-LDV-MIC-Experiment \\
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \\
  --out_dir results/ldv_informed_micmic_20260213_120000
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
from scipy.signal import savgol_filter, welch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Plan-locked constants
# ─────────────────────────────────────────────────────────────────────

FS_EXPECTED = 48_000
WINDOW_SEC = 5.0
CENTER_START_SEC = 100.0
CENTER_END_SEC = 600.0
CENTER_STEP_SEC = 1.0

BAND_HZ = (500.0, 2000.0)
GCC_MAX_LAG_MS = 10.0
GCC_GUIDED_RADIUS_MS = 0.3
PSR_EXCLUDE_SAMPLES = 50

WELCH_NPERSEG = 8192
WELCH_NOVERLAP = 4096

SILENCE_PERCENT = 1.0
SPEECH_RMS_PERCENTILE = 50.0

N_BANDS = 64
RIDGE_EPS = 1e-12


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


def dataset_fingerprint(files: list[Path], *, root: Path) -> str:
    entries: list[str] = []
    for fp in sorted(files, key=lambda p: p.as_posix().lower()):
        entries.append(f"{sha256_file(fp)} {fp.relative_to(root).as_posix()}")
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


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


def list_triplet_files(data_root: Path, speaker: str) -> tuple[Path, Path, Path]:
    sp_dir = data_root / speaker
    ldv_files = sorted(sp_dir.glob("*LDV*.wav"))
    micl_files = sorted(sp_dir.glob("*LEFT*.wav"))
    micr_files = sorted(sp_dir.glob("*RIGHT*.wav"))
    if not ldv_files or not micl_files or not micr_files:
        raise FileNotFoundError(
            f"Missing WAV triplet in {sp_dir} (LDV={len(ldv_files)}, LEFT={len(micl_files)}, RIGHT={len(micr_files)})"
        )
    return ldv_files[0], micl_files[0], micr_files[0]


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


def rms(x: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(x * x) + 1e-20))


# ─────────────────────────────────────────────────────────────────────
# Geometry + truth
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
# Welch PSD + weights
# ─────────────────────────────────────────────────────────────────────


def compute_psd_welch(x: np.ndarray, *, fs: int) -> tuple[np.ndarray, np.ndarray]:
    freqs, pxx = welch(
        x.astype(np.float64, copy=False),
        fs=float(fs),
        window="hann",
        nperseg=int(WELCH_NPERSEG),
        noverlap=int(WELCH_NOVERLAP),
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )
    return freqs.astype(np.float64, copy=False), pxx.astype(np.float64, copy=False)


def smooth_clip_weight(w: np.ndarray) -> np.ndarray:
    w = w.astype(np.float64, copy=False)
    if w.size < 31:
        out = w.copy()
    else:
        out = savgol_filter(w, window_length=31, polyorder=3, mode="interp")
    return np.clip(out, 0.0, 1.0)


def band_mask(freqs_hz: np.ndarray, band_hz: tuple[float, float]) -> np.ndarray:
    lo, hi = float(band_hz[0]), float(band_hz[1])
    return (freqs_hz >= lo) & (freqs_hz <= hi)


def band_edges_linear(band_hz: tuple[float, float], n_bands: int) -> np.ndarray:
    lo, hi = float(band_hz[0]), float(band_hz[1])
    return np.linspace(lo, hi, int(n_bands) + 1, dtype=np.float64)


def band_means(values: np.ndarray, freqs_hz: np.ndarray, edges_hz: np.ndarray) -> np.ndarray:
    out = np.zeros((len(edges_hz) - 1,), dtype=np.float64)
    for i in range(len(out)):
        f0, f1 = float(edges_hz[i]), float(edges_hz[i + 1])
        m = (freqs_hz >= f0) & (freqs_hz < f1 if i < len(out) - 1 else freqs_hz <= f1)
        if not np.any(m):
            out[i] = 0.0
        else:
            out[i] = float(np.mean(values[m]))
    return out


# ─────────────────────────────────────────────────────────────────────
# GCC (frequency-domain weighting)
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GCCResult:
    tau_sec: float
    psr_db: float


def estimate_tau_from_weighted_phat(
    R_w: np.ndarray,
    *,
    fs: int,
    n_fft: int,
    max_tau_sec: float,
    guided_tau_sec: float,
    guided_radius_sec: float,
    psr_exclude_samples: int,
) -> GCCResult:
    if R_w.ndim != 1:
        raise ValueError(f"Expected 1D spectrum, got shape={R_w.shape}")
    cc = np.fft.irfft(R_w, n_fft)
    cc = np.real(cc)

    max_shift = int(round(float(max_tau_sec) * float(fs)))
    if max_shift <= 0:
        raise ValueError("max_shift must be > 0")
    if max_shift >= len(cc) // 2:
        raise ValueError("max_shift too large for correlation length")

    cc_win = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    abs_cc = np.abs(cc_win)

    guided_center = int(round(float(guided_tau_sec) * float(fs))) + max_shift
    guided_radius = int(round(float(guided_radius_sec) * float(fs)))
    lo = max(0, guided_center - guided_radius)
    hi = min(len(abs_cc) - 1, guided_center + guided_radius)
    if lo > hi:
        raise ValueError("Invalid guided search window")

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
    exc = int(psr_exclude_samples)
    lo_e = max(0, peak_idx - exc)
    hi_e = min(len(abs_cc), peak_idx + exc + 1)
    mask[lo_e:hi_e] = False
    sidelobe_max = float(abs_cc[mask].max()) if np.any(mask) else 0.0
    peak_val = float(abs_cc[peak_idx])
    psr_db = 20.0 * float(np.log10(peak_val / (sidelobe_max + 1e-10)))
    return GCCResult(tau_sec=float(tau_sec), psr_db=float(psr_db))


def gcc_phat_weighted(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fs: int,
    max_tau_sec: float,
    guided_tau_sec: float,
    guided_radius_sec: float,
    weight_fft: np.ndarray,
) -> GCCResult:
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x={x.shape}, y={y.shape}")
    n_fft = int(x.size + y.size)
    X = np.fft.rfft(x.astype(np.float64, copy=False), n_fft)
    Y = np.fft.rfft(y.astype(np.float64, copy=False), n_fft)
    R = X * np.conj(Y)
    R_phat = R / (np.abs(R) + RIDGE_EPS)
    if weight_fft.shape != R_phat.shape:
        raise ValueError(f"weight_fft shape {weight_fft.shape} != spectrum shape {R_phat.shape}")
    return estimate_tau_from_weighted_phat(
        (R_phat * weight_fft).astype(np.complex128, copy=False),
        fs=fs,
        n_fft=n_fft,
        max_tau_sec=max_tau_sec,
        guided_tau_sec=guided_tau_sec,
        guided_radius_sec=guided_radius_sec,
        psr_exclude_samples=int(PSR_EXCLUDE_SAMPLES),
    )


def fft_weight_from_welch(
    *,
    freqs_welch_hz: np.ndarray,
    w_welch: np.ndarray,
    freqs_fft_hz: np.ndarray,
) -> np.ndarray:
    w = np.interp(
        freqs_fft_hz.astype(np.float64, copy=False),
        freqs_welch_hz.astype(np.float64, copy=False),
        w_welch.astype(np.float64, copy=False),
        left=0.0,
        right=0.0,
    )
    return w.astype(np.float64, copy=False)


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────


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
    parser = argparse.ArgumentParser(description="LDV-informed GCC-PHAT teacher for MIC-MIC on speech")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--truth_ref_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--speakers",
        type=str,
        nargs="+",
        default=["18-0.1V", "19-0.1V", "20-0.1V", "21-0.1V", "22-0.1V"],
        help="Speaker subdirectories to process (default: 18..22 at 0.1V)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    configure_logging(out_dir)
    write_code_state(out_dir, Path(__file__))

    data_root = Path(args.data_root)
    truth_ref_root = Path(args.truth_ref_root)
    speakers = list(args.speakers)

    run_config = {
        "generated": datetime.now().isoformat(),
        "data_root": str(data_root),
        "truth_ref_root": str(truth_ref_root),
        "speakers": speakers,
        "fs_expected": int(FS_EXPECTED),
        "window_sec": float(WINDOW_SEC),
        "center_start_sec": float(CENTER_START_SEC),
        "center_end_sec": float(CENTER_END_SEC),
        "center_step_sec": float(CENTER_STEP_SEC),
        "band_hz": [float(BAND_HZ[0]), float(BAND_HZ[1])],
        "welch": {"nperseg": int(WELCH_NPERSEG), "noverlap": int(WELCH_NOVERLAP)},
        "gcc": {
            "max_lag_ms": float(GCC_MAX_LAG_MS),
            "guided_radius_ms": float(GCC_GUIDED_RADIUS_MS),
            "psr_exclude_samples": int(PSR_EXCLUDE_SAMPLES),
        },
        "silence_percent": float(SILENCE_PERCENT),
        "speech_rms_percentile": float(SPEECH_RMS_PERCENTILE),
        "n_bands": int(N_BANDS),
    }
    write_json(out_dir / "run_config.json", run_config)

    per_speaker_dir = out_dir / "per_speaker"
    per_speaker_dir.mkdir(parents=True, exist_ok=True)

    all_files: list[Path] = []
    all_X: list[np.ndarray] = []
    all_Y: list[np.ndarray] = []
    all_speaker_ids: list[str] = []
    all_centers_sec: list[float] = []

    pooled = {
        "baseline": {"theta_err_ref": [], "theta_err_geo": []},
        "teacher": {"theta_err_ref": [], "theta_err_geo": []},
    }

    edges_hz = band_edges_linear(BAND_HZ, N_BANDS)

    for speaker in speakers:
        ldv_path, micl_path, micr_path = list_triplet_files(data_root, speaker)
        all_files.extend([ldv_path, micl_path, micr_path])

        truth_ref = load_truth_reference(truth_ref_root / speaker / "summary.json")
        geom = compute_geometry_truth(speaker)

        sr_ldv, ldv = load_wav_mono(ldv_path)
        sr_l, micl = load_wav_mono(micl_path)
        sr_r, micr = load_wav_mono(micr_path)
        if not (sr_ldv == sr_l == sr_r == FS_EXPECTED):
            raise ValueError(
                f"Sample rate mismatch for {speaker}: ldv={sr_ldv}, micl={sr_l}, micr={sr_r}, expected={FS_EXPECTED}"
            )

        duration_sec = min(len(ldv), len(micl), len(micr)) / float(FS_EXPECTED)
        logger.info("Speaker %s duration: %.2f s", speaker, duration_sec)

        centers = np.arange(CENTER_START_SEC, CENTER_END_SEC + 1e-9, CENTER_STEP_SEC, dtype=np.float64)
        candidates: list[dict[str, Any]] = []
        for csec in centers.tolist():
            try:
                seg_micl = extract_centered_window(micl, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)
                seg_ldv = extract_centered_window(ldv, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)
                seg_micr = extract_centered_window(micr, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)
                _ = (seg_ldv, seg_micr)  # silence lint
            except ValueError:
                continue
            candidates.append({"center_sec": float(csec), "rms_micl": rms(seg_micl)})

        if not candidates:
            raise RuntimeError(f"No valid windows found for speaker={speaker} in {CENTER_START_SEC}..{CENTER_END_SEC}s")

        rms_all = np.array([c["rms_micl"] for c in candidates], dtype=np.float64)
        speech_thresh = float(np.percentile(rms_all, SPEECH_RMS_PERCENTILE))
        speech_centers = [c["center_sec"] for c in candidates if float(c["rms_micl"]) >= speech_thresh]
        if not speech_centers:
            raise RuntimeError(f"Speaker {speaker}: selected 0 speech windows with RMS>=p{SPEECH_RMS_PERCENTILE}")

        n_silence = max(1, int(np.ceil(len(candidates) * (SILENCE_PERCENT / 100.0))))
        idx_sorted = np.argsort(rms_all)
        silence_centers = [float(candidates[i]["center_sec"]) for i in idx_sorted[:n_silence]]

        logger.info(
            "%s: %d candidate windows, %d speech-active (RMS>=%.3e), %d silence windows (bottom %.1f%%)",
            speaker,
            len(candidates),
            len(speech_centers),
            speech_thresh,
            len(silence_centers),
            SILENCE_PERCENT,
        )

        # Noise PSD estimates from silence windows.
        noise_ldv_list: list[np.ndarray] = []
        noise_micavg_list: list[np.ndarray] = []
        freqs_ref: np.ndarray | None = None
        for csec in silence_centers:
            seg_ldv = extract_centered_window(ldv, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)
            seg_micl = extract_centered_window(micl, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)
            seg_micr = extract_centered_window(micr, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)
            f_ldv, p_ldv = compute_psd_welch(seg_ldv, fs=FS_EXPECTED)
            f_l, p_l = compute_psd_welch(seg_micl, fs=FS_EXPECTED)
            f_r, p_r = compute_psd_welch(seg_micr, fs=FS_EXPECTED)
            if not (np.allclose(f_ldv, f_l) and np.allclose(f_ldv, f_r)):
                raise RuntimeError("Welch frequency grids do not match (unexpected)")
            freqs_ref = f_ldv
            noise_ldv_list.append(p_ldv)
            noise_micavg_list.append(0.5 * (p_l + p_r))

        assert freqs_ref is not None
        N_ldv = np.mean(np.stack(noise_ldv_list, axis=0), axis=0)
        N_mic_avg = np.mean(np.stack(noise_micavg_list, axis=0), axis=0)

        # FFT frequency grid (fixed by window length).
        n_fft = int(round(WINDOW_SEC * FS_EXPECTED) * 2)
        freqs_fft = np.fft.rfftfreq(n_fft, d=1.0 / float(FS_EXPECTED)).astype(np.float64, copy=False)
        band_fft = band_mask(freqs_fft, BAND_HZ).astype(np.float64)

        guided_tau_sec = float(truth_ref["tau_ref_ms"]) / 1000.0
        guided_radius_sec = float(GCC_GUIDED_RADIUS_MS) / 1000.0
        max_tau_sec = float(GCC_MAX_LAG_MS) / 1000.0

        speaker_out = per_speaker_dir / speaker
        speaker_out.mkdir(parents=True, exist_ok=True)
        windows_path = speaker_out / "windows.jsonl"
        windows_f = windows_path.open("w", encoding="utf-8")

        speaker_err_ref_base: list[float] = []
        speaker_err_ref_teacher: list[float] = []
        speaker_err_geo_base: list[float] = []
        speaker_err_geo_teacher: list[float] = []

        speaker_X: list[np.ndarray] = []
        speaker_Y: list[np.ndarray] = []
        speaker_centers: list[float] = []

        for csec in speech_centers:
            seg_ldv = extract_centered_window(ldv, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)
            seg_micl = extract_centered_window(micl, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)
            seg_micr = extract_centered_window(micr, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)

            # Teacher weight from LDV PSD (Welch grid).
            freqs_w, P_ldv = compute_psd_welch(seg_ldv, fs=FS_EXPECTED)
            if not np.allclose(freqs_w, freqs_ref):
                raise RuntimeError("Welch frequency grids changed unexpectedly")
            S_hat = np.maximum(P_ldv - N_ldv, 0.0)
            W = S_hat / (S_hat + N_mic_avg + RIDGE_EPS)
            W = smooth_clip_weight(W)
            W = np.where(band_mask(freqs_w, BAND_HZ), W, 0.0)

            # Map to FFT grid for weighted GCC.
            W_fft = fft_weight_from_welch(freqs_welch_hz=freqs_w, w_welch=W, freqs_fft_hz=freqs_fft)
            W_fft = (W_fft * band_fft).astype(np.float64, copy=False)

            # Baseline: uniform in band.
            base = gcc_phat_weighted(
                seg_micl,
                seg_micr,
                fs=FS_EXPECTED,
                max_tau_sec=max_tau_sec,
                guided_tau_sec=guided_tau_sec,
                guided_radius_sec=guided_radius_sec,
                weight_fft=band_fft,
            )
            base_theta = tau_to_theta_deg(base.tau_sec)

            teacher = gcc_phat_weighted(
                seg_micl,
                seg_micr,
                fs=FS_EXPECTED,
                max_tau_sec=max_tau_sec,
                guided_tau_sec=guided_tau_sec,
                guided_radius_sec=guided_radius_sec,
                weight_fft=W_fft,
            )
            teacher_theta = tau_to_theta_deg(teacher.tau_sec)

            err_ref_base = abs(base_theta - float(truth_ref["theta_ref_deg"]))
            err_ref_teacher = abs(teacher_theta - float(truth_ref["theta_ref_deg"]))
            err_geo_base = abs(base_theta - float(geom["theta_true_deg"]))
            err_geo_teacher = abs(teacher_theta - float(geom["theta_true_deg"]))

            speaker_err_ref_base.append(float(err_ref_base))
            speaker_err_ref_teacher.append(float(err_ref_teacher))
            speaker_err_geo_base.append(float(err_geo_base))
            speaker_err_geo_teacher.append(float(err_geo_teacher))

            pooled["baseline"]["theta_err_ref"].append(float(err_ref_base))
            pooled["teacher"]["theta_err_ref"].append(float(err_ref_teacher))
            pooled["baseline"]["theta_err_geo"].append(float(err_geo_base))
            pooled["teacher"]["theta_err_geo"].append(float(err_geo_teacher))

            # Teacher dataset features/targets (band-aggregated).
            x_bands = np.log(band_means(P_ldv, freqs_w, edges_hz) + RIDGE_EPS)
            y_bands = band_means(W, freqs_w, edges_hz)
            speaker_X.append(x_bands.astype(np.float32, copy=False))
            speaker_Y.append(y_bands.astype(np.float32, copy=False))
            speaker_centers.append(float(csec))

            rec = {
                "speaker_id": speaker,
                "center_sec": float(csec),
                "rms_micl": float(rms(seg_micl)),
                "truth_reference": truth_ref,
                "geometry_truth": geom,
                "baseline": {
                    "tau_ms": float(base.tau_sec * 1000.0),
                    "psr_db": float(base.psr_db),
                    "theta_deg": float(base_theta),
                    "theta_error_ref_deg": float(err_ref_base),
                    "theta_error_geo_deg": float(err_geo_base),
                },
                "teacher": {
                    "tau_ms": float(teacher.tau_sec * 1000.0),
                    "psr_db": float(teacher.psr_db),
                    "theta_deg": float(teacher_theta),
                    "theta_error_ref_deg": float(err_ref_teacher),
                    "theta_error_geo_deg": float(err_geo_teacher),
                    "w_mean_inband": float(np.mean(W[band_mask(freqs_w, BAND_HZ)])),
                },
            }
            windows_f.write(json.dumps(rec) + "\n")

        windows_f.close()

        speaker_err_ref_base_arr = np.asarray(speaker_err_ref_base, dtype=np.float64)
        speaker_err_ref_teacher_arr = np.asarray(speaker_err_ref_teacher, dtype=np.float64)
        speaker_err_geo_base_arr = np.asarray(speaker_err_geo_base, dtype=np.float64)
        speaker_err_geo_teacher_arr = np.asarray(speaker_err_geo_teacher, dtype=np.float64)

        speaker_summary = {
            "generated": datetime.now().isoformat(),
            "speaker_id": speaker,
            "truth_reference": truth_ref,
            "geometry_truth": geom,
            "counts": {
                "n_candidate_windows": int(len(candidates)),
                "n_speech_windows": int(len(speech_centers)),
                "n_silence_windows": int(len(silence_centers)),
            },
            "rms_thresholds": {"speech_rms_micl_p50": float(speech_thresh)},
            "method": {
                "baseline": {
                    "theta_error_ref_deg": summarize_errors(speaker_err_ref_base_arr),
                    "theta_error_geo_deg": summarize_errors(speaker_err_geo_base_arr),
                    "fail_rate_ref_gt5deg": fail_rate(speaker_err_ref_base_arr, threshold_deg=5.0),
                },
                "teacher": {
                    "theta_error_ref_deg": summarize_errors(speaker_err_ref_teacher_arr),
                    "theta_error_geo_deg": summarize_errors(speaker_err_geo_teacher_arr),
                    "fail_rate_ref_gt5deg": fail_rate(speaker_err_ref_teacher_arr, threshold_deg=5.0),
                },
            },
        }
        write_json(speaker_out / "summary.json", speaker_summary)

        speaker_X_arr = np.stack(speaker_X, axis=0) if speaker_X else np.zeros((0, N_BANDS), dtype=np.float32)
        speaker_Y_arr = np.stack(speaker_Y, axis=0) if speaker_Y else np.zeros((0, N_BANDS), dtype=np.float32)
        all_X.append(speaker_X_arr)
        all_Y.append(speaker_Y_arr)
        all_speaker_ids.extend([speaker] * int(speaker_X_arr.shape[0]))
        all_centers_sec.extend(speaker_centers)

    # Manifest
    fingerprint = dataset_fingerprint(all_files, root=data_root)
    manifest = {
        "generated": datetime.now().isoformat(),
        "data_root": str(data_root),
        "speakers": speakers,
        "files": [
            {"rel_path": p.relative_to(data_root).as_posix(), "sha256": sha256_file(p)}
            for p in sorted(set(all_files), key=lambda x: x.as_posix().lower())
        ],
        "dataset_fingerprint_sha256": fingerprint,
    }
    write_json(out_dir / "manifest.json", manifest)

    # Teacher dataset
    X = np.concatenate(all_X, axis=0) if all_X else np.zeros((0, N_BANDS), dtype=np.float32)
    Y = np.concatenate(all_Y, axis=0) if all_Y else np.zeros((0, N_BANDS), dtype=np.float32)
    speaker_ids_arr = np.asarray(all_speaker_ids, dtype=str)
    centers_arr = np.asarray(all_centers_sec, dtype=np.float64)
    np.savez_compressed(
        out_dir / "teacher_dataset.npz",
        X=X,
        Y=Y,
        speaker_id=speaker_ids_arr,
        center_sec=centers_arr,
        band_edges_hz=edges_hz,
        metadata_json=json.dumps(run_config),
    )

    # Aggregate report + success criteria
    base_ref = np.asarray(pooled["baseline"]["theta_err_ref"], dtype=np.float64)
    teach_ref = np.asarray(pooled["teacher"]["theta_err_ref"], dtype=np.float64)
    base_geo = np.asarray(pooled["baseline"]["theta_err_geo"], dtype=np.float64)
    teach_geo = np.asarray(pooled["teacher"]["theta_err_geo"], dtype=np.float64)

    base_stats = summarize_errors(base_ref)
    teach_stats = summarize_errors(teach_ref)
    base_fail = fail_rate(base_ref, threshold_deg=5.0)
    teach_fail = fail_rate(teach_ref, threshold_deg=5.0)

    p95_impr = (float(base_stats.get("p95", np.nan)) - float(teach_stats.get("p95", np.nan))) / max(
        float(base_stats.get("p95", np.nan)), 1e-9
    )
    fail_impr = (float(base_fail) - float(teach_fail)) / max(float(base_fail), 1e-9)
    median_guard = (float(teach_stats.get("median", np.nan)) - float(base_stats.get("median", np.nan))) / max(
        float(base_stats.get("median", np.nan)), 1e-9
    )

    success = {
        "p95_improvement_frac_ge_0p15": bool(p95_impr >= 0.15),
        "fail_rate_improvement_frac_ge_0p20": bool(fail_impr >= 0.20),
        "median_not_worse_frac_le_0p05": bool(median_guard <= 0.05),
        "overall_pass": bool(p95_impr >= 0.15 and fail_impr >= 0.20 and median_guard <= 0.05),
        "computed": {
            "p95_improvement_frac": float(p95_impr),
            "fail_rate_improvement_frac": float(fail_impr),
            "median_worsening_frac": float(median_guard),
        },
    }

    summary = {
        "generated": datetime.now().isoformat(),
        "success_criteria": success,
        "pooled": {
            "baseline": {
                "theta_error_ref_deg": base_stats,
                "theta_error_geo_deg": summarize_errors(base_geo),
                "fail_rate_ref_gt5deg": float(base_fail),
            },
            "teacher": {
                "theta_error_ref_deg": teach_stats,
                "theta_error_geo_deg": summarize_errors(teach_geo),
                "fail_rate_ref_gt5deg": float(teach_fail),
            },
        },
    }
    write_json(out_dir / "summary.json", summary)

    lines: list[str] = []
    lines.append("# LDV-Informed MIC-MIC GCC Report")
    lines.append("")
    lines.append(f"Generated: {summary['generated']}")
    lines.append(f"Run dir: {out_dir}")
    lines.append("")
    lines.append("## Success Criteria (speech, pooled)")
    lines.append("")
    lines.append(f"- p95(theta_error_ref) improvement: {success['computed']['p95_improvement_frac']:.3f} (>= 0.150)")
    lines.append(f"- fail_rate_ref(theta_error_ref>5°) improvement: {success['computed']['fail_rate_improvement_frac']:.3f} (>= 0.200)")
    lines.append(f"- median(theta_error_ref) worsening: {success['computed']['median_worsening_frac']:.3f} (<= 0.050)")
    lines.append(f"- OVERALL: {'PASS' if success['overall_pass'] else 'FAIL'}")
    lines.append("")
    lines.append("## Pooled Metrics (vs chirp reference)")
    lines.append("")
    lines.append("| Method | count | median | p90 | p95 | fail_rate(>5°) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| baseline | {base_stats.get('count', 0)} | {base_stats.get('median', float('nan')):.3f} | {base_stats.get('p90', float('nan')):.3f} | {base_stats.get('p95', float('nan')):.3f} | {base_fail:.3f} |"
    )
    lines.append(
        f"| teacher | {teach_stats.get('count', 0)} | {teach_stats.get('median', float('nan')):.3f} | {teach_stats.get('p90', float('nan')):.3f} | {teach_stats.get('p95', float('nan')):.3f} | {teach_fail:.3f} |"
    )
    lines.append("")
    lines.append("## Pooled Metrics (vs geometry truth)")
    lines.append("")
    base_geo_stats = summarize_errors(base_geo)
    teach_geo_stats = summarize_errors(teach_geo)
    base_geo_fail = fail_rate(base_geo, threshold_deg=5.0)
    teach_geo_fail = fail_rate(teach_geo, threshold_deg=5.0)
    lines.append("| Method | count | median | p90 | p95 | fail_rate(>5°) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| baseline | {base_geo_stats.get('count', 0)} | {base_geo_stats.get('median', float('nan')):.3f} | {base_geo_stats.get('p90', float('nan')):.3f} | {base_geo_stats.get('p95', float('nan')):.3f} | {base_geo_fail:.3f} |"
    )
    lines.append(
        f"| teacher | {teach_geo_stats.get('count', 0)} | {teach_geo_stats.get('median', float('nan')):.3f} | {teach_geo_stats.get('p90', float('nan')):.3f} | {teach_geo_stats.get('p95', float('nan')):.3f} | {teach_geo_fail:.3f} |"
    )
    lines.append("")
    (out_dir / "grid_report.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info("Done. Results: %s", out_dir)
    logger.info("Success overall_pass=%s", success["overall_pass"])


if __name__ == "__main__":
    main()
