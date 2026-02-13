#!/usr/bin/env python
"""
Band-OMP teacher for MIC-MIC speech DoA (LDV+MIC features, band actions).

Goal
----
Produce OMP-like decision trajectories where each action selects a frequency band
to include in MIC-MIC GCC-PHAT, guided by chirp-reference truth (tau_ref_ms).

This script:
1) Defines a coupling-forbidden band mask per speaker using *silence windows*.
2) For each speech window, computes a truth-free observation vector obs[b] over bands.
3) Uses a greedy (OMP-like) teacher to select up to K bands maximizing a fixed score:
       score(S) = -( |tau_S_ms - tau_ref_ms| / 0.30 ) + ( psr_S_db / 3.0 )
4) Evaluates baseline (full-band MIC-MIC) vs teacher (selected bands) on speech tails.
5) Saves teacher trajectories (observations/actions/valid_len) for DTmin training.

CLI (plan-locked)
-----------------
python -u scripts/teacher_band_omp_micmic.py \\
  --data_root /home/sbplab/jiawei/data \\
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \\
  --out_dir results/band_omp_teacher_<YYYYMMDD_HHMMSS> \\
  --speakers 18-0.1V 19-0.1V 20-0.1V 21-0.1V 22-0.1V
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
from scipy.signal import csd, welch

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
N_BANDS = 64
K_HORIZON = 6

GCC_MAX_LAG_MS = 10.0
GCC_GUIDED_RADIUS_MS = 0.3
PSR_EXCLUDE_SAMPLES = 50

WELCH_NPERSEG = 8192
WELCH_NOVERLAP = 4096

SILENCE_PERCENT = 1.0
SPEECH_RMS_PERCENTILE = 50.0
COUPLING_FORBID_GAMMA2 = 0.20

RIDGE_EPS = 1e-12

TAU_NORM_MS = 0.30
PSR_NORM_DB = 3.0
STOP_GAIN_MIN = 0.01


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


def dataset_fingerprint(files: list[Path], *, root: Path) -> str:
    entries: list[str] = []
    for fp in sorted(files, key=lambda p: p.as_posix().lower()):
        entries.append(f"{sha256_file(fp)} {fp.relative_to(root).as_posix()}")
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


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
    return float(np.sqrt(np.mean(x * x)))


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
        "label": str(ref.get("label", "")),
    }


# ─────────────────────────────────────────────────────────────────────
# Bands + features
# ─────────────────────────────────────────────────────────────────────


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


def zscore_inband(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if not np.isfinite(sigma) or sigma < 1e-12:
        raise ValueError("zscore std too small / non-finite")
    return (x - mu) / sigma


def welch_psd(x: np.ndarray, *, fs: int) -> tuple[np.ndarray, np.ndarray]:
    f, p = welch(
        x.astype(np.float64, copy=False),
        fs=float(fs),
        window="hann",
        nperseg=int(WELCH_NPERSEG),
        noverlap=int(WELCH_NOVERLAP),
    )
    return f.astype(np.float64, copy=False), p.astype(np.float64, copy=False)


def welch_csd(x: np.ndarray, y: np.ndarray, *, fs: int) -> tuple[np.ndarray, np.ndarray]:
    f, pxy = csd(
        x.astype(np.float64, copy=False),
        y.astype(np.float64, copy=False),
        fs=float(fs),
        window="hann",
        nperseg=int(WELCH_NPERSEG),
        noverlap=int(WELCH_NOVERLAP),
    )
    return f.astype(np.float64, copy=False), pxy.astype(np.complex128, copy=False)


def gamma2_from_spectra(Pxx: np.ndarray, Pyy: np.ndarray, Pxy: np.ndarray) -> np.ndarray:
    denom = (Pxx * Pyy) + 1e-30
    return (np.abs(Pxy) ** 2) / denom


def compute_silence_coupling_mask(
    *,
    silence_windows: list[dict[str, Any]],
    edges_hz: np.ndarray,
    fs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      cpl_band: (B,) in [0,1] (band-mean max coherence across 3 pairs during silence)
      forbidden: (B,) bool where cpl_band >= threshold
    """
    if len(silence_windows) < 3:
        raise ValueError(f"Need at least 3 silence windows, got {len(silence_windows)}")

    acc_mic = None
    acc_ldvL = None
    acc_ldvR = None
    for w in silence_windows:
        ldv = w["ldv"]
        micl = w["micl"]
        micr = w["micr"]

        f, P_ldv = welch_psd(ldv, fs=fs)
        _f2, P_l = welch_psd(micl, fs=fs)
        _f3, P_r = welch_psd(micr, fs=fs)
        _f4, P_lr = welch_csd(micl, micr, fs=fs)
        _f5, P_ldvL = welch_csd(ldv, micl, fs=fs)
        _f6, P_ldvR = welch_csd(ldv, micr, fs=fs)

        g_mic = gamma2_from_spectra(P_l, P_r, P_lr)
        g_ldvL = gamma2_from_spectra(P_ldv, P_l, P_ldvL)
        g_ldvR = gamma2_from_spectra(P_ldv, P_r, P_ldvR)

        if acc_mic is None:
            acc_mic = g_mic.copy()
            acc_ldvL = g_ldvL.copy()
            acc_ldvR = g_ldvR.copy()
        else:
            acc_mic += g_mic
            acc_ldvL += g_ldvL
            acc_ldvR += g_ldvR

    assert acc_mic is not None
    g_mic_mean = acc_mic / float(len(silence_windows))
    g_ldvL_mean = acc_ldvL / float(len(silence_windows))
    g_ldvR_mean = acc_ldvR / float(len(silence_windows))

    g_max = np.maximum(g_mic_mean, np.maximum(g_ldvL_mean, g_ldvR_mean))
    cpl_band = band_means(g_max, f, edges_hz)
    forbidden_raw = cpl_band >= float(COUPLING_FORBID_GAMMA2)

    # Degenerate-case handling:
    # In this dataset, silence-window coherence can be high across nearly all bands due to
    # stable coupling/offset paths. A hard-forbid of all bands would make the teacher a no-op.
    # We therefore keep cpl_band as a *penalty feature* but disable hard forbids if they would
    # eliminate the entire action space.
    if int(np.sum(forbidden_raw)) >= int(len(forbidden_raw)):
        forbidden_effective = np.zeros_like(forbidden_raw, dtype=bool)
    else:
        forbidden_effective = forbidden_raw.copy()

    return (
        cpl_band.astype(np.float64, copy=False),
        forbidden_raw.astype(bool, copy=False),
        forbidden_effective.astype(bool, copy=False),
    )


def compute_obs_vector(
    *,
    ldv: np.ndarray,
    micl: np.ndarray,
    micr: np.ndarray,
    edges_hz: np.ndarray,
    cpl_band: np.ndarray,
    fs: int,
) -> np.ndarray:
    f, P_ldv = welch_psd(ldv, fs=fs)
    _f2, P_l = welch_psd(micl, fs=fs)
    _f3, P_r = welch_psd(micr, fs=fs)
    _f4, P_lr = welch_csd(micl, micr, fs=fs)

    mic_coh = gamma2_from_spectra(P_l, P_r, P_lr)
    P_mic_avg = 0.5 * (P_l + P_r)

    ldv_band = band_means(np.log(P_ldv + 1e-20), f, edges_hz)
    mic_band = band_means(np.log(P_mic_avg + 1e-20), f, edges_hz)
    coh_band = band_means(mic_coh, f, edges_hz)

    obs = (
        zscore_inband(ldv_band)
        + zscore_inband(mic_band)
        + zscore_inband(coh_band)
        - 2.0 * zscore_inband(cpl_band)
    )
    return obs.astype(np.float64, copy=False)


# ─────────────────────────────────────────────────────────────────────
# GCC utilities (FFT grid, band masks, tau/psr)
# ─────────────────────────────────────────────────────────────────────


def fft_freqs(fs: int, n_fft: int) -> np.ndarray:
    return np.fft.rfftfreq(int(n_fft), d=1.0 / float(fs)).astype(np.float64, copy=False)


def fft_band_masks(*, freqs_fft_hz: np.ndarray, edges_hz: np.ndarray) -> np.ndarray:
    masks = np.zeros((len(edges_hz) - 1, freqs_fft_hz.size), dtype=np.float64)
    for b in range(len(edges_hz) - 1):
        f0, f1 = float(edges_hz[b]), float(edges_hz[b + 1])
        m = (freqs_fft_hz >= f0) & (freqs_fft_hz < f1 if b < len(edges_hz) - 2 else freqs_fft_hz <= f1)
        masks[b, m] = 1.0
    return masks


@dataclass(frozen=True)
class TauPsr:
    tau_sec: float
    psr_db: float


def ccwin_from_spectrum(R_w: np.ndarray, *, n_fft: int, max_shift: int) -> np.ndarray:
    cc = np.fft.irfft(R_w, int(n_fft))
    cc = np.real(cc)
    return np.concatenate((cc[-max_shift:], cc[: max_shift + 1])).astype(np.float64, copy=False)


def estimate_tau_psr_from_ccwin(
    cc_win: np.ndarray,
    *,
    fs: int,
    max_shift: int,
    guided_tau_sec: float,
    guided_radius_sec: float,
    psr_exclude_samples: int,
) -> TauPsr:
    abs_cc = np.abs(cc_win)
    guided_center = int(round(float(guided_tau_sec) * float(fs))) + int(max_shift)
    guided_radius = int(round(float(guided_radius_sec) * float(fs)))
    lo = max(0, guided_center - guided_radius)
    hi = min(len(abs_cc) - 1, guided_center + guided_radius)
    if lo > hi:
        raise ValueError("Invalid guided window")

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
    return TauPsr(tau_sec=float(tau_sec), psr_db=float(psr_db))


def teacher_score(*, tau_ms: float, tau_ref_ms: float, psr_db: float) -> float:
    return float(-(abs(float(tau_ms) - float(tau_ref_ms)) / float(TAU_NORM_MS)) + (float(psr_db) / float(PSR_NORM_DB)))


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Band-OMP teacher for MIC-MIC guided GCC on speech")
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
        "center_grid": [float(CENTER_START_SEC), float(CENTER_END_SEC), float(CENTER_STEP_SEC)],
        "band_hz": [float(BAND_HZ[0]), float(BAND_HZ[1])],
        "n_bands": int(N_BANDS),
        "teacher": {
            "k_horizon": int(K_HORIZON),
            "tau_norm_ms": float(TAU_NORM_MS),
            "psr_norm_db": float(PSR_NORM_DB),
            "stop_gain_min": float(STOP_GAIN_MIN),
        },
        "welch": {"nperseg": int(WELCH_NPERSEG), "noverlap": int(WELCH_NOVERLAP)},
        "coupling": {
            "silence_percent": float(SILENCE_PERCENT),
            "speech_rms_percentile": float(SPEECH_RMS_PERCENTILE),
            "forbid_gamma2": float(COUPLING_FORBID_GAMMA2),
        },
        "gcc": {
            "max_lag_ms": float(GCC_MAX_LAG_MS),
            "guided_radius_ms": float(GCC_GUIDED_RADIUS_MS),
            "psr_exclude_samples": int(PSR_EXCLUDE_SAMPLES),
        },
    }
    write_json(out_dir / "run_config.json", run_config)

    edges_hz = band_edges_linear(BAND_HZ, N_BANDS)
    per_speaker_dir = out_dir / "per_speaker"
    per_speaker_dir.mkdir(parents=True, exist_ok=True)

    all_files: list[Path] = []
    pooled = {
        "baseline": {"theta_err_ref": [], "theta_err_geo": []},
        "teacher": {"theta_err_ref": [], "theta_err_geo": []},
    }

    traj_obs: list[np.ndarray] = []
    traj_act: list[np.ndarray] = []
    traj_len: list[int] = []
    traj_spk: list[str] = []
    traj_center: list[float] = []
    traj_forbidden: list[np.ndarray] = []

    centers_grid = np.arange(CENTER_START_SEC, CENTER_END_SEC + 1e-9, CENTER_STEP_SEC, dtype=np.float64)
    win_samples = int(round(WINDOW_SEC * FS_EXPECTED))
    n_fft = int(win_samples * 2)
    f_fft = fft_freqs(FS_EXPECTED, n_fft)
    analysis_mask_fft = ((f_fft >= BAND_HZ[0]) & (f_fft <= BAND_HZ[1])).astype(np.float64)
    band_masks_fft = fft_band_masks(freqs_fft_hz=f_fft, edges_hz=edges_hz)  # (B, n_bins)

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

        candidates: list[dict[str, Any]] = []
        for csec in centers_grid.tolist():
            try:
                seg_micl = extract_centered_window(micl, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)
                seg_micr = extract_centered_window(micr, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)
                seg_ldv = extract_centered_window(ldv, fs=FS_EXPECTED, center_sec=csec, window_sec=WINDOW_SEC)
            except ValueError:
                continue
            candidates.append(
                {
                    "center_sec": float(csec),
                    "micl": seg_micl.astype(np.float64, copy=False),
                    "micr": seg_micr.astype(np.float64, copy=False),
                    "ldv": seg_ldv.astype(np.float64, copy=False),
                    "rms_micl": rms(seg_micl),
                }
            )

        if not candidates:
            raise RuntimeError(f"No candidate windows for speaker {speaker}")

        rms_vals = np.asarray([c["rms_micl"] for c in candidates], dtype=np.float64)
        speech_thresh = float(np.percentile(rms_vals, SPEECH_RMS_PERCENTILE))
        speech_windows = [c for c in candidates if float(c["rms_micl"]) >= speech_thresh]
        if not speech_windows:
            raise RuntimeError(f"No speech windows after RMS filter for speaker {speaker}")

        n_silence = max(3, int(round(len(candidates) * (SILENCE_PERCENT / 100.0))))
        silence_windows = sorted(candidates, key=lambda d: float(d["rms_micl"]))[:n_silence]

        cpl_band, forbidden_raw, forbidden = compute_silence_coupling_mask(
            silence_windows=silence_windows, edges_hz=edges_hz, fs=FS_EXPECTED
        )
        forbidden_raw_idx = np.where(forbidden_raw)[0].tolist()
        forbidden_idx = np.where(forbidden)[0].tolist()
        forbid_degenerate = int(np.sum(forbidden_raw)) >= int(N_BANDS)
        logger.info(
            "%s: %d candidate windows, %d speech-active (RMS>=%.3e), %d silence windows; forbidden bands raw=%d effective=%d",
            speaker,
            len(candidates),
            len(speech_windows),
            speech_thresh,
            len(silence_windows),
            int(np.sum(forbidden_raw)),
            int(np.sum(forbidden)),
        )

        spk_out = per_speaker_dir / speaker
        spk_out.mkdir(parents=True, exist_ok=True)
        write_json(
            spk_out / "coupling_mask.json",
            {
                "generated": datetime.now().isoformat(),
                "speaker_id": speaker,
                "band_hz": [float(BAND_HZ[0]), float(BAND_HZ[1])],
                "n_bands": int(N_BANDS),
                "band_edges_hz": edges_hz.tolist(),
                "silence_percent": float(SILENCE_PERCENT),
                "forbid_gamma2": float(COUPLING_FORBID_GAMMA2),
                "cpl_band": cpl_band.tolist(),
                "forbidden_bands_raw": forbidden_raw_idx,
                "forbidden_bands_effective": forbidden_idx,
                "forbid_rule_degenerated": bool(forbid_degenerate),
            },
        )

        guided_tau_sec = float(truth_ref["tau_ref_ms"]) / 1000.0
        guided_radius_sec = float(GCC_GUIDED_RADIUS_MS) / 1000.0
        max_shift = int(round(float(GCC_MAX_LAG_MS) * float(FS_EXPECTED) / 1000.0))

        speaker_err_ref_base: list[float] = []
        speaker_err_ref_teacher: list[float] = []
        speaker_err_geo_base: list[float] = []
        speaker_err_geo_teacher: list[float] = []
        speaker_fail_ref_base: list[bool] = []
        speaker_fail_ref_teacher: list[bool] = []

        windows_path = spk_out / "windows.jsonl"
        with windows_path.open("w", encoding="utf-8") as f_jsonl:
            for w in speech_windows:
                center_sec = float(w["center_sec"])
                seg_micl = w["micl"]
                seg_micr = w["micr"]
                seg_ldv = w["ldv"]

                obs_vec = compute_obs_vector(
                    ldv=seg_ldv, micl=seg_micl, micr=seg_micr, edges_hz=edges_hz, cpl_band=cpl_band, fs=FS_EXPECTED
                )

                X = np.fft.rfft(seg_micl, n_fft)
                Y = np.fft.rfft(seg_micr, n_fft)
                R = X * np.conj(Y)
                R_phat = R / (np.abs(R) + RIDGE_EPS)

                # Baseline: full analysis band
                cc_base = ccwin_from_spectrum((R_phat * analysis_mask_fft).astype(np.complex128, copy=False), n_fft=n_fft, max_shift=max_shift)
                base_tp = estimate_tau_psr_from_ccwin(
                    cc_base,
                    fs=FS_EXPECTED,
                    max_shift=max_shift,
                    guided_tau_sec=guided_tau_sec,
                    guided_radius_sec=guided_radius_sec,
                    psr_exclude_samples=int(PSR_EXCLUDE_SAMPLES),
                )
                base_tau_ms = float(base_tp.tau_sec * 1000.0)
                base_theta = tau_to_theta_deg(base_tp.tau_sec)
                err_ref_base = abs(base_theta - float(truth_ref["theta_ref_deg"]))
                err_geo_base = abs(base_theta - float(geom["theta_true_deg"]))

                # Teacher: precompute per-band cc windows for allowed bands only.
                cc_band: list[np.ndarray] = []
                for b in range(N_BANDS):
                    if bool(forbidden[b]):
                        cc_band.append(np.full((2 * max_shift + 1,), np.nan, dtype=np.float64))
                        continue
                    R_b = R_phat * band_masks_fft[b]
                    cc_b = ccwin_from_spectrum(R_b.astype(np.complex128, copy=False), n_fft=n_fft, max_shift=max_shift)
                    cc_band.append(cc_b)

                # Greedy selection
                selected: list[int] = []
                step_scores: list[float] = []
                current_score = -1e30
                current_cc = None

                allowed_bands = [int(b) for b in range(N_BANDS) if not bool(forbidden[b])]
                for step in range(K_HORIZON):
                    best_b = None
                    best_score = None
                    best_cc = None
                    denom = float(len(selected) + 1)
                    for b in allowed_bands:
                        if b in selected:
                            continue
                        cc_b = cc_band[b]
                        if not np.all(np.isfinite(cc_b)):
                            continue
                        if current_cc is None:
                            cc_cand = cc_b
                        else:
                            cc_cand = (current_cc * float(len(selected)) + cc_b) / denom
                        tp = estimate_tau_psr_from_ccwin(
                            cc_cand,
                            fs=FS_EXPECTED,
                            max_shift=max_shift,
                            guided_tau_sec=guided_tau_sec,
                            guided_radius_sec=guided_radius_sec,
                            psr_exclude_samples=int(PSR_EXCLUDE_SAMPLES),
                        )
                        tau_ms = float(tp.tau_sec * 1000.0)
                        s = teacher_score(tau_ms=tau_ms, tau_ref_ms=float(truth_ref["tau_ref_ms"]), psr_db=float(tp.psr_db))
                        if best_score is None or s > best_score:
                            best_b = int(b)
                            best_score = float(s)
                            best_cc = cc_cand
                    if best_b is None or best_score is None or best_cc is None:
                        break
                    if best_score <= current_score + float(STOP_GAIN_MIN):
                        break
                    selected.append(int(best_b))
                    step_scores.append(float(best_score))
                    current_score = float(best_score)
                    current_cc = best_cc

                valid_len = int(len(selected))
                if current_cc is None:
                    # This should not happen when there is at least one allowed band.
                    current_cc = cc_base.copy()
                    selected = []
                    valid_len = 0

                teacher_tp = estimate_tau_psr_from_ccwin(
                    current_cc,
                    fs=FS_EXPECTED,
                    max_shift=max_shift,
                    guided_tau_sec=guided_tau_sec,
                    guided_radius_sec=guided_radius_sec,
                    psr_exclude_samples=int(PSR_EXCLUDE_SAMPLES),
                )
                teach_tau_ms = float(teacher_tp.tau_sec * 1000.0)
                teach_theta = tau_to_theta_deg(teacher_tp.tau_sec)
                err_ref_teacher = abs(teach_theta - float(truth_ref["theta_ref_deg"]))
                err_geo_teacher = abs(teach_theta - float(geom["theta_true_deg"]))

                pooled["baseline"]["theta_err_ref"].append(float(err_ref_base))
                pooled["baseline"]["theta_err_geo"].append(float(err_geo_base))
                pooled["teacher"]["theta_err_ref"].append(float(err_ref_teacher))
                pooled["teacher"]["theta_err_geo"].append(float(err_geo_teacher))

                speaker_err_ref_base.append(float(err_ref_base))
                speaker_err_ref_teacher.append(float(err_ref_teacher))
                speaker_err_geo_base.append(float(err_geo_base))
                speaker_err_geo_teacher.append(float(err_geo_teacher))
                speaker_fail_ref_base.append(bool(err_ref_base > 5.0))
                speaker_fail_ref_teacher.append(bool(err_ref_teacher > 5.0))

                # Save trajectories (obs repeated for each step)
                obs_KB = np.repeat(obs_vec[None, :], K_HORIZON, axis=0)
                act_K = np.full((K_HORIZON,), -1, dtype=np.int32)
                if valid_len > 0:
                    act_K[:valid_len] = np.asarray(selected[:valid_len], dtype=np.int32)
                traj_obs.append(obs_KB.astype(np.float32, copy=False))
                traj_act.append(act_K)
                traj_len.append(valid_len)
                traj_spk.append(str(speaker))
                traj_center.append(center_sec)
                traj_forbidden.append(forbidden.astype(bool, copy=False))

                record = {
                    "speaker_id": speaker,
                    "center_sec": center_sec,
                    "rms_micl": float(w["rms_micl"]),
                    "truth_reference": truth_ref,
                    "geometry_truth": geom,
                    "forbidden_bands": forbidden_idx,
                    "baseline": {
                        "tau_ms": base_tau_ms,
                        "psr_db": float(base_tp.psr_db),
                        "theta_deg": float(base_theta),
                        "tau_error_ref_ms": float(abs(base_tau_ms - float(truth_ref["tau_ref_ms"]))),
                        "theta_error_ref_deg": float(err_ref_base),
                        "theta_error_geo_deg": float(err_geo_base),
                    },
                    "teacher": {
                        "tau_ms": teach_tau_ms,
                        "psr_db": float(teacher_tp.psr_db),
                        "theta_deg": float(teach_theta),
                        "tau_error_ref_ms": float(abs(teach_tau_ms - float(truth_ref["tau_ref_ms"]))),
                        "theta_error_ref_deg": float(err_ref_teacher),
                        "theta_error_geo_deg": float(err_geo_teacher),
                        "selected_bands": selected,
                        "valid_len": valid_len,
                        "step_scores": step_scores,
                        "score_final": float(current_score),
                    },
                }
                f_jsonl.write(json.dumps(record) + "\n")

        def summarize(errors: list[float]) -> dict[str, float]:
            arr = np.asarray(errors, dtype=np.float64)
            if arr.size == 0:
                return {"count": 0}
            return {
                "count": int(arr.size),
                "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)),
            }

        speaker_summary = {
            "generated": datetime.now().isoformat(),
            "speaker_id": speaker,
            "truth_reference": truth_ref,
            "geometry_truth": geom,
            "counts": {
                "n_candidate_windows": int(len(candidates)),
                "n_speech_windows": int(len(speech_windows)),
                "n_silence_windows": int(len(silence_windows)),
            },
            "rms_thresholds": {"speech_rms_micl_p50": float(speech_thresh)},
            "coupling": {"forbidden_band_count": int(np.sum(forbidden))},
            "method": {
                "baseline": {
                    "theta_error_ref_deg": summarize(speaker_err_ref_base),
                    "theta_error_geo_deg": summarize(speaker_err_geo_base),
                    "fail_rate_ref_gt5deg": float(np.mean(np.asarray(speaker_fail_ref_base, dtype=bool))),
                },
                "teacher": {
                    "theta_error_ref_deg": summarize(speaker_err_ref_teacher),
                    "theta_error_geo_deg": summarize(speaker_err_geo_teacher),
                    "fail_rate_ref_gt5deg": float(np.mean(np.asarray(speaker_fail_ref_teacher, dtype=bool))),
                },
            },
        }
        write_json(spk_out / "summary.json", speaker_summary)

    # Save manifest
    manifest = {
        "generated": datetime.now().isoformat(),
        "data_root": str(data_root),
        "speakers": speakers,
        "files": [{"rel_path": str(p.relative_to(data_root)), "sha256": sha256_file(p)} for p in all_files],
        "dataset_fingerprint_sha256": dataset_fingerprint(all_files, root=data_root),
    }
    write_json(out_dir / "manifest.json", manifest)

    # Save trajectories
    if not traj_obs:
        raise RuntimeError("No trajectories collected")

    np.savez_compressed(
        out_dir / "teacher_trajectories.npz",
        observations=np.stack(traj_obs, axis=0).astype(np.float32, copy=False),
        actions=np.stack(traj_act, axis=0).astype(np.int32, copy=False),
        valid_len=np.asarray(traj_len, dtype=np.int32),
        speaker_id=np.asarray(traj_spk),
        center_sec=np.asarray(traj_center, dtype=np.float64),
        forbidden_mask=np.stack(traj_forbidden, axis=0).astype(bool, copy=False),
        band_edges_hz=edges_hz.astype(np.float64, copy=False),
    )

    valid_len_arr = np.asarray(traj_len, dtype=np.int32)
    trajectory_summary = {
        "generated": datetime.now().isoformat(),
        "run_dir": str(out_dir),
        "n_samples": int(valid_len_arr.size),
        "k_horizon": int(K_HORIZON),
        "n_bands": int(N_BANDS),
        "valid_len": {
            "mean": float(np.mean(valid_len_arr)),
            "median": float(np.median(valid_len_arr)),
            "p10": float(np.percentile(valid_len_arr, 10)),
            "p90": float(np.percentile(valid_len_arr, 90)),
            "frac_zero": float(np.mean(valid_len_arr == 0)),
        },
    }
    write_json(out_dir / "trajectory_summary.json", trajectory_summary)

    # Pooled summary + report
    def pooled_stats(x: list[float]) -> dict[str, float]:
        arr = np.asarray(x, dtype=np.float64)
        if arr.size == 0:
            return {"count": 0}
        return {
            "count": int(arr.size),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }

    base_ref = pooled_stats(pooled["baseline"]["theta_err_ref"])
    teach_ref = pooled_stats(pooled["teacher"]["theta_err_ref"])
    base_fail = float(np.mean(np.asarray(pooled["baseline"]["theta_err_ref"], dtype=np.float64) > 5.0))
    teach_fail = float(np.mean(np.asarray(pooled["teacher"]["theta_err_ref"], dtype=np.float64) > 5.0))

    # Teacher sanity gate: median not worse by >5%
    median_worsening_frac = float((teach_ref["median"] - base_ref["median"]) / (base_ref["median"] + 1e-12))
    teacher_gate_pass = bool(median_worsening_frac <= 0.05)

    summary = {
        "generated": datetime.now().isoformat(),
        "run_dir": str(out_dir),
        "pooled": {
            "baseline": {
                "theta_error_ref_deg": base_ref,
                "theta_error_geo_deg": pooled_stats(pooled["baseline"]["theta_err_geo"]),
                "fail_rate_ref_gt5deg": base_fail,
            },
            "teacher": {
                "theta_error_ref_deg": teach_ref,
                "theta_error_geo_deg": pooled_stats(pooled["teacher"]["theta_err_geo"]),
                "fail_rate_ref_gt5deg": teach_fail,
            },
        },
        "teacher_sanity_gate": {
            "median_not_worse_frac_le_0p05": teacher_gate_pass,
            "computed": {"median_worsening_frac": median_worsening_frac},
        },
    }
    write_json(out_dir / "summary.json", summary)

    lines = []
    lines.append("# Band-OMP Teacher Report")
    lines.append("")
    lines.append(f"Generated: {summary['generated']}")
    lines.append(f"Run dir: {out_dir}")
    lines.append("")
    lines.append("## Teacher sanity gate")
    lines.append("")
    lines.append(f"- median(theta_error_ref) worsening frac: {median_worsening_frac:.3f} (<= 0.050)")
    lines.append(f"- OVERALL: {'PASS' if teacher_gate_pass else 'FAIL'}")
    lines.append("")
    lines.append("## Pooled Metrics (vs chirp reference)")
    lines.append("")
    lines.append("| Method | count | median | p90 | p95 | fail_rate(>5°) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| baseline | {base_ref['count']} | {base_ref['median']:.3f} | {base_ref['p90']:.3f} | {base_ref['p95']:.3f} | {base_fail:.3f} |"
    )
    lines.append(
        f"| teacher | {teach_ref['count']} | {teach_ref['median']:.3f} | {teach_ref['p90']:.3f} | {teach_ref['p95']:.3f} | {teach_fail:.3f} |"
    )
    lines.append("")
    (out_dir / "grid_report.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info("Done. Results: %s", out_dir)
    logger.info("Teacher sanity gate pass=%s", teacher_gate_pass)


if __name__ == "__main__":
    main()
