#!/usr/bin/env python
"""
Root-cause audit for Claim-2 experiments (analysis-only; no model changes).

This script answers two questions on real WAVs:
1) Is failure primarily "uncertain-wrong" (low PSR, high error) or "confidently-wrong"
   (high PSR, high error)?
2) Is the global GCC peak (within ±maxlag) competing with the guided peak near tau_ref?

It replays the per-window mic corruption described in existing teacher/student JSONL
artifacts, reconstructs the corrupted MicL/MicR windows, and recomputes:
- guided peak (around tau_ref ± guided_radius)
- global peak (full ±maxlag window)
for MIC–MIC GCC-PHAT in the analysis band.

Outputs are written under --out_dir (never repo root).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


FS_EXPECTED = 48_000
WINDOW_SEC = 5.0
BAND_LO_HZ = 500.0
BAND_HI_HZ = 2000.0
MAX_LAG_MS = 10.0
GUIDED_RADIUS_MS = 0.3
PSR_EXCLUDE_SAMPLES = 50

THETA_FAIL_DEG = 4.0
PSR_GOOD_DB = 3.0
TAU_NEAR0_MS = 0.3


def configure_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_code_state(out_dir: Path) -> None:
    script_path = Path(__file__).resolve()
    try:
        git_head = (
            __import__("subprocess")
            .check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
        dirty = (
            __import__("subprocess")
            .check_output(["git", "status", "--porcelain"], text=True)
            .strip()
            != ""
        )
    except Exception:
        git_head, dirty = None, None
    write_json(
        out_dir / "code_state.json",
        {
            "script_path": str(script_path),
            "script_sha256": sha256_file(script_path),
            "git_head": git_head,
            "dirty": dirty,
            "timestamp": datetime.now().isoformat(),
        },
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return rows


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution so dataclasses (and other reflection) can resolve __module__.
    import sys as _sys

    _sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def tau_to_theta_deg(tau_sec: float, *, c: float = 343.0, d: float = 1.4) -> float:
    sin_theta = float(np.clip(float(tau_sec) * float(c) / float(d), -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_theta)))


def build_analysis_mask(freqs_hz: np.ndarray, *, lo_hz: float, hi_hz: float) -> np.ndarray:
    return ((freqs_hz >= float(lo_hz)) & (freqs_hz <= float(hi_hz))).astype(np.float64)


@dataclass(frozen=True)
class Peaks:
    tau_guided_ms: float
    psr_guided_db: float
    tau_global_ms: float
    psr_global_db: float


def psr_db_at_peak(abs_cc: np.ndarray, peak_idx: int, *, exclude: int) -> float:
    mask = np.ones_like(abs_cc, dtype=bool)
    lo_e = max(0, int(peak_idx) - int(exclude))
    hi_e = min(len(abs_cc), int(peak_idx) + int(exclude) + 1)
    mask[lo_e:hi_e] = False
    sidelobe_max = float(abs_cc[mask].max()) if np.any(mask) else 0.0
    peak_val = float(abs_cc[int(peak_idx)])
    return 20.0 * float(np.log10(peak_val / (sidelobe_max + 1e-10)))


def compute_peaks(
    *,
    micl: np.ndarray,
    micr: np.ndarray,
    fs: int,
    tau_ref_ms: float,
    mic_gcc_mod,
) -> Peaks:
    win_samples = int(round(float(WINDOW_SEC) * float(fs)))
    n_fft = int(win_samples * 2)
    freqs = np.fft.rfftfreq(int(n_fft), d=1.0 / float(fs))
    analysis_mask = build_analysis_mask(freqs, lo_hz=float(BAND_LO_HZ), hi_hz=float(BAND_HI_HZ))
    max_shift = int(round(float(MAX_LAG_MS) * float(fs) / 1000.0))

    X = np.fft.rfft(micl, n_fft)
    Y = np.fft.rfft(micr, n_fft)
    R = X * np.conj(Y)
    R_phat = R / (np.abs(R) + float(mic_gcc_mod.RIDGE_EPS))
    cc = mic_gcc_mod.ccwin_from_spectrum(
        (R_phat * analysis_mask).astype(np.complex128, copy=False),
        n_fft=n_fft,
        max_shift=max_shift,
    )

    guided = mic_gcc_mod.estimate_tau_psr_from_ccwin(
        cc,
        fs=int(fs),
        max_shift=int(max_shift),
        guided_tau_sec=float(tau_ref_ms) / 1000.0,
        guided_radius_sec=float(GUIDED_RADIUS_MS) / 1000.0,
        psr_exclude_samples=int(PSR_EXCLUDE_SAMPLES),
    )
    tau_guided_ms = float(guided.tau_sec) * 1000.0
    psr_guided_db = float(guided.psr_db)

    abs_cc = np.abs(cc)
    global_idx = int(np.argmax(abs_cc))
    tau_global_ms = float((global_idx - int(max_shift)) / float(fs) * 1000.0)
    psr_global_db = psr_db_at_peak(abs_cc, global_idx, exclude=int(PSR_EXCLUDE_SAMPLES))

    return Peaks(
        tau_guided_ms=float(tau_guided_ms),
        psr_guided_db=float(psr_guided_db),
        tau_global_ms=float(tau_global_ms),
        psr_global_db=float(psr_global_db),
    )


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


def load_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    from scipy.io import wavfile

    sr, data = wavfile.read(str(path))
    if data.ndim != 1:
        raise ValueError(f"Expected mono WAV: {path} (shape={data.shape})")
    if data.dtype == np.int16:
        data = (data.astype(np.float32) / 32768.0).astype(np.float32, copy=False)
    elif data.dtype == np.int32:
        data = (data.astype(np.float32) / 2147483648.0).astype(np.float32, copy=False)
    else:
        data = data.astype(np.float32, copy=False)
    return int(sr), data.astype(np.float32, copy=False)


def extract_centered_window(signal: np.ndarray, *, fs: int, center_sec: float, window_sec: float) -> np.ndarray:
    win_samples = int(round(float(window_sec) * float(fs)))
    center_samp = int(round(float(center_sec) * float(fs)))
    start = int(center_samp - win_samples // 2)
    end = int(start + win_samples)
    if start < 0 or end > len(signal):
        raise ValueError(
            f"Window out of bounds: center_sec={center_sec}, start={start}, end={end}, len={len(signal)}"
        )
    return signal[start:end]


def summarize_array(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
    }


def case_rows_from_dir(case_dir: Path) -> list[dict[str, Any]]:
    test_windows = case_dir / "test_windows.jsonl"
    if test_windows.exists():
        return load_jsonl(test_windows)
    per_speaker = case_dir / "per_speaker"
    if per_speaker.exists():
        out: list[dict[str, Any]] = []
        for spk_dir in sorted(per_speaker.iterdir()):
            if not spk_dir.is_dir():
                continue
            out.extend(load_jsonl(spk_dir / "windows.jsonl"))
        if not out:
            raise ValueError(f"No rows under per_speaker: {case_dir}")
        return out
    raise FileNotFoundError(f"Unsupported case dir (no test_windows.jsonl or per_speaker): {case_dir}")


def replay_mic_corruption(
    *,
    micl_clean: np.ndarray,
    micr_clean: np.ndarray,
    micl_full: np.ndarray,
    micr_full: np.ndarray,
    center_sec: float,
    corruption: dict[str, Any] | None,
    mic_corrupt_mod,
    fs: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Reconstruct the corrupted MicL/MicR windows from the JSONL corruption record.
    Returns (micl_corr, micr_corr, diag_record).
    """
    corr = corruption or {}

    occl = corr.get("occlusion", None) or {}
    occl_enabled = bool(occl.get("enabled", False))
    occl_target = str(occl.get("target", "micr"))
    occl_kind = str(occl.get("kind", "lowpass"))
    occl_lowpass = float(occl.get("lowpass_hz", 800.0))
    occl_tilt_k = float(occl.get("tilt_k", 2.0))
    occl_tilt_pivot = float(occl.get("tilt_pivot_hz", 800.0))

    sigL_mix = micl_clean.astype(np.float64, copy=False)
    sigR_mix = micr_clean.astype(np.float64, copy=False)
    if occl_enabled and occl_target == "micl":
        sigL_mix = mic_corrupt_mod.apply_occlusion_fft(
            sigL_mix,
            fs=int(fs),
            kind=str(occl_kind),
            lowpass_hz=float(occl_lowpass),
            tilt_k=float(occl_tilt_k),
            tilt_pivot_hz=float(occl_tilt_pivot),
        )
    if occl_enabled and occl_target == "micr":
        sigR_mix = mic_corrupt_mod.apply_occlusion_fft(
            sigR_mix,
            fs=int(fs),
            kind=str(occl_kind),
            lowpass_hz=float(occl_lowpass),
            tilt_k=float(occl_tilt_k),
            tilt_pivot_hz=float(occl_tilt_pivot),
        )

    cm = corr.get("common_mode_interference", None)
    cm_record = None
    if isinstance(cm, dict) and bool(cm.get("enabled", False)):
        nccm = float(cm["noise_center_sec"])
        noise_cm = extract_centered_window(
            micl_full.astype(np.float32, copy=False), fs=int(fs), center_sec=float(nccm), window_sec=float(WINDOW_SEC)
        ).astype(np.float64, copy=False)
        cfg = mic_corrupt_mod.CommonModeInterferenceConfig(
            snr_db=float(cm.get("snr_target_db", cm.get("snr_db"))),
            band_lo_hz=float(cm.get("diag", {}).get("band_lo_hz", BAND_LO_HZ)),
            band_hi_hz=float(cm.get("diag", {}).get("band_hi_hz", BAND_HI_HZ)),
            seed=int(cm.get("seed", 0)),
        )
        (sigL_mix, sigR_mix), diag_cm = mic_corrupt_mod.add_common_mode_interference(
            sigL_mix,
            sigR_mix,
            noise_cm,
            cfg=cfg,
            fs=int(fs),
            signal_for_alpha=micl_clean.astype(np.float64, copy=False),
        )
        cm_record = {"enabled": True, "noise_center_sec": float(nccm), "diag": diag_cm}

    cd = corr.get("delayed_coherent_interference", None)
    cd_record = None
    if isinstance(cd, dict) and bool(cd.get("enabled", False)):
        nccd = float(cd["noise_center_sec"])
        noise_cd = extract_centered_window(
            micl_full.astype(np.float32, copy=False), fs=int(fs), center_sec=float(nccd), window_sec=float(WINDOW_SEC)
        ).astype(np.float64, copy=False)
        cfg = mic_corrupt_mod.DelayedCoherentInterferenceConfig(
            snr_db=float(cd.get("snr_target_db", cd.get("snr_db"))),
            band_lo_hz=float(cd.get("diag", {}).get("band_lo_hz", BAND_LO_HZ)),
            band_hi_hz=float(cd.get("diag", {}).get("band_hi_hz", BAND_HI_HZ)),
            delay_ms=float(cd.get("delay_ms", 0.0)),
            target_delayed=str(cd.get("target_delayed", "micr")),
            seed=int(cd.get("seed", 0)),
        )
        (sigL_mix, sigR_mix), diag_cd = mic_corrupt_mod.add_delayed_coherent_interference(
            sigL_mix,
            sigR_mix,
            noise_cd,
            cfg=cfg,
            fs=int(fs),
            signal_for_alpha=micl_clean.astype(np.float64, copy=False),
        )
        cd_record = {"enabled": True, "noise_center_sec": float(nccd), "diag": diag_cd}

    enabled = bool(corr.get("enabled", False))
    if not enabled:
        return (
            sigL_mix.astype(np.float64, copy=False),
            sigR_mix.astype(np.float64, copy=False),
            {
                "enabled": False,
                "occlusion": occl,
                "common_mode_interference": cm_record,
                "delayed_coherent_interference": cd_record,
            },
        )

    ncl = float(corr["noise_center_sec_L"])
    ncr = float(corr["noise_center_sec_R"])
    noiseL = extract_centered_window(
        micl_full.astype(np.float32, copy=False), fs=int(fs), center_sec=float(ncl), window_sec=float(WINDOW_SEC)
    ).astype(np.float64, copy=False)
    noiseR = extract_centered_window(
        micr_full.astype(np.float32, copy=False), fs=int(fs), center_sec=float(ncr), window_sec=float(WINDOW_SEC)
    ).astype(np.float64, copy=False)

    micl_diag = corr.get("micl", {}) or {}
    micr_diag = corr.get("micr", {}) or {}
    snr_db = float(micl_diag.get("snr_target_db", micr_diag.get("snr_target_db", 0.0)))
    preclip_gain = float(micl_diag.get("preclip_gain", 100.0))
    clip_limit = float(micl_diag.get("clip_limit", 0.99))
    cfg_ind = mic_corrupt_mod.MicCorruptionConfig(
        snr_db=float(snr_db),
        band_lo_hz=float(micl_diag.get("band_lo_hz", BAND_LO_HZ)),
        band_hi_hz=float(micl_diag.get("band_hi_hz", BAND_HI_HZ)),
        preclip_gain=float(preclip_gain),
        clip_limit=float(clip_limit),
        seed=0,
    )

    micl_corr, diagL = mic_corrupt_mod.apply_mic_corruption(
        micl_clean.astype(np.float64, copy=False),
        noiseL,
        cfg=cfg_ind,
        fs=int(fs),
        signal_for_alpha=micl_clean.astype(np.float64, copy=False),
        signal_for_mix=sigL_mix.astype(np.float64, copy=False),
    )
    micr_corr, diagR = mic_corrupt_mod.apply_mic_corruption(
        micr_clean.astype(np.float64, copy=False),
        noiseR,
        cfg=cfg_ind,
        fs=int(fs),
        signal_for_alpha=micr_clean.astype(np.float64, copy=False),
        signal_for_mix=sigR_mix.astype(np.float64, copy=False),
    )

    return (
        micl_corr.astype(np.float64, copy=False),
        micr_corr.astype(np.float64, copy=False),
        {
            "enabled": True,
            "noise_center_sec_L": float(ncl),
            "noise_center_sec_R": float(ncr),
            "common_mode_interference": cm_record,
            "delayed_coherent_interference": cd_record,
            "occlusion": occl,
            "micl": diagL,
            "micr": diagR,
        },
    )


def analyze_case(
    *,
    label: str,
    case_dir: Path,
    data_root: Path,
    truth_ref_root: Path,
    out_dir: Path,
    test_min_center_sec: float,
    mic_corrupt_mod,
    mic_gcc_mod,
) -> dict[str, Any]:
    rows = case_rows_from_dir(case_dir)
    case_out = out_dir / "cases" / label
    case_out.mkdir(parents=True, exist_ok=True)
    write_json(case_out / "input.json", {"label": label, "case_dir": str(case_dir), "n_rows": int(len(rows))})

    # Cache WAVs per speaker
    cache: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    wav_files: list[Path] = []

    per_window_path = case_out / "per_window.jsonl"
    n_used = 0

    # Metrics accumulators
    guided_err = []
    guided_psr = []
    global_psr = []
    global_tau = []
    near0_global = 0
    cw = 0
    cr = 0
    uw = 0
    total = 0

    per_speaker_counts: dict[str, dict[str, int]] = {}

    with per_window_path.open("w", encoding="utf-8") as f:
        for r in rows:
            spk = str(r.get("speaker_id") or r.get("speaker") or "")
            if not spk:
                raise ValueError(f"Missing speaker_id in row for case={label}")
            center_sec = float(r["center_sec"])
            if center_sec <= float(test_min_center_sec):
                continue

            truth_ref = r.get("truth_reference", None)
            if not isinstance(truth_ref, dict):
                # Fall back to chirp summary (must exist)
                truth_ref = json.loads((truth_ref_root / spk / "summary.json").read_text(encoding="utf-8"))["truth_reference"]
            tau_ref_ms = float(truth_ref["tau_ref_ms"])
            theta_ref_deg = float(truth_ref["theta_ref_deg"])

            if spk not in cache:
                _ldv_path, micl_path, micr_path = list_triplet_files(data_root, spk)
                wav_files.extend([micl_path, micr_path])
                sr_l, micl_full = load_wav_mono(micl_path)
                sr_r, micr_full = load_wav_mono(micr_path)
                if sr_l != int(FS_EXPECTED) or sr_r != int(FS_EXPECTED):
                    raise ValueError(f"fs mismatch for {spk}: micl={sr_l}, micr={sr_r}, expected={FS_EXPECTED}")
                cache[spk] = (micl_full.astype(np.float32, copy=False), micr_full.astype(np.float32, copy=False), int(sr_l))

            micl_full, micr_full, fs = cache[spk]
            micl_clean = extract_centered_window(micl_full, fs=int(fs), center_sec=float(center_sec), window_sec=float(WINDOW_SEC))
            micr_clean = extract_centered_window(micr_full, fs=int(fs), center_sec=float(center_sec), window_sec=float(WINDOW_SEC))

            corruption = r.get("corruption", None)
            micl_corr, micr_corr, corr_diag = replay_mic_corruption(
                micl_clean=micl_clean,
                micr_clean=micr_clean,
                micl_full=micl_full,
                micr_full=micr_full,
                center_sec=float(center_sec),
                corruption=corruption if isinstance(corruption, dict) else None,
                mic_corrupt_mod=mic_corrupt_mod,
                fs=int(fs),
            )

            peaks = compute_peaks(
                micl=micl_corr,
                micr=micr_corr,
                fs=int(fs),
                tau_ref_ms=float(tau_ref_ms),
                mic_gcc_mod=mic_gcc_mod,
            )

            theta_guided = tau_to_theta_deg(peaks.tau_guided_ms / 1000.0)
            theta_global = tau_to_theta_deg(peaks.tau_global_ms / 1000.0)
            err_guided = abs(float(theta_guided) - float(theta_ref_deg))
            err_global = abs(float(theta_global) - float(theta_ref_deg))

            is_psr_good = float(peaks.psr_guided_db) >= float(PSR_GOOD_DB)
            is_wrong = float(err_guided) >= float(THETA_FAIL_DEG)
            if is_psr_good and is_wrong:
                cls = "CW"
            elif is_psr_good and not is_wrong:
                cls = "CR"
            elif (not is_psr_good) and is_wrong:
                cls = "UW"
            else:
                cls = "UR"

            gnear0 = (abs(float(peaks.tau_global_ms)) < float(TAU_NEAR0_MS)) and (float(peaks.psr_global_db) >= float(PSR_GOOD_DB))

            rec = {
                "label": label,
                "case_dir": str(case_dir),
                "speaker_id": spk,
                "center_sec": float(center_sec),
                "truth_reference": {"tau_ref_ms": float(tau_ref_ms), "theta_ref_deg": float(theta_ref_deg)},
                "peaks": {
                    "guided": {
                        "tau_ms": float(peaks.tau_guided_ms),
                        "psr_db": float(peaks.psr_guided_db),
                        "theta_deg": float(theta_guided),
                        "theta_error_ref_deg": float(err_guided),
                    },
                    "global": {
                        "tau_ms": float(peaks.tau_global_ms),
                        "psr_db": float(peaks.psr_global_db),
                        "theta_deg": float(theta_global),
                        "theta_error_ref_deg": float(err_global),
                    },
                },
                "flags": {
                    "class_guided": cls,
                    "global_near0_psr_gt3": bool(gnear0),
                    "guided_wrong": bool(is_wrong),
                    "guided_psr_good": bool(is_psr_good),
                    "global_minus_ref_ms": float(peaks.tau_global_ms - float(tau_ref_ms)),
                    "guided_minus_ref_ms": float(peaks.tau_guided_ms - float(tau_ref_ms)),
                },
                "corruption_replay": corr_diag,
            }
            f.write(json.dumps(rec) + "\n")

            n_used += 1
            total += 1
            guided_err.append(float(err_guided))
            guided_psr.append(float(peaks.psr_guided_db))
            global_psr.append(float(peaks.psr_global_db))
            global_tau.append(float(peaks.tau_global_ms))
            near0_global += int(bool(gnear0))

            if cls == "CW":
                cw += 1
            elif cls == "CR":
                cr += 1
            elif cls == "UW":
                uw += 1

            d = per_speaker_counts.setdefault(spk, {"n": 0, "CW": 0, "CR": 0, "UW": 0, "near0_global": 0})
            d["n"] += 1
            d["CW"] += int(cls == "CW")
            d["CR"] += int(cls == "CR")
            d["UW"] += int(cls == "UW")
            d["near0_global"] += int(bool(gnear0))

    if n_used == 0:
        raise RuntimeError(f"No rows analyzed for case={label}; check test_min_center_sec={test_min_center_sec}")

    guided_err_arr = np.asarray(guided_err, dtype=np.float64)
    guided_psr_arr = np.asarray(guided_psr, dtype=np.float64)
    global_psr_arr = np.asarray(global_psr, dtype=np.float64)
    global_tau_arr = np.asarray(global_tau, dtype=np.float64)

    summary = {
        "label": label,
        "case_dir": str(case_dir),
        "test_min_center_sec": float(test_min_center_sec),
        "n_analyzed": int(n_used),
        "guided": {
            "theta_error_ref_deg": summarize_array(guided_err_arr),
            "psr_db": summarize_array(guided_psr_arr),
            "p_confident_wrong": float(cw) / float(total),
            "p_confident_right": float(cr) / float(total),
            "p_uncertain_wrong": float(uw) / float(total),
        },
        "global": {
            "tau_ms": summarize_array(global_tau_arr),
            "psr_db": summarize_array(global_psr_arr),
            "p_near0_psr_gt3": float(near0_global) / float(total),
        },
        "per_speaker": {},
    }
    for spk, c in sorted(per_speaker_counts.items()):
        n = max(1, int(c["n"]))
        summary["per_speaker"][spk] = {
            "n": int(c["n"]),
            "p_CW": float(c["CW"]) / float(n),
            "p_CR": float(c["CR"]) / float(n),
            "p_UW": float(c["UW"]) / float(n),
            "p_global_near0_psr_gt3": float(c["near0_global"]) / float(n),
        }

    write_json(case_out / "summary.json", summary)

    # Manifest (WAVs + input JSONLs)
    files = []
    input_files = []
    if (case_dir / "test_windows.jsonl").exists():
        input_files.append(case_dir / "test_windows.jsonl")
    else:
        for spk_dir in sorted((case_dir / "per_speaker").iterdir()):
            if spk_dir.is_dir():
                input_files.append(spk_dir / "windows.jsonl")

    for p in sorted(set(wav_files + input_files)):
        if p.exists():
            files.append({"path": str(p), "sha256": sha256_file(p)})

    write_json(case_out / "manifest.json", {"generated": datetime.now().isoformat(), "files": files})
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Claim-2 failure modes (confident-wrong vs uncertain-wrong; global vs guided peaks)")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--truth_ref_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--case",
        type=str,
        action="append",
        default=[],
        help="Case in the form label=PATH. PATH can be a teacher dir (per_speaker/*/windows.jsonl) or a student dir (test_windows.jsonl). Repeatable.",
    )
    ap.add_argument("--test_min_center_sec", type=float, default=450.0, help="Analyze windows with center_sec > this value. Default: 450.")
    args = ap.parse_args()

    if not args.case:
        raise ValueError("At least one --case label=PATH is required")

    out_dir = Path(args.out_dir).expanduser().resolve()
    configure_logging(out_dir)
    write_code_state(out_dir)

    data_root = Path(args.data_root).expanduser().resolve()
    truth_ref_root = Path(args.truth_ref_root).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[1]
    mic_corrupt_mod = load_module_from_path("mic_corruption_mod", repo_root / "scripts" / "mic_corruption.py")
    mic_gcc_mod = load_module_from_path("mic_gcc_mod", repo_root / "scripts" / "teacher_band_omp_micmic.py")

    run_cfg = {
        "generated": datetime.now().isoformat(),
        "data_root": str(data_root),
        "truth_ref_root": str(truth_ref_root),
        "window_sec": float(WINDOW_SEC),
        "fs_expected": int(FS_EXPECTED),
        "analysis_band_hz": [float(BAND_LO_HZ), float(BAND_HI_HZ)],
        "max_lag_ms": float(MAX_LAG_MS),
        "guided_radius_ms": float(GUIDED_RADIUS_MS),
        "theta_fail_deg": float(THETA_FAIL_DEG),
        "psr_good_db": float(PSR_GOOD_DB),
        "tau_near0_ms": float(TAU_NEAR0_MS),
        "test_min_center_sec": float(args.test_min_center_sec),
        "cases": list(args.case),
    }
    write_json(out_dir / "run_config.json", run_cfg)

    summaries = []
    for item in args.case:
        if "=" not in item:
            raise ValueError(f"Invalid --case format (expected label=PATH): {item}")
        label, path = item.split("=", 1)
        label = label.strip()
        case_dir = Path(path).expanduser().resolve()
        logger.info("Analyzing case %s: %s", label, case_dir)
        summaries.append(
            analyze_case(
                label=label,
                case_dir=case_dir,
                data_root=data_root,
                truth_ref_root=truth_ref_root,
                out_dir=out_dir,
                test_min_center_sec=float(args.test_min_center_sec),
                mic_corrupt_mod=mic_corrupt_mod,
                mic_gcc_mod=mic_gcc_mod,
            )
        )

    write_json(out_dir / "summary.json", {"generated": datetime.now().isoformat(), "summaries": summaries})

    # Report
    lines = []
    lines.append("# Claim-2 Root-Cause Audit: Confidently-Wrong vs Uncertainly-Wrong, Global vs Guided Peaks\n\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n\n")
    lines.append(f"Run dir: `{out_dir.as_posix()}`\n\n")
    lines.append("## Definitions (guided peak)\n\n")
    lines.append(f"- Confidently-wrong (CW): PSR >= {PSR_GOOD_DB:.1f} dB AND theta_error_ref >= {THETA_FAIL_DEG:.1f} deg\n")
    lines.append(f"- Confidently-right (CR): PSR >= {PSR_GOOD_DB:.1f} dB AND theta_error_ref < {THETA_FAIL_DEG:.1f} deg\n")
    lines.append(f"- Uncertainly-wrong (UW): PSR < {PSR_GOOD_DB:.1f} dB AND theta_error_ref >= {THETA_FAIL_DEG:.1f} deg\n\n")
    lines.append("## Case summary (pooled)\n\n")
    lines.append("| case | n | guided p95 err (deg) | P(CW) | P(UW) | guided PSR median | P(global near0 & PSR>3) | global tau median (ms) |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for s in summaries:
        g = s["guided"]
        gl = s["global"]
        lines.append(
            f"| {s['label']} | {s['n_analyzed']} | {g['theta_error_ref_deg']['p95']:.3f} | "
            f"{g['p_confident_wrong']:.3f} | {g['p_uncertain_wrong']:.3f} | {g['psr_db']['median']:.2f} | "
            f"{gl['p_near0_psr_gt3']:.3f} | {gl['tau_ms']['median']:.3f} |\n"
        )
    lines.append("\n")
    lines.append("## Per-speaker CW rates (guided)\n\n")
    for s in summaries:
        lines.append(f"### {s['label']}\n\n")
        lines.append("| speaker | n | P(CW) | P(UW) | P(global near0 & PSR>3) |\n")
        lines.append("| --- | ---: | ---: | ---: | ---: |\n")
        for spk, d in s["per_speaker"].items():
            lines.append(
                f"| {spk} | {d['n']} | {d['p_CW']:.3f} | {d['p_UW']:.3f} | {d['p_global_near0_psr_gt3']:.3f} |\n"
            )
        lines.append("\n")

    (out_dir / "report.md").write_text("".join(lines), encoding="utf-8")
    logger.info("Wrote report: %s", (out_dir / "report.md").as_posix())


if __name__ == "__main__":
    main()
