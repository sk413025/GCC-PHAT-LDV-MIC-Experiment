#!/usr/bin/env python
"""
Train and evaluate a lightweight DTmin band-selection policy from Band-OMP teacher trajectories.

Input
-----
Teacher trajectories produced by:
  scripts/teacher_band_omp_micmic.py

Trajectory contents (plan-locked)
--------------------------------
- observations: (N, K, B) float32
- actions     : (N, K) int32 (band index), padded with -1
- valid_len   : (N,) int32
- speaker_id  : (N,) string
- center_sec  : (N,) float
- forbidden_mask: (N, B) bool
- band_edges_hz: (B+1,) float64

Student model (plan-locked)
---------------------------
Step-conditioned nearest-centroid classifier over band actions:
  - one centroid per (step k, action b)
  - predict action by argmin ||obs - centroid|| over valid actions

Evaluation (plan-locked)
------------------------
Time split per speaker:
  - Train: center_sec <= 450
  - Test : center_sec > 450

Metrics (plan-locked)
---------------------
Compare baseline MIC-MIC (full 500–2000 Hz) vs student-selected bands on test windows,
guided by chirp-reference truth tau_ref_ms with radius 0.3 ms.

CLI (plan-locked)
-----------------
python -u scripts/train_dtmin_from_band_trajectories.py \\
  --traj_path results/band_omp_teacher_<ts>/teacher_trajectories.npz \\
  --data_root /home/sbplab/jiawei/data \\
  --truth_ref_root results/ldv_vs_mic_grid_20260211_093529/micl_micr_chirp_tau2_band500_2000 \\
  --out_dir results/band_dtmin_student_<YYYYMMDD_HHMMSS>
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

try:
    # When executed as a script: `python scripts/train_dtmin_from_band_trajectories.py`
    from mic_corruption import (  # type: ignore
        CommonModeInterferenceConfig,
        DelayedCoherentInterferenceConfig,
        MicCorruptionConfig,
        add_common_mode_interference,
        add_delayed_coherent_interference,
        apply_mic_corruption,
        apply_occlusion_fft,
    )
except ImportError:  # pragma: no cover
    # When imported from repo root: `import scripts.train_dtmin_from_band_trajectories`
    from scripts.mic_corruption import (  # type: ignore
        CommonModeInterferenceConfig,
        DelayedCoherentInterferenceConfig,
        MicCorruptionConfig,
        add_common_mode_interference,
        add_delayed_coherent_interference,
        apply_mic_corruption,
        apply_occlusion_fft,
    )

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
RIDGE_EPS = 1e-12

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
        "label": str(ref.get("label", "")),
    }


# ─────────────────────────────────────────────────────────────────────
# GCC utilities (band masks, tau/psr)
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
            # Quadratic interpolation can extrapolate far outside the guided window when the
            # argmax hits the window boundary. Clamp to a local sub-sample refinement only.
            shift = float(np.clip(shift, -0.5, 0.5))

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


# ─────────────────────────────────────────────────────────────────────
# DTmin band policy
# ─────────────────────────────────────────────────────────────────────


def train_nearest_centroid(
    observations: np.ndarray, actions: np.ndarray, valid_len: np.ndarray, *, horizon: int, n_actions: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroids = np.zeros((horizon, n_actions, observations.shape[-1]), dtype=np.float32)
    action_valid = np.zeros((horizon, n_actions), dtype=bool)
    action_counts = np.zeros((horizon, n_actions), dtype=np.int32)

    for step_idx in range(horizon):
        step_mask = valid_len > step_idx
        obs_step = observations[step_mask, step_idx, :]
        act_step = actions[step_mask, step_idx]
        if obs_step.shape[0] == 0:
            continue
        for a in range(n_actions):
            a_mask = act_step == a
            count = int(np.sum(a_mask))
            action_counts[step_idx, a] = count
            if count > 0:
                centroids[step_idx, a, :] = obs_step[a_mask].mean(axis=0)
                action_valid[step_idx, a] = True

    return centroids, action_valid, action_counts


def predict_sequence(
    obs_kb: np.ndarray,
    *,
    centroids: np.ndarray,
    action_valid: np.ndarray,
    forbidden_mask: np.ndarray,
    stop_action_id: int | None = None,
    stop_preference_delta: np.ndarray | None = None,
) -> list[int]:
    """
    Predict a variable-length band set with optional STOP action.

    Selection rule (fail-fast, no fallbacks outside the action set):
    - Evaluate distances to all valid action centroids for each step.
    - Choose the closest *allowed* band action, skipping forbidden/already-selected bands.
    - If STOP is enabled and sufficiently close to the best-allowed band (per step delta),
      stop early.
    - If no allowed band actions exist for a step, stop.
    """
    horizon, n_actions, dim = centroids.shape
    assert obs_kb.shape == (horizon, dim)
    if stop_preference_delta is not None:
        if stop_preference_delta.shape != (horizon,):
            raise ValueError(f"stop_preference_delta must have shape ({horizon},), got {stop_preference_delta.shape}")
    selected: list[int] = []
    for step_idx in range(horizon):
        if not np.any(action_valid[step_idx]):
            break
        o = obs_kb[step_idx].astype(np.float32, copy=False)
        diffs = centroids[step_idx] - o[None, :]
        d2 = np.sum(diffs * diffs, axis=1)  # (n_actions,)
        d2 = np.where(action_valid[step_idx], d2, np.inf)

        # Identify best-allowed band candidate (skip forbidden/already-selected).
        order = np.argsort(d2)
        best_band: int | None = None
        best_band_d2: float | None = None
        for a in order:
            if not np.isfinite(d2[a]):
                break
            a = int(a)
            if stop_action_id is not None and int(a) == int(stop_action_id):
                continue
            if a < 0 or a >= int(forbidden_mask.size):
                continue
            if bool(forbidden_mask[a]) or a in selected:
                continue
            best_band = a
            best_band_d2 = float(d2[a])
            break

        if best_band is None or best_band_d2 is None:
            break

        # Optional STOP preference: stop if STOP is close to the best allowed band.
        if stop_action_id is not None and stop_preference_delta is not None:
            if action_valid[step_idx, stop_action_id]:
                d2_stop = float(d2[int(stop_action_id)])
                if np.isfinite(d2_stop):
                    delta = float(stop_preference_delta[step_idx])
                    if d2_stop <= best_band_d2 + delta:
                        break

        selected.append(int(best_band))
    return selected


def calibrate_stop_preference_delta(
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    valid_len: np.ndarray,
    centroids: np.ndarray,
    action_valid: np.ndarray,
    stop_action_id: int,
    stop_target_frac_add: float = 0.0,
) -> np.ndarray:
    """
    Calibrate a per-step STOP preference delta so that the *frequency* of STOP decisions
    roughly matches the teacher on the training split.

    Rule: stop if d2(stop) <= d2(best_nonstop) + delta[step].
    """
    horizon, n_actions, dim = centroids.shape
    if observations.shape[1] != horizon or observations.shape[2] != dim:
        raise ValueError(f"observations shape mismatch: {observations.shape} vs ({observations.shape[0]},{horizon},{dim})")
    if actions.shape[1] != horizon:
        raise ValueError(f"actions shape mismatch: {actions.shape}")
    if int(stop_action_id) < 0 or int(stop_action_id) >= int(n_actions):
        raise ValueError(f"Invalid stop_action_id={stop_action_id} for n_actions={n_actions}")
    deltas = np.zeros((horizon,), dtype=np.float32)

    for step_idx in range(horizon):
        step_mask = valid_len > step_idx
        if not np.any(step_mask):
            deltas[step_idx] = 0.0
            continue
        if not bool(action_valid[step_idx, stop_action_id]):
            # No STOP examples at this step; discourage STOP.
            deltas[step_idx] = -1e6
            continue

        obs_step = observations[step_mask, step_idx, :].astype(np.float32, copy=False)
        act_step = actions[step_mask, step_idx].astype(np.int32, copy=False)

        # Compute d2 to STOP and to the best non-STOP action for each sample.
        diffs = centroids[step_idx][None, :, :] - obs_step[:, None, :]
        d2_all = np.sum(diffs * diffs, axis=2)  # (Ns, n_actions)
        valid_actions = action_valid[step_idx].astype(bool, copy=False)
        d2_all = np.where(valid_actions[None, :], d2_all, np.inf)

        d2_stop = d2_all[:, int(stop_action_id)]
        # Best non-stop
        d2_nonstop = d2_all.copy()
        d2_nonstop[:, int(stop_action_id)] = np.inf
        d2_best = np.min(d2_nonstop, axis=1)

        finite = np.isfinite(d2_stop) & np.isfinite(d2_best)
        if not np.any(finite):
            deltas[step_idx] = -1e6
            continue

        diff = (d2_stop[finite] - d2_best[finite]).astype(np.float64)
        target_frac = float(np.mean(act_step[finite] == int(stop_action_id)))
        target_frac = float(np.clip(target_frac + float(stop_target_frac_add), 0.0, 1.0))

        if target_frac <= 0.0:
            deltas[step_idx] = float(np.min(diff) - 1e-6)
            continue
        if target_frac >= 1.0:
            deltas[step_idx] = float(np.max(diff) + 1e-6)
            continue

        q = float(np.clip(target_frac * 100.0, 0.0, 100.0))
        deltas[step_idx] = float(np.percentile(diff, q))

    return deltas.astype(np.float32, copy=False)


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
    parser = argparse.ArgumentParser(description="Train DTmin band policy from Band-OMP trajectories")
    parser.add_argument("--traj_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--truth_ref_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--train_max_center_sec",
        type=float,
        default=float(TRAIN_MAX_CENTER_SEC),
        help="Time split threshold per speaker (train: <=, test: >). Default: 450.",
    )
    parser.add_argument(
        "--use_traj_corruption",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, apply mic corruption as specified by the trajectory NPZ (Claim-2). Default: 1.",
    )
    parser.add_argument(
        "--stop_to_teacher_valid_len",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, truncate the student's selected band sequence to the teacher's valid_len per sample (analysis-only).",
    )
    parser.add_argument(
        "--stop_target_frac_add",
        type=float,
        default=0.0,
        help="If STOP is enabled, increase the per-step teacher STOP rate target by this additive amount when calibrating stop_preference_delta. Default: 0.0.",
    )
    parser.add_argument(
        "--inference_psr_stop_enable",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, apply a truth-free inference-time filter: stop adding bands if the guided PSR does not improve. Default: 0.",
    )
    parser.add_argument(
        "--inference_psr_stop_min_gain_db",
        type=float,
        default=0.01,
        help="Minimum guided-PSR improvement (dB) required to accept adding the next band when --inference_psr_stop_enable=1. Default: 0.01.",
    )
    parser.add_argument(
        "--inference_psr_stop_mode",
        type=str,
        default="monotone",
        choices=["monotone", "best_prefix"],
        help="PSR-based selection mode when --inference_psr_stop_enable=1. monotone: stop on first non-improving step. best_prefix: choose prefix that maximizes PSR. Default: monotone.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    use_traj_corruption = bool(int(args.use_traj_corruption))
    stop_to_teacher_valid_len = bool(int(args.stop_to_teacher_valid_len))
    train_max_center_sec = float(args.train_max_center_sec)
    stop_target_frac_add = float(args.stop_target_frac_add)
    inference_psr_stop_enable = bool(int(args.inference_psr_stop_enable))
    inference_psr_stop_min_gain_db = float(args.inference_psr_stop_min_gain_db)
    inference_psr_stop_mode = str(args.inference_psr_stop_mode)
    configure_logging(out_dir)
    write_code_state(out_dir, Path(__file__))

    traj_path = Path(args.traj_path)
    payload = np.load(traj_path, allow_pickle=False)
    observations = payload["observations"].astype(np.float32, copy=False)  # (N, K, B)
    actions = payload["actions"].astype(np.int32, copy=False)  # (N, K)
    valid_len = payload["valid_len"].astype(np.int32, copy=False)  # (N,)
    speaker_id = payload["speaker_id"]
    center_sec = payload["center_sec"].astype(np.float64, copy=False)
    forbidden_mask = payload["forbidden_mask"].astype(bool, copy=False)  # (N, B)
    edges_hz = payload["band_edges_hz"].astype(np.float64, copy=False)
    stop_action_id = None
    n_actions = None
    if "stop_action_id" in payload and "n_actions" in payload:
        stop_action_id = int(payload["stop_action_id"].item())
        n_actions = int(payload["n_actions"].item())

    noise_center_sec_L = None
    noise_center_sec_R = None
    cm_noise_center_sec = None
    cd_noise_center_sec = None
    corruption_cfg = None
    occlusion_cfg = None
    cm_interf_cfg = None
    cd_interf_cfg = None
    if use_traj_corruption:
        required = ["noise_center_sec_L", "noise_center_sec_R", "corruption_config_json"]
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(
                f"Trajectory NPZ missing corruption keys: {missing}. "
                f"Re-run teacher with corruption enabled or pass --use_traj_corruption=0."
        )
        noise_center_sec_L = payload["noise_center_sec_L"].astype(np.float64, copy=False)
        noise_center_sec_R = payload["noise_center_sec_R"].astype(np.float64, copy=False)
        if "cm_noise_center_sec" in payload:
            cm_noise_center_sec = payload["cm_noise_center_sec"].astype(np.float64, copy=False)
        if "cd_noise_center_sec" in payload:
            cd_noise_center_sec = payload["cd_noise_center_sec"].astype(np.float64, copy=False)
        cfg_json = str(payload["corruption_config_json"].item())
        if cfg_json.strip():
            cfg_payload = json.loads(cfg_json)
            corruption_cfg = MicCorruptionConfig(
                snr_db=float(cfg_payload["snr_db"]),
                band_lo_hz=float(cfg_payload["band_lo_hz"]),
                band_hi_hz=float(cfg_payload["band_hi_hz"]),
                preclip_gain=float(cfg_payload["preclip_gain"]),
                clip_limit=float(cfg_payload["clip_limit"]),
                seed=int(cfg_payload["seed"]),
            )
            occlusion_cfg = cfg_payload.get("occlusion", None)
            cm_payload = cfg_payload.get("common_mode_interference", None)
            if isinstance(cm_payload, dict) and bool(cm_payload.get("enabled", False)):
                if cm_noise_center_sec is None:
                    raise ValueError("Trajectory NPZ missing cm_noise_center_sec for enabled common-mode interference")
                if cm_payload.get("snr_db", None) is None:
                    raise ValueError("common_mode_interference.snr_db is required when enabled")
                cm_interf_cfg = CommonModeInterferenceConfig(
                    snr_db=float(cm_payload["snr_db"]),
                    band_lo_hz=float(cm_payload.get("band_lo_hz", BAND_HZ[0])),
                    band_hi_hz=float(cm_payload.get("band_hi_hz", BAND_HZ[1])),
                    seed=int(cm_payload.get("seed", 0)),
                )

            cd_payload = cfg_payload.get("delayed_coherent_interference", None)
            if isinstance(cd_payload, dict) and bool(cd_payload.get("enabled", False)):
                if cd_noise_center_sec is None:
                    raise ValueError("Trajectory NPZ missing cd_noise_center_sec for enabled delayed coherent interference")
                if cd_payload.get("snr_db", None) is None:
                    raise ValueError("delayed_coherent_interference.snr_db is required when enabled")
                cd_interf_cfg = DelayedCoherentInterferenceConfig(
                    snr_db=float(cd_payload["snr_db"]),
                    band_lo_hz=float(cd_payload.get("band_lo_hz", BAND_HZ[0])),
                    band_hi_hz=float(cd_payload.get("band_hi_hz", BAND_HZ[1])),
                    delay_ms=float(cd_payload.get("delay_ms", 0.0)),
                    target_delayed=str(cd_payload.get("target_delayed", "micr")),
                    seed=int(cd_payload.get("seed", 0)),
                )

    if observations.ndim != 3:
        raise ValueError(f"Expected observations (N,K,B), got shape={observations.shape}")
    N, K, obs_dim = observations.shape
    B_bands = int(len(edges_hz) - 1)
    if B_bands <= 0:
        raise ValueError("Invalid band_edges_hz (need >=2 edges)")
    if forbidden_mask.shape != (N, B_bands):
        raise ValueError(f"forbidden_mask shape {forbidden_mask.shape} != {(N, B_bands)}")
    if n_actions is None:
        n_actions = int(B_bands)
    if stop_action_id is None:
        stop_action_id = int(B_bands) if int(n_actions) == int(B_bands + 1) else None
    if stop_action_id is not None:
        if int(n_actions) != int(B_bands + 1) or int(stop_action_id) != int(B_bands):
            raise ValueError(f"Inconsistent stop action config: B={B_bands}, n_actions={n_actions}, stop_action_id={stop_action_id}")
    if actions.shape != (N, K):
        raise ValueError(f"actions shape {actions.shape} != {(N, K)}")

    # Validate action range
    act = actions.astype(np.int32, copy=False)
    ok = (act == -1) | ((act >= 0) & (act < int(n_actions)))
    if not bool(np.all(ok)):
        bad = int(np.sum(~ok))
        raise ValueError(f"Actions out of range: n_actions={n_actions}, bad_count={bad}")

    logger.info("Dataset: N=%d, horizon=%d, bands=%d, obs_dim=%d, n_actions=%d", N, K, B_bands, obs_dim, int(n_actions))

    # Train/test split (per speaker, time-based)
    train_mask = center_sec <= float(train_max_center_sec)
    test_mask = ~train_mask
    if int(np.sum(train_mask)) == 0 or int(np.sum(test_mask)) == 0:
        raise RuntimeError("Empty train/test split; check center_sec values")
    logger.info("Train: %d samples, Test: %d samples", int(np.sum(train_mask)), int(np.sum(test_mask)))

    centroids, action_valid, action_counts = train_nearest_centroid(
        observations[train_mask], actions[train_mask], valid_len[train_mask], horizon=K, n_actions=int(n_actions)
    )

    stop_preference_delta = None
    if stop_action_id is not None:
        stop_preference_delta = calibrate_stop_preference_delta(
            observations=observations[train_mask],
            actions=actions[train_mask],
            valid_len=valid_len[train_mask],
            centroids=centroids,
            action_valid=action_valid,
            stop_action_id=int(stop_action_id),
            stop_target_frac_add=stop_target_frac_add,
        )

    model_path = out_dir / f"model_dtmin_band_policy_k{K}.npz"
    metadata = {
        "generated": datetime.now().isoformat(),
        "traj_path": str(traj_path),
        "n_samples": int(N),
        "horizon": int(K),
        "n_bands": int(B_bands),
        "n_actions": int(n_actions),
        "stop_action_id": None if stop_action_id is None else int(stop_action_id),
        "stop_target_frac_add": float(stop_target_frac_add),
        "stop_preference_delta": None if stop_preference_delta is None else stop_preference_delta.tolist(),
        "inference_psr_stop_enable": bool(inference_psr_stop_enable),
        "inference_psr_stop_min_gain_db": float(inference_psr_stop_min_gain_db),
        "inference_psr_stop_mode": str(inference_psr_stop_mode),
        "model_type": "nearest_centroid_stepwise",
        "train_max_center_sec": float(train_max_center_sec),
    }
    np.savez_compressed(
        model_path,
        centroids=centroids,
        action_valid=action_valid,
        action_counts=action_counts,
        band_edges_hz=edges_hz,
        stop_preference_delta=(np.array([], dtype=np.float32) if stop_preference_delta is None else stop_preference_delta),
        metadata_json=json.dumps(metadata),
    )
    write_json(
        out_dir / "train_summary.json",
        {
            **metadata,
            "non_empty_actions_per_step": [int(np.sum(action_valid[k])) for k in range(K)],
            "model_path": str(model_path),
        },
    )
    logger.info("Saved model: %s", model_path)

    # Prepare FFT masks for evaluation (one IFFT per window)
    win_samples = int(round(WINDOW_SEC * FS_EXPECTED))
    n_fft = int(win_samples * 2)
    f_fft = fft_freqs(FS_EXPECTED, n_fft)
    analysis_mask_fft = ((f_fft >= BAND_HZ[0]) & (f_fft <= BAND_HZ[1])).astype(np.float64)
    band_masks_fft = fft_band_masks(freqs_fft_hz=f_fft, edges_hz=edges_hz)  # (B, n_bins)
    max_shift = int(round(float(GCC_MAX_LAG_MS) * float(FS_EXPECTED) / 1000.0))
    guided_radius_sec = float(GCC_GUIDED_RADIUS_MS) / 1000.0

    data_root = Path(args.data_root)
    truth_ref_root = Path(args.truth_ref_root)

    wav_files: list[Path] = []

    base_err_ref: list[float] = []
    stud_err_ref: list[float] = []
    base_err_geo: list[float] = []
    stud_err_geo: list[float] = []

    snr_ach_test_L: list[float] = []
    snr_ach_test_R: list[float] = []
    clip_frac_test_L: list[float] = []
    clip_frac_test_R: list[float] = []

    test_windows_path = out_dir / "test_windows.jsonl"
    with test_windows_path.open("w", encoding="utf-8") as f_jsonl:
        for i in np.where(test_mask)[0].tolist():
            spk = str(speaker_id[i])
            center = float(center_sec[i])
            forbid = forbidden_mask[i]

            truth_ref = load_truth_reference(truth_ref_root / spk / "summary.json")
            geom = compute_geometry_truth(spk)

            micl_path, micr_path = list_mic_files(data_root, spk)
            wav_files.extend([micl_path, micr_path])
            sr_l, micl = load_wav_mono(micl_path)
            sr_r, micr = load_wav_mono(micr_path)
            if not (sr_l == sr_r == FS_EXPECTED):
                raise ValueError(f"Sample rate mismatch for {spk}: {sr_l},{sr_r} expected {FS_EXPECTED}")

            seg_l = extract_centered_window(micl, fs=FS_EXPECTED, center_sec=center, window_sec=WINDOW_SEC).astype(
                np.float64, copy=False
            )
            seg_r = extract_centered_window(micr, fs=FS_EXPECTED, center_sec=center, window_sec=WINDOW_SEC).astype(
                np.float64, copy=False
            )

            corruption_record: dict[str, Any] | None = None
            if use_traj_corruption and corruption_cfg is not None:
                assert noise_center_sec_L is not None
                assert noise_center_sec_R is not None
                ncl = float(noise_center_sec_L[i])
                ncr = float(noise_center_sec_R[i])
                noise_l = extract_centered_window(micl, fs=FS_EXPECTED, center_sec=ncl, window_sec=WINDOW_SEC).astype(
                    np.float64, copy=False
                )
                noise_r = extract_centered_window(micr, fs=FS_EXPECTED, center_sec=ncr, window_sec=WINDOW_SEC).astype(
                    np.float64, copy=False
                )
                sig_l_mix = seg_l
                sig_r_mix = seg_r
                if isinstance(occlusion_cfg, dict) and bool(occlusion_cfg.get("enabled", False)):
                    target = str(occlusion_cfg.get("target", "micr"))
                    kind = str(occlusion_cfg.get("kind", "lowpass"))
                    lowpass_hz = float(occlusion_cfg.get("lowpass_hz", 800.0))
                    tilt_k = float(occlusion_cfg.get("tilt_k", 2.0))
                    tilt_pivot_hz = float(occlusion_cfg.get("tilt_pivot_hz", 800.0))
                    if target == "micl":
                        sig_l_mix = apply_occlusion_fft(
                            sig_l_mix,
                            fs=FS_EXPECTED,
                            kind=kind,
                            lowpass_hz=lowpass_hz,
                            tilt_k=tilt_k,
                            tilt_pivot_hz=tilt_pivot_hz,
                        )
                    elif target == "micr":
                        sig_r_mix = apply_occlusion_fft(
                            sig_r_mix,
                            fs=FS_EXPECTED,
                            kind=kind,
                            lowpass_hz=lowpass_hz,
                            tilt_k=tilt_k,
                            tilt_pivot_hz=tilt_pivot_hz,
                        )
                    else:
                        raise ValueError(f"Unknown occlusion target: {target}")

                cm_record = None
                if cm_interf_cfg is not None:
                    assert cm_noise_center_sec is not None
                    nccm = float(cm_noise_center_sec[i])
                    if not np.isfinite(nccm):
                        raise ValueError("Non-finite cm_noise_center_sec for enabled common-mode interference")
                    noise_cm = extract_centered_window(micl, fs=FS_EXPECTED, center_sec=nccm, window_sec=WINDOW_SEC).astype(
                        np.float64, copy=False
                    )
                    (sig_l_mix, sig_r_mix), diag_cm = add_common_mode_interference(
                        sig_l_mix,
                        sig_r_mix,
                        noise_cm,
                        cfg=cm_interf_cfg,
                        fs=FS_EXPECTED,
                        signal_for_alpha=seg_l,
                    )
                    cm_record = {
                        "enabled": True,
                        "noise_center_sec": float(nccm),
                        "snr_target_db": float(cm_interf_cfg.snr_db),
                        "diag": diag_cm,
                    }

                cd_record = None
                if cd_interf_cfg is not None:
                    assert cd_noise_center_sec is not None
                    nccd = float(cd_noise_center_sec[i])
                    if not np.isfinite(nccd):
                        raise ValueError("Non-finite cd_noise_center_sec for enabled delayed coherent interference")
                    noise_cd = extract_centered_window(micl, fs=FS_EXPECTED, center_sec=nccd, window_sec=WINDOW_SEC).astype(
                        np.float64, copy=False
                    )
                    (sig_l_mix, sig_r_mix), diag_cd = add_delayed_coherent_interference(
                        sig_l_mix,
                        sig_r_mix,
                        noise_cd,
                        cfg=cd_interf_cfg,
                        fs=FS_EXPECTED,
                        signal_for_alpha=seg_l,
                    )
                    cd_record = {
                        "enabled": True,
                        "noise_center_sec": float(nccd),
                        "snr_target_db": float(cd_interf_cfg.snr_db),
                        "delay_ms": float(cd_interf_cfg.delay_ms),
                        "target_delayed": str(cd_interf_cfg.target_delayed),
                        "diag": diag_cd,
                    }

                seg_l, diag_l = apply_mic_corruption(
                    seg_l,
                    noise_l,
                    cfg=corruption_cfg,
                    fs=FS_EXPECTED,
                    signal_for_alpha=seg_l,
                    signal_for_mix=sig_l_mix,
                )
                seg_r, diag_r = apply_mic_corruption(
                    seg_r,
                    noise_r,
                    cfg=corruption_cfg,
                    fs=FS_EXPECTED,
                    signal_for_alpha=seg_r,
                    signal_for_mix=sig_r_mix,
                )
                corruption_record = {
                    "enabled": True,
                    "noise_center_sec_L": ncl,
                    "noise_center_sec_R": ncr,
                    "occlusion": occlusion_cfg,
                    "common_mode_interference": cm_record,
                    "delayed_coherent_interference": cd_record,
                    "micl": diag_l,
                    "micr": diag_r,
                }
                snr_ach_test_L.append(float(diag_l["snr_achieved_db_preclip"]))
                snr_ach_test_R.append(float(diag_r["snr_achieved_db_preclip"]))
                clip_frac_test_L.append(float(diag_l["clip_frac"]))
                clip_frac_test_R.append(float(diag_r["clip_frac"]))

            guided_tau_sec = float(truth_ref["tau_ref_ms"]) / 1000.0

            X = np.fft.rfft(seg_l, n_fft)
            Y = np.fft.rfft(seg_r, n_fft)
            R = X * np.conj(Y)
            R_phat = R / (np.abs(R) + RIDGE_EPS)

            # Baseline (full analysis band)
            cc_base = ccwin_from_spectrum((R_phat * analysis_mask_fft).astype(np.complex128, copy=False), n_fft=n_fft, max_shift=max_shift)
            base_tp = estimate_tau_psr_from_ccwin(
                cc_base,
                fs=FS_EXPECTED,
                max_shift=max_shift,
                guided_tau_sec=guided_tau_sec,
                guided_radius_sec=guided_radius_sec,
                psr_exclude_samples=int(PSR_EXCLUDE_SAMPLES),
            )
            base_theta = tau_to_theta_deg(base_tp.tau_sec)
            err_ref_b = abs(base_theta - float(truth_ref["theta_ref_deg"]))
            err_geo_b = abs(base_theta - float(geom["theta_true_deg"]))

            # Student predicted bands
            obs_kb = observations[i]
            selected = predict_sequence(
                obs_kb,
                centroids=centroids,
                action_valid=action_valid,
                forbidden_mask=forbid,
                stop_action_id=stop_action_id,
                stop_preference_delta=stop_preference_delta,
            )
            if stop_to_teacher_valid_len:
                k_stop = int(valid_len[i])
                if k_stop < 0:
                    raise ValueError(f"Invalid teacher valid_len at i={i}: {k_stop}")
                selected = list(selected[:k_stop])

            if inference_psr_stop_enable and selected:
                if inference_psr_stop_mode == "monotone":
                    spec_sum = np.zeros_like(R_phat, dtype=np.complex128)
                    kept: list[int] = []
                    prev_psr = -1e9
                    for b in selected:
                        b = int(b)
                        bm = band_masks_fft[b].astype(np.float64, copy=False)
                        spec_sum = spec_sum + (R_phat * bm).astype(np.complex128, copy=False)
                        k_sel = float(len(kept) + 1)
                        spec_mean = spec_sum / k_sel
                        cc_tmp = ccwin_from_spectrum(
                            spec_mean.astype(np.complex128, copy=False), n_fft=n_fft, max_shift=max_shift
                        )
                        tp_tmp = estimate_tau_psr_from_ccwin(
                            cc_tmp,
                            fs=FS_EXPECTED,
                            max_shift=max_shift,
                            guided_tau_sec=guided_tau_sec,
                            guided_radius_sec=guided_radius_sec,
                            psr_exclude_samples=int(PSR_EXCLUDE_SAMPLES),
                        )
                        if float(tp_tmp.psr_db) < float(prev_psr) + float(inference_psr_stop_min_gain_db):
                            break
                        kept.append(int(b))
                        prev_psr = float(tp_tmp.psr_db)
                    selected = kept
                elif inference_psr_stop_mode == "best_prefix":
                    # Choose the prefix length that maximizes guided PSR.
                    spec_sum = np.zeros_like(R_phat, dtype=np.complex128)
                    prefixes: list[list[int]] = []
                    psr_vals: list[float] = []
                    kept: list[int] = []
                    for b in selected:
                        b = int(b)
                        bm = band_masks_fft[b].astype(np.float64, copy=False)
                        spec_sum = spec_sum + (R_phat * bm).astype(np.complex128, copy=False)
                        k_sel = float(len(kept) + 1)
                        spec_mean = spec_sum / k_sel
                        cc_tmp = ccwin_from_spectrum(
                            spec_mean.astype(np.complex128, copy=False), n_fft=n_fft, max_shift=max_shift
                        )
                        tp_tmp = estimate_tau_psr_from_ccwin(
                            cc_tmp,
                            fs=FS_EXPECTED,
                            max_shift=max_shift,
                            guided_tau_sec=guided_tau_sec,
                            guided_radius_sec=guided_radius_sec,
                            psr_exclude_samples=int(PSR_EXCLUDE_SAMPLES),
                        )
                        kept.append(int(b))
                        prefixes.append(list(kept))
                        psr_vals.append(float(tp_tmp.psr_db))
                    if not psr_vals:
                        selected = []
                    else:
                        best_idx = int(np.argmax(np.asarray(psr_vals, dtype=np.float64)))
                        if best_idx > 0 and (psr_vals[best_idx] - psr_vals[0]) < float(inference_psr_stop_min_gain_db):
                            selected = prefixes[0]
                        else:
                            selected = prefixes[best_idx]
                else:
                    raise ValueError(f"Unknown inference_psr_stop_mode: {inference_psr_stop_mode}")

            if selected:
                mask_sel = np.mean(band_masks_fft[np.asarray(selected, dtype=np.int32)], axis=0)
                cc_stud = ccwin_from_spectrum(
                    (R_phat * mask_sel).astype(np.complex128, copy=False), n_fft=n_fft, max_shift=max_shift
                )
            else:
                cc_stud = cc_base
            stud_tp = estimate_tau_psr_from_ccwin(
                cc_stud,
                fs=FS_EXPECTED,
                max_shift=max_shift,
                guided_tau_sec=guided_tau_sec,
                guided_radius_sec=guided_radius_sec,
                psr_exclude_samples=int(PSR_EXCLUDE_SAMPLES),
            )
            stud_theta = tau_to_theta_deg(stud_tp.tau_sec)
            err_ref_s = abs(stud_theta - float(truth_ref["theta_ref_deg"]))
            err_geo_s = abs(stud_theta - float(geom["theta_true_deg"]))

            base_err_ref.append(float(err_ref_b))
            stud_err_ref.append(float(err_ref_s))
            base_err_geo.append(float(err_geo_b))
            stud_err_geo.append(float(err_geo_s))

            record = {
                "speaker_id": spk,
                "center_sec": center,
                "corruption": corruption_record,
                "truth_reference": truth_ref,
                "geometry_truth": geom,
                "baseline": {
                    "tau_ms": float(base_tp.tau_sec * 1000.0),
                    "psr_db": float(base_tp.psr_db),
                    "theta_deg": float(base_theta),
                    "theta_error_ref_deg": float(err_ref_b),
                    "theta_error_geo_deg": float(err_geo_b),
                },
                "student": {
                    "tau_ms": float(stud_tp.tau_sec * 1000.0),
                    "psr_db": float(stud_tp.psr_db),
                    "theta_deg": float(stud_theta),
                    "theta_error_ref_deg": float(err_ref_s),
                    "theta_error_geo_deg": float(err_geo_s),
                    "selected_bands": selected,
                    "valid_len": int(len(selected)),
                },
            }
            f_jsonl.write(json.dumps(record) + "\n")

    # Manifest
    manifest = {
        "generated": datetime.now().isoformat(),
        "data_root": str(data_root),
        "files": [{"rel_path": str(p.relative_to(data_root)), "sha256": sha256_file(p)} for p in sorted(set(wav_files))],
        "traj_path": str(traj_path),
    }
    write_json(out_dir / "manifest.json", manifest)

    base_ref = np.asarray(base_err_ref, dtype=np.float64)
    stud_ref = np.asarray(stud_err_ref, dtype=np.float64)
    base_geo = np.asarray(base_err_geo, dtype=np.float64)
    stud_geo = np.asarray(stud_err_geo, dtype=np.float64)

    # Win condition (plan-locked)
    p95_improvement_frac = float((np.percentile(base_ref, 95) - np.percentile(stud_ref, 95)) / (np.percentile(base_ref, 95) + 1e-12))
    fail_improvement_frac = float((fail_rate(base_ref, threshold_deg=5.0) - fail_rate(stud_ref, threshold_deg=5.0)) / (fail_rate(base_ref, threshold_deg=5.0) + 1e-12))
    median_worsening_frac = float((np.median(stud_ref) - np.median(base_ref)) / (np.median(base_ref) + 1e-12))

    win = {
        "p95_improvement_frac_ge_0p15": bool(p95_improvement_frac >= 0.15),
        "fail_rate_improvement_frac_ge_0p20": bool(fail_improvement_frac >= 0.20),
        "median_not_worse_frac_le_0p05": bool(median_worsening_frac <= 0.05),
        "overall_pass": bool((p95_improvement_frac >= 0.15) and (fail_improvement_frac >= 0.20) and (median_worsening_frac <= 0.05)),
        "computed": {
            "p95_improvement_frac": p95_improvement_frac,
            "fail_rate_improvement_frac": fail_improvement_frac,
            "median_worsening_frac": median_worsening_frac,
        },
    }

    summary = {
        "generated": datetime.now().isoformat(),
        "run_dir": str(out_dir),
        "acceptance": win,
        "stop_to_teacher_valid_len": bool(stop_to_teacher_valid_len),
        "corruption": {
            "use_traj_corruption": bool(use_traj_corruption),
            "config": None
            if corruption_cfg is None
            else {
                "snr_db": float(corruption_cfg.snr_db),
                "band_lo_hz": float(corruption_cfg.band_lo_hz),
                "band_hi_hz": float(corruption_cfg.band_hi_hz),
                "preclip_gain": float(corruption_cfg.preclip_gain),
                "clip_limit": float(corruption_cfg.clip_limit),
                "seed": int(corruption_cfg.seed),
            },
            "snr_achieved_db_preclip_test": {
                "micl_median": float(np.median(np.asarray(snr_ach_test_L, dtype=np.float64))) if snr_ach_test_L else None,
                "micr_median": float(np.median(np.asarray(snr_ach_test_R, dtype=np.float64))) if snr_ach_test_R else None,
            },
            "clip_frac_test": {
                "micl_mean": float(np.mean(np.asarray(clip_frac_test_L, dtype=np.float64))) if clip_frac_test_L else None,
                "micr_mean": float(np.mean(np.asarray(clip_frac_test_R, dtype=np.float64))) if clip_frac_test_R else None,
            },
        },
        "pooled": {
            "baseline": {
                "theta_error_ref_deg": summarize_errors(base_ref),
                "theta_error_geo_deg": summarize_errors(base_geo),
                "fail_rate_ref_gt5deg": fail_rate(base_ref, threshold_deg=5.0),
            },
            "student": {
                "theta_error_ref_deg": summarize_errors(stud_ref),
                "theta_error_geo_deg": summarize_errors(stud_geo),
                "fail_rate_ref_gt5deg": fail_rate(stud_ref, threshold_deg=5.0),
            },
        },
    }
    write_json(out_dir / "summary.json", summary)

    # Report
    lines = []
    lines.append("# Band-DTmin Student Report")
    lines.append("")
    lines.append(f"Generated: {summary['generated']}")
    lines.append(f"Run dir: {out_dir}")
    lines.append("")
    lines.append("## Success Criteria (pooled test windows, vs chirp reference)")
    lines.append("")
    lines.append(f"- p95(theta_error_ref) improvement: {win['computed']['p95_improvement_frac']:.3f} (>= 0.150)")
    lines.append(f"- fail_rate_ref(theta_error_ref>5°) improvement: {win['computed']['fail_rate_improvement_frac']:.3f} (>= 0.200)")
    lines.append(f"- median(theta_error_ref) worsening: {win['computed']['median_worsening_frac']:.3f} (<= 0.050)")
    lines.append(f"- OVERALL: {'PASS' if win['overall_pass'] else 'FAIL'}")
    lines.append("")
    lines.append("## Test Metrics (vs chirp reference)")
    lines.append("")
    lines.append("| Method | count | median | p90 | p95 | fail_rate(>5°) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    b = summary["pooled"]["baseline"]["theta_error_ref_deg"]
    s = summary["pooled"]["student"]["theta_error_ref_deg"]
    lines.append(
        f"| baseline | {b['count']} | {b['median']:.3f} | {b['p90']:.3f} | {b['p95']:.3f} | {summary['pooled']['baseline']['fail_rate_ref_gt5deg']:.3f} |"
    )
    lines.append(
        f"| student | {s['count']} | {s['median']:.3f} | {s['p90']:.3f} | {s['p95']:.3f} | {summary['pooled']['student']['fail_rate_ref_gt5deg']:.3f} |"
    )
    lines.append("")
    (out_dir / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info("Done. Results: %s", out_dir)
    logger.info("Acceptance overall_pass=%s", bool(win["overall_pass"]))


if __name__ == "__main__":
    main()
