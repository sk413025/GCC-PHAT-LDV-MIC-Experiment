#!/usr/bin/env python
"""
Generate OMP teacher trajectories (K-step lag selections) for DTmin training.

Output:
- lag_trajectories.npz
- trajectory_summary.json
- subset_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import stft

import stage4_doa_ldv_vs_mic_comparison as stage4

logger = logging.getLogger(__name__)


def parse_speakers(s: str) -> list[str]:
    out = [x.strip() for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError("speakers list is empty")
    return out


def list_triplet_files(data_root: Path, speaker: str) -> tuple[Path, Path, Path]:
    sp_dir = data_root / speaker
    ldv_files = list(sp_dir.glob("*LDV*.wav"))
    mic_l_files = list(sp_dir.glob("*LEFT*.wav"))
    mic_r_files = list(sp_dir.glob("*RIGHT*.wav"))
    if not ldv_files or not mic_l_files or not mic_r_files:
        raise FileNotFoundError(f"Missing wav triplet in {sp_dir}")
    return ldv_files[0], mic_l_files[0], mic_r_files[0]


def compute_fixed_segment_centers(duration_s: float, config: dict, n_segments: int) -> list[float]:
    slice_samples = int(float(config["analysis_slice_sec"]) * int(config["fs"]))
    half_slice_sec = (slice_samples / int(config["fs"])) / 2.0
    min_center = half_slice_sec
    max_center = max(min_center, duration_s - half_slice_sec)
    start_center = max(min_center, float(config["segment_offset_sec"]))
    spacing = float(config["segment_spacing_sec"])
    if n_segments <= 1:
        return [min(max_center, max(min_center, duration_s / 2.0))]
    centers = [start_center + i * spacing for i in range(n_segments)]
    centers = [float(t) for t in centers if t <= max_center]
    return centers


def collect_omp_trajectory_for_freq(
    Dict_f: np.ndarray,
    Y_f: np.ndarray,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    # Dict_f shape: (n_lags, tw), Y_f: (tw,)
    n_lags, _ = Dict_f.shape
    D = Dict_f.T
    D_norms = np.linalg.norm(np.abs(D), axis=0, keepdims=True) + 1e-10
    D_normalized = D / D_norms
    residual = Y_f.copy()
    selected_lags: list[int] = []

    observations = np.zeros((max_k, n_lags), dtype=np.float32)
    actions = np.full((max_k,), -1, dtype=np.int32)
    valid_len = 0

    for step_idx in range(max_k):
        corrs = np.abs(D_normalized.conj().T @ residual).astype(np.float32, copy=False)
        for lag in selected_lags:
            corrs[lag] = -np.inf
        best_lag = int(np.argmax(corrs))
        if best_lag in selected_lags:
            break
        selected_lags.append(best_lag)
        observations[step_idx] = corrs
        actions[step_idx] = best_lag
        valid_len += 1

        A = D[:, selected_lags]
        coeffs, _, _, _ = np.linalg.lstsq(A, Y_f, rcond=None)
        residual = Y_f - (A @ coeffs)
    return observations, actions, valid_len


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OMP teacher trajectories for DTmin")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--speakers", type=str, required=True, help="Comma-separated speaker ids")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_k", type=int, default=3)
    parser.add_argument("--n_segments", type=int, default=5)
    parser.add_argument("--analysis_slice_sec", type=float, default=5.0)
    parser.add_argument("--segment_spacing_sec", type=float, default=50.0)
    parser.add_argument("--segment_offset_sec", type=float, default=100.0)
    parser.add_argument("--ldv_prealign", type=str, choices=["none", "gcc_phat"], default="gcc_phat")
    parser.add_argument("--freq_min", type=float, default=100.0)
    parser.add_argument("--freq_max", type=float, default=8000.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    config = stage4.DEFAULT_CONFIG.copy()
    config["max_k"] = int(args.max_k)
    config["analysis_slice_sec"] = float(args.analysis_slice_sec)
    config["segment_spacing_sec"] = float(args.segment_spacing_sec)
    config["segment_offset_sec"] = float(args.segment_offset_sec)
    config["ldv_prealign"] = str(args.ldv_prealign)
    config["freq_min"] = float(args.freq_min)
    config["freq_max"] = float(args.freq_max)

    speakers = parse_speakers(args.speakers)
    data_root = Path(args.data_root)

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_valid_len: list[int] = []
    meta_rows: list[dict] = []
    subset_manifest: dict[str, dict] = {}

    max_lag = int(config["max_lag"])
    n_lags = 2 * max_lag + 1
    tw = int(config["tw"])
    fs = int(config["fs"])
    n_fft = int(config["n_fft"])
    hop = int(config["hop_length"])
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    freq_mask = (freqs >= float(config["freq_min"])) & (freqs <= float(config["freq_max"]))
    freq_indices = np.where(freq_mask)[0]

    logger.info("Speakers: %s", speakers)
    logger.info("Config: max_k=%d, n_lags=%d, n_freq_selected=%d", args.max_k, n_lags, len(freq_indices))

    for speaker in speakers:
        ldv_path, mic_l_path, mic_r_path = list_triplet_files(data_root, speaker)
        subset_manifest[speaker] = {
            "ldv": str(ldv_path),
            "mic_left": str(mic_l_path),
            "mic_right": str(mic_r_path),
        }
        sr_ldv, ldv_signal = stage4.load_wav(str(ldv_path))
        sr_mic_l, mic_l_signal = stage4.load_wav(str(mic_l_path))
        _sr_mic_r, _mic_r_signal = stage4.load_wav(str(mic_r_path))
        if not (sr_ldv == sr_mic_l == fs):
            raise ValueError(f"Sample rate mismatch for {speaker}: {sr_ldv}, {sr_mic_l}, expected {fs}")

        duration_s = min(len(ldv_signal), len(mic_l_signal)) / fs
        centers = compute_fixed_segment_centers(duration_s, config, int(args.n_segments))
        logger.info("%s: %d segments", speaker, len(centers))

        for seg_idx, center_sec in enumerate(centers):
            center_sample = int(center_sec * fs)
            slice_samples = int(float(config["analysis_slice_sec"]) * fs)
            slice_start, slice_end = stage4.extract_centered_slice(
                [ldv_signal, mic_l_signal],
                center_sample=center_sample,
                slice_samples=slice_samples,
            )
            ldv_slice = ldv_signal[slice_start:slice_end]
            mic_l_slice = mic_l_signal[slice_start:slice_end]
            desired_center_in_slice = int(center_sample - slice_start)

            if str(config["ldv_prealign"]).lower() == "gcc_phat":
                eval_window_samples = int(float(config["eval_window_sec"]) * fs)
                eval_center = int(np.clip(desired_center_in_slice, 0, max(0, len(ldv_slice) - 1)))
                t_start = max(0, eval_center - eval_window_samples // 2)
                t_end = min(len(ldv_slice), t_start + eval_window_samples)
                if t_end > t_start:
                    tau_sec, _ = stage4.gcc_phat_full_analysis(
                        ldv_slice[t_start:t_end].astype(np.float64, copy=False),
                        mic_l_slice[t_start:t_end].astype(np.float64, copy=False),
                        fs,
                        max_tau=float(config["gcc_max_lag_ms"]) / 1000.0,
                        bandpass=None,
                        psr_exclude_samples=int(config["psr_exclude_samples"]),
                    )
                    ldv_slice = stage4.apply_fractional_delay_fd(ldv_slice, fs, -float(tau_sec))

            _, _, Zxx_ldv = stft(
                ldv_slice,
                fs=fs,
                nperseg=n_fft,
                noverlap=n_fft - hop,
                window="hann",
            )
            _, _, Zxx_mic_l = stft(
                mic_l_slice,
                fs=fs,
                nperseg=n_fft,
                noverlap=n_fft - hop,
                window="hann",
            )

            n_time = min(Zxx_ldv.shape[1], Zxx_mic_l.shape[1])
            desired_frame = int(round((desired_center_in_slice - n_fft // 2) / hop))
            start_t = desired_frame - tw // 2
            start_t = max(start_t, max_lag + 1)
            start_t = min(start_t, n_time - tw - max_lag - 1)
            if start_t < max_lag + 1:
                logger.warning("Skipping short-support segment: speaker=%s seg=%d", speaker, seg_idx)
                continue

            Dict_full = stage4.build_lagged_dictionary(Zxx_ldv, max_lag, tw, start_t)
            Dict_selected = Dict_full[freq_mask, :, :]
            for local_f_idx, global_f_idx in enumerate(freq_indices):
                Dict_f = Dict_selected[local_f_idx]
                Y_f = Zxx_mic_l[global_f_idx, start_t : start_t + tw]
                Dict_norm, _ = stage4.normalize_per_freq_maxabs(Dict_f)
                Y_norm, _ = stage4.normalize_per_freq_maxabs(Y_f)
                obs, actions, valid_len = collect_omp_trajectory_for_freq(Dict_norm, Y_norm, int(args.max_k))
                all_obs.append(obs)
                all_actions.append(actions)
                all_valid_len.append(valid_len)
                meta_rows.append(
                    {
                        "speaker": speaker,
                        "segment_index": int(seg_idx),
                        "center_sec": float(center_sec),
                        "freq_bin": int(global_f_idx),
                        "freq_hz": float(freqs[global_f_idx]),
                    }
                )

    if not all_obs:
        raise RuntimeError("No trajectories generated")

    observations = np.stack(all_obs, axis=0).astype(np.float32, copy=False)
    actions = np.stack(all_actions, axis=0).astype(np.int32, copy=False)
    valid_len = np.asarray(all_valid_len, dtype=np.int32)

    npz_path = out_dir / "lag_trajectories.npz"
    np.savez_compressed(
        npz_path,
        observations=observations,
        actions=actions,
        valid_len=valid_len,
        n_lags=np.asarray([n_lags], dtype=np.int32),
        max_k=np.asarray([int(args.max_k)], dtype=np.int32),
    )

    (out_dir / "trajectory_metadata.json").write_text(json.dumps(meta_rows, indent=2), encoding="utf-8")
    (out_dir / "subset_manifest.json").write_text(json.dumps(subset_manifest, indent=2), encoding="utf-8")

    action_hist = np.bincount(actions[actions >= 0], minlength=n_lags).tolist()
    summary = {
        "generated": datetime.now().isoformat(),
        "data_root": str(data_root),
        "speakers": speakers,
        "n_samples": int(observations.shape[0]),
        "max_k": int(args.max_k),
        "n_lags": int(n_lags),
        "freq_min": float(config["freq_min"]),
        "freq_max": float(config["freq_max"]),
        "ldv_prealign": str(config["ldv_prealign"]),
        "npz_path": str(npz_path),
        "action_histogram": action_hist,
    }
    (out_dir / "trajectory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved trajectories: %s", npz_path)
    logger.info("n_samples=%d, max_k=%d, n_lags=%d", observations.shape[0], args.max_k, n_lags)


if __name__ == "__main__":
    main()

