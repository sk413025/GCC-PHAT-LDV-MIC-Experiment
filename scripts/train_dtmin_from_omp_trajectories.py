#!/usr/bin/env python
"""
Train a lightweight DTmin lag policy from OMP teacher trajectories.

Model:
- Step-conditioned nearest-centroid classifier over lag actions.
- One centroid per action index per step.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DTmin policy from OMP trajectories")
    parser.add_argument("--traj_path", type=str, required=True, help="Path to lag_trajectories.npz")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=3, help="Policy horizon (K steps)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    payload = np.load(args.traj_path, allow_pickle=False)
    observations = payload["observations"]  # (N, K, n_lags)
    actions = payload["actions"]  # (N, K)
    valid_len = payload["valid_len"]  # (N,)
    n_lags = int(payload["n_lags"][0])
    max_k_data = int(payload["max_k"][0])
    horizon = int(args.horizon)
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if horizon > max_k_data:
        raise ValueError(f"horizon={horizon} exceeds trajectory max_k={max_k_data}")
    if observations.shape[1] < horizon:
        raise ValueError(
            f"Trajectory K dimension {observations.shape[1]} smaller than horizon={horizon}"
        )

    centroids = np.zeros((horizon, n_lags, n_lags), dtype=np.float32)
    action_valid = np.zeros((horizon, n_lags), dtype=bool)
    action_counts = np.zeros((horizon, n_lags), dtype=np.int32)

    for step_idx in range(horizon):
        step_mask = valid_len > step_idx
        obs_step = observations[step_mask, step_idx, :]
        act_step = actions[step_mask, step_idx]
        if obs_step.shape[0] == 0:
            continue
        for lag_idx in range(n_lags):
            lag_mask = act_step == lag_idx
            count = int(np.sum(lag_mask))
            action_counts[step_idx, lag_idx] = count
            if count > 0:
                centroids[step_idx, lag_idx, :] = obs_step[lag_mask].mean(axis=0)
                action_valid[step_idx, lag_idx] = True

    model_path = out_dir / "model_dtmin_policy_k3.npz"
    metadata = {
        "generated": datetime.now().isoformat(),
        "traj_path": str(Path(args.traj_path)),
        "n_samples": int(observations.shape[0]),
        "horizon": int(horizon),
        "n_lags": int(n_lags),
        "model_type": "nearest_centroid_stepwise",
    }
    np.savez_compressed(
        model_path,
        centroids=centroids,
        action_valid=action_valid,
        action_counts=action_counts,
        metadata_json=json.dumps(metadata),
    )

    summary = {
        **metadata,
        "non_empty_actions_per_step": [int(np.sum(action_valid[k])) for k in range(horizon)],
        "model_path": str(model_path),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved model: %s", model_path)
    logger.info("Non-empty actions per step: %s", summary["non_empty_actions_per_step"])


if __name__ == "__main__":
    main()

