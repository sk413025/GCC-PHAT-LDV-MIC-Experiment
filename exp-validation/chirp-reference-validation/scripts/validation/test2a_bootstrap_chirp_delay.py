#!/usr/bin/env python3
"""
Test 2A: Bootstrap Stability Test for Chirp Calibration Delay

Bootstraps LDV delay estimates from chirp calibration events and reports
mean/std/CI/CV stability per position.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Obs:
    pair: Tuple[str, str]
    tau_geom_ms: float
    tau_meas_ms: float
    psr_db: float


def solve_sensor_delays_ms(observations: List[Obs]) -> Dict[str, float]:
    """Weighted least squares for delta_R and delta_LDV (MicL fixed to 0)."""
    if not observations:
        return {"LEFT-MIC": 0.0, "RIGHT-MIC": 0.0, "LDV": 0.0}

    A_rows = []
    b = []
    w = []

    for obs in observations:
        r = float(obs.tau_meas_ms - obs.tau_geom_ms)
        weight = max(0.1, float(obs.psr_db))
        a, bb = obs.pair

        if (a, bb) == ("LEFT-MIC", "RIGHT-MIC"):
            A_rows.append([-1.0, 0.0])
            b.append(r)
            w.append(weight)
        elif (a, bb) == ("LDV", "LEFT-MIC"):
            A_rows.append([0.0, 1.0])
            b.append(r)
            w.append(weight)
        elif (a, bb) == ("LDV", "RIGHT-MIC"):
            A_rows.append([-1.0, 1.0])
            b.append(r)
            w.append(weight)

    if not A_rows:
        return {"LEFT-MIC": 0.0, "RIGHT-MIC": 0.0, "LDV": 0.0}

    A = np.asarray(A_rows, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    W = np.sqrt(w)[:, None]
    Aw = A * W
    bw = b * W[:, 0]

    u, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    delta_R, delta_LDV = [float(v) for v in u]

    return {"LEFT-MIC": 0.0, "RIGHT-MIC": delta_R, "LDV": delta_LDV}


def load_events(summary_path: Path, positions: List[str]) -> Dict[str, List[dict]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    pos_data = data.get("positions", {})
    out = {}
    for pos in positions:
        entry = pos_data.get(pos, {})
        events = entry.get("events", [])
        out[pos] = events
    return out


def events_to_observations(events: List[dict]) -> List[Obs]:
    obs = []
    for ev in events:
        pairs = ev.get("pairs", {})
        for key in ["LEFT-MIC_RIGHT-MIC", "LDV_LEFT-MIC", "LDV_RIGHT-MIC"]:
            p = pairs.get(key)
            if p is None:
                continue
            if key == "LEFT-MIC_RIGHT-MIC":
                pair = ("LEFT-MIC", "RIGHT-MIC")
            elif key == "LDV_LEFT-MIC":
                pair = ("LDV", "LEFT-MIC")
            else:
                pair = ("LDV", "RIGHT-MIC")
            obs.append(
                Obs(
                    pair=pair,
                    tau_geom_ms=float(p.get("tau_geom_ms")),
                    tau_meas_ms=float(p.get("tau_meas_ms")),
                    psr_db=float(p.get("psr_db")),
                )
            )
    return obs


def bootstrap_delays(events: List[dict], iters: int, subsample: float, rng: np.random.Generator) -> List[float]:
    if not events:
        return []

    n = len(events)
    k = max(1, int(np.ceil(subsample * n)))
    delays = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=k)
        sample = [events[i] for i in idx]
        obs = events_to_observations(sample)
        delays.append(solve_sensor_delays_ms(obs)["LDV"])
    return delays


def compute_stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {
            "n": 0,
            "mean_ms": None,
            "std_ms": None,
            "p2_5_ms": None,
            "p97_5_ms": None,
            "cv_percent": None,
        }
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    cv = None if mean == 0 else float(abs(std / mean) * 100.0)
    return {
        "n": int(arr.size),
        "mean_ms": mean,
        "std_ms": std,
        "p2_5_ms": float(np.percentile(arr, 2.5)),
        "p97_5_ms": float(np.percentile(arr, 97.5)),
        "cv_percent": cv,
    }


def classify_stability(stats: Dict[str, float]) -> str:
    std = stats.get("std_ms")
    cv = stats.get("cv_percent")
    if std is None or cv is None:
        return "insufficient"
    if std < 0.10 and cv < 10:
        return "stable"
    if std < 0.20 and cv < 20:
        return "marginal"
    return "unstable"


def plot_distributions(results: Dict[str, Dict[str, List[float]]], out_path: Path) -> None:
    datasets = list(results.keys())
    positions = list(next(iter(results.values())).keys()) if results else []

    nrows = len(datasets)
    ncols = len(positions)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for r, ds in enumerate(datasets):
        for c, pos in enumerate(positions):
            ax = axes[r][c]
            vals = results[ds][pos]
            if vals:
                ax.hist(vals, bins=30, alpha=0.8, color="#4C78A8")
            ax.set_title(f"{ds} {pos}")
            ax.set_xlabel("LDV delay (ms)")
            ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ci(summary: Dict[str, Dict[str, Dict]], out_path: Path) -> None:
    datasets = list(summary.keys())
    positions = list(next(iter(summary.values())).keys()) if summary else []

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(positions))
    width = 0.35

    for i, ds in enumerate(datasets):
        means = []
        yerr_low = []
        yerr_high = []
        for pos in positions:
            st = summary[ds][pos]
            means.append(st.get("mean_ms", np.nan))
            p2 = st.get("p2_5_ms")
            p97 = st.get("p97_5_ms")
            mean = st.get("mean_ms")
            if p2 is None or p97 is None or mean is None:
                yerr_low.append(0.0)
                yerr_high.append(0.0)
            else:
                yerr_low.append(mean - p2)
                yerr_high.append(p97 - mean)
        offset = (i - 0.5) * width
        ax.errorbar(x + offset, means, yerr=[yerr_low, yerr_high], fmt="o", capsize=4, label=ds)

    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_ylabel("LDV delay (ms)")
    ax.set_title("Bootstrap mean and 95% CI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test 2A: Bootstrap stability for chirp delay")
    parser.add_argument("--chirp_summary", type=Path, default=Path("dataset/chirp/results/chirp_calibration_summary.json"))
    parser.add_argument("--chirp2_summary", type=Path, default=Path("dataset/chirp_2/results/chirp_calibration_summary.json"))
    parser.add_argument("--positions", nargs="*", default=["+0.0", "+0.4", "+0.8"])
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--subsample", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("results") / f"test2a_bootstrap_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    datasets = {
        "chirp": args.chirp_summary,
        "chirp_2": args.chirp2_summary,
    }

    results = {}
    summary = {}

    for name, path in datasets.items():
        events_by_pos = load_events(path, args.positions)
        results[name] = {}
        summary[name] = {}

        for pos in args.positions:
            events_all = events_by_pos[pos]
            events_pass = [e for e in events_all if e.get("event_pass")]

            # If no pass events, fall back to all events but flag it
            use_events = events_pass if events_pass else events_all

            delays = bootstrap_delays(use_events, args.iters, args.subsample, rng)
            results[name][pos] = delays

            stats = compute_stats(delays)
            stats["n_events_total"] = len(events_all)
            stats["n_events_used"] = len(use_events)
            stats["used_pass_events_only"] = bool(events_pass)
            stats["subsample_n"] = max(1, int(np.ceil(args.subsample * max(1, len(use_events)))))
            stats["stability"] = classify_stability(stats)
            summary[name][pos] = stats

    # Save JSON
    (out_dir / "test2a_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "test2a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plots
    plot_distributions(results, out_dir / "test2a_distributions.png")
    plot_ci(summary, out_dir / "test2a_ci_plot.png")

    print(f"Wrote results to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
