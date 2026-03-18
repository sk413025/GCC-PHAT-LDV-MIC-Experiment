#!/usr/bin/env python3
"""
Pairing and scoring family sweep for 0223 LDV-MIC delta-tau recovery.

This script focuses on pair selection logic using the baseline LDV diff
preprocessing and tests stronger pair scoring and sub-band voting strategies.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import filter_sweep_0223_delta_tau as base
import peak_pair_sweep_0223_delta_tau as pair


def score_single_band(
    vl: dict[str, float],
    vr: dict[str, float],
    *,
    strategy: str,
    delta_scale_ms: float,
    mean_tau_center_ms: float,
    mean_tau_scale_ms: float,
) -> float:
    dt = vr["tau_ms"] - vl["tau_ms"]
    mean_tau = 0.5 * (vl["tau_ms"] + vr["tau_ms"])
    prod = vl["amp"] * vr["amp"]
    prom_prod = max(vl["prom"], 1e-6) * max(vr["prom"], 1e-6)
    balance = min(vl["amp"], vr["amp"]) / max(max(vl["amp"], vr["amp"]), 1e-9)
    delta_quad = math.exp(-((dt / max(delta_scale_ms, 1e-6)) ** 2))
    mean_tau_penalty = math.exp(-(((mean_tau - mean_tau_center_ms) / max(mean_tau_scale_ms, 1e-6)) ** 2))

    if strategy == "baseline":
        return prod * delta_quad * mean_tau_penalty
    if strategy == "prominence_weighted":
        return prod * (1.0 + prom_prod) * delta_quad * mean_tau_penalty
    if strategy == "balanced":
        return prod * balance * delta_quad * mean_tau_penalty
    if strategy == "tight_mean_tau":
        tight_penalty = math.exp(-(((mean_tau - mean_tau_center_ms) / 0.35) ** 2))
        return prod * delta_quad * tight_penalty
    if strategy == "wide_mean_tau":
        wide_penalty = math.exp(-(((mean_tau - mean_tau_center_ms) / 0.60) ** 2))
        return prod * delta_quad * wide_penalty
    raise ValueError(f"Unknown strategy: {strategy}")


def select_for_band(
    ldv: np.ndarray,
    mic_l: np.ndarray,
    mic_r: np.ndarray,
    *,
    bandpass: tuple[float, float],
    strategy: str,
    delta_limit_ms: float,
    delta_scale_ms: float,
    mean_tau_center_ms: float,
    mean_tau_scale_ms: float,
) -> dict[str, Any] | None:
    lag_vl, cc_vl = pair.gcc_curve(ldv, mic_l, 48000, max_lag_ms=10.0, bandpass=bandpass)
    lag_vr, cc_vr = pair.gcc_curve(ldv, mic_r, 48000, max_lag_ms=10.0, bandpass=bandpass)
    cand_vl = pair.extract_candidates(lag_vl, cc_vl, lag_min_ms=4.4, lag_max_ms=6.5, top_k=8)
    cand_vr = pair.extract_candidates(lag_vr, cc_vr, lag_min_ms=4.4, lag_max_ms=6.5, top_k=8)
    best_row = None
    for i, vl in enumerate(cand_vl, start=1):
        for j, vr in enumerate(cand_vr, start=1):
            dt = vr["tau_ms"] - vl["tau_ms"]
            if abs(dt) > delta_limit_ms:
                continue
            score = score_single_band(
                vl,
                vr,
                strategy=strategy,
                delta_scale_ms=delta_scale_ms,
                mean_tau_center_ms=mean_tau_center_ms,
                mean_tau_scale_ms=mean_tau_scale_ms,
            )
            row = {
                "vl_rank": i,
                "vr_rank": j,
                "tau_vl_ms": float(vl["tau_ms"]),
                "tau_vr_ms": float(vr["tau_ms"]),
                "delta_tau_ms": float(dt),
                "score": float(score),
            }
            if best_row is None or row["score"] > best_row["score"]:
                best_row = row
    return best_row


def aggregate_pairs(pairs: list[dict[str, Any]], mode: str) -> dict[str, Any] | None:
    valid = [p for p in pairs if p is not None]
    if not valid:
        return None
    tau_vl = np.array([p["tau_vl_ms"] for p in valid], dtype=np.float64)
    tau_vr = np.array([p["tau_vr_ms"] for p in valid], dtype=np.float64)
    if mode == "median":
        agg_vl = float(np.median(tau_vl))
        agg_vr = float(np.median(tau_vr))
    elif mode == "trimmed_mean":
        if len(valid) >= 3:
            agg_vl = float(np.mean(np.sort(tau_vl)[1:-1]))
            agg_vr = float(np.mean(np.sort(tau_vr)[1:-1]))
        else:
            agg_vl = float(np.mean(tau_vl))
            agg_vr = float(np.mean(tau_vr))
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")
    return {
        "tau_vl_ms": agg_vl,
        "tau_vr_ms": agg_vr,
        "delta_tau_ms": agg_vr - agg_vl,
        "num_votes": len(valid),
    }


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "single_band_baseline",
        "ldv_ops": ["diff"],
        "mic_ops": [],
        "bands": [(500.0, 2000.0)],
        "strategy": "baseline",
        "aggregate": "median",
        "delta_scale_ms": 0.6,
        "mean_tau_center_ms": 4.8,
        "mean_tau_scale_ms": 0.45,
    },
    {
        "name": "single_band_prominence_weighted",
        "ldv_ops": ["diff"],
        "mic_ops": [],
        "bands": [(500.0, 2000.0)],
        "strategy": "prominence_weighted",
        "aggregate": "median",
        "delta_scale_ms": 0.6,
        "mean_tau_center_ms": 4.8,
        "mean_tau_scale_ms": 0.45,
    },
    {
        "name": "single_band_balanced",
        "ldv_ops": ["diff"],
        "mic_ops": [],
        "bands": [(500.0, 2000.0)],
        "strategy": "balanced",
        "aggregate": "median",
        "delta_scale_ms": 0.6,
        "mean_tau_center_ms": 4.8,
        "mean_tau_scale_ms": 0.45,
    },
    {
        "name": "single_band_tight_mean_tau",
        "ldv_ops": ["diff"],
        "mic_ops": [],
        "bands": [(500.0, 2000.0)],
        "strategy": "tight_mean_tau",
        "aggregate": "median",
        "delta_scale_ms": 0.6,
        "mean_tau_center_ms": 4.8,
        "mean_tau_scale_ms": 0.35,
    },
    {
        "name": "single_band_wide_mean_tau",
        "ldv_ops": ["diff"],
        "mic_ops": [],
        "bands": [(500.0, 2000.0)],
        "strategy": "wide_mean_tau",
        "aggregate": "median",
        "delta_scale_ms": 0.6,
        "mean_tau_center_ms": 4.8,
        "mean_tau_scale_ms": 0.60,
    },
    {
        "name": "subband_vote_median",
        "ldv_ops": ["diff"],
        "mic_ops": [],
        "bands": [(400.0, 1400.0), (700.0, 1800.0), (1000.0, 2600.0)],
        "strategy": "baseline",
        "aggregate": "median",
        "delta_scale_ms": 0.6,
        "mean_tau_center_ms": 4.8,
        "mean_tau_scale_ms": 0.45,
    },
    {
        "name": "subband_vote_trimmed_mean",
        "ldv_ops": ["diff"],
        "mic_ops": [],
        "bands": [(400.0, 1400.0), (700.0, 1800.0), (1000.0, 2600.0)],
        "strategy": "baseline",
        "aggregate": "trimmed_mean",
        "delta_scale_ms": 0.6,
        "mean_tau_center_ms": 4.8,
        "mean_tau_scale_ms": 0.45,
    },
    {
        "name": "subband_prominence_vote",
        "ldv_ops": ["diff"],
        "mic_ops": [],
        "bands": [(500.0, 1500.0), (800.0, 2000.0), (1200.0, 2800.0)],
        "strategy": "prominence_weighted",
        "aggregate": "median",
        "delta_scale_ms": 0.6,
        "mean_tau_center_ms": 4.8,
        "mean_tau_scale_ms": 0.45,
    },
]


def evaluate_variant(case: base.CaseRef, signals: dict[str, np.ndarray], variant: dict[str, Any]) -> dict[str, Any]:
    ref = base.compute_reference(case)
    ldv = base.apply_ops(signals["ldv"], list(variant["ldv_ops"]))
    mic_l = base.apply_ops(signals["mic_l"], list(variant["mic_ops"]))
    mic_r = base.apply_ops(signals["mic_r"], list(variant["mic_ops"]))

    votes = []
    for band in variant["bands"]:
        votes.append(
            select_for_band(
                ldv,
                mic_l,
                mic_r,
                bandpass=band,
                strategy=variant["strategy"],
                delta_limit_ms=1.0,
                delta_scale_ms=float(variant["delta_scale_ms"]),
                mean_tau_center_ms=float(variant["mean_tau_center_ms"]),
                mean_tau_scale_ms=float(variant["mean_tau_scale_ms"]),
            )
        )
    agg = aggregate_pairs(votes, variant["aggregate"])
    row = {"case_id": case.case_id, "variant": variant["name"], "reference": ref, "selected": agg}
    if agg is None:
        return row
    theta_v = float(np.degrees(np.arcsin(np.clip((agg["delta_tau_ms"] / 1000.0) * base.C / 1.4, -1.0, 1.0))))
    agg["theta_v_deg"] = theta_v
    agg["delta_tau_abs_err_ms"] = abs(agg["delta_tau_ms"] - ref["delta_tau_ms"])
    agg["theta_v_abs_err_deg"] = abs(theta_v - ref["theta_v_deg"])
    agg["physical_positive_lags"] = bool(agg["tau_vl_ms"] > 0.0 and agg["tau_vr_ms"] > 0.0)
    agg["physical_small_delta"] = bool(abs(agg["delta_tau_ms"]) <= 1.0)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    summary = []
    for variant, items in grouped.items():
        valid = [it for it in items if it["selected"] is not None]
        if not valid:
            summary.append({"variant": variant, "valid_cases": 0, "physical_count": 0, "delta_tau_mae_ms": None, "theta_v_mae_deg": None})
            continue
        delta_err = np.array([it["selected"]["delta_tau_abs_err_ms"] for it in valid], dtype=np.float64)
        theta_err = np.array([it["selected"]["theta_v_abs_err_deg"] for it in valid], dtype=np.float64)
        physical_count = sum(1 for it in valid if it["selected"]["physical_positive_lags"] and it["selected"]["physical_small_delta"])
        summary.append(
            {
                "variant": variant,
                "valid_cases": len(valid),
                "physical_count": physical_count,
                "delta_tau_mae_ms": float(np.mean(delta_err)),
                "theta_v_mae_deg": float(np.mean(theta_err)),
                "max_delta_tau_abs_err_ms": float(np.max(delta_err)),
                "max_theta_v_abs_err_deg": float(np.max(theta_err)),
            }
        )
    summary.sort(
        key=lambda x: (
            1 if x["delta_tau_mae_ms"] is None else 0,
            float("inf") if x["delta_tau_mae_ms"] is None else x["delta_tau_mae_ms"],
            float("inf") if x["theta_v_mae_deg"] is None else x["theta_v_mae_deg"],
        )
    )
    return {"variants": summary}


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# 0223 Pairing Family Sweep",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        f"- Slice sec: `{payload['slice_sec']}`",
        "",
        "## Summary",
        "",
        "| variant | valid_cases | physical_count | delta_tau_mae_ms | theta_v_mae_deg | max_delta_tau_err_ms | max_theta_v_err_deg |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["variants"]:
        dt = "NA" if row["delta_tau_mae_ms"] is None else f"{row['delta_tau_mae_ms']:.3f}"
        th = "NA" if row["theta_v_mae_deg"] is None else f"{row['theta_v_mae_deg']:.3f}"
        max_dt = "NA" if row.get("max_delta_tau_abs_err_ms") is None else f"{row['max_delta_tau_abs_err_ms']:.3f}"
        max_th = "NA" if row.get("max_theta_v_abs_err_deg") is None else f"{row['max_theta_v_abs_err_deg']:.3f}"
        lines.append(f"| {row['variant']} | {row['valid_cases']} | {row['physical_count']} | {dt} | {th} | {max_dt} | {max_th} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep pair scoring families for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--slice_sec", type=float, default=5.0)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (Path(__file__).resolve().parent.parent / "results" / f"peak_pair_family_0223_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    signals_cache = {case.case_id: base.load_case_signals(case, args.data_root, args.slice_sec, 48000) for case in base.CASES}
    rows = []
    for variant in VARIANTS:
        for case in base.CASES:
            rows.append(evaluate_variant(case, signals_cache[case.case_id], variant))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "slice_sec": float(args.slice_sec),
        "variants": VARIANTS,
        "results": rows,
        "summary": summarize(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir / "report.md", payload)
    print(f"[ok] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
