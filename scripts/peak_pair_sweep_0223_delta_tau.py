#!/usr/bin/env python3
"""
Automated LDV-MIC peak-pair selection sweep for 0223 delta-tau estimation.

This script builds on filter_sweep_0223_delta_tau.py. Instead of reading only
the single global GCC-PHAT argmax, it extracts multiple positive-lag candidates
for VL/VR and scores candidate pairs under simple physical constraints.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import find_peaks

import filter_sweep_0223_delta_tau as base


def gcc_curve(
    sig1: np.ndarray,
    sig2: np.ndarray,
    fs: int,
    *,
    max_lag_ms: float,
    bandpass: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(sig1, dtype=np.float64)
    y = np.asarray(sig2, dtype=np.float64)
    if bandpass is not None:
        x = base.stage4.bandpass_filter(x, bandpass[0], bandpass[1], fs)
        y = base.stage4.bandpass_filter(y, bandpass[0], bandpass[1], fs)
    n = len(x) + len(y)
    X = fft(x, n)
    Y = fft(y, n)
    R = X * np.conj(Y)
    R = R / (np.abs(R) + 1e-10)
    cc = np.real(ifft(R))
    max_shift = int(round(max_lag_ms * fs / 1000.0))
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    lags = np.arange(-max_shift, max_shift + 1, dtype=np.int64)
    lag_ms = lags.astype(np.float64) * 1000.0 / float(fs)
    return lag_ms, np.abs(cc)


def extract_candidates(
    lag_ms: np.ndarray,
    abs_cc: np.ndarray,
    *,
    lag_min_ms: float,
    lag_max_ms: float,
    top_k: int,
) -> list[dict[str, float]]:
    mask = (lag_ms >= lag_min_ms) & (lag_ms <= lag_max_ms)
    if not np.any(mask):
        return []
    lag_win = lag_ms[mask]
    cc_win = abs_cc[mask]
    peak_idx, props = find_peaks(cc_win)
    if peak_idx.size == 0:
        peak_idx = np.array([int(np.argmax(cc_win))], dtype=np.int64)
        prominences = np.array([0.0], dtype=np.float64)
    else:
        prominences = props.get("prominences")
        if prominences is None:
            prominences = np.zeros_like(peak_idx, dtype=np.float64)
    candidates = []
    for i, idx in enumerate(peak_idx.tolist()):
        candidates.append(
            {
                "tau_ms": float(lag_win[idx]),
                "amp": float(cc_win[idx]),
                "prom": float(prominences[i]) if i < len(prominences) else 0.0,
            }
        )
    candidates.sort(key=lambda x: (x["amp"], x["prom"]), reverse=True)
    return candidates[:top_k]


def score_pair(
    vl: dict[str, float],
    vr: dict[str, float],
    strategy: str,
    delta_scale_ms: float,
    mean_tau_center_ms: float,
    mean_tau_scale_ms: float,
) -> float:
    dt = vr["tau_ms"] - vl["tau_ms"]
    mean_tau = 0.5 * (vl["tau_ms"] + vr["tau_ms"])
    prod = vl["amp"] * vr["amp"]
    prom_prod = max(vl["prom"], 1e-6) * max(vr["prom"], 1e-6)
    amp_sum = vl["amp"] + vr["amp"]
    delta_penalty = math.exp(-abs(dt) / max(delta_scale_ms, 1e-6))
    delta_quad = math.exp(-((dt / max(delta_scale_ms, 1e-6)) ** 2))
    mean_tau_penalty = math.exp(-(((mean_tau - mean_tau_center_ms) / max(mean_tau_scale_ms, 1e-6)) ** 2))
    balance = min(vl["amp"], vr["amp"]) / max(max(vl["amp"], vr["amp"]), 1e-9)

    if strategy == "amp_product":
        return prod
    if strategy == "amp_product_delta":
        return prod * delta_penalty
    if strategy == "amp_product_delta_quad":
        return prod * delta_quad
    if strategy == "amp_sum_delta":
        return amp_sum * delta_penalty
    if strategy == "prom_product_delta":
        return prod * (1.0 + prom_prod) * delta_penalty
    if strategy == "balanced_product_delta":
        return prod * balance * delta_penalty
    if strategy == "amp_product_mean_tau_delta_quad":
        return prod * mean_tau_penalty * delta_quad
    if strategy == "balanced_mean_tau_delta_quad":
        return prod * balance * mean_tau_penalty * delta_quad
    raise ValueError(f"Unknown strategy: {strategy}")


def select_best_pair(
    candidates_vl: list[dict[str, float]],
    candidates_vr: list[dict[str, float]],
    *,
    strategy: str,
    delta_limit_ms: float,
    delta_scale_ms: float,
    mean_tau_center_ms: float,
    mean_tau_scale_ms: float,
) -> dict[str, Any] | None:
    pairs = []
    for i, vl in enumerate(candidates_vl, start=1):
        for j, vr in enumerate(candidates_vr, start=1):
            dt = vr["tau_ms"] - vl["tau_ms"]
            if abs(dt) > delta_limit_ms:
                continue
            score = score_pair(
                vl,
                vr,
                strategy,
                delta_scale_ms,
                mean_tau_center_ms,
                mean_tau_scale_ms,
            )
            pairs.append(
                {
                    "vl_rank": i,
                    "vr_rank": j,
                    "tau_vl_ms": float(vl["tau_ms"]),
                    "tau_vr_ms": float(vr["tau_ms"]),
                    "delta_tau_ms": float(dt),
                    "vl_amp": float(vl["amp"]),
                    "vr_amp": float(vr["amp"]),
                    "vl_prom": float(vl["prom"]),
                    "vr_prom": float(vr["prom"]),
                    "score": float(score),
                }
            )
    if not pairs:
        return None
    pairs.sort(key=lambda x: x["score"], reverse=True)
    return {"best": pairs[0], "top_pairs": pairs[:10], "num_pairs": len(pairs)}


def evaluate_strategy(
    case: base.CaseRef,
    variant: dict[str, Any],
    signals: dict[str, np.ndarray],
    *,
    lag_min_ms: float,
    lag_max_ms: float,
    top_k: int,
    delta_limit_ms: float,
    delta_scale_ms: float,
    mean_tau_center_ms: float,
    mean_tau_scale_ms: float,
    strategy: str,
    max_lag_ms: float,
) -> dict[str, Any]:
    ref = base.compute_reference(case)
    ldv = base.apply_ops(signals["ldv"], list(variant["ldv_ops"]))
    mic_l = base.apply_ops(signals["mic_l"], list(variant["mic_ops"]))
    mic_r = base.apply_ops(signals["mic_r"], list(variant["mic_ops"]))

    lag_vl, cc_vl = gcc_curve(ldv, mic_l, 48000, max_lag_ms=max_lag_ms, bandpass=variant["bandpass"])
    lag_vr, cc_vr = gcc_curve(ldv, mic_r, 48000, max_lag_ms=max_lag_ms, bandpass=variant["bandpass"])
    cand_vl = extract_candidates(lag_vl, cc_vl, lag_min_ms=lag_min_ms, lag_max_ms=lag_max_ms, top_k=top_k)
    cand_vr = extract_candidates(lag_vr, cc_vr, lag_min_ms=lag_min_ms, lag_max_ms=lag_max_ms, top_k=top_k)
    selected = select_best_pair(
        cand_vl,
        cand_vr,
        strategy=strategy,
        delta_limit_ms=delta_limit_ms,
        delta_scale_ms=delta_scale_ms,
        mean_tau_center_ms=mean_tau_center_ms,
        mean_tau_scale_ms=mean_tau_scale_ms,
    )
    row = {
        "case_id": case.case_id,
        "variant": variant["name"],
        "strategy": strategy,
        "lag_min_ms": lag_min_ms,
        "delta_scale_ms": delta_scale_ms,
        "mean_tau_center_ms": mean_tau_center_ms,
        "mean_tau_scale_ms": mean_tau_scale_ms,
        "reference": ref,
        "selected": selected,
        "num_candidates_vl": len(cand_vl),
        "num_candidates_vr": len(cand_vr),
    }
    if selected is None:
        return row

    best_pair = selected["best"]
    theta_v = float(np.degrees(np.arcsin(np.clip((best_pair["delta_tau_ms"] / 1000.0) * base.C / 1.4, -1.0, 1.0))))
    best_pair["theta_v_deg"] = theta_v
    best_pair["delta_tau_abs_err_ms"] = abs(best_pair["delta_tau_ms"] - ref["delta_tau_ms"])
    best_pair["theta_v_abs_err_deg"] = abs(theta_v - ref["theta_v_deg"])
    best_pair["physical_positive_lags"] = bool(best_pair["tau_vl_ms"] > 0.0 and best_pair["tau_vr_ms"] > 0.0)
    best_pair["physical_small_delta"] = bool(abs(best_pair["delta_tau_ms"]) <= delta_limit_ms)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, float, float, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["variant"],
            row["strategy"],
            float(row["lag_min_ms"]),
            float(row["delta_scale_ms"]),
            float(row["mean_tau_center_ms"]),
            float(row["mean_tau_scale_ms"]),
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    for (variant, strategy, lag_min_ms, delta_scale_ms, mean_tau_center_ms, mean_tau_scale_ms), items in grouped.items():
        valid = [it for it in items if it["selected"] is not None]
        if not valid:
            summary.append(
                {
                    "variant": variant,
                    "strategy": strategy,
                    "lag_min_ms": lag_min_ms,
                    "delta_scale_ms": delta_scale_ms,
                    "mean_tau_center_ms": mean_tau_center_ms,
                    "mean_tau_scale_ms": mean_tau_scale_ms,
                    "valid_cases": 0,
                    "physical_count": 0,
                    "delta_tau_mae_ms": None,
                    "theta_v_mae_deg": None,
                    "max_delta_tau_abs_err_ms": None,
                    "max_theta_v_abs_err_deg": None,
                }
            )
            continue

        delta_err = np.array([it["selected"]["best"]["delta_tau_abs_err_ms"] for it in valid], dtype=np.float64)
        theta_err = np.array([it["selected"]["best"]["theta_v_abs_err_deg"] for it in valid], dtype=np.float64)
        physical_count = sum(
            1
            for it in valid
            if it["selected"]["best"]["physical_positive_lags"] and it["selected"]["best"]["physical_small_delta"]
        )
        summary.append(
            {
                "variant": variant,
                "strategy": strategy,
                "lag_min_ms": lag_min_ms,
                "delta_scale_ms": delta_scale_ms,
                "mean_tau_center_ms": mean_tau_center_ms,
                "mean_tau_scale_ms": mean_tau_scale_ms,
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
            -x["physical_count"],
        )
    )
    return {"strategies": summary}


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# 0223 Automated Peak-Pair Sweep",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        f"- Slice sec: `{payload['slice_sec']}`",
        f"- Lag min grid: `{payload['lag_min_grid']}`",
        f"- Lag max: `{payload['lag_max_ms']}` ms",
        f"- Top-k per side: `{payload['top_k']}`",
        f"- Delta gate: `|delta_tau| <= {payload['delta_limit_ms']}` ms",
        "",
        "## Summary",
        "",
        "| variant | strategy | lag_min_ms | delta_scale_ms | mean_tau_center_ms | mean_tau_scale_ms | valid_cases | physical_count | delta_tau_mae_ms | theta_v_mae_deg | max_delta_tau_err_ms | max_theta_v_err_deg |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["strategies"]:
        delta_txt = "NA" if row["delta_tau_mae_ms"] is None else f"{row['delta_tau_mae_ms']:.3f}"
        theta_txt = "NA" if row["theta_v_mae_deg"] is None else f"{row['theta_v_mae_deg']:.3f}"
        max_dt_txt = "NA" if row["max_delta_tau_abs_err_ms"] is None else f"{row['max_delta_tau_abs_err_ms']:.3f}"
        max_th_txt = "NA" if row["max_theta_v_abs_err_deg"] is None else f"{row['max_theta_v_abs_err_deg']:.3f}"
        lines.append(
            f"| {row['variant']} | {row['strategy']} | {row['lag_min_ms']:.3f} | {row['delta_scale_ms']:.3f} | "
            f"{row['mean_tau_center_ms']:.3f} | {row['mean_tau_scale_ms']:.3f} | {row['valid_cases']} | "
            f"{row['physical_count']} | {delta_txt} | {theta_txt} | {max_dt_txt} | {max_th_txt} |"
        )

    lines.extend(["", "## Per-Case Selections", ""])
    for row in payload["results"]:
        lines.extend(
            [
                f"### {row['case_id']} / {row['variant']} / {row['strategy']}",
                "",
                f"- lag_min_ms={row['lag_min_ms']:.3f}, delta_scale_ms={row['delta_scale_ms']:.3f}, "
                f"mean_tau_center_ms={row['mean_tau_center_ms']:.3f}, mean_tau_scale_ms={row['mean_tau_scale_ms']:.3f}",
            ]
        )
        if row["selected"] is None:
            lines.extend(["- no valid pair found", ""])
            continue
        best = row["selected"]["best"]
        ref = row["reference"]
        lines.extend(
            [
                f"- reference: tau_vl={ref['tau_vl_ms']:.3f} ms, tau_vr={ref['tau_vr_ms']:.3f} ms, "
                f"delta_tau={ref['delta_tau_ms']:.3f} ms, theta_v={ref['theta_v_deg']:.3f} deg",
                f"- selected: vl_rank={best['vl_rank']}, vr_rank={best['vr_rank']}, "
                f"tau_vl={best['tau_vl_ms']:.3f} ms, tau_vr={best['tau_vr_ms']:.3f} ms, "
                f"delta_tau={best['delta_tau_ms']:.3f} ms, theta_v={best['theta_v_deg']:.3f} deg, score={best['score']:.6f}",
                f"- errors: delta_tau_abs_err={best['delta_tau_abs_err_ms']:.3f} ms, "
                f"theta_v_abs_err={best['theta_v_abs_err_deg']:.3f} deg",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep positive-lag peak-pair strategies for 0223 delta-tau.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--slice_sec", type=float, default=5.0)
    parser.add_argument("--lag_min_grid", type=str, default="3.5,4.0,4.4")
    parser.add_argument("--lag_max_ms", type=float, default=6.5)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--delta_limit_ms", type=float, default=1.0)
    parser.add_argument("--max_lag_ms", type=float, default=10.0)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    lag_min_values = [float(part.strip()) for part in args.lag_min_grid.split(",") if part.strip()]
    strategies = [
        ("amp_product", 0.50, 4.8, 10.0),
        ("amp_product_delta", 0.25, 4.8, 10.0),
        ("amp_product_delta", 0.50, 4.8, 10.0),
        ("amp_product_delta", 1.00, 4.8, 10.0),
        ("amp_product_delta_quad", 0.25, 4.8, 10.0),
        ("amp_product_delta_quad", 0.50, 4.8, 10.0),
        ("amp_sum_delta", 0.50, 4.8, 10.0),
        ("prom_product_delta", 0.50, 4.8, 10.0),
        ("balanced_product_delta", 0.50, 4.8, 10.0),
        ("amp_product_mean_tau_delta_quad", 0.60, 4.8, 0.45),
        ("balanced_mean_tau_delta_quad", 0.60, 4.8, 0.45),
    ]
    variants = [
        v
        for v in base.VARIANTS
        if v["name"] in {"raw_fullband", "bp_500_2000", "bp_1000_3000", "ldv_preemph_bp500_2000", "ldv_diff_bp500_2000"}
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (Path(__file__).resolve().parent.parent / "results" / f"peak_pair_sweep_0223_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    signals_cache = {case.case_id: base.load_case_signals(case, args.data_root, args.slice_sec, 48000) for case in base.CASES}
    rows = []
    for variant in variants:
        for lag_min_ms in lag_min_values:
            for strategy, delta_scale_ms, mean_tau_center_ms, mean_tau_scale_ms in strategies:
                for case in base.CASES:
                    rows.append(
                        evaluate_strategy(
                            case,
                            variant,
                            signals_cache[case.case_id],
                            lag_min_ms=lag_min_ms,
                            lag_max_ms=args.lag_max_ms,
                            top_k=args.top_k,
                            delta_limit_ms=args.delta_limit_ms,
                            delta_scale_ms=delta_scale_ms,
                            mean_tau_center_ms=mean_tau_center_ms,
                            mean_tau_scale_ms=mean_tau_scale_ms,
                            strategy=strategy,
                            max_lag_ms=args.max_lag_ms,
                        )
                    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "slice_sec": float(args.slice_sec),
        "lag_min_grid": lag_min_values,
        "lag_max_ms": float(args.lag_max_ms),
        "top_k": int(args.top_k),
        "delta_limit_ms": float(args.delta_limit_ms),
        "max_lag_ms": float(args.max_lag_ms),
        "cases": [asdict(c) for c in base.CASES],
        "variants": variants,
        "results": rows,
        "summary": summarize(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir / "report.md", payload)
    print(f"[ok] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
