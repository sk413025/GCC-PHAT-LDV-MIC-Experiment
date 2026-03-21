#!/usr/bin/env python3
"""
Round 3 lane 3: same-anchor competition-gap scoring for 0223.

This lane fixes:

- front-end: diff_len80_ldvonly_bp700_1800
- candidate extraction: same as round3 lane 2

The only search axis is the blind pair scorer. The new score families explicitly
model same-anchor escape routes and local competition gaps on the hard cases.
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
import round3_blind_proxy_0223 as lane2


HARD_CASES = {"block6_n04_19", "block7_n08_20"}
SAME_ANCHOR_DELTA_SHIFT_MIN_MS = 0.20
MEAN_TAU_CORRIDOR_MS = 0.22
ZERO_AVOID_CENTER_MS = 0.12
ZERO_AVOID_SPAN_MS = 0.40


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "lane2_control",
        "worst_margin_weight": 0.0,
        "bilateral_gate_weight": 0.0,
        "shift_cost_weight": 0.0,
        "corridor_margin_weight": 0.0,
        "consensus_weight": 0.0,
        "rank_discount_pow": 0.0,
        "zero_avoid_weight": 0.0,
    },
    {
        "name": "worst_constraint_margin",
        "worst_margin_weight": 0.90,
        "bilateral_gate_weight": 0.0,
        "shift_cost_weight": 0.0,
        "corridor_margin_weight": 0.0,
        "consensus_weight": 0.0,
        "rank_discount_pow": 0.0,
        "zero_avoid_weight": 0.0,
    },
    {
        "name": "bilateral_ownership_gate",
        "worst_margin_weight": 0.0,
        "bilateral_gate_weight": 0.80,
        "shift_cost_weight": 0.0,
        "corridor_margin_weight": 0.0,
        "consensus_weight": 0.0,
        "rank_discount_pow": 0.0,
        "zero_avoid_weight": 0.0,
    },
    {
        "name": "delta_shift_per_cost",
        "worst_margin_weight": 0.0,
        "bilateral_gate_weight": 0.0,
        "shift_cost_weight": 0.90,
        "corridor_margin_weight": 0.0,
        "consensus_weight": 0.0,
        "rank_discount_pow": 0.0,
        "zero_avoid_weight": 0.0,
    },
    {
        "name": "corridor_relative_margin",
        "worst_margin_weight": 0.0,
        "bilateral_gate_weight": 0.0,
        "shift_cost_weight": 0.0,
        "corridor_margin_weight": 0.90,
        "consensus_weight": 0.0,
        "rank_discount_pow": 0.0,
        "zero_avoid_weight": 0.0,
    },
    {
        "name": "local_consensus_gap",
        "worst_margin_weight": 0.0,
        "bilateral_gate_weight": 0.0,
        "shift_cost_weight": 0.0,
        "corridor_margin_weight": 0.0,
        "consensus_weight": 0.70,
        "rank_discount_pow": 0.0,
        "zero_avoid_weight": 0.0,
    },
    {
        "name": "hybrid_gap_zero0",
        "worst_margin_weight": 0.50,
        "bilateral_gate_weight": 0.35,
        "shift_cost_weight": 0.75,
        "corridor_margin_weight": 0.55,
        "consensus_weight": 0.30,
        "rank_discount_pow": 0.0,
        "zero_avoid_weight": 0.0,
    },
    {
        "name": "hybrid_gap_zero05",
        "worst_margin_weight": 0.50,
        "bilateral_gate_weight": 0.35,
        "shift_cost_weight": 0.75,
        "corridor_margin_weight": 0.55,
        "consensus_weight": 0.30,
        "rank_discount_pow": 0.0,
        "zero_avoid_weight": 0.5,
    },
    {
        "name": "hybrid_gap_zero10",
        "worst_margin_weight": 0.50,
        "bilateral_gate_weight": 0.35,
        "shift_cost_weight": 0.75,
        "corridor_margin_weight": 0.55,
        "consensus_weight": 0.30,
        "rank_discount_pow": 0.0,
        "zero_avoid_weight": 1.0,
    },
    {
        "name": "hybrid_gap_zero10_rankguard",
        "worst_margin_weight": 0.50,
        "bilateral_gate_weight": 0.35,
        "shift_cost_weight": 0.75,
        "corridor_margin_weight": 0.55,
        "consensus_weight": 0.30,
        "rank_discount_pow": 0.20,
        "zero_avoid_weight": 1.0,
    },
]


def ratio_feature(value: float, clip_hi: float) -> float:
    return float(np.clip(np.log1p(max(value, 0.0)), 0.0, clip_hi))


def zero_avoid_signal(delta_tau_ms: float) -> float:
    shifted = (abs(delta_tau_ms) - ZERO_AVOID_CENTER_MS) / max(ZERO_AVOID_SPAN_MS, 1e-9)
    return float(np.clip(shifted, 0.0, 1.0))


def enrich_competition_features(pair_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    if not pair_rows:
        return []

    enriched = []
    for row in pair_rows:
        current = dict(row)
        current["core_score"] = lane2.base_score(current, mean_tau_scale_ms=0.45)
        same_vl = []
        same_vr = []
        same_corridor = []
        same_vl_shift = []
        same_vr_shift = []
        for other in pair_rows:
            if other is row:
                continue
            delta_shift = abs(other["delta_tau_ms"]) - abs(current["delta_tau_ms"])
            if abs(other["tau_vl_ms"] - current["tau_vl_ms"]) < 1e-9:
                same_vl.append(other)
                if delta_shift >= SAME_ANCHOR_DELTA_SHIFT_MIN_MS:
                    same_vl_shift.append(other)
            if abs(other["tau_vr_ms"] - current["tau_vr_ms"]) < 1e-9:
                same_vr.append(other)
                if delta_shift >= SAME_ANCHOR_DELTA_SHIFT_MIN_MS:
                    same_vr_shift.append(other)
            if abs(other["mean_tau_ms"] - current["mean_tau_ms"]) <= MEAN_TAU_CORRIDOR_MS:
                same_corridor.append(other)

        def max_core(candidates: list[dict[str, float]]) -> float:
            if not candidates:
                return 0.0
            return max(lane2.base_score(other, mean_tau_scale_ms=0.45) for other in candidates)

        def best_ratio(candidates: list[dict[str, float]]) -> tuple[float, float]:
            if not candidates:
                return 0.0, 0.0
            ratios = []
            creds = []
            for other in candidates:
                other_core = lane2.base_score(other, mean_tau_scale_ms=0.45)
                ratios.append(other_core / max(current["core_score"], 1e-9))
                creds.append(float(other["psr_geom"] * other["amp_ratio_geom"]))
            return max(ratios), max(creds)

        same_vl_ratio, same_vl_rescue = best_ratio(same_vl_shift)
        same_vr_ratio, same_vr_rescue = best_ratio(same_vr_shift)

        shift_per_cost = 0.0
        for group in (same_vl_shift, same_vr_shift):
            for other in group:
                delta_shift = max(abs(other["delta_tau_ms"]) - abs(current["delta_tau_ms"]) - 0.15, 0.0)
                prod_ratio = other["prod"] / max(current["prod"], 1e-9)
                support_ratio = other.get("window_support", 0.0) / max(current.get("window_support", 0.0), 1e-9)
                cost = 0.15 + max(1.0 - prod_ratio, 0.0) + 0.5 * max(1.0 - support_ratio, 0.0)
                shift_per_cost = max(shift_per_cost, delta_shift / max(cost, 1e-9))

        corridor_mass = 0.0
        weighted_deltas = []
        for other in same_corridor:
            other_core = lane2.base_score(other, mean_tau_scale_ms=0.45)
            corridor_mass += other_core
            weighted_deltas.append((other["delta_tau_ms"], other_core))
        local_mean_tau_comp_gap = corridor_mass / max(current["core_score"], 1e-9)

        def normalized_margin(best_other_core: float) -> float:
            if best_other_core <= 0.0:
                return 1.5
            gap = (current["core_score"] - best_other_core) / max(current["core_score"], 1e-9)
            return float(np.clip(gap, -1.5, 1.5))

        same_vl_margin = normalized_margin(max_core(same_vl))
        same_vr_margin = normalized_margin(max_core(same_vr))
        tau_band_margin = normalized_margin(max_core(same_corridor))
        worst_constraint_margin = min(same_vl_margin, same_vr_margin, tau_band_margin)
        endpoint_asymmetry_gap = abs(same_vl_margin - same_vr_margin)

        local_consensus_gap = 0.0
        if weighted_deltas:
            weighted_deltas.sort(key=lambda x: x[0])
            total_w = sum(w for _, w in weighted_deltas)
            acc = 0.0
            median_delta = weighted_deltas[-1][0]
            for delta_val, weight in weighted_deltas:
                acc += weight
                if acc >= 0.5 * total_w:
                    median_delta = delta_val
                    break
            local_consensus_gap = abs(current["delta_tau_ms"] - median_delta)

        current["same_vl_escape_gap"] = ratio_feature(same_vl_ratio, 2.0)
        current["same_vr_escape_gap"] = ratio_feature(same_vr_ratio, 2.0)
        current["same_anchor_escape_gap"] = max(current["same_vl_escape_gap"], current["same_vr_escape_gap"])
        current["delta_shift_per_cost"] = float(np.clip(shift_per_cost, 0.0, 2.5))
        current["local_mean_tau_comp_gap"] = ratio_feature(local_mean_tau_comp_gap, 2.5)
        current["rescue_peak_credibility"] = float(np.clip(max(same_vl_rescue, same_vr_rescue), 0.0, 2.0))
        current["same_vl_margin"] = same_vl_margin
        current["same_vr_margin"] = same_vr_margin
        current["tau_band_margin"] = tau_band_margin
        current["worst_constraint_margin"] = worst_constraint_margin
        current["endpoint_asymmetry_gap"] = float(np.clip(endpoint_asymmetry_gap, 0.0, 3.0))
        current["local_consensus_gap"] = float(np.clip(local_consensus_gap, 0.0, 1.0))
        current["zero_avoid_signal"] = zero_avoid_signal(current["delta_tau_ms"])
        enriched.append(current)
    return enriched


def score_pair_variant(pair_row: dict[str, float], variant: dict[str, Any]) -> dict[str, float]:
    core = lane2.base_score(pair_row, mean_tau_scale_ms=0.45)
    worst_margin_signal = max(pair_row["worst_constraint_margin"], -1.0)
    corridor_margin_signal = max(pair_row["tau_band_margin"], -1.0)
    gate_l = 1.0 / (1.0 + math.exp(-4.0 * pair_row["same_vl_margin"]))
    gate_r = 1.0 / (1.0 + math.exp(-4.0 * pair_row["same_vr_margin"]))
    bilateral_gate = gate_l * gate_r
    worst_term = 1.0 + float(variant["worst_margin_weight"]) * max(worst_margin_signal, 0.0)
    gate_term = 1.0 + float(variant["bilateral_gate_weight"]) * bilateral_gate
    shift_cost_term = 1.0 + float(variant["shift_cost_weight"]) * pair_row["delta_shift_per_cost"]
    corridor_term = 1.0 + float(variant["corridor_margin_weight"]) * max(corridor_margin_signal, 0.0)
    consensus_term = 1.0 + float(variant["consensus_weight"]) * max(0.5 - pair_row["local_consensus_gap"], 0.0)
    rank_discount = float((pair_row["vl_rank"] * pair_row["vr_rank"]) ** float(variant["rank_discount_pow"])) if float(variant["rank_discount_pow"]) > 0.0 else 1.0
    zero_term = 1.0 + float(variant["zero_avoid_weight"]) * pair_row["zero_avoid_signal"]

    core_mult = worst_term * gate_term * shift_cost_term * corridor_term * consensus_term
    core_score = core * core_mult / max(rank_discount, 1e-9)
    zero_contrib = core_score * (zero_term - 1.0)
    final_score = core_score * zero_term
    return {
        "score": float(final_score),
        "core_competition_score": float(core_score),
        "zero_penalty_contrib": float(zero_contrib),
        "zero_penalty_share": float(zero_contrib / max(final_score, 1e-9)),
    }


def oracle_match_rank(pair_rows: list[dict[str, float]], ref: dict[str, float], variant: dict[str, Any]) -> dict[str, float] | None:
    scored = []
    for row in pair_rows:
        item = dict(row)
        item.update(score_pair_variant(item, variant))
        scored.append(item)
    scored.sort(key=lambda x: x["score"], reverse=True)
    match = None
    match_rank = None
    for idx, row in enumerate(scored, start=1):
        if abs(row["tau_vl_ms"] - ref["tau_vl_ms"]) <= lane2.ORACLE_RADIUS_MS and abs(row["tau_vr_ms"] - ref["tau_vr_ms"]) <= lane2.ORACLE_RADIUS_MS:
            match = row
            match_rank = idx
            break
    if match is None:
        return None
    best_wrong = max((row["score"] for row in scored if row is not match), default=0.0)
    return {
        "correct_pair_rank": int(match_rank),
        "correct_pair_margin": float(match["score"] / max(best_wrong, 1e-9)),
    }


def select_best_pair(pair_rows: list[dict[str, float]], variant: dict[str, Any]) -> dict[str, Any] | None:
    if not pair_rows:
        return None
    scored = []
    for row in pair_rows:
        item = dict(row)
        item.update(score_pair_variant(item, variant))
        scored.append(item)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"best": scored[0], "top_pairs": scored[:10], "num_pairs": len(scored)}


def evaluate_variant(case_windows: list[dict[str, Any]], variant: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in case_windows:
        pair_rows = enrich_competition_features(item["pair_rows"])
        selected = select_best_pair(pair_rows, variant)
        ref = item["reference"]
        row = {
            "case_id": item["case_id"],
            "variant": variant["name"],
            "offset_sec": item["offset_sec"],
            "reference": ref,
            "selected": selected,
            "num_candidates_vl": item["num_candidates_vl"],
            "num_candidates_vr": item["num_candidates_vr"],
        }
        if selected is not None:
            best = selected["best"]
            theta_v = float(np.degrees(np.arcsin(np.clip((best["delta_tau_ms"] / 1000.0) * base.C / 1.4, -1.0, 1.0))))
            best["theta_v_deg"] = theta_v
            best["delta_tau_abs_err_ms"] = abs(best["delta_tau_ms"] - ref["delta_tau_ms"])
            best["theta_v_abs_err_deg"] = abs(theta_v - ref["theta_v_deg"])
            best["physical_positive_lags"] = bool(best["tau_vl_ms"] > 0.0 and best["tau_vr_ms"] > 0.0)
            best["physical_small_delta"] = bool(abs(best["delta_tau_ms"]) <= lane2.DELTA_LIMIT_MS)
            best["physical_valid"] = bool(best["physical_positive_lags"] and best["physical_small_delta"])
        if abs(item["offset_sec"]) < 1e-9:
            row["oracle_match"] = oracle_match_rank(pair_rows, ref, variant)
        else:
            row["oracle_match"] = None
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)

    summary = []
    for variant, items in grouped.items():
        central = [it for it in items if abs(it["offset_sec"]) < 1e-9]
        central_valid = [it["selected"]["best"] for it in central if it["selected"] is not None]
        if not central_valid:
            summary.append(
                {
                    "variant": variant,
                    "central_valid_cases": 0,
                    "central_physical_count": 0,
                    "central_delta_tau_mae_ms": None,
                    "central_theta_v_mae_deg": None,
                    "central_max_delta_tau_abs_err_ms": None,
                    "central_correct_pair_rank_mean": None,
                    "central_correct_pair_margin_mean": None,
                    "hard_case_delta_tau_mae_ms": None,
                    "hard_case_win_rate": None,
                    "window_delta_tau_mae_ms": None,
                    "window_stability_mean_std_ms": None,
                    "window_stability_max_std_ms": None,
                    "selected_zero_penalty_share_median": None,
                    "selected_zero_penalty_share_p90": None,
                }
            )
            continue

        dt = np.array([it["delta_tau_abs_err_ms"] for it in central_valid], dtype=np.float64)
        th = np.array([it["theta_v_abs_err_deg"] for it in central_valid], dtype=np.float64)
        physical_count = sum(1 for it in central_valid if it["physical_valid"])
        oracle_rows = [it["oracle_match"] for it in central if it.get("oracle_match") is not None]

        hard_dt = [
            it["selected"]["best"]["delta_tau_abs_err_ms"]
            for it in central
            if it["case_id"] in HARD_CASES and it["selected"] is not None
        ]
        hard_win = [
            1.0 if it["selected"]["best"]["delta_tau_abs_err_ms"] <= 0.224 else 0.0
            for it in central
            if it["case_id"] in HARD_CASES and it["selected"] is not None
        ]

        window_valid = [it for it in items if it["selected"] is not None]
        window_dt = np.array([it["selected"]["best"]["delta_tau_abs_err_ms"] for it in window_valid], dtype=np.float64)
        case_std = []
        penalty_shares = []
        for case_id in sorted({it["case_id"] for it in window_valid}):
            vals = [it["selected"]["best"]["delta_tau_ms"] for it in window_valid if it["case_id"] == case_id]
            if len(vals) >= 2:
                case_std.append(float(np.std(np.asarray(vals, dtype=np.float64))))
        for item in central_valid:
            penalty_shares.append(float(item.get("zero_penalty_share", 0.0)))

        summary.append(
            {
                "variant": variant,
                "central_valid_cases": len(central_valid),
                "central_physical_count": physical_count,
                "central_delta_tau_mae_ms": float(np.mean(dt)),
                "central_theta_v_mae_deg": float(np.mean(th)),
                "central_max_delta_tau_abs_err_ms": float(np.max(dt)),
                "central_correct_pair_rank_mean": float(np.mean([it["correct_pair_rank"] for it in oracle_rows])) if oracle_rows else None,
                "central_correct_pair_margin_mean": float(np.mean([it["correct_pair_margin"] for it in oracle_rows])) if oracle_rows else None,
                "hard_case_delta_tau_mae_ms": float(np.mean(np.asarray(hard_dt, dtype=np.float64))) if hard_dt else None,
                "hard_case_win_rate": float(np.mean(np.asarray(hard_win, dtype=np.float64))) if hard_win else None,
                "window_delta_tau_mae_ms": float(np.mean(window_dt)),
                "window_stability_mean_std_ms": float(np.mean(case_std)) if case_std else None,
                "window_stability_max_std_ms": float(np.max(case_std)) if case_std else None,
                "selected_zero_penalty_share_median": float(np.median(np.asarray(penalty_shares, dtype=np.float64))) if penalty_shares else None,
                "selected_zero_penalty_share_p90": float(np.percentile(np.asarray(penalty_shares, dtype=np.float64), 90.0)) if penalty_shares else None,
            }
        )

    summary.sort(
        key=lambda x: (
            1 if x["central_delta_tau_mae_ms"] is None else 0,
            float("inf") if x["hard_case_delta_tau_mae_ms"] is None else x["hard_case_delta_tau_mae_ms"],
            float("inf") if x["central_delta_tau_mae_ms"] is None else x["central_delta_tau_mae_ms"],
        )
    )
    return {"variants": summary}


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 3 Competition-Gap Scoring Sweep",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        f"- Fixed front-end: `diff_len80_ldvonly_bp700_1800`",
        "",
        "## Summary",
        "",
        "| variant | central_dt_mae_ms | hard_dt_mae_ms | hard_win_rate | central_max_dt_ms | window_dt_mae_ms | window_std_mean_ms | zero_share_median | zero_share_p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["variants"]:
        def fmt(name: str) -> str:
            value = row.get(name)
            if value is None:
                return "NA"
            return f"{float(value):.3f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    fmt("central_delta_tau_mae_ms"),
                    fmt("hard_case_delta_tau_mae_ms"),
                    fmt("hard_case_win_rate"),
                    fmt("central_max_delta_tau_abs_err_ms"),
                    fmt("window_delta_tau_mae_ms"),
                    fmt("window_stability_mean_std_ms"),
                    fmt("selected_zero_penalty_share_median"),
                    fmt("selected_zero_penalty_share_p90"),
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 3 competition-gap scoring sweep for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--window_sec", type=float, default=5.0)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round3_competition_gap_0223_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    full_signals = {
        case.case_id: lane2.lane1.load_full_case_signals(case, args.data_root)
        for case in base.CASES
    }
    case_windows = {
        case.case_id: lane2.evaluate_case_windows(case, full_signals[case.case_id], window_sec=args.window_sec)
        for case in base.CASES
    }

    rows = []
    for variant in VARIANTS:
        for case in base.CASES:
            rows.extend(evaluate_variant(case_windows[case.case_id], variant))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "window_sec": float(args.window_sec),
        "variants": VARIANTS,
        "results": rows,
        "summary": summarize(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir / "report.md", payload)
    print(f"[ok] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
