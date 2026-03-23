#!/usr/bin/env python3
"""
Round 5 lane 3: same-anchor replacement on top of the soft pool winner.

This lane fixes:

- front-end: diff_len80_ldvonly_bp700_1800
- branch pool: base_only
- pair formation pool: cond_prune_soft
- baseline scorer: soft_anchor_pow_1p0

The search axis is a very narrow block6-like mechanism:
repair a same-VL double-top monopoly without reopening broader pruning.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import filter_sweep_0223_delta_tau as base
import round3_blind_proxy_0223 as lane2
import round5_pair_formation_0223 as lane5a


SOFT_POOL = next(variant for variant in lane5a.VARIANTS if variant["name"] == "cond_prune_soft")


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "soft_anchor_reference",
        "replace_bonus": 0.0,
        "monopoly_penalty": 0.0,
        "ratio_floor": 0.18,
        "delta_lift_min_ms": 0.20,
        "mean_tau_corridor_ms": 0.25,
    },
    {
        "name": "same_vl_replace_w4_p2",
        "replace_bonus": 4.0,
        "monopoly_penalty": 2.0,
        "ratio_floor": 0.18,
        "delta_lift_min_ms": 0.20,
        "mean_tau_corridor_ms": 0.25,
    },
    {
        "name": "same_vl_replace_w6_p2",
        "replace_bonus": 6.0,
        "monopoly_penalty": 2.0,
        "ratio_floor": 0.15,
        "delta_lift_min_ms": 0.20,
        "mean_tau_corridor_ms": 0.25,
    },
    {
        "name": "same_vl_replace_w6_p3",
        "replace_bonus": 6.0,
        "monopoly_penalty": 3.0,
        "ratio_floor": 0.15,
        "delta_lift_min_ms": 0.20,
        "mean_tau_corridor_ms": 0.25,
    },
]


def baseline_score(pair_row: dict[str, float]) -> float:
    core = lane5a.base_core_score(pair_row)
    min_rank = float(min(pair_row["vl_rank"], pair_row["vr_rank"]))
    return float(core / max(min_rank, 1e-9))


def strict_ranks(sorted_rows: list[dict[str, float]], case_id: str) -> dict[str, Any] | None:
    if case_id not in lane5a.STRICT_TARGETS:
        return None
    target = lane5a.STRICT_TARGETS[case_id]
    seen_vl: list[float] = []
    seen_vr: list[float] = []
    vl_rank = None
    vr_rank = None
    pair_rank = None
    for idx, row in enumerate(sorted_rows, start=1):
        tau_vl = float(row["tau_vl_ms"])
        tau_vr = float(row["tau_vr_ms"])
        if all(abs(tau_vl - seen) > 1e-9 for seen in seen_vl):
            seen_vl.append(tau_vl)
            if vl_rank is None and abs(tau_vl - target["tau_vl_ms"]) <= lane5a.STRICT_RADIUS_MS:
                vl_rank = len(seen_vl)
        if all(abs(tau_vr - seen) > 1e-9 for seen in seen_vr):
            seen_vr.append(tau_vr)
            if vr_rank is None and abs(tau_vr - target["tau_vr_ms"]) <= lane5a.STRICT_RADIUS_MS:
                vr_rank = len(seen_vr)
        if (
            pair_rank is None
            and abs(tau_vl - target["tau_vl_ms"]) <= lane5a.STRICT_RADIUS_MS
            and abs(tau_vr - target["tau_vr_ms"]) <= lane5a.STRICT_RADIUS_MS
        ):
            pair_rank = idx
    return {"rescue_vl_rank": vl_rank, "rescue_vr_rank": vr_rank, "rescue_pair_rank": pair_rank}


def enrich_same_anchor_features(pair_rows: list[dict[str, float]], variant: dict[str, Any]) -> list[dict[str, float]]:
    rows = []
    for row in pair_rows:
        item = dict(row)
        item["baseline_score"] = baseline_score(item)
        rows.append(item)

    by_vl: dict[float, list[dict[str, float]]] = {}
    by_vr: dict[float, list[dict[str, float]]] = {}
    for row in rows:
        by_vl.setdefault(float(row["tau_vl_ms"]), []).append(row)
        by_vr.setdefault(float(row["tau_vr_ms"]), []).append(row)

    for vl_tau, group in by_vl.items():
        group.sort(key=lambda x: x["baseline_score"], reverse=True)
        leader = group[0]
        leader_sign = np.sign(leader["delta_tau_ms"])
        for row in group:
            row["same_vl_replacement_bonus"] = 0.0
            row["same_vl_monopoly_penalty"] = 0.0
            row["same_vl_double_top_context"] = bool(leader["vl_rank"] == 1 and leader["vr_rank"] == 1)
            if not row["same_vl_double_top_context"]:
                continue
            if row is leader:
                continue
            if abs(row["mean_tau_ms"] - leader["mean_tau_ms"]) > float(variant["mean_tau_corridor_ms"]):
                continue
            if np.sign(row["delta_tau_ms"]) != leader_sign:
                continue
            delta_lift = abs(row["delta_tau_ms"]) - abs(leader["delta_tau_ms"])
            if delta_lift < float(variant["delta_lift_min_ms"]):
                continue
            ratio = row["baseline_score"] / max(leader["baseline_score"], 1e-9)
            if ratio < float(variant["ratio_floor"]):
                continue
            lift_term = min(delta_lift / max(float(variant["delta_lift_min_ms"]), 1e-9), 2.0)
            row["same_vl_replacement_bonus"] = float(variant["replace_bonus"]) * ratio * lift_term
            leader["same_vl_monopoly_penalty"] = max(
                float(leader["same_vl_monopoly_penalty"]),
                float(variant["monopoly_penalty"]) * ratio * lift_term,
            )

    for vr_tau, group in by_vr.items():
        signs = {int(np.sign(row["delta_tau_ms"])) for row in group if abs(row["delta_tau_ms"]) > 1e-9}
        mixed_sign = len(signs) >= 2
        if not mixed_sign:
            continue
        group.sort(key=lambda x: x["baseline_score"], reverse=True)
        leader = group[0]
        leader["vr_family_mixed_sign"] = True
    return rows


def score_pair_variant(pair_row: dict[str, float]) -> float:
    score = float(pair_row["baseline_score"])
    score *= 1.0 + float(pair_row.get("same_vl_replacement_bonus", 0.0))
    score /= 1.0 + float(pair_row.get("same_vl_monopoly_penalty", 0.0))
    return float(score)


def evaluate_case_windows(case: base.CaseRef, full_signals: dict[str, np.ndarray], *, window_sec: float) -> list[dict[str, Any]]:
    return lane5a.evaluate_case_windows(case, full_signals, window_sec=window_sec)


def evaluate_variant(case_windows: list[dict[str, Any]], variant: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in case_windows:
        base_pair_rows = lane2.build_pair_rows(item["base_candidates_vl"], item["base_candidates_vr"])
        formed = lane5a.form_pairs(base_pair_rows, SOFT_POOL)
        enriched = enrich_same_anchor_features(formed, variant)
        scored = []
        for row in enriched:
            ranked = dict(row)
            ranked["score"] = score_pair_variant(ranked)
            scored.append(ranked)
        scored.sort(key=lambda x: x["score"], reverse=True)
        selected = {"best": scored[0], "top_pairs": scored[:10], "num_pairs": len(scored)} if scored else None
        ref = item["reference"]
        row = {
            "case_id": item["case_id"],
            "variant": variant["name"],
            "offset_sec": item["offset_sec"],
            "reference": ref,
            "selected": selected,
            "num_pairs_after": len(formed),
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
            row["strict_ranks"] = strict_ranks(scored, item["case_id"])
        else:
            row["strict_ranks"] = None
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
            summary.append({"variant": variant, "central_delta_tau_mae_ms": None})
            continue
        dt = np.array([it["delta_tau_abs_err_ms"] for it in central_valid], dtype=np.float64)
        th = np.array([it["theta_v_abs_err_deg"] for it in central_valid], dtype=np.float64)
        physical_count = sum(1 for it in central_valid if it["physical_valid"])
        hard_items = [it for it in central if it["case_id"] in lane5a.HARD_CASES and it["selected"] is not None]
        hard_dt = np.array([it["selected"]["best"]["delta_tau_abs_err_ms"] for it in hard_items], dtype=np.float64)
        hard_win = np.array([1.0 if it["selected"]["best"]["delta_tau_abs_err_ms"] <= 0.224 else 0.0 for it in hard_items], dtype=np.float64)
        strict_by_case = {it["case_id"]: it.get("strict_ranks") for it in central if it["case_id"] in lane5a.HARD_CASES}
        summary.append(
            {
                "variant": variant,
                "central_valid_cases": len(central_valid),
                "central_physical_count": physical_count,
                "central_delta_tau_mae_ms": float(np.mean(dt)),
                "central_theta_v_mae_deg": float(np.mean(th)),
                "central_max_delta_tau_abs_err_ms": float(np.max(dt)),
                "hard_case_delta_tau_mae_ms": float(np.mean(hard_dt)) if hard_dt.size else None,
                "hard_case_win_rate": float(np.mean(hard_win)) if hard_win.size else None,
                "block6_rescue_pair_rank": strict_by_case.get("block6_n04_19", {}).get("rescue_pair_rank") if strict_by_case.get("block6_n04_19") else None,
                "block7_rescue_pair_rank": strict_by_case.get("block7_n08_20", {}).get("rescue_pair_rank") if strict_by_case.get("block7_n08_20") else None,
                "block6_selected_delta_tau_ms": next((it["selected"]["best"]["delta_tau_ms"] for it in central if it["case_id"] == "block6_n04_19" and it["selected"] is not None), None),
                "block7_selected_delta_tau_ms": next((it["selected"]["best"]["delta_tau_ms"] for it in central if it["case_id"] == "block7_n08_20" and it["selected"] is not None), None),
            }
        )
    summary.sort(
        key=lambda x: (
            1 if x["hard_case_delta_tau_mae_ms"] is None else 0,
            float("inf") if x["hard_case_delta_tau_mae_ms"] is None else x["hard_case_delta_tau_mae_ms"],
            float("inf") if x["central_delta_tau_mae_ms"] is None else x["central_delta_tau_mae_ms"],
        )
    )
    return {"variants": summary}


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 5 Same-Anchor Replacement Sweep",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        f"- Fixed pool: `cond_prune_soft`",
        f"- Fixed baseline scorer: `soft_anchor_pow_1p0`",
        "",
        "## Summary",
        "",
        "| variant | central_dt_mae_ms | hard_dt_mae_ms | hard_win_rate | central_max_dt_ms | block6_pair_rank | block7_pair_rank | block6_dt | block7_dt |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["variants"]:
        def fmt(name: str) -> str:
            value = row.get(name)
            if value is None:
                return "NA"
            if isinstance(value, int):
                return str(value)
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
                    fmt("block6_rescue_pair_rank"),
                    fmt("block7_rescue_pair_rank"),
                    fmt("block6_selected_delta_tau_ms"),
                    fmt("block7_selected_delta_tau_ms"),
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 5 same-anchor replacement sweep for 0223.")
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
        Path(__file__).resolve().parent.parent / "results" / f"round5_same_anchor_replacement_0223_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    full_signals = {
        case.case_id: lane2.lane1.load_full_case_signals(case, args.data_root)
        for case in base.CASES
    }
    case_windows = {
        case.case_id: evaluate_case_windows(case, full_signals[case.case_id], window_sec=args.window_sec)
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
