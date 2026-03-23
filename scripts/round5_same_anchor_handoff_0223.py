#!/usr/bin/env python3
"""
Round 5 lane 4: same-anchor handoff selector on top of the lane 3 scorer.

This lane keeps:

- front-end: diff_len80_ldvonly_bp700_1800
- pair formation pool: cond_prune_soft
- lane 3 scorer: same_vl_replace_w6_p3

The only search axis is a final handoff rule:
when a same-VL rescue pair has already reached rank 2 and nearly matches the
monopoly top1, should it take over.
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
import round5_same_anchor_replacement_0223 as lane5c


LANE3_VARIANT = next(v for v in lane5c.VARIANTS if v["name"] == "same_vl_replace_w6_p3")


VARIANTS: list[dict[str, Any]] = [
    {"name": "lane3_reference", "handoff_ratio": 0.0, "delta_lift_min_ms": 0.20},
    {"name": "handoff_ratio_0p90", "handoff_ratio": 0.90, "delta_lift_min_ms": 0.20},
    {"name": "handoff_ratio_0p93", "handoff_ratio": 0.93, "delta_lift_min_ms": 0.20},
    {"name": "handoff_ratio_0p95", "handoff_ratio": 0.95, "delta_lift_min_ms": 0.20},
]


def strict_ranks(sorted_rows: list[dict[str, float]], case_id: str) -> dict[str, Any] | None:
    return lane5c.strict_ranks(sorted_rows, case_id)


def evaluate_case_windows(case: base.CaseRef, full_signals: dict[str, np.ndarray], *, window_sec: float) -> list[dict[str, Any]]:
    return lane5a.evaluate_case_windows(case, full_signals, window_sec=window_sec)


def build_scored_pool(item: dict[str, Any]) -> list[dict[str, float]]:
    base_pair_rows = lane2.build_pair_rows(item["base_candidates_vl"], item["base_candidates_vr"])
    formed = lane5a.form_pairs(base_pair_rows, lane5c.SOFT_POOL)
    enriched = lane5c.enrich_same_anchor_features(formed, LANE3_VARIANT)
    scored = []
    for row in enriched:
        ranked = dict(row)
        ranked["score"] = lane5c.score_pair_variant(ranked)
        scored.append(ranked)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def apply_handoff(scored: list[dict[str, float]], variant: dict[str, Any]) -> list[dict[str, float]]:
    if not scored or float(variant["handoff_ratio"]) <= 0.0:
        return scored

    top1 = scored[0]
    if not bool(top1.get("same_vl_double_top_context", False)):
        return scored

    replacement_idx = None
    for idx, row in enumerate(scored[1:], start=1):
        if abs(row["tau_vl_ms"] - top1["tau_vl_ms"]) > 1e-9:
            continue
        if np.sign(row["delta_tau_ms"]) != np.sign(top1["delta_tau_ms"]):
            continue
        if abs(row["delta_tau_ms"]) < abs(top1["delta_tau_ms"]) + float(variant["delta_lift_min_ms"]):
            continue
        if row["score"] < float(variant["handoff_ratio"]) * top1["score"]:
            continue
        replacement_idx = idx
        break

    if replacement_idx is None:
        return scored

    reordered = list(scored)
    reordered[0], reordered[replacement_idx] = reordered[replacement_idx], reordered[0]
    return reordered


def evaluate_variant(case_windows: list[dict[str, Any]], variant: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in case_windows:
        scored = build_scored_pool(item)
        selected_pool = apply_handoff(scored, variant)
        selected = {"best": selected_pool[0], "top_pairs": selected_pool[:10], "num_pairs": len(selected_pool)} if selected_pool else None
        ref = item["reference"]
        row = {
            "case_id": item["case_id"],
            "variant": variant["name"],
            "offset_sec": item["offset_sec"],
            "reference": ref,
            "selected": selected,
            "num_pairs_after": len(selected_pool),
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
            row["strict_ranks"] = strict_ranks(selected_pool, item["case_id"])
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
        "# Round 5 Same-Anchor Handoff Sweep",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        f"- Fixed pool: `cond_prune_soft`",
        f"- Fixed lane 3 scorer: `same_vl_replace_w6_p3`",
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
    parser = argparse.ArgumentParser(description="Round 5 same-anchor handoff sweep for 0223.")
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
        Path(__file__).resolve().parent.parent / "results" / f"round5_same_anchor_handoff_0223_{timestamp}"
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
