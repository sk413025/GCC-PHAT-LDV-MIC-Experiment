#!/usr/bin/env python3
"""
Round 5 lane 1: branch-aware pair formation with conditional corridor pruning.

This lane fixes:

- front-end: diff_len80_ldvonly_bp700_1800
- branch pool: base_only
- downstream scorer: len80_current_score

The only search axis is pair formation before final blind scoring.
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
import round4_candidate_generation_0223 as lane4a


STRICT_RADIUS_MS = 0.20
HARD_CASES = {"block6_n04_19", "block7_n08_20"}
STRICT_TARGETS = {
    "block6_n04_19": {"tau_vl_ms": 4.854166666666667, "tau_vr_ms": 5.4375},
    "block7_n08_20": {"tau_vl_ms": 4.833333333333333, "tau_vr_ms": 4.520833333333333},
}
CURRENT_SCORE_VARIANT = lane4a.CURRENT_SCORE_VARIANT


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "base_control",
        "delta_prune_max_ms": -1.0,
        "min_shift_ms": 0.20,
        "alt_ratio_floor": 0.20,
        "mean_tau_corridor_ms": 0.25,
        "mutual_top_m": 0,
    },
    {
        "name": "cond_prune_soft",
        "delta_prune_max_ms": 0.18,
        "min_shift_ms": 0.20,
        "alt_ratio_floor": 0.30,
        "mean_tau_corridor_ms": 0.25,
        "mutual_top_m": 0,
    },
    {
        "name": "cond_prune_medium",
        "delta_prune_max_ms": 0.30,
        "min_shift_ms": 0.20,
        "alt_ratio_floor": 0.18,
        "mean_tau_corridor_ms": 0.25,
        "mutual_top_m": 0,
    },
    {
        "name": "cond_prune_medium_mutual2",
        "delta_prune_max_ms": 0.30,
        "min_shift_ms": 0.20,
        "alt_ratio_floor": 0.18,
        "mean_tau_corridor_ms": 0.25,
        "mutual_top_m": 2,
    },
    {
        "name": "cond_prune_medium_mutual3",
        "delta_prune_max_ms": 0.30,
        "min_shift_ms": 0.20,
        "alt_ratio_floor": 0.18,
        "mean_tau_corridor_ms": 0.25,
        "mutual_top_m": 3,
    },
    {
        "name": "cond_prune_hard_mutual2",
        "delta_prune_max_ms": 0.35,
        "min_shift_ms": 0.20,
        "alt_ratio_floor": 0.12,
        "mean_tau_corridor_ms": 0.30,
        "mutual_top_m": 2,
    },
]


def base_core_score(row: dict[str, float]) -> float:
    return float(lane2.base_score(row, mean_tau_scale_ms=float(CURRENT_SCORE_VARIANT["mean_tau_scale_ms"])))


def same_anchor_alternative_exists(
    current: dict[str, float],
    pair_rows: list[dict[str, float]],
    *,
    side: str,
    delta_prune_max_ms: float,
    min_shift_ms: float,
    alt_ratio_floor: float,
    mean_tau_corridor_ms: float,
) -> bool:
    if delta_prune_max_ms < 0.0:
        return False
    if abs(current["delta_tau_ms"]) > delta_prune_max_ms:
        return False

    current_core = current["core_score"]
    current_tau = current["tau_vl_ms"] if side == "vl" else current["tau_vr_ms"]
    for other in pair_rows:
        if other is current:
            continue
        other_tau = other["tau_vl_ms"] if side == "vl" else other["tau_vr_ms"]
        if abs(other_tau - current_tau) > 1e-9:
            continue
        if abs(other["mean_tau_ms"] - current["mean_tau_ms"]) > mean_tau_corridor_ms:
            continue
        if abs(other["delta_tau_ms"]) < abs(current["delta_tau_ms"]) + min_shift_ms:
            continue
        if other["core_score"] < current_core * alt_ratio_floor:
            continue
        return True
    return False


def apply_mutual_top_m(pair_rows: list[dict[str, float]], top_m: int) -> list[dict[str, float]]:
    if top_m <= 0 or not pair_rows:
        return pair_rows

    by_vl: dict[float, list[dict[str, float]]] = {}
    by_vr: dict[float, list[dict[str, float]]] = {}
    for row in pair_rows:
        by_vl.setdefault(float(row["tau_vl_ms"]), []).append(row)
        by_vr.setdefault(float(row["tau_vr_ms"]), []).append(row)

    allowed: set[int] = set()
    vl_top: dict[tuple[float, float], int] = {}
    vr_top: dict[tuple[float, float], int] = {}
    for tau_vl, rows in by_vl.items():
        rows_sorted = sorted(rows, key=lambda x: (x["core_score"], abs(x["delta_tau_ms"])), reverse=True)
        for rank, row in enumerate(rows_sorted[:top_m], start=1):
            vl_top[(tau_vl, float(row["tau_vr_ms"]))] = rank
    for tau_vr, rows in by_vr.items():
        rows_sorted = sorted(rows, key=lambda x: (x["core_score"], abs(x["delta_tau_ms"])), reverse=True)
        for rank, row in enumerate(rows_sorted[:top_m], start=1):
            vr_top[(float(row["tau_vl_ms"]), tau_vr)] = rank

    kept = []
    for row in pair_rows:
        key_vl = (float(row["tau_vl_ms"]), float(row["tau_vr_ms"]))
        key_vr = (float(row["tau_vl_ms"]), float(row["tau_vr_ms"]))
        if key_vl in vl_top and key_vr in vr_top:
            kept.append(row)
    return kept if kept else pair_rows


def form_pairs(pair_rows: list[dict[str, float]], variant: dict[str, Any]) -> list[dict[str, float]]:
    working = []
    for row in pair_rows:
        item = dict(row)
        item["core_score"] = base_core_score(item)
        item["pruned"] = False
        item["prune_reason"] = None
        working.append(item)

    if variant["delta_prune_max_ms"] >= 0.0:
        for row in working:
            same_vl = same_anchor_alternative_exists(
                row,
                working,
                side="vl",
                delta_prune_max_ms=float(variant["delta_prune_max_ms"]),
                min_shift_ms=float(variant["min_shift_ms"]),
                alt_ratio_floor=float(variant["alt_ratio_floor"]),
                mean_tau_corridor_ms=float(variant["mean_tau_corridor_ms"]),
            )
            same_vr = same_anchor_alternative_exists(
                row,
                working,
                side="vr",
                delta_prune_max_ms=float(variant["delta_prune_max_ms"]),
                min_shift_ms=float(variant["min_shift_ms"]),
                alt_ratio_floor=float(variant["alt_ratio_floor"]),
                mean_tau_corridor_ms=float(variant["mean_tau_corridor_ms"]),
            )
            if same_vl or same_vr:
                row["pruned"] = True
                row["prune_reason"] = "same_anchor_corridor_prune"

    kept = [row for row in working if not row["pruned"]]
    kept = apply_mutual_top_m(kept, int(variant["mutual_top_m"]))
    kept.sort(key=lambda x: x["core_score"], reverse=True)
    return kept


def strict_ranks(pair_rows: list[dict[str, float]], case_id: str) -> dict[str, Any] | None:
    if case_id not in STRICT_TARGETS:
        return None
    target = STRICT_TARGETS[case_id]
    sorted_rows = sorted(pair_rows, key=lambda x: x["core_score"], reverse=True)

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
            if vl_rank is None and abs(tau_vl - target["tau_vl_ms"]) <= STRICT_RADIUS_MS:
                vl_rank = len(seen_vl)
        if all(abs(tau_vr - seen) > 1e-9 for seen in seen_vr):
            seen_vr.append(tau_vr)
            if vr_rank is None and abs(tau_vr - target["tau_vr_ms"]) <= STRICT_RADIUS_MS:
                vr_rank = len(seen_vr)
        if (
            pair_rank is None
            and abs(tau_vl - target["tau_vl_ms"]) <= STRICT_RADIUS_MS
            and abs(tau_vr - target["tau_vr_ms"]) <= STRICT_RADIUS_MS
        ):
            pair_rank = idx
    return {"rescue_vl_rank": vl_rank, "rescue_vr_rank": vr_rank, "rescue_pair_rank": pair_rank}


def evaluate_case_windows(case: base.CaseRef, full_signals: dict[str, np.ndarray], *, window_sec: float) -> list[dict[str, Any]]:
    return lane4a.evaluate_case_windows(case, full_signals, window_sec=window_sec)


def evaluate_variant(case_windows: list[dict[str, Any]], variant: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in case_windows:
        base_pair_rows = lane2.build_pair_rows(item["base_candidates_vl"], item["base_candidates_vr"])
        formed = form_pairs(base_pair_rows, variant)
        selected = lane2.select_best_pair(formed, CURRENT_SCORE_VARIANT)
        ref = item["reference"]
        row = {
            "case_id": item["case_id"],
            "variant": variant["name"],
            "offset_sec": item["offset_sec"],
            "reference": ref,
            "num_pairs_before": len(base_pair_rows),
            "num_pairs_after": len(formed),
            "selected": selected,
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
            row["strict_ranks"] = strict_ranks(formed, item["case_id"])
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
        hard_items = [it for it in central if it["case_id"] in HARD_CASES and it["selected"] is not None]
        hard_dt = np.array([it["selected"]["best"]["delta_tau_abs_err_ms"] for it in hard_items], dtype=np.float64)
        hard_win = np.array([1.0 if it["selected"]["best"]["delta_tau_abs_err_ms"] <= 0.224 else 0.0 for it in hard_items], dtype=np.float64)

        strict_by_case = {it["case_id"]: it.get("strict_ranks") for it in central if it["case_id"] in HARD_CASES}
        window_hits: dict[str, int] = {}
        for case_id in HARD_CASES:
            count = 0
            for it in items:
                if it["case_id"] != case_id:
                    continue
                ranks = strict_ranks(it["selected"]["top_pairs"], case_id) if it["selected"] is not None else None
                if ranks is not None and ranks["rescue_pair_rank"] is not None and ranks["rescue_pair_rank"] <= 5:
                    count += 1
            window_hits[case_id] = count

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
                "block6_rescue_pair_hit5": int(window_hits["block6_n04_19"]),
                "block7_rescue_pair_hit5": int(window_hits["block7_n08_20"]),
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
        "# Round 5 Pair Formation Sweep",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        f"- Fixed front-end: `diff_len80_ldvonly_bp700_1800`",
        f"- Fixed scorer: `len80_current_score`",
        "",
        "## Summary",
        "",
        "| variant | central_dt_mae_ms | hard_dt_mae_ms | hard_win_rate | central_max_dt_ms | block6_pair_rank | block7_pair_rank | block6_dt | block7_dt | block6_hit5 | block7_hit5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
                    fmt("block6_rescue_pair_hit5"),
                    fmt("block7_rescue_pair_hit5"),
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 5 pair formation sweep for 0223.")
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
        Path(__file__).resolve().parent.parent / "results" / f"round5_pair_formation_0223_{timestamp}"
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
