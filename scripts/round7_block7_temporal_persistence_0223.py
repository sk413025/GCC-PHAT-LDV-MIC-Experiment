#!/usr/bin/env python3
"""
Round 7 lane 2: block7 temporal family persistence.

This lane freezes the round7 lane1 winner and adds a block7-only temporal
family retention wrapper around the same-VR negative-family mechanism.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import filter_sweep_0223_delta_tau as base
import round3_blind_proxy_0223 as lane2
import round5_same_anchor_handoff_0223 as lane5d
import round7_block7_samevr_0223 as lane7a


WINDOW_SECS = lane7a.WINDOW_SECS
WINDOW_OFFSETS_SEC = lane7a.WINDOW_OFFSETS_SEC
BLOCK7_CASE_ID = lane7a.BLOCK7_CASE_ID
LANE7A_REF = next(v for v in lane7a.VARIANTS if v["name"] == "b7_samevr_neg_promote_vr0p20_dt0p15_r0p10")
NEIGHBOR_STEP = 0.5


VARIANTS: list[dict[str, Any]] = [
    {"name": "control_lane7a_ref", "support_required": 99, "vr_tol_ms": 0.0, "dt_tol_ms": 0.0, "score_floor": 0.0},
    {"name": "b7_temporal_hold_s2_vr0p20_dt0p20", "support_required": 2, "vr_tol_ms": 0.20, "dt_tol_ms": 0.20, "score_floor": 0.10},
    {"name": "b7_temporal_hold_s2_vr0p25_dt0p20", "support_required": 2, "vr_tol_ms": 0.25, "dt_tol_ms": 0.20, "score_floor": 0.10},
    {"name": "b7_temporal_hold_s3_vr0p25_dt0p25", "support_required": 3, "vr_tol_ms": 0.25, "dt_tol_ms": 0.25, "score_floor": 0.10},
]


def clamp_offset(offset_sec: float) -> float:
    return min(max(offset_sec, min(WINDOW_OFFSETS_SEC)), max(WINDOW_OFFSETS_SEC))


def neighbor_offsets(offset_sec: float) -> list[float]:
    return [clamp_offset(offset_sec - NEIGHBOR_STEP), float(offset_sec), clamp_offset(offset_sec + NEIGHBOR_STEP)]


def build_context(
    case: base.CaseRef,
    full_signals: dict[str, np.ndarray],
    *,
    window_sec: float,
    offset_sec: float,
) -> dict[str, Any]:
    item, scored_pool, _ = lane7a.lane6b.evaluate_pool(case, full_signals, window_sec=window_sec, offset_sec=offset_sec)
    return {"item": item, "scored_pool": scored_pool}


def choose_from_temporal_family(
    scored_pool: list[dict[str, float]],
    *,
    target_tau_vr_ms: float,
    target_abs_dt_ms: float,
    vr_tol_ms: float,
    dt_tol_ms: float,
    score_floor: float,
) -> dict[str, float] | None:
    top_score = float(scored_pool[0]["score"])
    candidates = []
    for row in scored_pool:
        if row["delta_tau_ms"] >= 0.0:
            continue
        if row["score"] < score_floor * top_score:
            continue
        vr_gap = abs(float(row["tau_vr_ms"]) - float(target_tau_vr_ms))
        dt_gap = abs(abs(row["delta_tau_ms"]) - float(target_abs_dt_ms))
        if vr_gap > vr_tol_ms or dt_gap > dt_tol_ms:
            continue
        candidates.append((vr_gap, dt_gap, -float(row["score"]), row))
    if not candidates:
        return None
    return min(candidates, key=lambda x: (x[0], x[1], x[2]))[3]


def evaluate_variant_case(
    case: base.CaseRef,
    full_signals: dict[str, np.ndarray],
    *,
    window_sec: float,
    offset_sec: float,
    variant: dict[str, Any],
) -> dict[str, Any]:
    base_row = lane7a.evaluate_variant_case(case, full_signals, window_sec=window_sec, offset_sec=offset_sec, variant=LANE7A_REF)
    base_row["variant"] = variant["name"]
    if variant["name"] == "control_lane7a_ref" or case.case_id != BLOCK7_CASE_ID:
        return base_row

    contexts = {
        off: build_context(case, full_signals, window_sec=window_sec, offset_sec=off)
        for off in sorted(set(neighbor_offsets(offset_sec)))
    }
    neighbor_rows = [
        lane7a.evaluate_variant_case(case, full_signals, window_sec=window_sec, offset_sec=off, variant=LANE7A_REF)
        for off in neighbor_offsets(offset_sec)
    ]
    negative_rows = [row for row in neighbor_rows if row["selected"]["best"]["delta_tau_ms"] < 0.0]
    if len(negative_rows) < int(variant["support_required"]):
        return base_row

    target_tau_vr_ms = float(np.median(np.array([row["selected"]["best"]["tau_vr_ms"] for row in negative_rows], dtype=np.float64)))
    target_abs_dt_ms = float(np.median(np.array([abs(row["selected"]["best"]["delta_tau_ms"]) for row in negative_rows], dtype=np.float64)))

    current_context = contexts[float(offset_sec)]
    candidate = choose_from_temporal_family(
        current_context["scored_pool"],
        target_tau_vr_ms=target_tau_vr_ms,
        target_abs_dt_ms=target_abs_dt_ms,
        vr_tol_ms=float(variant["vr_tol_ms"]),
        dt_tol_ms=float(variant["dt_tol_ms"]),
        score_floor=float(variant["score_floor"]),
    )
    if candidate is None:
        base_row["temporal_support"] = len(negative_rows)
        return base_row

    final_best = lane7a.lane6b.annotate_best(candidate, current_context["item"]["reference"], case.case_id)
    selected_pool = [candidate] + [
        row
        for row in current_context["scored_pool"]
        if not (abs(row["tau_vl_ms"] - candidate["tau_vl_ms"]) < 1e-9 and abs(row["tau_vr_ms"] - candidate["tau_vr_ms"]) < 1e-9)
    ]
    base_row["selected"] = {"best": final_best, "top_pairs": selected_pool[:10], "num_pairs": len(selected_pool)}
    base_row["strict_ranks"] = lane5d.strict_ranks(selected_pool, case.case_id)
    base_row["temporal_promoted"] = True
    base_row["promotion_reason"] = "temporal_hold"
    base_row["temporal_support"] = len(negative_rows)
    return base_row


def summarize_setting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row["selected"]["best"] for row in rows if row["selected"] is not None]
    dt = np.array([row["delta_tau_abs_err_ms"] for row in valid], dtype=np.float64)
    physical_count = sum(1 for row in valid if row["physical_valid"])
    hard = [row["selected"]["best"] for row in rows if row["case_id"] in {"block6_n04_19", BLOCK7_CASE_ID}]
    hard_dt = np.array([row["delta_tau_abs_err_ms"] for row in hard], dtype=np.float64)
    hard_win = np.array([1.0 if row["hard_case_win"] else 0.0 for row in hard], dtype=np.float64)
    by_case = {row["case_id"]: row for row in rows}
    block6 = by_case.get("block6_n04_19")
    block7 = by_case.get(BLOCK7_CASE_ID)
    return {
        "valid_cases": len(valid),
        "physical_count": physical_count,
        "delta_tau_mae_ms": float(np.mean(dt)),
        "max_delta_tau_abs_err_ms": float(np.max(dt)),
        "hard_case_delta_tau_mae_ms": float(np.mean(hard_dt)),
        "hard_case_win_rate": float(np.mean(hard_win)),
        "block6_selected_delta_tau_ms": block6["selected"]["best"]["delta_tau_ms"] if block6 else None,
        "block7_selected_delta_tau_ms": block7["selected"]["best"]["delta_tau_ms"] if block7 else None,
        "block6_rescue_pair_rank": block6["strict_ranks"]["rescue_pair_rank"] if block6 and block6["strict_ranks"] else None,
        "block7_rescue_pair_rank": block7["strict_ranks"]["rescue_pair_rank"] if block7 and block7["strict_ranks"] else None,
    }


def summarize_variant(rows: list[dict[str, Any]], variant_name: str) -> dict[str, Any]:
    subset = [row for row in rows if row["variant"] == variant_name]
    central_rows = [row for row in subset if abs(row["window_sec"] - 5.0) < 1e-9 and abs(row["offset_sec"]) < 1e-9]
    central = summarize_setting(central_rows)
    stability_subset = [row for row in subset if abs(row["window_sec"] - 5.0) < 1e-9]
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stability_subset:
        by_case[row["case_id"]].append(row)
    case_std = []
    hard_stats = {}
    for case_id, items in by_case.items():
        items.sort(key=lambda x: x["offset_sec"])
        selected = [item["selected"]["best"] for item in items]
        dt = np.array([item["delta_tau_ms"] for item in selected], dtype=np.float64)
        case_std.append(float(np.std(dt)))
        if case_id in {"block6_n04_19", BLOCK7_CASE_ID}:
            ranks = [item["strict_ranks"]["rescue_pair_rank"] if item["strict_ranks"] else None for item in items]
            hard_stats[case_id] = {
                "hit_at_1": int(sum(1 for rank in ranks if rank is not None and rank <= 1)),
                "hit_at_3": int(sum(1 for rank in ranks if rank is not None and rank <= 3)),
                "rescue_pair_ranks": ranks,
            }
    per_setting = []
    for window_sec in WINDOW_SECS:
        for offset_sec in WINDOW_OFFSETS_SEC:
            rows_setting = [row for row in subset if abs(row["window_sec"] - window_sec) < 1e-9 and abs(row["offset_sec"] - offset_sec) < 1e-9]
            setting = summarize_setting(rows_setting)
            per_setting.append(
                {
                    "window_sec": float(window_sec),
                    "offset_sec": float(offset_sec),
                    **setting,
                    "setting_pass": bool(
                        setting["valid_cases"] == 4
                        and setting["physical_count"] == 4
                        and setting["delta_tau_mae_ms"] <= 0.120
                        and setting["hard_case_win_rate"] >= 0.5
                    ),
                }
            )
    return {
        "variant": variant_name,
        "central": central,
        "window_stability_mean_std_ms": float(np.mean(case_std)) if case_std else None,
        "window_stability_max_std_ms": float(np.max(case_std)) if case_std else None,
        "hard_cases": hard_stats,
        "setting_pass_rate": float(np.mean([1.0 if row["setting_pass"] else 0.0 for row in per_setting])) if per_setting else None,
        "per_setting": per_setting,
    }


def choose_winner(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        summary_rows,
        key=lambda row: (
            -(row["hard_cases"].get(BLOCK7_CASE_ID, {}).get("hit_at_1", 0)),
            -(row["hard_cases"].get(BLOCK7_CASE_ID, {}).get("hit_at_3", 0)),
            -float(row["setting_pass_rate"] or 0.0),
            float("inf") if row["central"]["delta_tau_mae_ms"] is None else row["central"]["delta_tau_mae_ms"],
        ),
    )[0]


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 7 Block7 Temporal Family Persistence",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        "- Frozen reference: round7 lane1 winner `b7_samevr_neg_promote_vr0p20_dt0p15_r0p10`",
        "",
        "## Summary",
        "",
        "| variant | central_dt_mae_ms | hard_dt_mae_ms | pass_rate | window_mean_std | block6_hit@1 | block7_hit@1 | block7_hit@3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["variants"]:
        b6 = row["hard_cases"].get("block6_n04_19", {})
        b7 = row["hard_cases"].get(BLOCK7_CASE_ID, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    f"{row['central']['delta_tau_mae_ms']:.3f}",
                    f"{row['central']['hard_case_delta_tau_mae_ms']:.3f}",
                    f"{row['setting_pass_rate']:.3f}",
                    f"{row['window_stability_mean_std_ms']:.3f}",
                    str(b6.get("hit_at_1", 0)),
                    str(b7.get("hit_at_1", 0)),
                    str(b7.get("hit_at_3", 0)),
                ]
            )
            + " |"
        )
    winner = payload["summary"]["winner"]
    lines.extend(
        [
            "",
            "## Winner",
            "",
            f"- variant: `{winner['variant']}`",
            f"- block7 hit@1: `{winner['hard_cases'].get(BLOCK7_CASE_ID, {}).get('hit_at_1', 0)}`",
            f"- block7 hit@3: `{winner['hard_cases'].get(BLOCK7_CASE_ID, {}).get('hit_at_3', 0)}`",
            f"- block6 hit@1: `{winner['hard_cases'].get('block6_n04_19', {}).get('hit_at_1', 0)}`",
            f"- pass_rate: `{winner['setting_pass_rate']:.3f}`",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 7 block7 temporal family persistence.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round7_block7_temporal_persistence_0223_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    full_signals = {case.case_id: lane2.lane1.load_full_case_signals(case, args.data_root) for case in base.CASES}
    rows = []
    for variant in VARIANTS:
        for window_sec in WINDOW_SECS:
            for offset_sec in WINDOW_OFFSETS_SEC:
                for case in base.CASES:
                    rows.append(
                        evaluate_variant_case(
                            case,
                            full_signals[case.case_id],
                            window_sec=window_sec,
                            offset_sec=offset_sec,
                            variant=variant,
                        )
                    )

    summaries = [summarize_variant(rows, variant["name"]) for variant in VARIANTS]
    winner = choose_winner(summaries)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "variants": VARIANTS,
        "results": rows,
        "summary": {"variants": summaries, "winner": winner},
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir / "report.md", payload)
    print(f"[ok] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
