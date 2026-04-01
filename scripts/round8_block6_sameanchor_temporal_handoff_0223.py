#!/usr/bin/env python3
"""
Round 8 lane 3: block6 same-anchor temporal handoff.

This lane freezes round6 lane3 `sign_veto_poslift_r0p10` as the base and only
adds a block6-focused temporal persistence gate on top of the same-anchor
handoff idea from round5 lane4.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import filter_sweep_0223_delta_tau as base
import round3_blind_proxy_0223 as lane2
import round5_same_anchor_handoff_0223 as lane5d
import round6_sign_veto_handoff_0223 as lane6c
import round6_temporal_handoff_0223 as lane6b


WINDOW_SECS = lane6c.WINDOW_SECS
WINDOW_OFFSETS_SEC = lane6c.WINDOW_OFFSETS_SEC
BLOCK6_CASE_ID = "block6_n04_19"
BLOCK7_CASE_ID = "block7_n08_20"
LANE6_REF = next(v for v in lane6c.VARIANTS if v["name"] == "sign_veto_poslift_r0p10")
FAMILY_SUPPORT_OFFSETS = (0.0, 0.5, 1.0)


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "control_sign_veto_poslift_r0p10",
        "support_required": 99,
        "handoff_ratio": 0.0,
        "delta_lift_min_ms": 9.99,
        "anchor_gap_ms": 0.0,
        "score_floor": 0.0,
    },
    {
        "name": "b6_sameanchor_hold_s2_r0p08",
        "support_required": 2,
        "handoff_ratio": 0.08,
        "delta_lift_min_ms": 0.05,
        "anchor_gap_ms": 0.18,
        "vr_gap_ms": 0.20,
        "score_floor": 0.05,
    },
    {
        "name": "b6_sameanchor_hold_s2_r0p10",
        "support_required": 2,
        "handoff_ratio": 0.10,
        "delta_lift_min_ms": 0.05,
        "anchor_gap_ms": 0.18,
        "vr_gap_ms": 0.20,
        "score_floor": 0.05,
    },
    {
        "name": "b6_sameanchor_hold_s3_r0p10",
        "support_required": 3,
        "handoff_ratio": 0.10,
        "delta_lift_min_ms": 0.05,
        "anchor_gap_ms": 0.15,
        "vr_gap_ms": 0.18,
        "score_floor": 0.05,
    },
]


FAMILY_REFERENCE_CACHE: dict[tuple[str, float], dict[str, float]] = {}


def build_context(
    case: base.CaseRef,
    full_signals: dict[str, np.ndarray],
    *,
    window_sec: float,
    offset_sec: float,
) -> dict[str, Any]:
    item, scored_pool, baseline_pool = lane6b.evaluate_pool(case, full_signals, window_sec=window_sec, offset_sec=offset_sec)
    baseline_best = lane6b.annotate_best(baseline_pool[0], item["reference"], case.case_id)
    return {"item": item, "scored_pool": scored_pool, "baseline_best": baseline_best}


def family_reference(
    case: base.CaseRef,
    full_signals: dict[str, np.ndarray],
    *,
    window_sec: float,
) -> dict[str, float] | None:
    key = (case.case_id, float(window_sec))
    if key not in FAMILY_REFERENCE_CACHE:
        ref_row = lane6c.evaluate_variant_case(case, full_signals, window_sec=window_sec, offset_sec=0.0, variant=LANE6_REF)
        if ref_row["selected"] is None:
            return None
        FAMILY_REFERENCE_CACHE[key] = dict(ref_row["selected"]["best"])
    return FAMILY_REFERENCE_CACHE[key]


def choose_sameanchor_candidate(
    scored_pool: list[dict[str, float]],
    *,
    target_tau_vl_ms: float,
    target_tau_vr_ms: float,
    target_abs_dt_ms: float,
    handoff_ratio: float,
    delta_lift_min_ms: float,
    anchor_gap_ms: float,
    vr_gap_ms: float,
    score_floor: float,
) -> dict[str, float] | None:
    top_score = float(scored_pool[0]["score"])
    candidates = []
    for row in scored_pool:
        if row["delta_tau_ms"] <= 0.0:
            continue
        if row["score"] < score_floor * top_score:
            continue
        anchor_gap = abs(float(row["tau_vl_ms"]) - float(target_tau_vl_ms))
        if anchor_gap > anchor_gap_ms:
            continue
        vr_gap = abs(float(row["tau_vr_ms"]) - float(target_tau_vr_ms))
        if vr_gap > vr_gap_ms:
            continue
        if abs(float(row["delta_tau_ms"])) < float(target_abs_dt_ms) + float(delta_lift_min_ms):
            continue
        if row["score"] < handoff_ratio * top_score:
            continue
        candidates.append((anchor_gap, vr_gap, -float(row["score"]), -abs(float(row["delta_tau_ms"])), row))
    if not candidates:
        return None
    return min(candidates, key=lambda x: (x[0], x[1], x[2], x[3]))[4]


def evaluate_variant_case(
    case: base.CaseRef,
    full_signals: dict[str, np.ndarray],
    *,
    window_sec: float,
    offset_sec: float,
    variant: dict[str, Any],
) -> dict[str, Any]:
    base_row = lane6c.evaluate_variant_case(case, full_signals, window_sec=window_sec, offset_sec=offset_sec, variant=LANE6_REF)
    base_row["variant"] = variant["name"]
    if variant["name"] == "control_sign_veto_poslift_r0p10" or case.case_id != BLOCK6_CASE_ID:
        return base_row

    ref = family_reference(case, full_signals, window_sec=window_sec)
    if ref is None:
        base_row["neighbor_support"] = 0
        return base_row

    support_rows = []
    for off in FAMILY_SUPPORT_OFFSETS:
        support_rows.append(
            lane6c.evaluate_variant_case(case, full_signals, window_sec=window_sec, offset_sec=off, variant=LANE6_REF)
        )
    positive_neighbors = [
        row
        for row in support_rows
        if row["selected"]["best"]["delta_tau_ms"] > 0.0
        and abs(row["selected"]["best"]["tau_vl_ms"] - float(ref["tau_vl_ms"])) <= float(variant["anchor_gap_ms"])
        and abs(row["selected"]["best"]["tau_vr_ms"] - float(ref["tau_vr_ms"])) <= float(variant["vr_gap_ms"])
    ]
    if len(positive_neighbors) < int(variant["support_required"]):
        base_row["neighbor_support"] = len(positive_neighbors)
        base_row["family_tau_vl_ms"] = float(ref["tau_vl_ms"])
        base_row["family_tau_vr_ms"] = float(ref["tau_vr_ms"])
        return base_row

    current_context = build_context(case, full_signals, window_sec=window_sec, offset_sec=offset_sec)
    candidate = choose_sameanchor_candidate(
        current_context["scored_pool"],
        target_tau_vl_ms=float(ref["tau_vl_ms"]),
        target_tau_vr_ms=float(ref["tau_vr_ms"]),
        target_abs_dt_ms=float(abs(ref["delta_tau_ms"])),
        handoff_ratio=float(variant["handoff_ratio"]),
        delta_lift_min_ms=float(variant["delta_lift_min_ms"]),
        anchor_gap_ms=float(variant["anchor_gap_ms"]),
        vr_gap_ms=float(variant["vr_gap_ms"]),
        score_floor=float(variant["score_floor"]),
    )
    if candidate is None:
        base_row["neighbor_support"] = len(positive_neighbors)
        base_row["family_tau_vl_ms"] = float(ref["tau_vl_ms"])
        base_row["family_tau_vr_ms"] = float(ref["tau_vr_ms"])
        return base_row

    final_best = lane6b.annotate_best(candidate, current_context["item"]["reference"], case.case_id)
    selected_pool = [candidate] + [
        row
        for row in current_context["scored_pool"]
        if not (abs(row["tau_vl_ms"] - candidate["tau_vl_ms"]) < 1e-9 and abs(row["tau_vr_ms"] - candidate["tau_vr_ms"]) < 1e-9)
    ]
    base_row["selected"] = {"best": final_best, "top_pairs": selected_pool[:10], "num_pairs": len(selected_pool)}
    base_row["strict_ranks"] = lane5d.strict_ranks(selected_pool, case.case_id)
    base_row["temporal_promoted"] = True
    base_row["promotion_reason"] = "sameanchor_temporal_handoff"
    base_row["neighbor_support"] = len(positive_neighbors)
    base_row["family_tau_vl_ms"] = float(ref["tau_vl_ms"])
    base_row["family_tau_vr_ms"] = float(ref["tau_vr_ms"])
    return base_row


def summarize_setting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row["selected"]["best"] for row in rows if row["selected"] is not None]
    if not valid:
        return {
            "valid_cases": 0,
            "physical_count": 0,
            "delta_tau_mae_ms": None,
            "max_delta_tau_abs_err_ms": None,
            "hard_case_delta_tau_mae_ms": None,
            "hard_case_win_rate": None,
            "block6_selected_delta_tau_ms": None,
            "block7_selected_delta_tau_ms": None,
            "block6_rescue_pair_rank": None,
            "block7_rescue_pair_rank": None,
            "promoted_cases": 0,
        }
    dt = np.array([row["delta_tau_abs_err_ms"] for row in valid], dtype=np.float64)
    physical_count = sum(1 for row in valid if row["physical_valid"])
    hard = [row["selected"]["best"] for row in rows if row["case_id"] in {BLOCK6_CASE_ID, BLOCK7_CASE_ID} and row["selected"] is not None]
    hard_dt = np.array([row["delta_tau_abs_err_ms"] for row in hard], dtype=np.float64)
    hard_win = np.array([1.0 if row["hard_case_win"] else 0.0 for row in hard], dtype=np.float64)
    by_case = {row["case_id"]: row for row in rows}
    block6 = by_case.get(BLOCK6_CASE_ID)
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
        "promoted_cases": int(sum(1 for row in rows if row["temporal_promoted"])),
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
        if case_id in {BLOCK6_CASE_ID, BLOCK7_CASE_ID}:
            ranks = [item["strict_ranks"]["rescue_pair_rank"] if item["strict_ranks"] else None for item in items]
            hard_stats[case_id] = {
                "hit_at_1": int(sum(1 for rank in ranks if rank is not None and rank <= 1)),
                "hit_at_3": int(sum(1 for rank in ranks if rank is not None and rank <= 3)),
                "rescue_pair_ranks": ranks,
                "selected_delta_tau_ms": [item["selected"]["best"]["delta_tau_ms"] for item in items],
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
                        and setting["delta_tau_mae_ms"] is not None
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
            -(row["hard_cases"].get(BLOCK6_CASE_ID, {}).get("hit_at_1", 0)),
            -(row["hard_cases"].get(BLOCK6_CASE_ID, {}).get("hit_at_3", 0)),
            -float(row["setting_pass_rate"] or 0.0),
            float("inf") if row["central"]["delta_tau_mae_ms"] is None else row["central"]["delta_tau_mae_ms"],
            -(row["hard_cases"].get(BLOCK7_CASE_ID, {}).get("hit_at_1", 0)),
        ),
    )[0]


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 8 Block6 Same-Anchor Temporal Handoff",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        "- Frozen reference: `sign_veto_poslift_r0p10`",
        "",
        "## Summary",
        "",
        "| variant | central_dt_mae_ms | hard_dt_mae_ms | pass_rate | window_mean_std | block6_hit@1 | block6_hit@3 | block7_hit@1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["variants"]:
        b6 = row["hard_cases"].get(BLOCK6_CASE_ID, {})
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
                    str(b6.get("hit_at_3", 0)),
                    str(b7.get("hit_at_1", 0)),
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
            f"- block6 hit@1: `{winner['hard_cases'].get(BLOCK6_CASE_ID, {}).get('hit_at_1', 0)}`",
            f"- block6 hit@3: `{winner['hard_cases'].get(BLOCK6_CASE_ID, {}).get('hit_at_3', 0)}`",
            f"- block7 hit@1: `{winner['hard_cases'].get(BLOCK7_CASE_ID, {}).get('hit_at_1', 0)}`",
            f"- pass_rate: `{winner['setting_pass_rate']:.3f}`",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 8 block6 same-anchor temporal handoff for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round8_block6_sameanchor_temporal_handoff_0223_{timestamp}"
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
