#!/usr/bin/env python3
"""
Round 9 lane 2: block7 nearest negative-family retention.

This lane freezes the round9 lane1 winner and only changes the block7 rule.
It uses the nearest supported negative neighbors and prefers candidates that
keep a sufficiently large negative magnitude instead of collapsing to shallow
negative branches.
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
import round5_pair_formation_0223 as lane5a
import round5_same_anchor_handoff_0223 as lane5d
import round9_block7_negative_family_target_hold_0223 as lane9a


WINDOW_SECS = lane9a.WINDOW_SECS
WINDOW_OFFSETS_SEC = lane9a.WINDOW_OFFSETS_SEC
BASELINE_VARIANT = next(v for v in lane9a.VARIANTS if v["name"] == "b7_neg_hold_span1_support2_floor0p08")
BLOCK7_CASE_ID = lane9a.BLOCK7_CASE_ID


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "control_round9a_ref",
        "score_floor": 1.0,
        "vr_tol_ms": 0.0,
        "min_abs_dt_ratio": 9.99,
    },
    {
        "name": "b7_nearest_neg_r0p08_ratio0p80_vr0p20",
        "score_floor": 0.08,
        "vr_tol_ms": 0.20,
        "min_abs_dt_ratio": 0.80,
    },
    {
        "name": "b7_nearest_neg_r0p08_ratio0p70_vr0p20",
        "score_floor": 0.08,
        "vr_tol_ms": 0.20,
        "min_abs_dt_ratio": 0.70,
    },
    {
        "name": "b7_nearest_neg_r0p05_ratio0p75_vr0p25",
        "score_floor": 0.05,
        "vr_tol_ms": 0.25,
        "min_abs_dt_ratio": 0.75,
    },
]


def nearest_negative_anchor(
    case: base.CaseRef,
    *,
    window_sec: float,
    offset_sec: float,
    baseline_rows: dict[tuple[str, float, float], dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = []
    current_idx = list(WINDOW_OFFSETS_SEC).index(float(offset_sec))
    for idx, neighbor_offset in enumerate(WINDOW_OFFSETS_SEC):
        if idx == current_idx:
            continue
        row = baseline_rows[(case.case_id, float(window_sec), float(neighbor_offset))]
        if row["selected"] is None:
            continue
        best = row["selected"]["best"]
        if best["delta_tau_ms"] >= 0.0:
            continue
        if not bool(best["physical_valid"]):
            continue
        distance = abs(float(neighbor_offset) - float(offset_sec))
        candidates.append((distance, -abs(float(best["delta_tau_ms"])), -float(best["score"]), best))
    if not candidates:
        return None
    best = min(candidates, key=lambda x: (x[0], x[1], x[2]))[3]
    return {
        "target_tau_vr_ms": float(best["tau_vr_ms"]),
        "target_abs_dt_ms": float(abs(best["delta_tau_ms"])),
        "target_tau_vl_ms": float(best["tau_vl_ms"]),
    }


def choose_candidate(
    scored_pool: list[dict[str, float]],
    anchor: dict[str, Any] | None,
    *,
    score_floor: float,
    vr_tol_ms: float,
    min_abs_dt_ratio: float,
) -> dict[str, float] | None:
    if anchor is None or not scored_pool:
        return None
    top_score = float(scored_pool[0]["score"])
    min_abs_dt = float(min_abs_dt_ratio) * float(anchor["target_abs_dt_ms"])
    candidates = []
    for row in scored_pool:
        if row["delta_tau_ms"] >= 0.0:
            continue
        if row["score"] < float(score_floor) * top_score:
            continue
        if abs(float(row["delta_tau_ms"])) < min_abs_dt:
            continue
        vr_gap = abs(float(row["tau_vr_ms"]) - float(anchor["target_tau_vr_ms"]))
        if vr_gap > float(vr_tol_ms):
            continue
        dt_gap = abs(abs(float(row["delta_tau_ms"])) - float(anchor["target_abs_dt_ms"]))
        candidates.append((vr_gap, -abs(float(row["delta_tau_ms"])), dt_gap, -float(row["score"]), row))
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
    baseline_rows: dict[tuple[str, float, float], dict[str, Any]],
) -> dict[str, Any]:
    baseline_row = baseline_rows[(case.case_id, float(window_sec), float(offset_sec))]
    row = dict(baseline_row)
    row["variant"] = variant["name"]
    if variant["name"] == "control_round9a_ref" or case.case_id != BLOCK7_CASE_ID:
        return row

    item, scored_pool, _ = lane9a.lane8.evaluate_pool(case, full_signals, window_sec=window_sec, offset_sec=offset_sec)
    anchor = nearest_negative_anchor(case, window_sec=window_sec, offset_sec=offset_sec, baseline_rows=baseline_rows)
    candidate = choose_candidate(
        scored_pool,
        anchor,
        score_floor=float(variant["score_floor"]),
        vr_tol_ms=float(variant["vr_tol_ms"]),
        min_abs_dt_ratio=float(variant["min_abs_dt_ratio"]),
    )
    if candidate is None:
        row["nearest_anchor"] = anchor
        row["nearest_promoted"] = False
        row["promotion_reason"] = None
        return row

    final_best = lane9a.lane8.annotate_best(candidate, item["reference"], case.case_id)
    selected_pool = [candidate] + [
        cand
        for cand in scored_pool
        if not (abs(cand["tau_vl_ms"] - candidate["tau_vl_ms"]) < 1e-9 and abs(cand["tau_vr_ms"] - candidate["tau_vr_ms"]) < 1e-9)
    ]
    row["selected"] = {"best": final_best, "top_pairs": selected_pool[:10], "num_pairs": len(selected_pool)}
    row["strict_ranks"] = lane5d.strict_ranks(selected_pool, case.case_id)
    row["nearest_anchor"] = anchor
    row["nearest_promoted"] = True
    row["promotion_reason"] = "nearest_negative_family"
    return row


def summarize_setting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return lane9a.summarize_setting(rows)


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
        if case_id in lane5a.HARD_CASES:
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
            rows_setting = [
                row for row in subset if abs(row["window_sec"] - window_sec) < 1e-9 and abs(row["offset_sec"] - offset_sec) < 1e-9
            ]
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
                        and setting["hard_case_win_rate"] is not None
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
            float("inf") if row["central"].get("delta_tau_mae_ms") is None else row["central"]["delta_tau_mae_ms"],
        ),
    )[0]


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 9 Block7 Nearest Negative-Family Retention",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        "- Frozen reference: round9 lane1 winner `b7_neg_hold_span1_support2_floor0p08`",
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
            f"- block6 hit@1: `{winner['hard_cases'].get('block6_n04_19', {}).get('hit_at_1', 0)}`",
            f"- block7 hit@1: `{winner['hard_cases'].get(BLOCK7_CASE_ID, {}).get('hit_at_1', 0)}`",
            f"- block7 hit@3: `{winner['hard_cases'].get(BLOCK7_CASE_ID, {}).get('hit_at_3', 0)}`",
            f"- pass_rate: `{winner['setting_pass_rate']:.3f}`",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 9 block7 nearest negative-family retention for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round9_block7_nearest_negative_family_0223_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    full_signals = {case.case_id: lane9a.lane8.lane2.lane1.load_full_case_signals(case, args.data_root) for case in base.CASES}

    lane6_rows: dict[tuple[str, float, float], dict[str, Any]] = {}
    for case in base.CASES:
        for window_sec in WINDOW_SECS:
            for offset_sec in WINDOW_OFFSETS_SEC:
                lane6_rows[(case.case_id, float(window_sec), float(offset_sec))] = lane9a.lane8.lane6c.evaluate_variant_case(
                    case,
                    full_signals[case.case_id],
                    window_sec=window_sec,
                    offset_sec=offset_sec,
                    variant=lane9a.lane8.BASELINE_VARIANT,
                )

    round8_rows: dict[tuple[str, float, float], dict[str, Any]] = {}
    for case in base.CASES:
        for window_sec in WINDOW_SECS:
            for offset_sec in WINDOW_OFFSETS_SEC:
                round8_rows[(case.case_id, float(window_sec), float(offset_sec))] = lane9a.lane8.evaluate_variant_case(
                    case,
                    full_signals[case.case_id],
                    window_sec=window_sec,
                    offset_sec=offset_sec,
                    variant=lane9a.BASELINE_VARIANT,
                    baseline_rows=lane6_rows,
                )

    baseline_rows: dict[tuple[str, float, float], dict[str, Any]] = {}
    for case in base.CASES:
        for window_sec in WINDOW_SECS:
            for offset_sec in WINDOW_OFFSETS_SEC:
                baseline_rows[(case.case_id, float(window_sec), float(offset_sec))] = lane9a.evaluate_variant_case(
                    case,
                    full_signals[case.case_id],
                    window_sec=window_sec,
                    offset_sec=offset_sec,
                    variant=BASELINE_VARIANT,
                    baseline_rows=round8_rows,
                )

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
                            baseline_rows=baseline_rows,
                        )
                    )

    summaries = [summarize_variant(rows, variant["name"]) for variant in VARIANTS]
    winner = choose_winner(summaries)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "baseline_variant": BASELINE_VARIANT["name"],
        "variants": VARIANTS,
        "results": rows,
        "summary": {"variants": summaries, "winner": winner},
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir / "report.md", payload)
    print(f"[ok] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
