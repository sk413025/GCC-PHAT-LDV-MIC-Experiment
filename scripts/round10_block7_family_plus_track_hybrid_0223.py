#!/usr/bin/env python3
"""
Round 10 lane 5: block7 negative-family plus short-track hybrid.

This lane freezes the round9 family-hold winner and only changes the block7
decision rule. A candidate is only promoted when it matches both:

1. the supported negative family across nearby windows
2. a short immediate-neighbor tracklet

Within that gated set, the short-track fit is ranked before the wider family
fit to test a narrow track-before-select hybrid.
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
        "neighbor_span": 0,
        "support_required": 99,
        "track_span": 0,
        "track_support_required": 99,
        "score_floor": 1.0,
        "vr_tol_ms": 0.0,
        "abs_dt_tol_ms": 0.0,
        "vl_tol_ms": 0.0,
        "track_vr_tol_ms": 0.0,
        "track_dt_tol_ms": 0.0,
        "track_vl_tol_ms": 0.0,
    },
    {
        "name": "b7_hybrid_span1_track1_floor0p08",
        "neighbor_span": 1,
        "support_required": 2,
        "track_span": 1,
        "track_support_required": 1,
        "score_floor": 0.08,
        "vr_tol_ms": 0.20,
        "abs_dt_tol_ms": 0.20,
        "vl_tol_ms": 0.45,
        "track_vr_tol_ms": 0.20,
        "track_dt_tol_ms": 0.20,
        "track_vl_tol_ms": 0.45,
    },
    {
        "name": "b7_hybrid_span1_track1_relaxed",
        "neighbor_span": 1,
        "support_required": 2,
        "track_span": 1,
        "track_support_required": 1,
        "score_floor": 0.08,
        "vr_tol_ms": 0.22,
        "abs_dt_tol_ms": 0.22,
        "vl_tol_ms": 0.50,
        "track_vr_tol_ms": 0.25,
        "track_dt_tol_ms": 0.25,
        "track_vl_tol_ms": 0.55,
    },
    {
        "name": "b7_hybrid_span1_track2_floor0p08",
        "neighbor_span": 1,
        "support_required": 2,
        "track_span": 1,
        "track_support_required": 2,
        "score_floor": 0.08,
        "vr_tol_ms": 0.20,
        "abs_dt_tol_ms": 0.20,
        "vl_tol_ms": 0.45,
        "track_vr_tol_ms": 0.20,
        "track_dt_tol_ms": 0.20,
        "track_vl_tol_ms": 0.45,
    },
]


def support_offsets(offset_sec: float, *, neighbor_span: int) -> list[float]:
    if neighbor_span <= 0:
        return []
    idx = list(WINDOW_OFFSETS_SEC).index(float(offset_sec))
    offsets = []
    for step in range(1, neighbor_span + 1):
        if idx - step >= 0:
            offsets.append(float(WINDOW_OFFSETS_SEC[idx - step]))
        if idx + step < len(WINDOW_OFFSETS_SEC):
            offsets.append(float(WINDOW_OFFSETS_SEC[idx + step]))
    return sorted(set(offsets))


def build_negative_anchor(
    case: base.CaseRef,
    *,
    window_sec: float,
    offset_sec: float,
    neighbor_span: int,
    baseline_rows: dict[tuple[str, float, float], dict[str, Any]],
) -> dict[str, Any] | None:
    support_rows = []
    for neighbor_offset in support_offsets(offset_sec, neighbor_span=neighbor_span):
        row = baseline_rows[(case.case_id, float(window_sec), float(neighbor_offset))]
        if row["selected"] is None:
            continue
        best = row["selected"]["best"]
        if best["delta_tau_ms"] >= 0.0:
            continue
        if not bool(best["physical_valid"]):
            continue
        support_rows.append(best)
    if not support_rows:
        return None
    return {
        "support_count": int(len(support_rows)),
        "target_tau_vr_ms": float(np.median(np.array([row["tau_vr_ms"] for row in support_rows], dtype=np.float64))),
        "target_tau_vl_ms": float(np.median(np.array([row["tau_vl_ms"] for row in support_rows], dtype=np.float64))),
        "target_abs_dt_ms": float(np.median(np.array([abs(row["delta_tau_ms"]) for row in support_rows], dtype=np.float64))),
    }


def build_track_anchor(
    case: base.CaseRef,
    *,
    window_sec: float,
    offset_sec: float,
    track_span: int,
    baseline_rows: dict[tuple[str, float, float], dict[str, Any]],
) -> dict[str, Any] | None:
    track_rows = []
    for neighbor_offset in support_offsets(offset_sec, neighbor_span=track_span):
        row = baseline_rows[(case.case_id, float(window_sec), float(neighbor_offset))]
        if row["selected"] is None:
            continue
        best = row["selected"]["best"]
        if best["delta_tau_ms"] >= 0.0:
            continue
        if not bool(best["physical_valid"]):
            continue
        track_rows.append(best)
    if not track_rows:
        return None
    return {
        "support_count": int(len(track_rows)),
        "target_tau_vr_ms": float(np.median(np.array([row["tau_vr_ms"] for row in track_rows], dtype=np.float64))),
        "target_tau_vl_ms": float(np.median(np.array([row["tau_vl_ms"] for row in track_rows], dtype=np.float64))),
        "target_delta_tau_ms": float(np.median(np.array([row["delta_tau_ms"] for row in track_rows], dtype=np.float64))),
    }


def choose_hybrid_candidate(
    scored_pool: list[dict[str, float]],
    family_anchor: dict[str, Any] | None,
    track_anchor: dict[str, Any] | None,
    *,
    support_required: int,
    track_support_required: int,
    score_floor: float,
    vr_tol_ms: float,
    abs_dt_tol_ms: float,
    vl_tol_ms: float,
    track_vr_tol_ms: float,
    track_dt_tol_ms: float,
    track_vl_tol_ms: float,
) -> dict[str, float] | None:
    if family_anchor is None or int(family_anchor["support_count"]) < int(support_required) or not scored_pool:
        return None
    if track_anchor is None or int(track_anchor["support_count"]) < int(track_support_required):
        return None
    top_score = float(scored_pool[0]["score"])
    candidates = []
    for row in scored_pool:
        if row["delta_tau_ms"] >= 0.0:
            continue
        if row["score"] < float(score_floor) * top_score:
            continue

        family_vr_gap = abs(float(row["tau_vr_ms"]) - float(family_anchor["target_tau_vr_ms"]))
        family_vl_gap = abs(float(row["tau_vl_ms"]) - float(family_anchor["target_tau_vl_ms"]))
        family_dt_gap = abs(abs(float(row["delta_tau_ms"])) - float(family_anchor["target_abs_dt_ms"]))
        if (
            family_vr_gap > float(vr_tol_ms)
            or family_dt_gap > float(abs_dt_tol_ms)
            or family_vl_gap > float(vl_tol_ms)
        ):
            continue

        track_vr_gap = abs(float(row["tau_vr_ms"]) - float(track_anchor["target_tau_vr_ms"]))
        track_vl_gap = abs(float(row["tau_vl_ms"]) - float(track_anchor["target_tau_vl_ms"]))
        track_dt_gap = abs(float(row["delta_tau_ms"]) - float(track_anchor["target_delta_tau_ms"]))
        if (
            track_vr_gap > float(track_vr_tol_ms)
            or track_dt_gap > float(track_dt_tol_ms)
            or track_vl_gap > float(track_vl_tol_ms)
        ):
            continue

        candidates.append(
            (
                track_vr_gap,
                track_dt_gap,
                track_vl_gap,
                family_vr_gap,
                family_dt_gap,
                family_vl_gap,
                -float(row["score"]),
                row,
            )
        )
    if not candidates:
        return None
    return min(candidates, key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6]))[7]


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
    family_anchor = build_negative_anchor(
        case,
        window_sec=window_sec,
        offset_sec=offset_sec,
        neighbor_span=int(variant["neighbor_span"]),
        baseline_rows=baseline_rows,
    )
    track_anchor = build_track_anchor(
        case,
        window_sec=window_sec,
        offset_sec=offset_sec,
        track_span=int(variant["track_span"]),
        baseline_rows=baseline_rows,
    )
    candidate = choose_hybrid_candidate(
        scored_pool,
        family_anchor,
        track_anchor,
        support_required=int(variant["support_required"]),
        track_support_required=int(variant["track_support_required"]),
        score_floor=float(variant["score_floor"]),
        vr_tol_ms=float(variant["vr_tol_ms"]),
        abs_dt_tol_ms=float(variant["abs_dt_tol_ms"]),
        vl_tol_ms=float(variant["vl_tol_ms"]),
        track_vr_tol_ms=float(variant["track_vr_tol_ms"]),
        track_dt_tol_ms=float(variant["track_dt_tol_ms"]),
        track_vl_tol_ms=float(variant["track_vl_tol_ms"]),
    )
    if candidate is None:
        row["negative_anchor"] = family_anchor
        row["track_anchor"] = track_anchor
        row["hybrid_promoted"] = False
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
    row["negative_anchor"] = family_anchor
    row["track_anchor"] = track_anchor
    row["hybrid_promoted"] = True
    row["promotion_reason"] = "family_plus_short_track"
    return row


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
    hard = [row["selected"]["best"] for row in rows if row["case_id"] in lane5a.HARD_CASES and row["selected"] is not None]
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
        "hard_case_delta_tau_mae_ms": float(np.mean(hard_dt)) if hard_dt.size else None,
        "hard_case_win_rate": float(np.mean(hard_win)) if hard_win.size else None,
        "block6_selected_delta_tau_ms": block6["selected"]["best"]["delta_tau_ms"] if block6 else None,
        "block7_selected_delta_tau_ms": block7["selected"]["best"]["delta_tau_ms"] if block7 else None,
        "block6_rescue_pair_rank": block6["strict_ranks"]["rescue_pair_rank"] if block6 and block6["strict_ranks"] else None,
        "block7_rescue_pair_rank": block7["strict_ranks"]["rescue_pair_rank"] if block7 and block7["strict_ranks"] else None,
        "promoted_cases": int(sum(1 for row in rows if row.get("hybrid_promoted", False))),
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
        "# Round 10 Block7 Family Plus Track Hybrid",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        "- Frozen reference: round9 lane1 winner `b7_neg_hold_span1_support2_floor0p08`",
        "",
        "## Summary",
        "",
        "| variant | central_dt_mae_ms | hard_dt_mae_ms | pass_rate | window_mean_std | promoted_cases | block6_hit@1 | block7_hit@1 | block7_hit@3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
                    str(row["central"]["promoted_cases"]),
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
    parser = argparse.ArgumentParser(description="Round 10 block7 family plus short-track hybrid for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round10_block7_family_plus_track_hybrid_0223_{timestamp}"
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
