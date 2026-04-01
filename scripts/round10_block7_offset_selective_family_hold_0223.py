#!/usr/bin/env python3
"""
Round 10 lane 2: block7 offset-selective family hold.

This lane freezes the round9 lane1 winner and only revisits off-center block7
windows that still miss the strict rescue pair. It uses an asymmetric family
anchor:

- negative offsets must be corroborated by a same-VL negative family
- positive offsets must be corroborated by a same-VR negative family

Promotion is only allowed when neighboring offsets agree on both the family
anchor and the expected negative delta-tau range.
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
        "rank_gate": -1,
        "score_floor": 1.0,
        "family_tol_ms": 0.0,
        "companion_tol_ms": 0.0,
        "dt_slack_ms": 0.0,
    },
    {
        "name": "b7_offset_family_hold_gate1_span1_r0p08_f0p06_c0p18_dt0p04",
        "neighbor_span": 1,
        "support_required": 2,
        "rank_gate": 1,
        "score_floor": 0.08,
        "family_tol_ms": 0.06,
        "companion_tol_ms": 0.18,
        "dt_slack_ms": 0.04,
    },
    {
        "name": "b7_offset_family_hold_gate1_span2_r0p08_f0p08_c0p22_dt0p06",
        "neighbor_span": 2,
        "support_required": 2,
        "rank_gate": 1,
        "score_floor": 0.08,
        "family_tol_ms": 0.08,
        "companion_tol_ms": 0.22,
        "dt_slack_ms": 0.06,
    },
    {
        "name": "b7_offset_family_hold_gate3_span2_r0p05_f0p10_c0p25_dt0p08",
        "neighbor_span": 2,
        "support_required": 2,
        "rank_gate": 3,
        "score_floor": 0.05,
        "family_tol_ms": 0.10,
        "companion_tol_ms": 0.25,
        "dt_slack_ms": 0.08,
    },
]


def family_axes(offset_sec: float) -> tuple[str, str] | None:
    if float(offset_sec) < 0.0:
        return "tau_vl_ms", "tau_vr_ms"
    if float(offset_sec) > 0.0:
        return "tau_vr_ms", "tau_vl_ms"
    return None


def corroboration_offsets(offset_sec: float, *, neighbor_span: int) -> list[float]:
    if neighbor_span <= 0:
        return []
    idx = list(WINDOW_OFFSETS_SEC).index(float(offset_sec))
    allowed_sign = int(np.sign(float(offset_sec)))
    offsets = []
    for step in range(1, neighbor_span + 1):
        for neighbor_idx in (idx - step, idx + step):
            if neighbor_idx < 0 or neighbor_idx >= len(WINDOW_OFFSETS_SEC):
                continue
            neighbor_offset = float(WINDOW_OFFSETS_SEC[neighbor_idx])
            neighbor_sign = int(np.sign(neighbor_offset))
            if neighbor_sign not in {0, allowed_sign}:
                continue
            offsets.append(neighbor_offset)
    return sorted(set(offsets))


def build_corroborated_anchor(
    case: base.CaseRef,
    *,
    window_sec: float,
    offset_sec: float,
    neighbor_span: int,
    support_required: int,
    family_tol_ms: float,
    baseline_rows: dict[tuple[str, float, float], dict[str, Any]],
) -> dict[str, Any] | None:
    axes = family_axes(offset_sec)
    if axes is None:
        return None
    family_key, companion_key = axes

    support_rows = []
    for neighbor_offset in corroboration_offsets(offset_sec, neighbor_span=neighbor_span):
        row = baseline_rows[(case.case_id, float(window_sec), float(neighbor_offset))]
        if row["selected"] is None:
            continue
        best = dict(row["selected"]["best"])
        if best["delta_tau_ms"] >= 0.0:
            continue
        if not bool(best["physical_valid"]):
            continue
        best["offset_sec"] = float(neighbor_offset)
        support_rows.append(best)
    if len(support_rows) < int(support_required):
        return None

    clusters = []
    for seed in support_rows:
        cluster = [row for row in support_rows if abs(float(row[family_key]) - float(seed[family_key])) <= float(family_tol_ms)]
        if len(cluster) < int(support_required):
            continue
        family_vals = np.array([float(row[family_key]) for row in cluster], dtype=np.float64)
        companion_vals = np.array([float(row[companion_key]) for row in cluster], dtype=np.float64)
        dt_vals = np.array([abs(float(row["delta_tau_ms"])) for row in cluster], dtype=np.float64)
        clusters.append(
            {
                "support_count": int(len(cluster)),
                "family_key": family_key,
                "companion_key": companion_key,
                "target_family_ms": float(np.median(family_vals)),
                "target_companion_ms": float(np.median(companion_vals)),
                "min_abs_dt_ms": float(np.min(dt_vals)),
                "max_abs_dt_ms": float(np.max(dt_vals)),
                "target_abs_dt_ms": float(np.median(dt_vals)),
                "family_spread_ms": float(np.max(family_vals) - np.min(family_vals)),
                "dt_spread_ms": float(np.max(dt_vals) - np.min(dt_vals)),
                "neighbor_offsets": sorted(
                    {
                        float(row.get("offset_sec", np.nan))
                        for row in cluster
                        if "offset_sec" in row and np.isfinite(float(row["offset_sec"]))
                    }
                ),
                "support_scores": [float(row["score"]) for row in cluster],
            }
        )
    if not clusters:
        return None

    best_cluster = sorted(
        clusters,
        key=lambda row: (
            -int(row["support_count"]),
            float(row["family_spread_ms"]),
            float(row["dt_spread_ms"]),
            -float(np.mean(np.array(row["support_scores"], dtype=np.float64))),
        ),
    )[0]
    return best_cluster


def range_gap(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return float(lo - value)
    if value > hi:
        return float(value - hi)
    return 0.0


def candidate_tuple(
    row: dict[str, float],
    anchor: dict[str, Any],
    *,
    companion_tol_ms: float,
    dt_slack_ms: float,
) -> tuple[float, float, float, float, float]:
    family_key = str(anchor["family_key"])
    companion_key = str(anchor["companion_key"])
    abs_dt = abs(float(row["delta_tau_ms"]))
    lo = float(anchor["min_abs_dt_ms"]) - float(dt_slack_ms)
    hi = float(anchor["max_abs_dt_ms"]) + float(dt_slack_ms)
    return (
        range_gap(abs_dt, lo, hi),
        abs(float(row[family_key]) - float(anchor["target_family_ms"])),
        abs(float(row[companion_key]) - float(anchor["target_companion_ms"])),
        abs(abs_dt - float(anchor["target_abs_dt_ms"])),
        -float(row["score"]),
    )


def choose_corroborated_candidate(
    scored_pool: list[dict[str, float]],
    anchor: dict[str, Any] | None,
    baseline_best: dict[str, Any],
    *,
    score_floor: float,
    family_tol_ms: float,
    companion_tol_ms: float,
    dt_slack_ms: float,
) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    if anchor is None or not scored_pool:
        return None, None

    top_score = float(scored_pool[0]["score"])
    candidates = []
    for row in scored_pool:
        if row["delta_tau_ms"] >= 0.0:
            continue
        if row["score"] < float(score_floor) * top_score:
            continue
        metrics = candidate_tuple(row, anchor, companion_tol_ms=companion_tol_ms, dt_slack_ms=dt_slack_ms)
        if metrics[0] > 1e-9:
            continue
        if metrics[1] > float(family_tol_ms):
            continue
        if metrics[2] > float(companion_tol_ms):
            continue
        candidates.append((metrics, row))
    if not candidates:
        return None, None

    chosen_metrics, chosen_row = min(candidates, key=lambda item: item[0])
    current_metrics = candidate_tuple(
        baseline_best,
        anchor,
        companion_tol_ms=companion_tol_ms,
        dt_slack_ms=dt_slack_ms,
    )
    if current_metrics <= chosen_metrics:
        return None, {
            "baseline_metrics": {
                "range_gap_ms": float(current_metrics[0]),
                "family_gap_ms": float(current_metrics[1]),
                "companion_gap_ms": float(current_metrics[2]),
                "target_abs_dt_gap_ms": float(current_metrics[3]),
            },
            "candidate_metrics": {
                "range_gap_ms": float(chosen_metrics[0]),
                "family_gap_ms": float(chosen_metrics[1]),
                "companion_gap_ms": float(chosen_metrics[2]),
                "target_abs_dt_gap_ms": float(chosen_metrics[3]),
            },
            "kept_baseline": True,
        }
    return chosen_row, {
        "baseline_metrics": {
            "range_gap_ms": float(current_metrics[0]),
            "family_gap_ms": float(current_metrics[1]),
            "companion_gap_ms": float(current_metrics[2]),
            "target_abs_dt_gap_ms": float(current_metrics[3]),
        },
        "candidate_metrics": {
            "range_gap_ms": float(chosen_metrics[0]),
            "family_gap_ms": float(chosen_metrics[1]),
            "companion_gap_ms": float(chosen_metrics[2]),
            "target_abs_dt_gap_ms": float(chosen_metrics[3]),
        },
        "kept_baseline": False,
    }


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
    if variant["name"] == "control_round9a_ref" or case.case_id != BLOCK7_CASE_ID or abs(float(offset_sec)) < 1e-9:
        return row

    current_rank = None
    if row.get("strict_ranks") is not None:
        current_rank = row["strict_ranks"].get("rescue_pair_rank")
    if current_rank is not None and int(current_rank) <= int(variant["rank_gate"]):
        row["offset_family_anchor"] = None
        row["offset_family_promoted"] = False
        row["promotion_reason"] = None
        row["promotion_skip_reason"] = "baseline_already_within_rank_gate"
        return row

    item, scored_pool, _ = lane9a.lane8.evaluate_pool(case, full_signals, window_sec=window_sec, offset_sec=offset_sec)
    anchor = build_corroborated_anchor(
        case,
        window_sec=window_sec,
        offset_sec=offset_sec,
        neighbor_span=int(variant["neighbor_span"]),
        support_required=int(variant["support_required"]),
        family_tol_ms=float(variant["family_tol_ms"]),
        baseline_rows=baseline_rows,
    )
    candidate, candidate_debug = choose_corroborated_candidate(
        scored_pool,
        anchor,
        baseline_row["selected"]["best"],
        score_floor=float(variant["score_floor"]),
        family_tol_ms=float(variant["family_tol_ms"]),
        companion_tol_ms=float(variant["companion_tol_ms"]),
        dt_slack_ms=float(variant["dt_slack_ms"]),
    )
    row["offset_family_anchor"] = anchor
    row["offset_family_candidate_debug"] = candidate_debug
    if candidate is None:
        row["offset_family_promoted"] = False
        row["promotion_reason"] = None
        row["promotion_skip_reason"] = (
            "baseline_better_match" if candidate_debug is not None and bool(candidate_debug.get("kept_baseline")) else "no_corroborated_candidate"
        )
        return row

    final_best = lane9a.lane8.annotate_best(candidate, item["reference"], case.case_id)
    selected_pool = [candidate] + [
        cand
        for cand in scored_pool
        if not (abs(cand["tau_vl_ms"] - candidate["tau_vl_ms"]) < 1e-9 and abs(cand["tau_vr_ms"] - candidate["tau_vr_ms"]) < 1e-9)
    ]
    row["selected"] = {"best": final_best, "top_pairs": selected_pool[:10], "num_pairs": len(selected_pool)}
    row["strict_ranks"] = lane5d.strict_ranks(selected_pool, case.case_id)
    row["offset_family_promoted"] = True
    row["promotion_reason"] = "offset_selective_family_hold"
    row["promotion_skip_reason"] = None
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
    promoted_rows = [row for row in subset if row.get("offset_family_promoted", False)]
    return {
        "variant": variant_name,
        "central": central,
        "window_stability_mean_std_ms": float(np.mean(case_std)) if case_std else None,
        "window_stability_max_std_ms": float(np.max(case_std)) if case_std else None,
        "hard_cases": hard_stats,
        "setting_pass_rate": float(np.mean([1.0 if row["setting_pass"] else 0.0 for row in per_setting])) if per_setting else None,
        "promoted_count": int(len(promoted_rows)),
        "promoted_offsets": [
            {"window_sec": float(row["window_sec"]), "offset_sec": float(row["offset_sec"])} for row in promoted_rows
        ],
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
        "# Round 10 Block7 Offset-Selective Family Hold",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        "- Frozen reference: round9 lane1 winner `b7_neg_hold_span1_support2_floor0p08`",
        "",
        "## Summary",
        "",
        "| variant | central_dt_mae_ms | hard_dt_mae_ms | pass_rate | window_mean_std | promoted | block6_hit@1 | block7_hit@1 | block7_hit@3 |",
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
                    str(row["promoted_count"]),
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
            f"- promoted_count: `{winner['promoted_count']}`",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 10 block7 offset-selective family hold for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round10_block7_offset_selective_family_hold_0223_{timestamp}"
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
