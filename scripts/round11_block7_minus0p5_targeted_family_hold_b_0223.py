#!/usr/bin/env python3
"""
Round 11: narrow block7 offset=-0.5 targeted family hold backup sweep.

This starts from the round10 offset-selective family-hold baseline and only
revisits the remaining block7 offset=-0.5 blocker. The added rule is narrow:

- only block7
- only offset=-0.5
- only immediate left/center support
- if the center baseline is positive, fall back to the best negative center
  support before building the asymmetric midpoint anchor
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
import round10_block7_offset_selective_family_hold_0223 as lane10


WINDOW_SECS = lane10.WINDOW_SECS
WINDOW_OFFSETS_SEC = lane10.WINDOW_OFFSETS_SEC
BASELINE_VARIANT = lane10.BASELINE_VARIANT
BLOCK7_CASE_ID = lane10.BLOCK7_CASE_ID
TARGET_OFFSET_SEC = -0.5
LEFT_SUPPORT_OFFSET_SEC = -1.0
CENTER_SUPPORT_OFFSET_SEC = 0.0


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "control_round10_ref",
        "score_floor": 1.0,
        "blend_left": 0.50,
        "family_tol_ms": 0.0,
        "companion_tol_ms": 0.0,
        "dt_tol_ms": 0.0,
    },
    {
        "name": "b7_m05_negctr_blend50_floor0p12_f0p40_c0p40_dt0p22",
        "score_floor": 0.12,
        "blend_left": 0.50,
        "family_tol_ms": 0.40,
        "companion_tol_ms": 0.40,
        "dt_tol_ms": 0.22,
    },
    {
        "name": "b7_m05_negctr_blend35_floor0p10_f0p45_c0p45_dt0p26",
        "score_floor": 0.10,
        "blend_left": 0.35,
        "family_tol_ms": 0.45,
        "companion_tol_ms": 0.45,
        "dt_tol_ms": 0.26,
    },
]


def best_negative_support(
    case: base.CaseRef,
    full_signals: dict[str, np.ndarray],
    *,
    window_sec: float,
    offset_sec: float,
    baseline_rows: dict[tuple[str, float, float], dict[str, Any]],
) -> dict[str, Any] | None:
    baseline_best = baseline_rows[(case.case_id, float(window_sec), float(offset_sec))]["selected"]["best"]
    if baseline_best["delta_tau_ms"] < 0.0 and bool(baseline_best["physical_valid"]):
        return dict(baseline_best)

    _, scored_pool, _ = lane9a.lane8.evaluate_pool(case, full_signals, window_sec=window_sec, offset_sec=offset_sec)
    for row in scored_pool:
        if float(row["delta_tau_ms"]) >= 0.0:
            continue
        if float(row["tau_vl_ms"]) <= 0.0 or float(row["tau_vr_ms"]) <= 0.0:
            continue
        if abs(float(row["delta_tau_ms"])) > lane9a.lane8.lane2.DELTA_LIMIT_MS:
            continue
        return dict(row)
    return None


def build_midpoint_anchor(
    case: base.CaseRef,
    full_signals: dict[str, np.ndarray],
    *,
    window_sec: float,
    variant: dict[str, Any],
    baseline_rows: dict[tuple[str, float, float], dict[str, Any]],
) -> dict[str, float] | None:
    left_best = best_negative_support(
        case,
        full_signals,
        window_sec=window_sec,
        offset_sec=LEFT_SUPPORT_OFFSET_SEC,
        baseline_rows=baseline_rows,
    )
    center_best = best_negative_support(
        case,
        full_signals,
        window_sec=window_sec,
        offset_sec=CENTER_SUPPORT_OFFSET_SEC,
        baseline_rows=baseline_rows,
    )
    if left_best is None or center_best is None:
        return None

    blend_left = float(variant["blend_left"])
    blend_center = 1.0 - blend_left
    return {
        "target_tau_vl_ms": blend_left * float(left_best["tau_vl_ms"]) + blend_center * float(center_best["tau_vl_ms"]),
        "target_tau_vr_ms": blend_left * float(left_best["tau_vr_ms"]) + blend_center * float(center_best["tau_vr_ms"]),
        "target_abs_dt_ms": blend_left * abs(float(left_best["delta_tau_ms"]))
        + blend_center * abs(float(center_best["delta_tau_ms"])),
        "left_support": {
            "tau_vl_ms": float(left_best["tau_vl_ms"]),
            "tau_vr_ms": float(left_best["tau_vr_ms"]),
            "delta_tau_ms": float(left_best["delta_tau_ms"]),
        },
        "center_support": {
            "tau_vl_ms": float(center_best["tau_vl_ms"]),
            "tau_vr_ms": float(center_best["tau_vr_ms"]),
            "delta_tau_ms": float(center_best["delta_tau_ms"]),
        },
    }


def choose_midpoint_candidate(
    scored_pool: list[dict[str, float]],
    anchor: dict[str, float] | None,
    *,
    variant: dict[str, Any],
) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    if anchor is None or not scored_pool:
        return None, None

    top_score = float(scored_pool[0]["score"])
    candidates = []
    for row in scored_pool:
        if float(row["delta_tau_ms"]) >= 0.0:
            continue
        if top_score > 0.0 and float(row["score"]) < float(variant["score_floor"]) * top_score:
            continue
        family_gap = abs(float(row["tau_vl_ms"]) - float(anchor["target_tau_vl_ms"]))
        companion_gap = abs(float(row["tau_vr_ms"]) - float(anchor["target_tau_vr_ms"]))
        dt_gap = abs(abs(float(row["delta_tau_ms"])) - float(anchor["target_abs_dt_ms"]))
        if family_gap > float(variant["family_tol_ms"]):
            continue
        if companion_gap > float(variant["companion_tol_ms"]):
            continue
        if dt_gap > float(variant["dt_tol_ms"]):
            continue
        candidates.append(((family_gap, companion_gap, dt_gap, -float(row["score"])), row))
    if not candidates:
        return None, None

    metrics, chosen = min(candidates, key=lambda item: item[0])
    return chosen, {
        "target_tau_vl_ms": float(anchor["target_tau_vl_ms"]),
        "target_tau_vr_ms": float(anchor["target_tau_vr_ms"]),
        "target_abs_dt_ms": float(anchor["target_abs_dt_ms"]),
        "family_gap_ms": float(metrics[0]),
        "companion_gap_ms": float(metrics[1]),
        "abs_dt_gap_ms": float(metrics[2]),
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
    if (
        variant["name"] == "control_round10_ref"
        or case.case_id != BLOCK7_CASE_ID
        or abs(float(offset_sec) - TARGET_OFFSET_SEC) > 1e-9
    ):
        return row

    anchor = build_midpoint_anchor(
        case,
        full_signals,
        window_sec=window_sec,
        variant=variant,
        baseline_rows=baseline_rows,
    )
    item, scored_pool, _ = lane9a.lane8.evaluate_pool(case, full_signals, window_sec=window_sec, offset_sec=offset_sec)
    candidate, candidate_debug = choose_midpoint_candidate(scored_pool, anchor, variant=variant)
    row["targeted_anchor"] = anchor
    row["targeted_candidate_debug"] = candidate_debug
    if candidate is None:
        row["targeted_promoted"] = False
        row["promotion_reason"] = None
        row["promotion_skip_reason"] = "no_targeted_candidate"
        return row

    final_best = lane9a.lane8.annotate_best(dict(candidate), item["reference"], case.case_id)
    selected_pool = [candidate] + [
        cand
        for cand in scored_pool
        if not (abs(cand["tau_vl_ms"] - candidate["tau_vl_ms"]) < 1e-9 and abs(cand["tau_vr_ms"] - candidate["tau_vr_ms"]) < 1e-9)
    ]
    row["selected"] = {"best": final_best, "top_pairs": selected_pool[:10], "num_pairs": len(selected_pool)}
    row["strict_ranks"] = lane5d.strict_ranks(selected_pool, case.case_id)
    row["targeted_promoted"] = True
    row["promotion_reason"] = "block7_minus0p5_negative_center_midpoint_hold"
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
    promoted_rows = [row for row in subset if row.get("targeted_promoted", False)]
    return {
        "variant": variant_name,
        "central": central,
        "window_stability_mean_std_ms": float(np.mean(case_std)) if case_std else None,
        "window_stability_max_std_ms": float(np.max(case_std)) if case_std else None,
        "hard_cases": hard_stats,
        "setting_pass_rate": float(np.mean([1.0 if row["setting_pass"] else 0.0 for row in per_setting])) if per_setting else None,
        "promoted_count": int(len(promoted_rows)),
        "per_setting": per_setting,
    }


def add_non_regression_flags(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next(row for row in summary_rows if row["variant"] == "control_round10_ref")
    baseline_b6 = baseline["hard_cases"].get("block6_n04_19", {})
    for row in summary_rows:
        row["non_regressing"] = bool(
            (row["central"].get("delta_tau_mae_ms") is not None)
            and (row["central"].get("hard_case_delta_tau_mae_ms") is not None)
            and row["central"]["delta_tau_mae_ms"] <= baseline["central"]["delta_tau_mae_ms"] + 1e-12
            and row["central"]["hard_case_delta_tau_mae_ms"] <= baseline["central"]["hard_case_delta_tau_mae_ms"] + 1e-12
            and float(row["setting_pass_rate"] or 0.0) >= float(baseline["setting_pass_rate"] or 0.0) - 1e-12
            and int(row["hard_cases"].get("block6_n04_19", {}).get("hit_at_1", 0))
            >= int(baseline_b6.get("hit_at_1", 0))
        )
    return summary_rows


def choose_winner(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        summary_rows,
        key=lambda row: (
            -int(bool(row.get("non_regressing"))),
            -float(row["setting_pass_rate"] or 0.0),
            -(row["hard_cases"].get(BLOCK7_CASE_ID, {}).get("hit_at_1", 0)),
            -(row["hard_cases"].get(BLOCK7_CASE_ID, {}).get("hit_at_3", 0)),
            float("inf") if row.get("window_stability_mean_std_ms") is None else row["window_stability_mean_std_ms"],
            float("inf") if row["central"].get("delta_tau_mae_ms") is None else row["central"]["delta_tau_mae_ms"],
        ),
    )[0]


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 11 Block7 Minus0.5 Targeted Family Hold B",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        f"- Frozen reference: round10 baseline winner `{payload['baseline_variant']}`",
        "",
        "## Summary",
        "",
        "| variant | non_regressing | central_dt_mae_ms | hard_dt_mae_ms | pass_rate | window_mean_std | block6_hit@1 | block7_hit@1 | block7_hit@3 | promoted |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["variants"]:
        b6 = row["hard_cases"].get("block6_n04_19", {})
        b7 = row["hard_cases"].get(BLOCK7_CASE_ID, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    str(row.get("non_regressing", False)),
                    f"{row['central']['delta_tau_mae_ms']:.3f}",
                    f"{row['central']['hard_case_delta_tau_mae_ms']:.3f}",
                    f"{row['setting_pass_rate']:.3f}",
                    f"{row['window_stability_mean_std_ms']:.3f}",
                    str(b6.get("hit_at_1", 0)),
                    str(b7.get("hit_at_1", 0)),
                    str(b7.get("hit_at_3", 0)),
                    str(row["promoted_count"]),
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
            f"- non_regressing: `{winner.get('non_regressing', False)}`",
            f"- pass_rate: `{winner['setting_pass_rate']:.3f}`",
            f"- block6 hit@1: `{winner['hard_cases'].get('block6_n04_19', {}).get('hit_at_1', 0)}`",
            f"- block7 hit@1: `{winner['hard_cases'].get(BLOCK7_CASE_ID, {}).get('hit_at_1', 0)}`",
            f"- block7 hit@3: `{winner['hard_cases'].get(BLOCK7_CASE_ID, {}).get('hit_at_3', 0)}`",
            f"- window_stability_mean_std_ms: `{winner['window_stability_mean_std_ms']:.3f}`",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 11 narrow backup sweep for block7 offset=-0.5 on 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round11_block7_minus0p5_targeted_family_hold_b_0223_{timestamp}"
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
    summaries = add_non_regression_flags(summaries)
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
