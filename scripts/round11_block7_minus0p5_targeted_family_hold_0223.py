#!/usr/bin/env python3
"""
Round 11: targeted block7 offset=-0.5 family-hold sweep.

This lane freezes the round10 winner
`b7_offset_family_hold_gate1_span1_r0p08_f0p06_c0p18_dt0p04` and only revisits
the remaining block7 miss at window=5.0, offset=-0.5. The intervention is
limited to immediate-neighbor corroboration inside that window.
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
import round10_block7_offset_selective_family_hold_0223 as round10


WINDOW_SECS = round10.WINDOW_SECS
WINDOW_OFFSETS_SEC = round10.WINDOW_OFFSETS_SEC
BLOCK7_CASE_ID = round10.BLOCK7_CASE_ID
TARGET_WINDOW_SEC = 5.0
TARGET_OFFSET_SEC = -0.5
ROUND10_WINNER = next(
    variant for variant in round10.VARIANTS if variant["name"] == "b7_offset_family_hold_gate1_span1_r0p08_f0p06_c0p18_dt0p04"
)


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "control_round10_winner",
        "mode": "control",
        "support_offsets": [],
        "support_required": 99,
        "aggregate": "mean",
        "score_floor": 1.0,
        "vl_tol_ms": 0.0,
        "vr_tol_ms": 0.0,
        "dt_tol_ms": 0.0,
    },
    {
        "name": "b7_m05_bridge_mean3_r0p15_vl0p20_vr0p22_dt0p10",
        "mode": "bridge",
        "support_offsets": [-1.0, 0.0, 0.5],
        "support_required": 3,
        "aggregate": "mean",
        "score_floor": 0.15,
        "vl_tol_ms": 0.20,
        "vr_tol_ms": 0.22,
        "dt_tol_ms": 0.10,
    },
    {
        "name": "b7_m05_bridge_mean2_r0p15_vl0p18_vr0p18_dt0p10",
        "mode": "bridge",
        "support_offsets": [-1.0, 0.0],
        "support_required": 2,
        "aggregate": "mean",
        "score_floor": 0.15,
        "vl_tol_ms": 0.18,
        "vr_tol_ms": 0.18,
        "dt_tol_ms": 0.10,
    },
    {
        "name": "b7_m05_bridge_median3_r0p20_vl0p18_vr0p15_dt0p12",
        "mode": "bridge",
        "support_offsets": [-1.0, 0.0, 0.5],
        "support_required": 3,
        "aggregate": "median",
        "score_floor": 0.20,
        "vl_tol_ms": 0.18,
        "vr_tol_ms": 0.15,
        "dt_tol_ms": 0.12,
    },
]


def target_row(case_id: str, window_sec: float, offset_sec: float) -> bool:
    return case_id == BLOCK7_CASE_ID and abs(window_sec - TARGET_WINDOW_SEC) < 1e-9 and abs(offset_sec - TARGET_OFFSET_SEC) < 1e-9


def aggregate_value(values: list[float], aggregate: str) -> float:
    array = np.array(values, dtype=np.float64)
    if aggregate == "median":
        return float(np.median(array))
    return float(np.mean(array))


def build_bridge_anchor(
    baseline_rows: dict[tuple[str, float, float], dict[str, Any]],
    *,
    case_id: str,
    window_sec: float,
    variant: dict[str, Any],
) -> dict[str, Any] | None:
    support_rows = []
    for support_offset in variant["support_offsets"]:
        row = baseline_rows[(case_id, float(window_sec), float(support_offset))]
        best = row["selected"]["best"]
        if float(best["delta_tau_ms"]) >= 0.0 or not bool(best["physical_valid"]):
            continue
        support_rows.append({"offset_sec": float(support_offset), "best": best})
    if len(support_rows) < int(variant["support_required"]):
        return None

    return {
        "support_offsets": [row["offset_sec"] for row in support_rows],
        "aggregate": variant["aggregate"],
        "target_tau_vl_ms": aggregate_value([float(row["best"]["tau_vl_ms"]) for row in support_rows], str(variant["aggregate"])),
        "target_tau_vr_ms": aggregate_value([float(row["best"]["tau_vr_ms"]) for row in support_rows], str(variant["aggregate"])),
        "target_abs_dt_ms": aggregate_value([abs(float(row["best"]["delta_tau_ms"])) for row in support_rows], str(variant["aggregate"])),
    }


def candidate_metrics(row: dict[str, float], anchor: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        abs(float(row["tau_vl_ms"]) - float(anchor["target_tau_vl_ms"])),
        abs(float(row["tau_vr_ms"]) - float(anchor["target_tau_vr_ms"])),
        abs(abs(float(row["delta_tau_ms"])) - float(anchor["target_abs_dt_ms"])),
        -float(row["score"]),
    )


def choose_bridge_candidate(
    scored_pool: list[dict[str, float]],
    anchor: dict[str, Any] | None,
    baseline_best: dict[str, Any],
    *,
    score_floor: float,
    vl_tol_ms: float,
    vr_tol_ms: float,
    dt_tol_ms: float,
) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    if anchor is None or not scored_pool:
        return None, None

    top_score = float(scored_pool[0]["score"])
    candidates = []
    for row in scored_pool:
        if float(row["delta_tau_ms"]) >= 0.0:
            continue
        if float(row["score"]) < float(score_floor) * top_score:
            continue
        metrics = candidate_metrics(row, anchor)
        if metrics[0] > float(vl_tol_ms) or metrics[1] > float(vr_tol_ms) or metrics[2] > float(dt_tol_ms):
            continue
        candidates.append((metrics, row))
    if not candidates:
        return None, None

    chosen_metrics, chosen_row = min(candidates, key=lambda item: item[0])
    current_metrics = candidate_metrics(baseline_best, anchor)
    debug = {
        "baseline_metrics": {
            "vl_gap_ms": float(current_metrics[0]),
            "vr_gap_ms": float(current_metrics[1]),
            "dt_gap_ms": float(current_metrics[2]),
        },
        "candidate_metrics": {
            "vl_gap_ms": float(chosen_metrics[0]),
            "vr_gap_ms": float(chosen_metrics[1]),
            "dt_gap_ms": float(chosen_metrics[2]),
        },
        "kept_baseline": bool(current_metrics <= chosen_metrics),
    }
    if current_metrics <= chosen_metrics:
        return None, debug
    return chosen_row, debug


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
    if variant["mode"] == "control" or not target_row(case.case_id, float(window_sec), float(offset_sec)):
        return row

    item, scored_pool, _ = round10.lane9a.lane8.evaluate_pool(case, full_signals, window_sec=window_sec, offset_sec=offset_sec)
    anchor = build_bridge_anchor(
        baseline_rows,
        case_id=case.case_id,
        window_sec=float(window_sec),
        variant=variant,
    )
    candidate, candidate_debug = choose_bridge_candidate(
        scored_pool,
        anchor,
        baseline_row["selected"]["best"],
        score_floor=float(variant["score_floor"]),
        vl_tol_ms=float(variant["vl_tol_ms"]),
        vr_tol_ms=float(variant["vr_tol_ms"]),
        dt_tol_ms=float(variant["dt_tol_ms"]),
    )
    row["target_bridge_anchor"] = anchor
    row["target_bridge_candidate_debug"] = candidate_debug
    if candidate is None:
        row["target_bridge_promoted"] = False
        row["promotion_reason"] = None
        row["promotion_skip_reason"] = (
            "baseline_better_anchor_match" if candidate_debug is not None and bool(candidate_debug.get("kept_baseline")) else "no_bridge_candidate"
        )
        return row

    final_best = round10.lane9a.lane8.annotate_best(candidate, item["reference"], case.case_id)
    selected_pool = [candidate] + [
        pool_row
        for pool_row in scored_pool
        if not (abs(pool_row["tau_vl_ms"] - candidate["tau_vl_ms"]) < 1e-9 and abs(pool_row["tau_vr_ms"] - candidate["tau_vr_ms"]) < 1e-9)
    ]
    row["selected"] = {"best": final_best, "top_pairs": selected_pool[:10], "num_pairs": len(selected_pool)}
    row["strict_ranks"] = lane5d.strict_ranks(selected_pool, case.case_id)
    row["target_bridge_promoted"] = True
    row["promotion_reason"] = "minus0p5_targeted_bridge_hold"
    row["promotion_skip_reason"] = None
    return row


def summarize_variant(rows: list[dict[str, Any]], variant_name: str) -> dict[str, Any]:
    subset = [row for row in rows if row["variant"] == variant_name]
    central_rows = [row for row in subset if abs(row["window_sec"] - 5.0) < 1e-9 and abs(row["offset_sec"]) < 1e-9]
    central = round10.summarize_setting(central_rows)

    stability_subset = [row for row in subset if abs(row["window_sec"] - 5.0) < 1e-9]
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stability_subset:
        by_case[row["case_id"]].append(row)

    case_std = []
    hard_stats = {}
    for case_id, items in by_case.items():
        items.sort(key=lambda item: item["offset_sec"])
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
            setting = round10.summarize_setting(rows_setting)
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

    blocker_row = next(
        row
        for row in subset
        if target_row(str(row["case_id"]), float(row["window_sec"]), float(row["offset_sec"]))
    )
    return {
        "variant": variant_name,
        "central": central,
        "window_stability_mean_std_ms": float(np.mean(case_std)) if case_std else None,
        "window_stability_max_std_ms": float(np.max(case_std)) if case_std else None,
        "hard_cases": hard_stats,
        "setting_pass_rate": float(np.mean([1.0 if row["setting_pass"] else 0.0 for row in per_setting])) if per_setting else None,
        "target_blocker": {
            "window_sec": TARGET_WINDOW_SEC,
            "offset_sec": TARGET_OFFSET_SEC,
            "delta_tau_ms": float(blocker_row["selected"]["best"]["delta_tau_ms"]),
            "delta_tau_abs_err_ms": float(blocker_row["selected"]["best"]["delta_tau_abs_err_ms"]),
            "rescue_pair_rank": None if blocker_row["strict_ranks"] is None else blocker_row["strict_ranks"].get("rescue_pair_rank"),
            "promoted": bool(blocker_row.get("target_bridge_promoted", False)),
        },
        "per_setting": per_setting,
    }


def non_regression(summary: dict[str, Any], baseline: dict[str, Any]) -> bool:
    block6 = summary["hard_cases"].get("block6_n04_19", {})
    baseline_block6 = baseline["hard_cases"].get("block6_n04_19", {})
    return bool(
        float(summary["setting_pass_rate"] or 0.0) >= float(baseline["setting_pass_rate"] or 0.0)
        and float(summary["central"]["delta_tau_mae_ms"] or np.inf) <= float(baseline["central"]["delta_tau_mae_ms"] or np.inf)
        and float(summary["central"]["hard_case_delta_tau_mae_ms"] or np.inf)
        <= float(baseline["central"]["hard_case_delta_tau_mae_ms"] or np.inf)
        and int(block6.get("hit_at_1", 0)) >= int(baseline_block6.get("hit_at_1", 0))
    )


def local_improvement(summary: dict[str, Any], baseline: dict[str, Any]) -> bool:
    current_rank = summary["target_blocker"]["rescue_pair_rank"]
    baseline_rank = baseline["target_blocker"]["rescue_pair_rank"]
    if baseline_rank is None and current_rank is not None:
        return True
    if baseline_rank is not None and current_rank is not None and int(current_rank) < int(baseline_rank):
        return True
    return bool(float(summary["target_blocker"]["delta_tau_abs_err_ms"]) < float(baseline["target_blocker"]["delta_tau_abs_err_ms"]))


def choose_winner(summary_rows: list[dict[str, Any]], baseline_name: str) -> dict[str, Any]:
    baseline = next(row for row in summary_rows if row["variant"] == baseline_name)
    improved = [row for row in summary_rows if row["variant"] != baseline_name and non_regression(row, baseline) and local_improvement(row, baseline)]
    if not improved:
        return baseline
    return sorted(
        improved,
        key=lambda row: (
            float(row["target_blocker"]["delta_tau_abs_err_ms"]),
            -(row["hard_cases"].get(BLOCK7_CASE_ID, {}).get("hit_at_1", 0)),
            -(row["hard_cases"].get(BLOCK7_CASE_ID, {}).get("hit_at_3", 0)),
            -float(row["setting_pass_rate"] or 0.0),
            float(row["window_stability_mean_std_ms"] or np.inf),
        ),
    )[0]


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    baseline_name = "control_round10_winner"
    baseline = next(row for row in payload["summary"]["variants"] if row["variant"] == baseline_name)
    winner = payload["summary"]["winner"]
    lines = [
        "# Round 11 Block7 Offset -0.5 Targeted Family Hold",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        f"- Frozen reference: round10 winner `{payload['baseline_variant']}`",
        "",
        "## Summary",
        "",
        "| variant | blocker_dt_err_ms | blocker_rank | promoted | central_dt_mae_ms | hard_dt_mae_ms | pass_rate | block6_hit@1 | block7_hit@1 | block7_hit@3 | stability |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["variants"]:
        block6 = row["hard_cases"].get("block6_n04_19", {})
        block7 = row["hard_cases"].get(BLOCK7_CASE_ID, {})
        blocker_rank = row["target_blocker"]["rescue_pair_rank"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    f"{row['target_blocker']['delta_tau_abs_err_ms']:.3f}",
                    "None" if blocker_rank is None else str(blocker_rank),
                    str(row["target_blocker"]["promoted"]),
                    f"{row['central']['delta_tau_mae_ms']:.3f}",
                    f"{row['central']['hard_case_delta_tau_mae_ms']:.3f}",
                    f"{row['setting_pass_rate']:.3f}",
                    str(block6.get("hit_at_1", 0)),
                    str(block7.get("hit_at_1", 0)),
                    str(block7.get("hit_at_3", 0)),
                    f"{row['window_stability_mean_std_ms']:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Winner",
            "",
            f"- baseline control: `{baseline['variant']}`",
            f"- selected winner: `{winner['variant']}`",
            f"- non-regression vs control: `{non_regression(winner, baseline)}`",
            f"- blocker improvement vs control: `{local_improvement(winner, baseline)}`",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 11 targeted block7 offset=-0.5 family-hold sweep for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round11_block7_minus0p5_targeted_family_hold_0223_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    full_signals = {case.case_id: round10.lane9a.lane8.lane2.lane1.load_full_case_signals(case, args.data_root) for case in base.CASES}

    lane6_rows: dict[tuple[str, float, float], dict[str, Any]] = {}
    for case in base.CASES:
        for window_sec in WINDOW_SECS:
            for offset_sec in WINDOW_OFFSETS_SEC:
                lane6_rows[(case.case_id, float(window_sec), float(offset_sec))] = round10.lane9a.lane8.lane6c.evaluate_variant_case(
                    case,
                    full_signals[case.case_id],
                    window_sec=window_sec,
                    offset_sec=offset_sec,
                    variant=round10.lane9a.lane8.BASELINE_VARIANT,
                )

    round8_rows: dict[tuple[str, float, float], dict[str, Any]] = {}
    for case in base.CASES:
        for window_sec in WINDOW_SECS:
            for offset_sec in WINDOW_OFFSETS_SEC:
                round8_rows[(case.case_id, float(window_sec), float(offset_sec))] = round10.lane9a.lane8.evaluate_variant_case(
                    case,
                    full_signals[case.case_id],
                    window_sec=window_sec,
                    offset_sec=offset_sec,
                    variant=round10.lane9a.BASELINE_VARIANT,
                    baseline_rows=lane6_rows,
                )

    round9_rows: dict[tuple[str, float, float], dict[str, Any]] = {}
    for case in base.CASES:
        for window_sec in WINDOW_SECS:
            for offset_sec in WINDOW_OFFSETS_SEC:
                round9_rows[(case.case_id, float(window_sec), float(offset_sec))] = round10.lane9a.evaluate_variant_case(
                    case,
                    full_signals[case.case_id],
                    window_sec=window_sec,
                    offset_sec=offset_sec,
                    variant=round10.BASELINE_VARIANT,
                    baseline_rows=round8_rows,
                )

    baseline_rows: dict[tuple[str, float, float], dict[str, Any]] = {}
    for case in base.CASES:
        for window_sec in WINDOW_SECS:
            for offset_sec in WINDOW_OFFSETS_SEC:
                baseline_rows[(case.case_id, float(window_sec), float(offset_sec))] = round10.evaluate_variant_case(
                    case,
                    full_signals[case.case_id],
                    window_sec=window_sec,
                    offset_sec=offset_sec,
                    variant=ROUND10_WINNER,
                    baseline_rows=round9_rows,
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
    winner = choose_winner(summaries, baseline_name="control_round10_winner")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "baseline_variant": ROUND10_WINNER["name"],
        "variants": VARIANTS,
        "results": rows,
        "summary": {"variants": summaries, "winner": winner},
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir / "report.md", payload)
    print(f"[ok] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
