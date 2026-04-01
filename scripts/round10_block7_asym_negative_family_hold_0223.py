#!/usr/bin/env python3
"""
Round 10 lane 1: block7 asymmetric negative-family carryover.

This lane freezes the round9 lane1 winner
`b7_neg_hold_span1_support2_floor0p08` and only changes the block7 decision
rule. It carries a selected negative-family state across adjacent windows with
bounded hysteresis, using tighter continuity on the right sweep and a broader
left sweep for the remaining off-center block7 failures.
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
ROUND8_BASELINE_VARIANT = lane9a.BASELINE_VARIANT
ROUND9_BASELINE_VARIANT = next(v for v in lane9a.VARIANTS if v["name"] == "b7_neg_hold_span1_support2_floor0p08")
BLOCK7_CASE_ID = lane9a.BLOCK7_CASE_ID
CENTER_OFFSET_SEC = 0.0
RIGHT_SWEEP_OFFSETS = [offset for offset in WINDOW_OFFSETS_SEC if offset > 0.0]
LEFT_SWEEP_OFFSETS = [offset for offset in reversed(WINDOW_OFFSETS_SEC) if offset < 0.0]


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "control_round9_lane1_ref",
        "score_floor": 1.0,
        "right_vr_tol_ms": 0.0,
        "right_vl_tol_ms": 0.0,
        "right_abs_dt_tol_ms": 0.0,
        "right_min_abs_dt_ratio": 9.99,
        "left_vr_tol_ms": 0.0,
        "left_vl_tol_ms": 0.0,
        "left_abs_dt_tol_ms": 0.0,
        "left_min_abs_dt_ratio": 9.99,
        "max_abs_dt_increase_ms": 0.0,
    },
    {
        "name": "b7_carry_sym_r0p05_vr0p12_vl0p15_dt0p15",
        "score_floor": 0.05,
        "right_vr_tol_ms": 0.12,
        "right_vl_tol_ms": 0.15,
        "right_abs_dt_tol_ms": 0.15,
        "right_min_abs_dt_ratio": 0.55,
        "left_vr_tol_ms": 0.12,
        "left_vl_tol_ms": 0.15,
        "left_abs_dt_tol_ms": 0.15,
        "left_min_abs_dt_ratio": 0.55,
        "max_abs_dt_increase_ms": 0.30,
    },
    {
        "name": "b7_carry_asym_left0p25_dt0p27",
        "score_floor": 0.05,
        "right_vr_tol_ms": 0.12,
        "right_vl_tol_ms": 0.15,
        "right_abs_dt_tol_ms": 0.15,
        "right_min_abs_dt_ratio": 0.55,
        "left_vr_tol_ms": 0.12,
        "left_vl_tol_ms": 0.25,
        "left_abs_dt_tol_ms": 0.27,
        "left_min_abs_dt_ratio": 0.55,
        "max_abs_dt_increase_ms": 0.30,
    },
    {
        "name": "b7_carry_asym_left0p25_dt0p30",
        "score_floor": 0.05,
        "right_vr_tol_ms": 0.12,
        "right_vl_tol_ms": 0.15,
        "right_abs_dt_tol_ms": 0.15,
        "right_min_abs_dt_ratio": 0.55,
        "left_vr_tol_ms": 0.12,
        "left_vl_tol_ms": 0.25,
        "left_abs_dt_tol_ms": 0.30,
        "left_min_abs_dt_ratio": 0.55,
        "max_abs_dt_increase_ms": 0.30,
    },
]


def state_from_best(best: dict[str, float] | None) -> dict[str, float] | None:
    if best is None:
        return None
    if float(best["delta_tau_ms"]) >= 0.0 or not bool(best["physical_valid"]):
        return None
    return {
        "target_tau_vl_ms": float(best["tau_vl_ms"]),
        "target_tau_vr_ms": float(best["tau_vr_ms"]),
        "target_abs_dt_ms": float(abs(best["delta_tau_ms"])),
    }


def choose_hysteresis_candidate(
    scored_pool: list[dict[str, float]],
    state: dict[str, float] | None,
    *,
    score_floor: float,
    vr_tol_ms: float,
    vl_tol_ms: float,
    abs_dt_tol_ms: float,
    min_abs_dt_ratio: float,
    max_abs_dt_increase_ms: float,
) -> dict[str, float] | None:
    if state is None or not scored_pool:
        return None
    top_score = float(scored_pool[0]["score"])
    candidates = []
    for row in scored_pool:
        if row["delta_tau_ms"] >= 0.0:
            continue
        if row["score"] < float(score_floor) * top_score:
            continue
        abs_dt = abs(float(row["delta_tau_ms"]))
        prev_abs_dt = float(state["target_abs_dt_ms"])
        if abs_dt < float(min_abs_dt_ratio) * prev_abs_dt:
            continue
        if abs_dt > prev_abs_dt + float(max_abs_dt_increase_ms):
            continue
        vr_gap = abs(float(row["tau_vr_ms"]) - float(state["target_tau_vr_ms"]))
        vl_gap = abs(float(row["tau_vl_ms"]) - float(state["target_tau_vl_ms"]))
        dt_gap = abs(abs_dt - prev_abs_dt)
        if vr_gap > float(vr_tol_ms) or vl_gap > float(vl_tol_ms) or dt_gap > float(abs_dt_tol_ms):
            continue
        candidates.append((vr_gap, vl_gap, dt_gap, -float(row["score"]), row))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))[4]


def block7_window_rows(
    case: base.CaseRef,
    full_signals: dict[str, np.ndarray],
    *,
    window_sec: float,
    variant: dict[str, Any],
    baseline_rows: dict[tuple[str, float, float], dict[str, Any]],
) -> list[dict[str, Any]]:
    offset_rows = {
        float(offset_sec): dict(baseline_rows[(case.case_id, float(window_sec), float(offset_sec))])
        for offset_sec in WINDOW_OFFSETS_SEC
    }
    for row in offset_rows.values():
        row["variant"] = variant["name"]
    if variant["name"] == "control_round9_lane1_ref":
        return [offset_rows[float(offset_sec)] for offset_sec in WINDOW_OFFSETS_SEC]

    contexts: dict[float, tuple[dict[str, Any], list[dict[str, float]]]] = {}
    for offset_sec in WINDOW_OFFSETS_SEC:
        item, scored_pool, _ = lane9a.lane8.evaluate_pool(case, full_signals, window_sec=window_sec, offset_sec=offset_sec)
        contexts[float(offset_sec)] = (item, scored_pool)

    seed_best = offset_rows[CENTER_OFFSET_SEC]["selected"]["best"]
    seed_state = state_from_best(seed_best)
    if seed_state is None:
        return [offset_rows[float(offset_sec)] for offset_sec in WINDOW_OFFSETS_SEC]
    offset_rows[CENTER_OFFSET_SEC]["carry_seed"] = True

    for side_name, sweep_offsets in (("right", RIGHT_SWEEP_OFFSETS), ("left", LEFT_SWEEP_OFFSETS)):
        state = dict(seed_state)
        for offset_sec in sweep_offsets:
            row = offset_rows[float(offset_sec)]
            item, scored_pool = contexts[float(offset_sec)]
            candidate = choose_hysteresis_candidate(
                scored_pool,
                state,
                score_floor=float(variant["score_floor"]),
                vr_tol_ms=float(variant[f"{side_name}_vr_tol_ms"]),
                vl_tol_ms=float(variant[f"{side_name}_vl_tol_ms"]),
                abs_dt_tol_ms=float(variant[f"{side_name}_abs_dt_tol_ms"]),
                min_abs_dt_ratio=float(variant[f"{side_name}_min_abs_dt_ratio"]),
                max_abs_dt_increase_ms=float(variant["max_abs_dt_increase_ms"]),
            )
            if candidate is None:
                baseline_best = row["selected"]["best"] if row["selected"] is not None else None
                carried = state_from_best(baseline_best)
                if carried is not None:
                    state = carried
                row["carry_promoted"] = False
                row["carry_direction"] = side_name
                row["promotion_reason"] = None
                continue

            selected_pool = [candidate] + [
                cand
                for cand in scored_pool
                if not (abs(cand["tau_vl_ms"] - candidate["tau_vl_ms"]) < 1e-9 and abs(cand["tau_vr_ms"] - candidate["tau_vr_ms"]) < 1e-9)
            ]
            row["selected"] = {
                "best": lane9a.lane8.annotate_best(candidate, item["reference"], case.case_id),
                "top_pairs": selected_pool[:10],
                "num_pairs": len(selected_pool),
            }
            row["strict_ranks"] = lane5d.strict_ranks(selected_pool, case.case_id)
            row["carry_promoted"] = True
            row["carry_direction"] = side_name
            row["promotion_reason"] = "family_state_carryover"
            row["carry_state"] = dict(state)
            state = state_from_best(row["selected"]["best"]) or state

    return [offset_rows[float(offset_sec)] for offset_sec in WINDOW_OFFSETS_SEC]


def evaluate_variant_rows(
    case: base.CaseRef,
    full_signals: dict[str, np.ndarray],
    *,
    variant: dict[str, Any],
    baseline_rows: dict[tuple[str, float, float], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for window_sec in WINDOW_SECS:
        if case.case_id != BLOCK7_CASE_ID:
            for offset_sec in WINDOW_OFFSETS_SEC:
                row = dict(baseline_rows[(case.case_id, float(window_sec), float(offset_sec))])
                row["variant"] = variant["name"]
                rows.append(row)
            continue
        rows.extend(
            block7_window_rows(
                case,
                full_signals,
                window_sec=window_sec,
                variant=variant,
                baseline_rows=baseline_rows,
            )
        )
    return rows


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
            float("inf") if row["window_stability_mean_std_ms"] is None else row["window_stability_mean_std_ms"],
        ),
    )[0]


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 10 Block7 Asymmetric Negative-Family Carryover",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        f"- Frozen reference: round9 lane1 winner `{ROUND9_BASELINE_VARIANT['name']}`",
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
    parser = argparse.ArgumentParser(description="Round 10 block7 asymmetric negative-family carryover for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round10_block7_asym_negative_family_hold_0223_{timestamp}"
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
                    variant=ROUND8_BASELINE_VARIANT,
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
                    variant=ROUND9_BASELINE_VARIANT,
                    baseline_rows=round8_rows,
                )

    rows = []
    for variant in VARIANTS:
        for case in base.CASES:
            rows.extend(
                evaluate_variant_rows(
                    case,
                    full_signals[case.case_id],
                    variant=variant,
                    baseline_rows=baseline_rows,
                )
            )

    summaries = [summarize_variant(rows, variant["name"]) for variant in VARIANTS]
    winner = choose_winner(summaries)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "round8_baseline_variant": ROUND8_BASELINE_VARIANT["name"],
        "baseline_variant": ROUND9_BASELINE_VARIANT["name"],
        "variants": VARIANTS,
        "results": rows,
        "summary": {"variants": summaries, "winner": winner},
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir / "report.md", payload)
    print(f"[ok] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
