#!/usr/bin/env python3
"""
Round 6 lane 3: temporal sign-veto plus narrow in-family positive lift.

This lane keeps:
- the frozen deployed stack
- the winning temporal consensus geometry from round6 lane2

It changes only the role of temporal consensus:
- temporal consensus becomes a sign-veto
- positive-family tie-breaking is handled by a very narrow same-VL lift
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
import round5_pair_formation_0223 as lane5a
import round6_temporal_handoff_0223 as lane6b


WINDOW_SECS = lane6b.WINDOW_SECS
WINDOW_OFFSETS_SEC = lane6b.WINDOW_OFFSETS_SEC
TEMPORAL_REF = next(v for v in lane6b.VARIANTS if v["name"] == "subwin60_vote2_ratio0p10_lift0p15")


VARIANTS: list[dict[str, Any]] = [
    {"name": "control_temporal_ref", "sign_veto": False, "lift_ratio_floor": 1.00, "lift_min_ms": 9.99},
    {"name": "sign_veto_only", "sign_veto": True, "lift_ratio_floor": 1.00, "lift_min_ms": 9.99},
    {"name": "sign_veto_poslift_r0p15", "sign_veto": True, "lift_ratio_floor": 0.15, "lift_min_ms": 0.15},
    {"name": "sign_veto_poslift_r0p10", "sign_veto": True, "lift_ratio_floor": 0.10, "lift_min_ms": 0.15},
]


def sign_filtered_best(scored_pool: list[dict[str, float]], consensus_sign: int | None) -> dict[str, float]:
    if consensus_sign is None:
        return scored_pool[0]
    for row in scored_pool:
        if int(np.sign(row["delta_tau_ms"])) == int(consensus_sign):
            return row
    return scored_pool[0]


def same_vl_positive_lift(
    base_row: dict[str, float],
    scored_pool: list[dict[str, float]],
    *,
    target_tau_vl_ms: float | None,
    ratio_floor: float,
    lift_min_ms: float,
) -> dict[str, float] | None:
    if base_row["delta_tau_ms"] <= 0.0:
        return None
    anchor_target = float(base_row["tau_vl_ms"]) if target_tau_vl_ms is None else float(target_tau_vl_ms)
    pool_score = float(base_row["score"])
    candidates = []
    for row in scored_pool:
        if row["delta_tau_ms"] <= 0.0:
            continue
        if row["score"] < ratio_floor * pool_score:
            continue
        if abs(row["delta_tau_ms"]) < abs(base_row["delta_tau_ms"]) + lift_min_ms:
            continue
        anchor_gap = abs(float(row["tau_vl_ms"]) - anchor_target)
        if anchor_gap > 0.25:
            continue
        candidates.append((anchor_gap, -float(row["score"]), -abs(float(row["delta_tau_ms"])), row))
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
    if variant["name"] == "control_temporal_ref":
        row = lane6b.evaluate_variant_case(case, full_signals, window_sec=window_sec, offset_sec=offset_sec, variant=TEMPORAL_REF)
        row["variant"] = variant["name"]
        return row

    item, scored_pool, baseline_pool = lane6b.evaluate_pool(case, full_signals, window_sec=window_sec, offset_sec=offset_sec)
    baseline_best = lane6b.annotate_best(baseline_pool[0], item["reference"], case.case_id)
    sub_rows = []
    sub_window_sec = max(3.0, float(window_sec) * float(TEMPORAL_REF["subwindow_frac"]))
    for rel_shift in lane6b.subwindow_offsets(window_sec, TEMPORAL_REF):
        sub_item, _, sub_pool = lane6b.evaluate_pool(
            case,
            full_signals,
            window_sec=sub_window_sec,
            offset_sec=float(offset_sec) + float(rel_shift),
        )
        sub_best = lane6b.annotate_best(sub_pool[0], sub_item["reference"], case.case_id)
        sub_rows.append({"offset_sec": float(offset_sec) + float(rel_shift), "selected": {"best": sub_best}})
    consensus = lane6b.consensus_target(sub_rows, TEMPORAL_REF)

    chosen = baseline_best
    if variant["sign_veto"] and consensus is not None:
        chosen = sign_filtered_best(scored_pool, int(consensus["consensus_sign"]))
    promoted = False
    promoted_reason = None
    if chosen is not baseline_best:
        promoted = True
        promoted_reason = "sign_veto"
    if variant["lift_ratio_floor"] < 1.0:
        lifted = same_vl_positive_lift(
            chosen,
            scored_pool,
            target_tau_vl_ms=(None if consensus is None else consensus["target_tau_vl_ms"]),
            ratio_floor=float(variant["lift_ratio_floor"]),
            lift_min_ms=float(variant["lift_min_ms"]),
        )
        if lifted is not None:
            chosen = lifted
            promoted = True
            promoted_reason = "positive_family_lift"

    final_best = lane6b.annotate_best(chosen, item["reference"], case.case_id)
    selected_pool = [chosen] + [
        row
        for row in scored_pool
        if not (abs(row["tau_vl_ms"] - chosen["tau_vl_ms"]) < 1e-9 and abs(row["tau_vr_ms"] - chosen["tau_vr_ms"]) < 1e-9)
    ]
    strict = lane5d.strict_ranks(selected_pool, case.case_id)
    return {
        "case_id": case.case_id,
        "variant": variant["name"],
        "window_sec": float(window_sec),
        "offset_sec": float(offset_sec),
        "reference": item["reference"],
        "selected": {"best": final_best, "top_pairs": selected_pool[:10], "num_pairs": len(selected_pool)},
        "baseline_best": baseline_best,
        "subwindow_rows": sub_rows,
        "consensus": consensus,
        "temporal_promoted": bool(promoted),
        "promotion_reason": promoted_reason,
        "strict_ranks": strict,
    }


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
    block7 = by_case.get("block7_n08_20")
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
            -row["setting_pass_rate"],
            float("inf") if row["window_stability_mean_std_ms"] is None else row["window_stability_mean_std_ms"],
            float("inf") if row["central"]["hard_case_delta_tau_mae_ms"] is None else row["central"]["hard_case_delta_tau_mae_ms"],
            float("inf") if row["central"]["delta_tau_mae_ms"] is None else row["central"]["delta_tau_mae_ms"],
        ),
    )[0]


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 6 Sign-Veto And In-Family Handoff",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        "- Temporal reference: `subwin60_vote2_ratio0p10_lift0p15`",
        "",
        "## Summary",
        "",
        "| variant | central_dt_mae_ms | hard_dt_mae_ms | hard_win_rate | window_mean_std | window_max_std | pass_rate | block6_hit@1 | block7_hit@1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["variants"]:
        central = row["central"]
        block6 = row["hard_cases"].get("block6_n04_19", {})
        block7 = row["hard_cases"].get("block7_n08_20", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    f"{central['delta_tau_mae_ms']:.3f}",
                    f"{central['hard_case_delta_tau_mae_ms']:.3f}",
                    f"{central['hard_case_win_rate']:.3f}",
                    f"{row['window_stability_mean_std_ms']:.3f}",
                    f"{row['window_stability_max_std_ms']:.3f}",
                    f"{row['setting_pass_rate']:.3f}",
                    str(block6.get("hit_at_1", 0)),
                    str(block7.get("hit_at_1", 0)),
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
            f"- central_delta_tau_mae_ms: `{winner['central']['delta_tau_mae_ms']:.3f}`",
            f"- hard_case_delta_tau_mae_ms: `{winner['central']['hard_case_delta_tau_mae_ms']:.3f}`",
            f"- window_stability_mean_std_ms: `{winner['window_stability_mean_std_ms']:.3f}`",
            f"- window_stability_max_std_ms: `{winner['window_stability_max_std_ms']:.3f}`",
            f"- setting_pass_rate: `{winner['setting_pass_rate']:.3f}`",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 6 sign-veto and in-family handoff for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round6_sign_veto_handoff_0223_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    full_signals = {
        case.case_id: lane2.lane1.load_full_case_signals(case, args.data_root)
        for case in base.CASES
    }

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
