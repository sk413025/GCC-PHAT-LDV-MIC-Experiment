#!/usr/bin/env python3
"""
Round 6 lane 4: family-split handoff.

Temporal consensus is used differently by sign:
- negative sign: target-refinement inside the negative family
- positive sign: same-VL positive-family lift
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
import round6_sign_veto_handoff_0223 as lane6c
import round6_temporal_handoff_0223 as lane6b
import round5_same_anchor_handoff_0223 as lane5d


WINDOW_SECS = lane6c.WINDOW_SECS
WINDOW_OFFSETS_SEC = lane6c.WINDOW_OFFSETS_SEC


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "control_sign_veto_poslift_r0p10",
        "neg_target_tolerance_ms": -1.0,
        "neg_score_floor": 0.0,
        "pos_lift_ratio_floor": 0.10,
        "pos_lift_min_ms": 0.15,
    },
    {
        "name": "split_neg_target_t0p18_poslift_r0p10",
        "neg_target_tolerance_ms": 0.18,
        "neg_score_floor": 0.10,
        "pos_lift_ratio_floor": 0.10,
        "pos_lift_min_ms": 0.15,
    },
    {
        "name": "split_neg_target_t0p25_poslift_r0p10",
        "neg_target_tolerance_ms": 0.25,
        "neg_score_floor": 0.10,
        "pos_lift_ratio_floor": 0.10,
        "pos_lift_min_ms": 0.15,
    },
    {
        "name": "split_neg_target_t0p18_poslift_r0p15",
        "neg_target_tolerance_ms": 0.18,
        "neg_score_floor": 0.10,
        "pos_lift_ratio_floor": 0.15,
        "pos_lift_min_ms": 0.15,
    },
]


def negative_target_choice(
    scored_pool: list[dict[str, float]],
    consensus: dict[str, Any] | None,
    *,
    tolerance_ms: float,
    score_floor: float,
) -> dict[str, float] | None:
    if consensus is None or int(consensus["consensus_sign"]) >= 0 or tolerance_ms < 0.0:
        return None
    top_score = float(scored_pool[0]["score"])
    candidates = []
    for row in scored_pool:
        if row["delta_tau_ms"] >= 0.0:
            continue
        if row["score"] < score_floor * top_score:
            continue
        target_gap = abs(abs(row["delta_tau_ms"]) - float(consensus["target_abs_dt_ms"]))
        if target_gap > tolerance_ms:
            continue
        anchor_gap = abs(float(row["tau_vl_ms"]) - float(consensus["target_tau_vl_ms"]))
        candidates.append((target_gap, anchor_gap, -float(row["score"]), row))
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
    if variant["name"] == "control_sign_veto_poslift_r0p10":
        row = lane6c.evaluate_variant_case(
            case,
            full_signals,
            window_sec=window_sec,
            offset_sec=offset_sec,
            variant=next(v for v in lane6c.VARIANTS if v["name"] == "sign_veto_poslift_r0p10"),
        )
        row["variant"] = variant["name"]
        return row

    item, scored_pool, baseline_pool = lane6b.evaluate_pool(case, full_signals, window_sec=window_sec, offset_sec=offset_sec)
    baseline_best = lane6b.annotate_best(baseline_pool[0], item["reference"], case.case_id)
    sub_rows = []
    sub_window_sec = max(3.0, float(window_sec) * float(lane6b.TEMPORAL_REF["subwindow_frac"]) if hasattr(lane6b, "TEMPORAL_REF") else float(window_sec) * 0.60)
    # round6_temporal_handoff_0223.py does not expose TEMPORAL_REF; use the winning config directly.
    sub_window_sec = max(3.0, float(window_sec) * 0.60)
    for rel_shift in lane6b.subwindow_offsets(window_sec, next(v for v in lane6b.VARIANTS if v["name"] == "subwin60_vote2_ratio0p10_lift0p15")):
        sub_item, _, sub_pool = lane6b.evaluate_pool(
            case,
            full_signals,
            window_sec=sub_window_sec,
            offset_sec=float(offset_sec) + float(rel_shift),
        )
        sub_best = lane6b.annotate_best(sub_pool[0], sub_item["reference"], case.case_id)
        sub_rows.append({"offset_sec": float(offset_sec) + float(rel_shift), "selected": {"best": sub_best}})
    consensus = lane6b.consensus_target(sub_rows, next(v for v in lane6b.VARIANTS if v["name"] == "subwin60_vote2_ratio0p10_lift0p15"))

    chosen = baseline_best
    promoted = False
    promoted_reason = None

    neg_choice = negative_target_choice(
        scored_pool,
        consensus,
        tolerance_ms=float(variant["neg_target_tolerance_ms"]),
        score_floor=float(variant["neg_score_floor"]),
    )
    if neg_choice is not None:
        chosen = neg_choice
        promoted = True
        promoted_reason = "negative_target"
    elif consensus is not None:
        chosen = lane6c.sign_filtered_best(scored_pool, int(consensus["consensus_sign"]))
        if chosen is not baseline_best:
            promoted = True
            promoted_reason = "sign_veto"
        lifted = lane6c.same_vl_positive_lift(
            chosen,
            scored_pool,
            target_tau_vl_ms=(None if consensus is None else consensus["target_tau_vl_ms"]),
            ratio_floor=float(variant["pos_lift_ratio_floor"]),
            lift_min_ms=float(variant["pos_lift_min_ms"]),
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


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 6 Family-Split Handoff",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
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
        def fmt(v: float | None) -> str:
            return "NA" if v is None else f"{float(v):.3f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    fmt(central.get("delta_tau_mae_ms")),
                    fmt(central.get("hard_case_delta_tau_mae_ms")),
                    fmt(central.get("hard_case_win_rate")),
                    fmt(row.get("window_stability_mean_std_ms")),
                    fmt(row.get("window_stability_max_std_ms")),
                    fmt(row.get("setting_pass_rate")),
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
            f"- central_delta_tau_mae_ms: `{winner['central'].get('delta_tau_mae_ms')}`",
            f"- hard_case_delta_tau_mae_ms: `{winner['central'].get('hard_case_delta_tau_mae_ms')}`",
            f"- window_stability_mean_std_ms: `{winner.get('window_stability_mean_std_ms')}`",
            f"- window_stability_max_std_ms: `{winner.get('window_stability_max_std_ms')}`",
            f"- setting_pass_rate: `{winner.get('setting_pass_rate')}`",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 6 family-split handoff for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round6_family_split_handoff_0223_{timestamp}"
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

    summaries = [lane6c.summarize_variant(rows, variant["name"]) for variant in VARIANTS]
    winner = lane6c.choose_winner(summaries)
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
