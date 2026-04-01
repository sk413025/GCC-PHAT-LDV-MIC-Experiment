#!/usr/bin/env python3
"""
Round 10 lane 3: anchor-driven temporal family promotion from central block7.

This lane freezes the round9 lane1 winner and only changes the block7 rule.
It promotes a strong negative-family choice from the recovered central window
to the immediate temporal neighbors with a decayed target magnitude and tighter
acceptance than the prior neighbor-driven hold rules.
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
BLOCK7_CASE_ID = lane9a.BLOCK7_CASE_ID
BASELINE_VARIANT = next(v for v in lane9a.VARIANTS if v["name"] == "b7_neg_hold_span1_support2_floor0p08")


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "control_round9a_ref",
        "target_offsets_sec": [],
        "anchor_offset_sec": 0.0,
        "anchor_rank_max": 0,
        "anchor_min_abs_dt_ms": 9.99,
        "score_floor": 1.0,
        "vr_tol_ms": 0.0,
        "vl_tol_ms": 0.0,
        "decay_ratio": 1.0,
        "abs_dt_tol_ms": 0.0,
        "min_abs_dt_ratio": 9.99,
        "max_abs_dt_ratio": 0.0,
    },
    {
        "name": "b7_central_decay_r0p95_floor0p20_vr0p08_vl0p35_dt0p16_cap1p40",
        "target_offsets_sec": [-0.5, 0.5],
        "anchor_offset_sec": 0.0,
        "anchor_rank_max": 1,
        "anchor_min_abs_dt_ms": 0.30,
        "score_floor": 0.20,
        "vr_tol_ms": 0.08,
        "vl_tol_ms": 0.35,
        "decay_ratio": 0.95,
        "abs_dt_tol_ms": 0.16,
        "min_abs_dt_ratio": 0.90,
        "max_abs_dt_ratio": 1.40,
    },
    {
        "name": "b7_central_nodecay_floor0p20_vr0p08_vl0p35_dt0p16_cap1p40",
        "target_offsets_sec": [-0.5, 0.5],
        "anchor_offset_sec": 0.0,
        "anchor_rank_max": 1,
        "anchor_min_abs_dt_ms": 0.30,
        "score_floor": 0.20,
        "vr_tol_ms": 0.08,
        "vl_tol_ms": 0.35,
        "decay_ratio": 1.00,
        "abs_dt_tol_ms": 0.16,
        "min_abs_dt_ratio": 0.90,
        "max_abs_dt_ratio": 1.40,
    },
    {
        "name": "b7_central_decay_r0p95_floor0p15_vr0p10_vl0p40_dt0p18_cap1p50",
        "target_offsets_sec": [-0.5, 0.5],
        "anchor_offset_sec": 0.0,
        "anchor_rank_max": 1,
        "anchor_min_abs_dt_ms": 0.25,
        "score_floor": 0.15,
        "vr_tol_ms": 0.10,
        "vl_tol_ms": 0.40,
        "decay_ratio": 0.95,
        "abs_dt_tol_ms": 0.18,
        "min_abs_dt_ratio": 0.80,
        "max_abs_dt_ratio": 1.50,
    },
]


def build_central_anchor(
    case: base.CaseRef,
    *,
    window_sec: float,
    variant: dict[str, Any],
    baseline_rows: dict[tuple[str, float, float], dict[str, Any]],
) -> dict[str, Any] | None:
    anchor_offset = float(variant["anchor_offset_sec"])
    anchor_row = baseline_rows[(case.case_id, float(window_sec), anchor_offset)]
    if anchor_row["selected"] is None:
        return None
    best = anchor_row["selected"]["best"]
    strict_ranks = anchor_row.get("strict_ranks") or {}
    anchor_rank = strict_ranks.get("rescue_pair_rank")
    if best["delta_tau_ms"] >= 0.0:
        return None
    if not bool(best["physical_valid"]):
        return None
    if anchor_rank is None or anchor_rank > int(variant["anchor_rank_max"]):
        return None
    if abs(float(best["delta_tau_ms"])) < float(variant["anchor_min_abs_dt_ms"]):
        return None
    return {
        "offset_sec": anchor_offset,
        "anchor_rank": int(anchor_rank),
        "target_tau_vl_ms": float(best["tau_vl_ms"]),
        "target_tau_vr_ms": float(best["tau_vr_ms"]),
        "target_abs_dt_ms": float(abs(best["delta_tau_ms"])),
        "source_score": float(best["score"]),
    }


def choose_promoted_candidate(
    scored_pool: list[dict[str, float]],
    anchor: dict[str, Any] | None,
    *,
    offset_sec: float,
    variant: dict[str, Any],
) -> dict[str, float] | None:
    if anchor is None or not scored_pool:
        return None
    if float(offset_sec) not in {float(v) for v in variant["target_offsets_sec"]}:
        return None

    top_score = float(scored_pool[0]["score"])
    neighbor_steps = int(round(abs(float(offset_sec) - float(anchor["offset_sec"])) / 0.5))
    target_abs_dt_ms = float(anchor["target_abs_dt_ms"]) * (float(variant["decay_ratio"]) ** max(neighbor_steps, 1))
    candidates = []
    for row in scored_pool:
        if row["delta_tau_ms"] >= 0.0:
            continue
        if top_score > 0.0 and row["score"] < float(variant["score_floor"]) * top_score:
            continue
        abs_dt_ms = abs(float(row["delta_tau_ms"]))
        if abs_dt_ms < float(variant["min_abs_dt_ratio"]) * float(anchor["target_abs_dt_ms"]):
            continue
        if abs_dt_ms > float(variant["max_abs_dt_ratio"]) * float(anchor["target_abs_dt_ms"]):
            continue
        vr_gap = abs(float(row["tau_vr_ms"]) - float(anchor["target_tau_vr_ms"]))
        vl_gap = abs(float(row["tau_vl_ms"]) - float(anchor["target_tau_vl_ms"]))
        dt_gap = abs(abs_dt_ms - target_abs_dt_ms)
        if vr_gap > float(variant["vr_tol_ms"]):
            continue
        if vl_gap > float(variant["vl_tol_ms"]):
            continue
        if dt_gap > float(variant["abs_dt_tol_ms"]):
            continue
        candidates.append((vr_gap, dt_gap, vl_gap, -float(row["score"]), row))
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

    anchor = build_central_anchor(case, window_sec=window_sec, variant=variant, baseline_rows=baseline_rows)
    item, scored_pool, _ = lane9a.lane8.evaluate_pool(case, full_signals, window_sec=window_sec, offset_sec=offset_sec)
    candidate = choose_promoted_candidate(scored_pool, anchor, offset_sec=offset_sec, variant=variant)
    if candidate is None:
        row["central_anchor"] = anchor
        row["central_promoted"] = False
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
    row["central_anchor"] = {
        **anchor,
        "effective_target_abs_dt_ms": float(anchor["target_abs_dt_ms"])
        * (float(variant["decay_ratio"]) ** max(int(round(abs(float(offset_sec) - float(anchor["offset_sec"])) / 0.5)), 1)),
    }
    row["central_promoted"] = True
    row["promotion_reason"] = "central_temporal_family_promotion"
    return row


def summarize_setting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return lane9a.summarize_setting(rows)


def summarize_variant(rows: list[dict[str, Any]], variant_name: str) -> dict[str, Any]:
    summary = lane9a.summarize_variant(rows, variant_name)
    subset = [row for row in rows if row["variant"] == variant_name]
    summary["promoted_neighbors"] = int(sum(1 for row in subset if row.get("central_promoted")))
    block7_5s = [
        row for row in subset if row["case_id"] == BLOCK7_CASE_ID and abs(row["window_sec"] - 5.0) < 1e-9
    ]
    block7_5s.sort(key=lambda row: row["offset_sec"])
    summary["block7_window5_selected_delta_tau_ms"] = [
        float(row["selected"]["best"]["delta_tau_ms"]) for row in block7_5s if row["selected"] is not None
    ]
    return summary


def choose_winner(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        summary_rows,
        key=lambda row: (
            -(row["hard_cases"].get(BLOCK7_CASE_ID, {}).get("hit_at_1", 0)),
            -(row["hard_cases"].get(BLOCK7_CASE_ID, {}).get("hit_at_3", 0)),
            -float(row["setting_pass_rate"] or 0.0),
            float("inf") if row["central"].get("delta_tau_mae_ms") is None else row["central"]["delta_tau_mae_ms"],
            float("inf") if row.get("window_stability_mean_std_ms") is None else row["window_stability_mean_std_ms"],
        ),
    )[0]


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 10 Block7 Temporal Family Promotion",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        "- Frozen reference: round9 lane1 winner `b7_neg_hold_span1_support2_floor0p08`",
        "",
        "## Summary",
        "",
        "| variant | central_dt_mae_ms | hard_dt_mae_ms | pass_rate | window_mean_std | block7_hit@1 | block7_hit@3 | promoted_neighbors | block7_5s_deltas |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["summary"]["variants"]:
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
                    str(b7.get("hit_at_1", 0)),
                    str(b7.get("hit_at_3", 0)),
                    str(row.get("promoted_neighbors", 0)),
                    "`" + ", ".join(f"{value:.3f}" for value in row.get("block7_window5_selected_delta_tau_ms", [])) + "`",
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
            f"- pass_rate: `{winner['setting_pass_rate']:.3f}`",
            f"- promoted_neighbors: `{winner.get('promoted_neighbors', 0)}`",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 10 block7 temporal family promotion for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round10_block7_temporal_family_promotion_0223_{timestamp}"
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
