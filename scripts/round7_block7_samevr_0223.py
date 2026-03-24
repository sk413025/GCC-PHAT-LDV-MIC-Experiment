#!/usr/bin/env python3
"""
Round 7 lane 1: block7-focused same-VR negative-family persistence.

This lane freezes round6 lane3 `sign_veto_poslift_r0p10` and only adds a
block7-only negative-family persistence rule on top of it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
TEMPORAL_REF = next(v for v in lane6b.VARIANTS if v["name"] == "subwin60_vote2_ratio0p10_lift0p15")
LANE6_REF = next(v for v in lane6c.VARIANTS if v["name"] == "sign_veto_poslift_r0p10")
BLOCK7_CASE_ID = "block7_n08_20"


VARIANTS: list[dict[str, Any]] = [
    {"name": "control_sign_veto_poslift_r0p10", "mode": "control", "vr_tol_ms": 0.0, "dt_tol_ms": 0.0, "score_floor": 0.0},
    {"name": "b7_samevr_neg_promote_vr0p15_dt0p15_r0p10", "mode": "promote", "vr_tol_ms": 0.15, "dt_tol_ms": 0.15, "score_floor": 0.10},
    {"name": "b7_samevr_neg_promote_vr0p20_dt0p15_r0p10", "mode": "promote", "vr_tol_ms": 0.20, "dt_tol_ms": 0.15, "score_floor": 0.10},
    {"name": "b7_samevr_neg_veto_vr0p15_dt0p15_r0p10", "mode": "veto", "vr_tol_ms": 0.15, "dt_tol_ms": 0.15, "score_floor": 0.10},
]


def subwindow_consensus(sub_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    selected = [row["selected"]["best"] for row in sub_rows if row["selected"] is not None]
    signs = [int(np.sign(row["delta_tau_ms"])) for row in selected if abs(row["delta_tau_ms"]) > 1e-9]
    if not signs:
        return None
    sign_counts = Counter(signs)
    consensus_sign, vote_count = sign_counts.most_common(1)[0]
    matching = [row for row in selected if int(np.sign(row["delta_tau_ms"])) == consensus_sign]
    if vote_count < 2:
        return None
    return {
        "consensus_sign": int(consensus_sign),
        "vote_count": int(vote_count),
        "target_tau_vr_ms": float(np.median(np.array([row["tau_vr_ms"] for row in matching], dtype=np.float64))),
        "target_abs_dt_ms": float(np.median(np.array([abs(row["delta_tau_ms"]) for row in matching], dtype=np.float64))),
    }


def choose_negative_same_vr(
    scored_pool: list[dict[str, float]],
    consensus: dict[str, Any] | None,
    *,
    vr_tol_ms: float,
    dt_tol_ms: float,
    score_floor: float,
) -> dict[str, float] | None:
    if consensus is None or int(consensus["consensus_sign"]) >= 0:
        return None
    top_score = float(scored_pool[0]["score"])
    candidates = []
    for row in scored_pool:
        if row["delta_tau_ms"] >= 0.0:
            continue
        if row["score"] < score_floor * top_score:
            continue
        vr_gap = abs(float(row["tau_vr_ms"]) - float(consensus["target_tau_vr_ms"]))
        dt_gap = abs(abs(row["delta_tau_ms"]) - float(consensus["target_abs_dt_ms"]))
        if vr_gap > vr_tol_ms or dt_gap > dt_tol_ms:
            continue
        candidates.append((vr_gap, dt_gap, -float(row["score"]), row))
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
    base_row = lane6c.evaluate_variant_case(case, full_signals, window_sec=window_sec, offset_sec=offset_sec, variant=LANE6_REF)
    base_row["variant"] = variant["name"]
    if variant["mode"] == "control" or case.case_id != BLOCK7_CASE_ID:
        return base_row

    item, scored_pool, _ = lane6b.evaluate_pool(case, full_signals, window_sec=window_sec, offset_sec=offset_sec)
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

    consensus = subwindow_consensus(sub_rows)
    candidate = choose_negative_same_vr(
        scored_pool,
        consensus,
        vr_tol_ms=float(variant["vr_tol_ms"]),
        dt_tol_ms=float(variant["dt_tol_ms"]),
        score_floor=float(variant["score_floor"]),
    )
    if candidate is None:
        base_row["consensus"] = consensus
        base_row["promotion_reason"] = None
        return base_row

    current_best = base_row["selected"]["best"]
    replace = False
    if variant["mode"] == "promote":
        replace = True
    elif variant["mode"] == "veto":
        vr_gap_current = abs(float(current_best["tau_vr_ms"]) - float(consensus["target_tau_vr_ms"])) if consensus is not None else np.inf
        dt_gap_current = abs(abs(current_best["delta_tau_ms"]) - float(consensus["target_abs_dt_ms"])) if consensus is not None else np.inf
        replace = bool(current_best["delta_tau_ms"] >= 0.0 or vr_gap_current > float(variant["vr_tol_ms"]) or dt_gap_current > float(variant["dt_tol_ms"]))

    if not replace:
        base_row["consensus"] = consensus
        base_row["promotion_reason"] = None
        return base_row

    final_best = lane6b.annotate_best(candidate, item["reference"], case.case_id)
    selected_pool = [candidate] + [
        row
        for row in scored_pool
        if not (abs(row["tau_vl_ms"] - candidate["tau_vl_ms"]) < 1e-9 and abs(row["tau_vr_ms"] - candidate["tau_vr_ms"]) < 1e-9)
    ]
    base_row["selected"] = {"best": final_best, "top_pairs": selected_pool[:10], "num_pairs": len(selected_pool)}
    base_row["strict_ranks"] = lane5d.strict_ranks(selected_pool, case.case_id)
    base_row["temporal_promoted"] = True
    base_row["promotion_reason"] = variant["mode"]
    base_row["consensus"] = consensus
    return base_row


def summarize_setting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row["selected"]["best"] for row in rows if row["selected"] is not None]
    if not valid:
        return {"valid_cases": 0, "physical_count": 0, "delta_tau_mae_ms": None}
    dt = np.array([row["delta_tau_abs_err_ms"] for row in valid], dtype=np.float64)
    physical_count = sum(1 for row in valid if row["physical_valid"])
    hard = [row["selected"]["best"] for row in rows if row["case_id"] in {"block6_n04_19", BLOCK7_CASE_ID}]
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
        "hard_case_delta_tau_mae_ms": float(np.mean(hard_dt)),
        "hard_case_win_rate": float(np.mean(hard_win)),
        "block6_selected_delta_tau_ms": block6["selected"]["best"]["delta_tau_ms"] if block6 else None,
        "block7_selected_delta_tau_ms": block7["selected"]["best"]["delta_tau_ms"] if block7 else None,
        "block6_rescue_pair_rank": block6["strict_ranks"]["rescue_pair_rank"] if block6 and block6["strict_ranks"] else None,
        "block7_rescue_pair_rank": block7["strict_ranks"]["rescue_pair_rank"] if block7 and block7["strict_ranks"] else None,
        "promoted_cases": int(sum(1 for row in rows if row.get("temporal_promoted", False))),
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
        if case_id in {"block6_n04_19", BLOCK7_CASE_ID}:
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
            -(row["hard_cases"].get(BLOCK7_CASE_ID, {}).get("hit_at_1", 0)),
            -(row["hard_cases"].get(BLOCK7_CASE_ID, {}).get("hit_at_3", 0)),
            -float(row["setting_pass_rate"] or 0.0),
            float("inf") if row["central"].get("delta_tau_mae_ms") is None else row["central"]["delta_tau_mae_ms"],
        ),
    )[0]


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 7 Block7 Same-VR Negative-Family Persistence",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        "- Frozen reference: `sign_veto_poslift_r0p10`",
        "",
        "## Summary",
        "",
        "| variant | central_dt_mae_ms | hard_dt_mae_ms | pass_rate | window_mean_std | block6_hit@1 | block7_hit@1 | block7_hit@3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["variants"]:
        block6 = row["hard_cases"].get("block6_n04_19", {})
        block7 = row["hard_cases"].get(BLOCK7_CASE_ID, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    f"{row['central']['delta_tau_mae_ms']:.3f}",
                    f"{row['central']['hard_case_delta_tau_mae_ms']:.3f}",
                    f"{row['setting_pass_rate']:.3f}",
                    f"{row['window_stability_mean_std_ms']:.3f}",
                    str(block6.get("hit_at_1", 0)),
                    str(block7.get("hit_at_1", 0)),
                    str(block7.get("hit_at_3", 0)),
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
            f"- block6 hit@1: `{winner['hard_cases'].get('block6_n04_19', {}).get('hit_at_1', 0)}`",
            f"- pass_rate: `{winner['setting_pass_rate']:.3f}`",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 7 block7 same-VR negative-family persistence.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round7_block7_samevr_0223_{timestamp}"
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
