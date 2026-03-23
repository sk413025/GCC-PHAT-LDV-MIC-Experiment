#!/usr/bin/env python3
"""
Round 6 lane 2: temporal handoff stabilization on top of the frozen baseline.

This lane keeps the promoted stack fixed and only changes the final handoff
decision by consulting three overlapping subwindows inside the current window.
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
import round6_baseline_robustness_0223 as lane6a
import round5_same_anchor_handoff_0223 as lane5d
import round5_pair_formation_0223 as lane5a


WINDOW_SECS = (4.0, 5.0, 6.0)
WINDOW_OFFSETS_SEC = (-1.0, -0.5, 0.0, 0.5, 1.0)
FROZEN_HANDOFF_RATIO = 0.90


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "control_frozen",
        "subwindow_frac": 0.70,
        "subwindow_shift_frac": 0.20,
        "min_votes": 3,
        "score_ratio_floor": 1.00,
        "lift_min_ms": 9.99,
        "target_tolerance_ms": 0.0,
    },
    {
        "name": "subwin70_vote2_ratio0p10_lift0p15",
        "subwindow_frac": 0.70,
        "subwindow_shift_frac": 0.20,
        "min_votes": 2,
        "score_ratio_floor": 0.10,
        "lift_min_ms": 0.15,
        "target_tolerance_ms": 0.18,
    },
    {
        "name": "subwin70_vote2_ratio0p10_lift0p20",
        "subwindow_frac": 0.70,
        "subwindow_shift_frac": 0.20,
        "min_votes": 2,
        "score_ratio_floor": 0.10,
        "lift_min_ms": 0.20,
        "target_tolerance_ms": 0.18,
    },
    {
        "name": "subwin60_vote2_ratio0p10_lift0p15",
        "subwindow_frac": 0.60,
        "subwindow_shift_frac": 0.18,
        "min_votes": 2,
        "score_ratio_floor": 0.10,
        "lift_min_ms": 0.15,
        "target_tolerance_ms": 0.18,
    },
    {
        "name": "subwin70_vote2_ratio0p15_lift0p15",
        "subwindow_frac": 0.70,
        "subwindow_shift_frac": 0.20,
        "min_votes": 2,
        "score_ratio_floor": 0.15,
        "lift_min_ms": 0.15,
        "target_tolerance_ms": 0.18,
    },
]


def evaluate_pool(
    case: base.CaseRef,
    full_signals: dict[str, np.ndarray],
    *,
    window_sec: float,
    offset_sec: float,
) -> tuple[dict[str, Any], list[dict[str, float]], list[dict[str, float]]]:
    ref = base.compute_reference(case)
    sliced = lane2.lane1.slice_signals(full_signals, window_sec=window_sec, offset_sec=offset_sec)
    base_ldv, base_mic_l, base_mic_r = lane6a.lane4a.base_frontend(sliced)
    lag_vl, cc_vl = lane2.lane1.pair.gcc_curve(
        base_ldv, base_mic_l, lane2.FS, max_lag_ms=lane2.MAX_LAG_MS, bandpass=lane6a.lane4a.BASE_BANDPASS
    )
    lag_vr, cc_vr = lane2.lane1.pair.gcc_curve(
        base_ldv, base_mic_r, lane2.FS, max_lag_ms=lane2.MAX_LAG_MS, bandpass=lane6a.lane4a.BASE_BANDPASS
    )
    base_cand_vl = lane2.extract_candidates_with_features(
        lag_vl, cc_vl, lag_min_ms=lane2.LAG_MIN_MS, lag_max_ms=lane2.LAG_MAX_MS, top_k=lane2.TOP_K
    )
    base_cand_vr = lane2.extract_candidates_with_features(
        lag_vr, cc_vr, lag_min_ms=lane2.LAG_MIN_MS, lag_max_ms=lane2.LAG_MAX_MS, top_k=lane2.TOP_K
    )
    item = {
        "case_id": case.case_id,
        "offset_sec": float(offset_sec),
        "reference": ref,
        "base_candidates_vl": base_cand_vl,
        "base_candidates_vr": base_cand_vr,
    }
    scored_pool = lane5d.build_scored_pool(item)
    selected_pool = lane5d.apply_handoff(scored_pool, lane6a.central_handoff_variant())
    return item, scored_pool, selected_pool


def annotate_best(best: dict[str, float], ref: dict[str, float], case_id: str) -> dict[str, float]:
    row = dict(best)
    theta_v = float(np.degrees(np.arcsin(np.clip((row["delta_tau_ms"] / 1000.0) * base.C / 1.4, -1.0, 1.0))))
    row["theta_v_deg"] = theta_v
    row["delta_tau_abs_err_ms"] = abs(row["delta_tau_ms"] - ref["delta_tau_ms"])
    row["theta_v_abs_err_deg"] = abs(theta_v - ref["theta_v_deg"])
    row["physical_positive_lags"] = bool(row["tau_vl_ms"] > 0.0 and row["tau_vr_ms"] > 0.0)
    row["physical_small_delta"] = bool(abs(row["delta_tau_ms"]) <= lane2.DELTA_LIMIT_MS)
    row["physical_valid"] = bool(row["physical_positive_lags"] and row["physical_small_delta"])
    row["hard_case_win"] = bool(row["delta_tau_abs_err_ms"] <= 0.224) if case_id in lane5a.HARD_CASES else None
    return row


def subwindow_offsets(window_sec: float, variant: dict[str, Any]) -> list[float]:
    shift = float(variant["subwindow_shift_frac"]) * float(window_sec)
    return [-shift, 0.0, shift]


def consensus_target(sub_rows: list[dict[str, Any]], variant: dict[str, Any]) -> dict[str, Any] | None:
    selected = [row["selected"]["best"] for row in sub_rows if row["selected"] is not None]
    if len(selected) < int(variant["min_votes"]):
        return None
    signs = [int(np.sign(row["delta_tau_ms"])) for row in selected if abs(row["delta_tau_ms"]) > 1e-9]
    if not signs:
        return None
    sign_counts = Counter(signs)
    consensus_sign, vote_count = sign_counts.most_common(1)[0]
    if vote_count < int(variant["min_votes"]):
        return None
    matching = [row for row in selected if int(np.sign(row["delta_tau_ms"])) == consensus_sign]
    target_abs_dt = float(np.median(np.array([abs(row["delta_tau_ms"]) for row in matching], dtype=np.float64)))
    target_tau_vl = float(np.median(np.array([row["tau_vl_ms"] for row in matching], dtype=np.float64)))
    return {
        "consensus_sign": int(consensus_sign),
        "vote_count": int(vote_count),
        "target_abs_dt_ms": target_abs_dt,
        "target_tau_vl_ms": target_tau_vl,
    }


def pick_temporal_candidate(
    scored_pool: list[dict[str, float]],
    baseline_best: dict[str, float],
    consensus: dict[str, Any] | None,
    variant: dict[str, Any],
) -> dict[str, float] | None:
    if consensus is None or not scored_pool:
        return None
    top_score = float(scored_pool[0]["score"])
    candidates = []
    for row in scored_pool:
        if int(np.sign(row["delta_tau_ms"])) != int(consensus["consensus_sign"]):
            continue
        if row["score"] < float(variant["score_ratio_floor"]) * top_score:
            continue
        target_gap = abs(abs(row["delta_tau_ms"]) - float(consensus["target_abs_dt_ms"]))
        if target_gap > float(variant["target_tolerance_ms"]):
            continue
        anchor_gap = abs(float(row["tau_vl_ms"]) - float(consensus["target_tau_vl_ms"]))
        candidates.append((target_gap, anchor_gap, -float(row["score"]), row))
    if not candidates:
        return None
    _, _, _, chosen = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    baseline_gap = abs(chosen["delta_tau_ms"]) - abs(baseline_best["delta_tau_ms"])
    if int(np.sign(chosen["delta_tau_ms"])) != int(np.sign(baseline_best["delta_tau_ms"])):
        return chosen
    if baseline_gap >= float(variant["lift_min_ms"]):
        return chosen
    return None


def evaluate_variant_case(
    case: base.CaseRef,
    full_signals: dict[str, np.ndarray],
    *,
    window_sec: float,
    offset_sec: float,
    variant: dict[str, Any],
) -> dict[str, Any]:
    item, scored_pool, baseline_pool = evaluate_pool(case, full_signals, window_sec=window_sec, offset_sec=offset_sec)
    baseline_best = annotate_best(baseline_pool[0], item["reference"], case.case_id)
    sub_rows = []
    sub_window_sec = max(3.0, float(window_sec) * float(variant["subwindow_frac"]))
    for rel_shift in subwindow_offsets(window_sec, variant):
        sub_item, _, sub_pool = evaluate_pool(
            case,
            full_signals,
            window_sec=sub_window_sec,
            offset_sec=float(offset_sec) + float(rel_shift),
        )
        sub_best = annotate_best(sub_pool[0], sub_item["reference"], case.case_id)
        sub_rows.append({"offset_sec": float(offset_sec) + float(rel_shift), "selected": {"best": sub_best}})

    consensus = consensus_target(sub_rows, variant)
    promoted = pick_temporal_candidate(scored_pool, baseline_best, consensus, variant)
    final_best = annotate_best(promoted if promoted is not None else baseline_pool[0], item["reference"], case.case_id)
    selected_pool = list(baseline_pool)
    if promoted is not None:
        remaining = [row for row in scored_pool if not (abs(row["tau_vl_ms"] - promoted["tau_vl_ms"]) < 1e-9 and abs(row["tau_vr_ms"] - promoted["tau_vr_ms"]) < 1e-9)]
        selected_pool = [promoted] + remaining
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
        "temporal_promoted": bool(promoted is not None),
        "strict_ranks": strict,
    }


def summarize_setting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row["selected"]["best"] for row in rows if row["selected"] is not None]
    if not valid:
        return {"valid_cases": 0, "physical_count": 0, "delta_tau_mae_ms": None}
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
        "temporal_promoted_cases": int(sum(1 for row in rows if row["temporal_promoted"])),
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
    all_physical = True
    for case_id, items in by_case.items():
        items.sort(key=lambda x: x["offset_sec"])
        selected = [item["selected"]["best"] for item in items]
        if any(not item["physical_valid"] for item in selected):
            all_physical = False
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

    pass_rate = float(np.mean([1.0 if row["setting_pass"] else 0.0 for row in per_setting])) if per_setting else None
    return {
        "variant": variant_name,
        "central": central,
        "window_stability_mean_std_ms": float(np.mean(case_std)) if case_std else None,
        "window_stability_max_std_ms": float(np.max(case_std)) if case_std else None,
        "all_physical_valid": bool(all_physical),
        "hard_cases": hard_stats,
        "setting_pass_rate": pass_rate,
        "per_setting": per_setting,
    }


def choose_winner(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        summary_rows,
        key=lambda row: (
            -row["setting_pass_rate"],
            float("inf") if row["window_stability_mean_std_ms"] is None else row["window_stability_mean_std_ms"],
            float("inf") if row["central"]["hard_case_delta_tau_mae_ms"] is None else row["central"]["hard_case_delta_tau_mae_ms"],
            float("inf") if row["central"]["delta_tau_mae_ms"] is None else row["central"]["delta_tau_mae_ms"],
        ),
    )
    return ordered[0]


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 6 Temporal Handoff Stabilization",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        "- Frozen baseline: `cond_prune_soft + same_vl_replace_w6_p3 + handoff_ratio_0p90`",
        "",
        "## Summary",
        "",
        "| variant | central_dt_mae_ms | hard_dt_mae_ms | hard_win_rate | window_mean_std | window_max_std | pass_rate | block6_hit@1 | block7_hit@1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["variants"]:
        central = row["central"]
        block6_hit1 = row["hard_cases"].get("block6_n04_19", {}).get("hit_at_1", 0)
        block7_hit1 = row["hard_cases"].get("block7_n08_20", {}).get("hit_at_1", 0)
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    "NA" if central["delta_tau_mae_ms"] is None else f"{central['delta_tau_mae_ms']:.3f}",
                    "NA" if central["hard_case_delta_tau_mae_ms"] is None else f"{central['hard_case_delta_tau_mae_ms']:.3f}",
                    "NA" if central["hard_case_win_rate"] is None else f"{central['hard_case_win_rate']:.3f}",
                    "NA" if row["window_stability_mean_std_ms"] is None else f"{row['window_stability_mean_std_ms']:.3f}",
                    "NA" if row["window_stability_max_std_ms"] is None else f"{row['window_stability_max_std_ms']:.3f}",
                    "NA" if row["setting_pass_rate"] is None else f"{row['setting_pass_rate']:.3f}",
                    str(block6_hit1),
                    str(block7_hit1),
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
    parser = argparse.ArgumentParser(description="Round 6 temporal handoff stabilization for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round6_temporal_handoff_0223_{timestamp}"
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
        "window_secs": list(WINDOW_SECS),
        "window_offsets_sec": list(WINDOW_OFFSETS_SEC),
        "variants": VARIANTS,
        "results": rows,
        "summary": {"variants": summaries, "winner": winner},
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir / "report.md", payload)
    print(f"[ok] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
