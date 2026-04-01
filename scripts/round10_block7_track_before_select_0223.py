#!/usr/bin/env python3
"""
Round 10 lane 4: block7 track-before-select family persistence.

This lane freezes the round9 lane1 winner and replaces independent block7
per-window promotion with a short temporal tracklet heuristic. The center
window is selected only after scoring compatible negative-family tracks across
neighboring offsets.
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
        "track_span": 0,
        "support_required": 99,
        "score_floor": 1.0,
        "candidate_limit": 1,
        "vr_tol_ms": 0.0,
        "vl_tol_ms": 0.0,
        "abs_dt_tol_ms": 0.0,
    },
    {
        "name": "b7_track_span1_support3_floor0p20",
        "track_span": 1,
        "support_required": 3,
        "score_floor": 0.20,
        "candidate_limit": 6,
        "vr_tol_ms": 0.25,
        "vl_tol_ms": 0.40,
        "abs_dt_tol_ms": 0.20,
    },
    {
        "name": "b7_track_span1_support3_floor0p15",
        "track_span": 1,
        "support_required": 3,
        "score_floor": 0.15,
        "candidate_limit": 8,
        "vr_tol_ms": 0.30,
        "vl_tol_ms": 0.45,
        "abs_dt_tol_ms": 0.25,
    },
    {
        "name": "b7_track_span2_support3_floor0p15",
        "track_span": 2,
        "support_required": 3,
        "score_floor": 0.15,
        "candidate_limit": 8,
        "vr_tol_ms": 0.30,
        "vl_tol_ms": 0.45,
        "abs_dt_tol_ms": 0.25,
    },
]


def candidate_key(row: dict[str, float]) -> tuple[float, float]:
    return (float(row["tau_vl_ms"]), float(row["tau_vr_ms"]))


def collect_track_candidates(
    scored_pool: list[dict[str, float]],
    *,
    score_floor: float,
    candidate_limit: int,
    baseline_best: dict[str, float] | None,
) -> tuple[list[dict[str, float]], float | None]:
    negative_rows = [row for row in scored_pool if row["delta_tau_ms"] < 0.0]
    negative_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    if not negative_rows:
        return [], None

    top_negative_score = float(negative_rows[0]["score"])
    keep = [row for row in negative_rows if float(row["score"]) >= float(score_floor) * top_negative_score]
    if not keep:
        keep = negative_rows[:1]

    if baseline_best is not None and baseline_best["delta_tau_ms"] < 0.0:
        keys = {candidate_key(row) for row in keep}
        if candidate_key(baseline_best) not in keys:
            keep.append(dict(baseline_best))

    deduped = []
    seen = set()
    for row in keep:
        key = candidate_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    deduped.sort(key=lambda row: float(row["score"]), reverse=True)
    return deduped[: int(candidate_limit)], top_negative_score


def build_contexts(
    case: base.CaseRef,
    full_signals: dict[str, np.ndarray],
    *,
    window_sec: float,
    variant: dict[str, Any],
    baseline_rows: dict[tuple[str, float, float], dict[str, Any]],
) -> dict[float, dict[str, Any]]:
    contexts = {}
    for offset_sec in WINDOW_OFFSETS_SEC:
        item, scored_pool, _ = lane9a.lane8.evaluate_pool(case, full_signals, window_sec=window_sec, offset_sec=float(offset_sec))
        baseline_best = baseline_rows[(case.case_id, float(window_sec), float(offset_sec))]["selected"]["best"]
        candidates, top_negative_score = collect_track_candidates(
            scored_pool,
            score_floor=float(variant["score_floor"]),
            candidate_limit=int(variant["candidate_limit"]),
            baseline_best=baseline_best,
        )
        wrapped_candidates = []
        if top_negative_score is not None and top_negative_score > 0.0:
            for idx, row in enumerate(candidates):
                wrapped_candidates.append(
                    {
                        "candidate_idx": idx,
                        "row": row,
                        "local_score": float(row["score"]) / top_negative_score,
                        "abs_dt_ms": abs(float(row["delta_tau_ms"])),
                    }
                )
        contexts[float(offset_sec)] = {
            "item": item,
            "scored_pool": scored_pool,
            "candidates": wrapped_candidates,
            "top_negative_score": top_negative_score,
        }
    return contexts


def edge_score(left_node: dict[str, Any], right_node: dict[str, Any], *, variant: dict[str, Any]) -> float | None:
    vr_gap = abs(float(left_node["row"]["tau_vr_ms"]) - float(right_node["row"]["tau_vr_ms"]))
    vl_gap = abs(float(left_node["row"]["tau_vl_ms"]) - float(right_node["row"]["tau_vl_ms"]))
    dt_gap = abs(float(left_node["abs_dt_ms"]) - float(right_node["abs_dt_ms"]))
    if (
        vr_gap > float(variant["vr_tol_ms"])
        or vl_gap > float(variant["vl_tol_ms"])
        or dt_gap > float(variant["abs_dt_tol_ms"])
    ):
        return None

    vr_score = 1.0 - (vr_gap / max(float(variant["vr_tol_ms"]), 1e-9))
    vl_score = 1.0 - (vl_gap / max(float(variant["vl_tol_ms"]), 1e-9))
    dt_score = 1.0 - (dt_gap / max(float(variant["abs_dt_tol_ms"]), 1e-9))
    return 0.50 * vr_score + 0.20 * vl_score + 0.30 * dt_score


def better_state(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return max(left, right, key=lambda item: (int(item["support_count"]), float(item["track_score"])))


def build_left_dp(candidate_lists: list[list[dict[str, Any]]], *, variant: dict[str, Any]) -> list[list[dict[str, Any]]]:
    states: list[list[dict[str, Any]]] = []
    for idx, nodes in enumerate(candidate_lists):
        row_states = []
        for node_idx, node in enumerate(nodes):
            best = {
                "support_count": 1,
                "track_score": float(node["local_score"]),
                "path": [(idx, node_idx)],
            }
            if idx > 0:
                for prev_idx, prev_node in enumerate(candidate_lists[idx - 1]):
                    link = edge_score(prev_node, node, variant=variant)
                    if link is None:
                        continue
                    prev_state = states[idx - 1][prev_idx]
                    candidate = {
                        "support_count": int(prev_state["support_count"]) + 1,
                        "track_score": float(prev_state["track_score"]) + float(node["local_score"]) + float(link),
                        "path": prev_state["path"] + [(idx, node_idx)],
                    }
                    best = better_state(best, candidate)
            row_states.append(best)
        states.append(row_states)
    return states


def best_track_for_center(
    contexts: dict[float, dict[str, Any]],
    *,
    center_offset_sec: float,
    variant: dict[str, Any],
) -> dict[str, Any] | None:
    offsets = [float(offset) for offset in WINDOW_OFFSETS_SEC]
    center_idx = offsets.index(float(center_offset_sec))
    span = int(variant["track_span"])
    start = max(0, center_idx - span)
    end = min(len(offsets), center_idx + span + 1)
    track_offsets = offsets[start:end]
    center_local_idx = center_idx - start

    candidate_lists = [contexts[offset]["candidates"] for offset in track_offsets]
    if len(candidate_lists[center_local_idx]) == 0 or len(track_offsets) < int(variant["support_required"]):
        return None

    left_dp = build_left_dp(candidate_lists, variant=variant)
    reversed_lists = list(reversed(candidate_lists))
    right_dp_rev = build_left_dp(reversed_lists, variant=variant)
    total_offsets = len(candidate_lists)

    options = []
    for center_node_idx, center_node in enumerate(candidate_lists[center_local_idx]):
        left_state = left_dp[center_local_idx][center_node_idx]
        rev_center_idx = total_offsets - 1 - center_local_idx
        right_state_rev = right_dp_rev[rev_center_idx][center_node_idx]
        right_path = [(total_offsets - 1 - idx, cand_idx) for idx, cand_idx in reversed(right_state_rev["path"])]
        full_path = left_state["path"] + right_path[1:]
        support_count = len(full_path)
        if support_count < int(variant["support_required"]):
            continue

        track_rows = [candidate_lists[path_idx][cand_idx]["row"] for path_idx, cand_idx in full_path]
        mean_abs_dt_ms = float(np.mean([abs(float(row["delta_tau_ms"])) for row in track_rows]))
        mean_tau_vr_ms = float(np.mean([float(row["tau_vr_ms"]) for row in track_rows]))
        center_abs_dt_gap = abs(abs(float(center_node["row"]["delta_tau_ms"])) - mean_abs_dt_ms)
        center_vr_gap = abs(float(center_node["row"]["tau_vr_ms"]) - mean_tau_vr_ms)
        track_score = float(left_state["track_score"]) + float(right_state_rev["track_score"]) - float(center_node["local_score"])
        options.append(
            {
                "center_node": center_node,
                "support_count": support_count,
                "track_score": track_score,
                "path": full_path,
                "track_offsets": [track_offsets[path_idx] for path_idx, _ in full_path],
                "track_rows": track_rows,
                "mean_abs_dt_ms": mean_abs_dt_ms,
                "mean_tau_vr_ms": mean_tau_vr_ms,
                "center_abs_dt_gap": center_abs_dt_gap,
                "center_vr_gap": center_vr_gap,
            }
        )

    if not options:
        return None
    return max(
        options,
        key=lambda item: (
            int(item["support_count"]),
            float(item["track_score"]),
            -float(item["center_abs_dt_gap"]),
            -float(item["center_vr_gap"]),
            float(item["center_node"]["local_score"]),
        ),
    )


def build_selected_payload(
    candidate: dict[str, float],
    *,
    scored_pool: list[dict[str, float]],
    reference: dict[str, float],
    case_id: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    final_best = lane9a.lane8.annotate_best(candidate, reference, case_id)
    selected_pool = [candidate] + [
        row
        for row in scored_pool
        if not (
            abs(float(row["tau_vl_ms"]) - float(candidate["tau_vl_ms"])) < 1e-9
            and abs(float(row["tau_vr_ms"]) - float(candidate["tau_vr_ms"])) < 1e-9
        )
    ]
    strict_ranks = lane5d.strict_ranks(selected_pool, case_id)
    return {"best": final_best, "top_pairs": selected_pool[:10], "num_pairs": len(selected_pool)}, strict_ranks


def evaluate_track_window(
    case: base.CaseRef,
    full_signals: dict[str, np.ndarray],
    *,
    window_sec: float,
    variant: dict[str, Any],
    baseline_rows: dict[tuple[str, float, float], dict[str, Any]],
) -> dict[float, dict[str, Any]]:
    contexts = build_contexts(case, full_signals, window_sec=window_sec, variant=variant, baseline_rows=baseline_rows)
    rows = {}
    for offset_sec in WINDOW_OFFSETS_SEC:
        baseline_row = baseline_rows[(case.case_id, float(window_sec), float(offset_sec))]
        row = dict(baseline_row)
        row["variant"] = variant["name"]

        best_track = best_track_for_center(contexts, center_offset_sec=float(offset_sec), variant=variant)
        if best_track is None:
            row["track_promoted"] = False
            row["track_support_count"] = 0
            row["track_offsets"] = []
            row["track_score"] = None
            rows[float(offset_sec)] = row
            continue

        candidate = best_track["center_node"]["row"]
        baseline_best = baseline_row["selected"]["best"]
        promoted = candidate_key(candidate) != candidate_key(baseline_best)
        if promoted:
            selected, strict_ranks = build_selected_payload(
                candidate,
                scored_pool=contexts[float(offset_sec)]["scored_pool"],
                reference=contexts[float(offset_sec)]["item"]["reference"],
                case_id=case.case_id,
            )
            row["selected"] = selected
            row["strict_ranks"] = strict_ranks
            row["promotion_reason"] = "track_before_select"

        row["track_promoted"] = bool(promoted)
        row["track_support_count"] = int(best_track["support_count"])
        row["track_offsets"] = [float(offset) for offset in best_track["track_offsets"]]
        row["track_score"] = float(best_track["track_score"])
        row["track_mean_abs_dt_ms"] = float(best_track["mean_abs_dt_ms"])
        row["track_mean_tau_vr_ms"] = float(best_track["mean_tau_vr_ms"])
        row["track_rows"] = [
            {
                "offset_sec": float(track_offset),
                "selected": {"best": track_row},
            }
            for track_offset, track_row in zip(best_track["track_offsets"], best_track["track_rows"])
        ]
        rows[float(offset_sec)] = row
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
        ),
    )[0]


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 10 Block7 Track-Before-Select",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        "- Frozen reference: round9 winner `b7_neg_hold_span1_support2_floor0p08`",
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
    parser = argparse.ArgumentParser(description="Round 10 block7 track-before-select family persistence for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round10_block7_track_before_select_0223_{timestamp}"
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

    track_rows: dict[tuple[str, float, float], dict[str, Any]] = {}
    block7_case = next(case for case in base.CASES if case.case_id == BLOCK7_CASE_ID)
    for variant in VARIANTS:
        if variant["name"] == "control_round9a_ref":
            continue
        for window_sec in WINDOW_SECS:
            window_rows = evaluate_track_window(
                block7_case,
                full_signals[block7_case.case_id],
                window_sec=window_sec,
                variant=variant,
                baseline_rows=baseline_rows,
            )
            for offset_sec, row in window_rows.items():
                track_rows[(variant["name"], float(window_sec), float(offset_sec))] = row

    rows = []
    for variant in VARIANTS:
        for window_sec in WINDOW_SECS:
            for offset_sec in WINDOW_OFFSETS_SEC:
                for case in base.CASES:
                    if case.case_id == BLOCK7_CASE_ID and variant["name"] != "control_round9a_ref":
                        rows.append(track_rows[(variant["name"], float(window_sec), float(offset_sec))])
                        continue
                    row = dict(baseline_rows[(case.case_id, float(window_sec), float(offset_sec))])
                    row["variant"] = variant["name"]
                    rows.append(row)

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
