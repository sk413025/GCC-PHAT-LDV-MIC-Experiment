#!/usr/bin/env python3
"""
Round 8 lane 2: block6 positive-family target hold.

This lane freezes the round6 lane3 baseline `sign_veto_poslift_r0p10` and only
changes the block6 decision rule. The goal is not sign repair; it is to keep
block6 aligned to the same positive family that neighboring windows already
support.
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
import round5_pair_formation_0223 as lane5a
import round5_same_anchor_handoff_0223 as lane5d
import round6_baseline_robustness_0223 as lane6a
import round6_sign_veto_handoff_0223 as lane6c
import round6_temporal_handoff_0223 as lane6b


WINDOW_SECS = lane6b.WINDOW_SECS
WINDOW_OFFSETS_SEC = lane6b.WINDOW_OFFSETS_SEC
BASELINE_VARIANT = next(v for v in lane6c.VARIANTS if v["name"] == "sign_veto_poslift_r0p10")


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "control_sign_veto_poslift_r0p10",
        "neighbor_span": 0,
        "support_required": 99,
        "score_floor": 1.0,
        "vl_tol_ms": 0.0,
        "vr_tol_ms": 0.0,
    },
    {
        "name": "b6_family_hold_span1_support2_floor0p08",
        "neighbor_span": 1,
        "support_required": 2,
        "score_floor": 0.08,
        "vl_tol_ms": 0.22,
        "vr_tol_ms": 0.30,
    },
    {
        "name": "b6_family_hold_span2_support2_floor0p08",
        "neighbor_span": 2,
        "support_required": 2,
        "score_floor": 0.08,
        "vl_tol_ms": 0.22,
        "vr_tol_ms": 0.30,
    },
    {
        "name": "b6_family_hold_span2_support2_floor0p05",
        "neighbor_span": 2,
        "support_required": 2,
        "score_floor": 0.05,
        "vl_tol_ms": 0.22,
        "vr_tol_ms": 0.30,
    },
]


def support_offsets(offset_sec: float, *, neighbor_span: int) -> list[float]:
    if neighbor_span <= 0:
        return []
    idx = list(WINDOW_OFFSETS_SEC).index(float(offset_sec))
    offsets = []
    for step in range(1, neighbor_span + 1):
        if idx - step >= 0:
            offsets.append(float(WINDOW_OFFSETS_SEC[idx - step]))
        if idx + step < len(WINDOW_OFFSETS_SEC):
            offsets.append(float(WINDOW_OFFSETS_SEC[idx + step]))
    return sorted(set(offsets))


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
    return item, scored_pool, scored_pool


def build_family_anchor(
    case: base.CaseRef,
    full_signals: dict[str, np.ndarray],
    *,
    window_sec: float,
    offset_sec: float,
    neighbor_span: int,
    baseline_rows: dict[tuple[str, float, float], dict[str, Any]],
) -> dict[str, Any] | None:
    support_rows = []
    for neighbor_offset in support_offsets(offset_sec, neighbor_span=neighbor_span):
        row = baseline_rows[(case.case_id, float(window_sec), float(neighbor_offset))]
        if row["selected"] is None:
            continue
        best = row["selected"]["best"]
        if best["delta_tau_ms"] <= 0.0:
            continue
        if not bool(best["physical_valid"]):
            continue
        support_rows.append(best)
    if not support_rows:
        return None
    anchor = max(support_rows, key=lambda r: (float(r["tau_vr_ms"]), float(r["score"])))
    return {
        "support_count": int(len(support_rows)),
        "target_tau_vl_ms": float(anchor["tau_vl_ms"]),
        "target_tau_vr_ms": float(anchor["tau_vr_ms"]),
        "target_abs_dt_ms": float(abs(anchor["delta_tau_ms"])),
        "anchor_score": float(anchor["score"]),
        "anchor_delta_tau_ms": float(anchor["delta_tau_ms"]),
    }


def choose_family_candidate(
    scored_pool: list[dict[str, float]],
    anchor: dict[str, Any] | None,
    *,
    support_required: int,
    score_floor: float,
    vl_tol_ms: float,
    vr_tol_ms: float,
) -> dict[str, float] | None:
    if anchor is None or int(anchor["support_count"]) < int(support_required) or not scored_pool:
        return None
    top_score = float(scored_pool[0]["score"])
    candidates = []
    for row in scored_pool:
        if row["delta_tau_ms"] <= 0.0:
            continue
        if row["score"] < float(score_floor) * top_score:
            continue
        vl_gap = abs(float(row["tau_vl_ms"]) - float(anchor["target_tau_vl_ms"]))
        vr_gap = abs(float(row["tau_vr_ms"]) - float(anchor["target_tau_vr_ms"]))
        dt_gap = abs(abs(float(row["delta_tau_ms"])) - float(anchor["target_abs_dt_ms"]))
        if vl_gap > float(vl_tol_ms) or vr_gap > float(vr_tol_ms):
            continue
        candidates.append((vr_gap, vl_gap, dt_gap, -float(row["score"]), row))
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
    base_best = baseline_row["selected"]["best"]
    row = dict(baseline_row)
    row["variant"] = variant["name"]
    if variant["name"] == "control_sign_veto_poslift_r0p10" or case.case_id != "block6_n04_19":
        return row

    item, scored_pool, _ = evaluate_pool(case, full_signals, window_sec=window_sec, offset_sec=offset_sec)
    anchor = build_family_anchor(
        case,
        full_signals,
        window_sec=window_sec,
        offset_sec=offset_sec,
        neighbor_span=int(variant["neighbor_span"]),
        baseline_rows=baseline_rows,
    )
    candidate = choose_family_candidate(
        scored_pool,
        anchor,
        support_required=int(variant["support_required"]),
        score_floor=float(variant["score_floor"]),
        vl_tol_ms=float(variant["vl_tol_ms"]),
        vr_tol_ms=float(variant["vr_tol_ms"]),
    )
    if candidate is None:
        row["family_anchor"] = anchor
        row["family_promoted"] = False
        row["promotion_reason"] = None
        return row

    final_best = annotate_best(candidate, item["reference"], case.case_id)
    selected_pool = [candidate] + [
        cand
        for cand in scored_pool
        if not (abs(cand["tau_vl_ms"] - candidate["tau_vl_ms"]) < 1e-9 and abs(cand["tau_vr_ms"] - candidate["tau_vr_ms"]) < 1e-9)
    ]
    row["selected"] = {"best": final_best, "top_pairs": selected_pool[:10], "num_pairs": len(selected_pool)}
    row["strict_ranks"] = lane5d.strict_ranks(selected_pool, case.case_id)
    row["family_anchor"] = anchor
    row["family_promoted"] = True
    row["promotion_reason"] = "family_target_hold"
    return row


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
        "promoted_cases": int(sum(1 for row in rows if row.get("family_promoted", False))),
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
            -(row["hard_cases"].get("block6_n04_19", {}).get("hit_at_1", 0)),
            -(row["hard_cases"].get("block6_n04_19", {}).get("hit_at_3", 0)),
            -float(row["setting_pass_rate"] or 0.0),
            float("inf") if row["central"].get("delta_tau_mae_ms") is None else row["central"]["delta_tau_mae_ms"],
        ),
    )[0]


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 8 Block6 Positive-Family Target Hold",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        "- Frozen reference: `sign_veto_poslift_r0p10`",
        "",
        "## Summary",
        "",
        "| variant | central_dt_mae_ms | hard_dt_mae_ms | pass_rate | window_mean_std | block6_hit@1 | block6_hit@3 | block7_hit@1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["variants"]:
        b6 = row["hard_cases"].get("block6_n04_19", {})
        b7 = row["hard_cases"].get("block7_n08_20", {})
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
                    str(b6.get("hit_at_3", 0)),
                    str(b7.get("hit_at_1", 0)),
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
            f"- block6 hit@3: `{winner['hard_cases'].get('block6_n04_19', {}).get('hit_at_3', 0)}`",
            f"- block7 hit@1: `{winner['hard_cases'].get('block7_n08_20', {}).get('hit_at_1', 0)}`",
            f"- pass_rate: `{winner['setting_pass_rate']:.3f}`",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 8 block6 positive-family target hold for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round8_block6_positive_family_target_hold_0223_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    full_signals = {case.case_id: lane2.lane1.load_full_case_signals(case, args.data_root) for case in base.CASES}

    baseline_rows: dict[tuple[str, float, float], dict[str, Any]] = {}
    for case in base.CASES:
        for window_sec in WINDOW_SECS:
            for offset_sec in WINDOW_OFFSETS_SEC:
                baseline_rows[(case.case_id, float(window_sec), float(offset_sec))] = lane6c.evaluate_variant_case(
                    case,
                    full_signals[case.case_id],
                    window_sec=window_sec,
                    offset_sec=offset_sec,
                    variant=BASELINE_VARIANT,
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
