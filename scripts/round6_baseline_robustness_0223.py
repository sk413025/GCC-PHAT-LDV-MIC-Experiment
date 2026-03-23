#!/usr/bin/env python3
"""
Round 6 lane 1: frozen-baseline robustness validation for the promoted 0223 stack.

This lane does not search for a new winner. It freezes the deployed stack:

- front-end: diff_len80_ldvonly_bp700_1800
- pair pool: cond_prune_soft
- scorer: same_vl_replace_w6_p3
- handoff selector: handoff_ratio_0p90

Then it measures whether the promoted result survives:

- neighboring window offsets
- neighboring window lengths
- mild handoff-ratio perturbations
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
import round4_candidate_generation_0223 as lane4a
import round5_pair_formation_0223 as lane5a
import round5_same_anchor_handoff_0223 as lane5d


DEPLOYED_HANDOFF_RATIO = 0.90
DELTA_LIFT_MIN_MS = 0.20
WINDOW_OFFSETS_SEC = (-1.0, -0.5, 0.0, 0.5, 1.0)
WINDOW_SECS = (4.0, 5.0, 6.0)
HANDOFF_RATIOS = (0.88, 0.90, 0.93, 0.95)

PROMOTED_REFERENCE = {
    "central_delta_tau_mae_ms": 0.05488870500594145,
    "hard_case_delta_tau_mae_ms": 0.07432639873206037,
    "hard_case_win_rate": 1.0,
    "central_max_delta_tau_abs_err_ms": 0.07974048900676944,
    "block6_selected_delta_tau_ms": 0.583333333333333,
    "block7_selected_delta_tau_ms": -0.375,
}


def central_handoff_variant() -> dict[str, float]:
    return {"name": "handoff_ratio_0p90", "handoff_ratio": DEPLOYED_HANDOFF_RATIO, "delta_lift_min_ms": DELTA_LIFT_MIN_MS}


def robustness_variant(handoff_ratio: float) -> dict[str, float]:
    return {"name": f"handoff_ratio_{handoff_ratio:.2f}", "handoff_ratio": float(handoff_ratio), "delta_lift_min_ms": DELTA_LIFT_MIN_MS}


def evaluate_window(
    case: base.CaseRef,
    full_signals: dict[str, np.ndarray],
    *,
    window_sec: float,
    offset_sec: float,
    handoff_ratio: float,
) -> dict[str, Any]:
    ref = base.compute_reference(case)
    sliced = lane2.lane1.slice_signals(full_signals, window_sec=window_sec, offset_sec=offset_sec)
    base_ldv, base_mic_l, base_mic_r = lane4a.base_frontend(sliced)
    lag_vl, cc_vl = lane2.lane1.pair.gcc_curve(
        base_ldv,
        base_mic_l,
        lane2.FS,
        max_lag_ms=lane2.MAX_LAG_MS,
        bandpass=lane4a.BASE_BANDPASS,
    )
    lag_vr, cc_vr = lane2.lane1.pair.gcc_curve(
        base_ldv,
        base_mic_r,
        lane2.FS,
        max_lag_ms=lane2.MAX_LAG_MS,
        bandpass=lane4a.BASE_BANDPASS,
    )
    base_cand_vl = lane2.extract_candidates_with_features(
        lag_vl,
        cc_vl,
        lag_min_ms=lane2.LAG_MIN_MS,
        lag_max_ms=lane2.LAG_MAX_MS,
        top_k=lane2.TOP_K,
    )
    base_cand_vr = lane2.extract_candidates_with_features(
        lag_vr,
        cc_vr,
        lag_min_ms=lane2.LAG_MIN_MS,
        lag_max_ms=lane2.LAG_MAX_MS,
        top_k=lane2.TOP_K,
    )
    item = {
        "case_id": case.case_id,
        "offset_sec": float(offset_sec),
        "reference": ref,
        "base_candidates_vl": base_cand_vl,
        "base_candidates_vr": base_cand_vr,
    }
    scored_pool = lane5d.build_scored_pool(item)
    selected_pool = lane5d.apply_handoff(scored_pool, robustness_variant(handoff_ratio))
    selected = {"best": selected_pool[0], "top_pairs": selected_pool[:10], "num_pairs": len(selected_pool)} if selected_pool else None
    strict = lane5d.strict_ranks(selected_pool, case.case_id)

    row = {
        "case_id": case.case_id,
        "window_sec": float(window_sec),
        "offset_sec": float(offset_sec),
        "handoff_ratio": float(handoff_ratio),
        "reference": ref,
        "selected": selected,
        "strict_ranks": strict,
        "num_pairs_after": len(selected_pool),
    }
    if selected is not None:
        best = selected["best"]
        theta_v = float(np.degrees(np.arcsin(np.clip((best["delta_tau_ms"] / 1000.0) * base.C / 1.4, -1.0, 1.0))))
        best["theta_v_deg"] = theta_v
        best["delta_tau_abs_err_ms"] = abs(best["delta_tau_ms"] - ref["delta_tau_ms"])
        best["theta_v_abs_err_deg"] = abs(theta_v - ref["theta_v_deg"])
        best["physical_positive_lags"] = bool(best["tau_vl_ms"] > 0.0 and best["tau_vr_ms"] > 0.0)
        best["physical_small_delta"] = bool(abs(best["delta_tau_ms"]) <= lane2.DELTA_LIMIT_MS)
        best["physical_valid"] = bool(best["physical_positive_lags"] and best["physical_small_delta"])
        best["hard_case_win"] = bool(best["delta_tau_abs_err_ms"] <= 0.224) if case.case_id in lane5a.HARD_CASES else None
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
        }
    dt = np.array([row["delta_tau_abs_err_ms"] for row in valid], dtype=np.float64)
    th = np.array([row["theta_v_abs_err_deg"] for row in valid], dtype=np.float64)
    physical_count = sum(1 for row in valid if row["physical_valid"])
    hard_rows = [row["selected"]["best"] for row in rows if row["case_id"] in lane5a.HARD_CASES and row["selected"] is not None]
    hard_dt = np.array([row["delta_tau_abs_err_ms"] for row in hard_rows], dtype=np.float64)
    hard_win = np.array([1.0 if row["hard_case_win"] else 0.0 for row in hard_rows], dtype=np.float64)
    by_case = {row["case_id"]: row for row in rows}
    block6 = by_case.get("block6_n04_19")
    block7 = by_case.get("block7_n08_20")
    return {
        "valid_cases": len(valid),
        "physical_count": physical_count,
        "delta_tau_mae_ms": float(np.mean(dt)),
        "theta_v_mae_deg": float(np.mean(th)),
        "max_delta_tau_abs_err_ms": float(np.max(dt)),
        "hard_case_delta_tau_mae_ms": float(np.mean(hard_dt)) if hard_dt.size else None,
        "hard_case_win_rate": float(np.mean(hard_win)) if hard_win.size else None,
        "block6_selected_delta_tau_ms": (
            block6["selected"]["best"]["delta_tau_ms"] if block6 and block6["selected"] is not None else None
        ),
        "block7_selected_delta_tau_ms": (
            block7["selected"]["best"]["delta_tau_ms"] if block7 and block7["selected"] is not None else None
        ),
        "block6_rescue_pair_rank": block6["strict_ranks"]["rescue_pair_rank"] if block6 and block6["strict_ranks"] else None,
        "block7_rescue_pair_rank": block7["strict_ranks"]["rescue_pair_rank"] if block7 and block7["strict_ranks"] else None,
        "rows": rows,
    }


def compute_window_stability(rows: list[dict[str, Any]], *, window_sec: float, handoff_ratio: float) -> dict[str, Any]:
    subset = [row for row in rows if abs(row["window_sec"] - window_sec) < 1e-9 and abs(row["handoff_ratio"] - handoff_ratio) < 1e-9]
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset:
        by_case[row["case_id"]].append(row)

    case_std = []
    hard_case_stats = {}
    all_physical_valid = True
    for case_id, items in by_case.items():
        items.sort(key=lambda x: x["offset_sec"])
        selected = [item["selected"]["best"] for item in items if item["selected"] is not None]
        if len(selected) != len(items):
            all_physical_valid = False
            continue
        dt = np.array([item["delta_tau_ms"] for item in selected], dtype=np.float64)
        case_std.append(float(np.std(dt)))
        if any(not item["physical_valid"] for item in selected):
            all_physical_valid = False
        if case_id in lane5a.HARD_CASES:
            ranks = [item["strict_ranks"]["rescue_pair_rank"] if item["strict_ranks"] else None for item in items]
            hard_case_stats[case_id] = {
                "hit_at_1": int(sum(1 for rank in ranks if rank is not None and rank <= 1)),
                "hit_at_3": int(sum(1 for rank in ranks if rank is not None and rank <= 3)),
                "selected_delta_tau_ms": [item["selected"]["best"]["delta_tau_ms"] for item in items],
                "rescue_pair_ranks": ranks,
            }

    return {
        "window_sec": float(window_sec),
        "handoff_ratio": float(handoff_ratio),
        "all_physical_valid": bool(all_physical_valid),
        "window_stability_mean_std_ms": float(np.mean(case_std)) if case_std else None,
        "window_stability_max_std_ms": float(np.max(case_std)) if case_std else None,
        "hard_cases": hard_case_stats,
    }


def summarize_perturbations(rows: list[dict[str, Any]], central_case_errors: dict[str, float]) -> dict[str, Any]:
    grouped: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (float(row["window_sec"]), float(row["handoff_ratio"]), float(row["offset_sec"]))
        grouped[key].append(row)

    per_setting = []
    hard_case_total = 0
    hard_case_wins = 0
    block6_rescue_like = 0
    block7_rescue_like = 0
    simultaneous_hard_failure = 0
    easy_regression_violations = 0
    for key, items in sorted(grouped.items()):
        setting = summarize_setting(items)
        block6_ok = (
            setting["block6_selected_delta_tau_ms"] is not None
            and setting["block6_selected_delta_tau_ms"] >= 0.45
            and (setting["block6_rescue_pair_rank"] is not None and setting["block6_rescue_pair_rank"] <= 3)
        )
        block7_ok = (
            setting["block7_selected_delta_tau_ms"] is not None
            and -0.45 <= setting["block7_selected_delta_tau_ms"] <= -0.20
            and (setting["block7_rescue_pair_rank"] is not None and setting["block7_rescue_pair_rank"] <= 4)
        )
        block6_rescue_like += int(block6_ok)
        block7_rescue_like += int(block7_ok)
        hard_case_total += 2
        hard_case_wins += int(block6_ok) + int(block7_ok)
        if not block6_ok and not block7_ok:
            simultaneous_hard_failure += 1

        easy_ok = True
        for item in items:
            if item["case_id"] not in {"block4_p08_17", "block5_p04_18"} or item["selected"] is None:
                continue
            case_err = item["selected"]["best"]["delta_tau_abs_err_ms"]
            if case_err > central_case_errors[item["case_id"]] + 0.03:
                easy_ok = False
                easy_regression_violations += 1

        setting_pass = (
            setting["valid_cases"] == 4
            and setting["physical_count"] == 4
            and (setting["delta_tau_mae_ms"] is not None and setting["delta_tau_mae_ms"] <= 0.090)
            and (setting["max_delta_tau_abs_err_ms"] is not None and setting["max_delta_tau_abs_err_ms"] <= 0.150)
            and block6_ok
            and block7_ok
            and easy_ok
        )
        per_setting.append(
            {
                "window_sec": key[0],
                "handoff_ratio": key[1],
                "offset_sec": key[2],
                "valid_cases": setting["valid_cases"],
                "physical_count": setting["physical_count"],
                "delta_tau_mae_ms": setting["delta_tau_mae_ms"],
                "max_delta_tau_abs_err_ms": setting["max_delta_tau_abs_err_ms"],
                "hard_case_delta_tau_mae_ms": setting["hard_case_delta_tau_mae_ms"],
                "hard_case_win_rate": setting["hard_case_win_rate"],
                "block6_selected_delta_tau_ms": setting["block6_selected_delta_tau_ms"],
                "block7_selected_delta_tau_ms": setting["block7_selected_delta_tau_ms"],
                "block6_rescue_pair_rank": setting["block6_rescue_pair_rank"],
                "block7_rescue_pair_rank": setting["block7_rescue_pair_rank"],
                "easy_case_non_regression": easy_ok,
                "setting_pass": setting_pass,
            }
        )

    dt_mae_values = [row["delta_tau_mae_ms"] for row in per_setting if row["delta_tau_mae_ms"] is not None]
    max_err_values = [row["max_delta_tau_abs_err_ms"] for row in per_setting if row["max_delta_tau_abs_err_ms"] is not None]
    return {
        "num_settings": len(per_setting),
        "pass_count": int(sum(1 for row in per_setting if row["setting_pass"])),
        "pass_rate": float(np.mean([1.0 if row["setting_pass"] else 0.0 for row in per_setting])) if per_setting else None,
        "median_delta_tau_mae_ms": float(np.median(np.array(dt_mae_values, dtype=np.float64))) if dt_mae_values else None,
        "worst_max_delta_tau_abs_err_ms": float(np.max(np.array(max_err_values, dtype=np.float64))) if max_err_values else None,
        "aggregate_hard_case_win_rate": float(hard_case_wins / max(hard_case_total, 1)),
        "block6_rescue_like_rate": float(block6_rescue_like / max(len(per_setting), 1)),
        "block7_rescue_like_rate": float(block7_rescue_like / max(len(per_setting), 1)),
        "simultaneous_hard_failure_settings": int(simultaneous_hard_failure),
        "easy_case_regression_violations": int(easy_regression_violations),
        "per_setting": per_setting,
    }


def build_gate_summary(
    central_summary: dict[str, Any],
    central_rows: list[dict[str, Any]],
    window_summary: dict[str, Any],
    perturbation_summary: dict[str, Any],
) -> dict[str, Any]:
    central_by_case = {row["case_id"]: row for row in central_rows}
    central_preservation_pass = (
        central_summary["valid_cases"] == 4
        and central_summary["physical_count"] == 4
        and central_summary["delta_tau_mae_ms"] <= 0.065
        and central_summary["max_delta_tau_abs_err_ms"] <= 0.100
        and central_summary["block6_selected_delta_tau_ms"] is not None
        and 0.50 <= central_summary["block6_selected_delta_tau_ms"] <= 0.66
        and (central_summary["block6_rescue_pair_rank"] is not None and central_summary["block6_rescue_pair_rank"] <= 2)
        and central_summary["block7_selected_delta_tau_ms"] is not None
        and -0.45 <= central_summary["block7_selected_delta_tau_ms"] <= -0.20
        and (central_summary["block7_rescue_pair_rank"] is not None and central_summary["block7_rescue_pair_rank"] <= 3)
    )
    hard_case_win_pass = (
        central_summary["hard_case_win_rate"] == 1.0
        and central_summary["hard_case_delta_tau_mae_ms"] is not None
        and central_summary["hard_case_delta_tau_mae_ms"] <= 0.090
        and central_summary["block6_selected_delta_tau_ms"] >= 0.50
        and central_summary["block7_selected_delta_tau_ms"] <= -0.20
    )
    window_stability_pass = (
        window_summary["all_physical_valid"]
        and window_summary["window_stability_mean_std_ms"] is not None
        and window_summary["window_stability_mean_std_ms"] <= 0.15
        and window_summary["window_stability_max_std_ms"] is not None
        and window_summary["window_stability_max_std_ms"] <= 0.25
        and window_summary["hard_cases"].get("block6_n04_19", {}).get("hit_at_1", 0) >= 4
        and window_summary["hard_cases"].get("block6_n04_19", {}).get("hit_at_3", 0) >= 5
        and window_summary["hard_cases"].get("block7_n08_20", {}).get("hit_at_1", 0) >= 3
        and window_summary["hard_cases"].get("block7_n08_20", {}).get("hit_at_3", 0) >= 4
    )
    perturbation_pass = (
        perturbation_summary["pass_rate"] is not None
        and perturbation_summary["pass_rate"] >= 0.80
        and perturbation_summary["median_delta_tau_mae_ms"] is not None
        and perturbation_summary["median_delta_tau_mae_ms"] <= 0.090
        and perturbation_summary["worst_max_delta_tau_abs_err_ms"] is not None
        and perturbation_summary["worst_max_delta_tau_abs_err_ms"] <= 0.150
        and perturbation_summary["aggregate_hard_case_win_rate"] >= 0.75
        and perturbation_summary["block6_rescue_like_rate"] >= 0.75
        and perturbation_summary["block7_rescue_like_rate"] >= 0.75
        and perturbation_summary["simultaneous_hard_failure_settings"] == 0
        and perturbation_summary["easy_case_regression_violations"] == 0
    )
    return {
        "central_preservation_pass": bool(central_preservation_pass),
        "hard_case_win_pass": bool(hard_case_win_pass),
        "window_stability_pass": bool(window_stability_pass),
        "perturbation_pass": bool(perturbation_pass),
        "overall_pass": bool(
            central_preservation_pass and hard_case_win_pass and window_stability_pass and perturbation_pass
        ),
    }


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    central = payload["central_summary"]
    window = payload["window_summary"]
    perturb = payload["perturbation_summary"]
    gates = payload["gates"]
    lines = [
        "# Round 6 Frozen-Baseline Robustness Validation",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        "- Frozen stack: `cond_prune_soft + same_vl_replace_w6_p3 + handoff_ratio_0p90`",
        f"- Window lengths: `{payload['window_secs']}`",
        f"- Window offsets: `{payload['window_offsets_sec']}`",
        f"- Handoff ratios: `{payload['handoff_ratios']}`",
        "",
        "## Central Preservation",
        "",
        f"- valid_cases: `{central['valid_cases']}`",
        f"- physical_count: `{central['physical_count']}`",
        f"- delta_tau_mae_ms: `{central['delta_tau_mae_ms']:.3f}`",
        f"- hard_case_delta_tau_mae_ms: `{central['hard_case_delta_tau_mae_ms']:.3f}`",
        f"- hard_case_win_rate: `{central['hard_case_win_rate']:.3f}`",
        f"- max_delta_tau_abs_err_ms: `{central['max_delta_tau_abs_err_ms']:.3f}`",
        f"- block6_selected_delta_tau_ms: `{central['block6_selected_delta_tau_ms']:.3f}`",
        f"- block7_selected_delta_tau_ms: `{central['block7_selected_delta_tau_ms']:.3f}`",
        "",
        "## Window Stability",
        "",
        f"- window_stability_mean_std_ms: `{window['window_stability_mean_std_ms']:.3f}`",
        f"- window_stability_max_std_ms: `{window['window_stability_max_std_ms']:.3f}`",
        f"- block6 hit@1 / hit@3: `{window['hard_cases'].get('block6_n04_19', {}).get('hit_at_1', 0)}` / `{window['hard_cases'].get('block6_n04_19', {}).get('hit_at_3', 0)}`",
        f"- block7 hit@1 / hit@3: `{window['hard_cases'].get('block7_n08_20', {}).get('hit_at_1', 0)}` / `{window['hard_cases'].get('block7_n08_20', {}).get('hit_at_3', 0)}`",
        "",
        "## Perturbation Suite",
        "",
        f"- num_settings: `{perturb['num_settings']}`",
        f"- pass_rate: `{perturb['pass_rate']:.3f}`",
        f"- median_delta_tau_mae_ms: `{perturb['median_delta_tau_mae_ms']:.3f}`",
        f"- worst_max_delta_tau_abs_err_ms: `{perturb['worst_max_delta_tau_abs_err_ms']:.3f}`",
        f"- aggregate_hard_case_win_rate: `{perturb['aggregate_hard_case_win_rate']:.3f}`",
        f"- block6_rescue_like_rate: `{perturb['block6_rescue_like_rate']:.3f}`",
        f"- block7_rescue_like_rate: `{perturb['block7_rescue_like_rate']:.3f}`",
        f"- simultaneous_hard_failure_settings: `{perturb['simultaneous_hard_failure_settings']}`",
        f"- easy_case_regression_violations: `{perturb['easy_case_regression_violations']}`",
        "",
        "## Gates",
        "",
        f"- central_preservation_pass: `{gates['central_preservation_pass']}`",
        f"- hard_case_win_pass: `{gates['hard_case_win_pass']}`",
        f"- window_stability_pass: `{gates['window_stability_pass']}`",
        f"- perturbation_pass: `{gates['perturbation_pass']}`",
        f"- overall_pass: `{gates['overall_pass']}`",
        "",
        "## Per-Setting Summary",
        "",
        "| window_sec | handoff_ratio | offset_sec | dt_mae_ms | max_dt_ms | hard_win_rate | block6_dt | block7_dt | pass |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in perturb["per_setting"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row['window_sec']:.1f}",
                    f"{row['handoff_ratio']:.2f}",
                    f"{row['offset_sec']:.1f}",
                    "NA" if row["delta_tau_mae_ms"] is None else f"{row['delta_tau_mae_ms']:.3f}",
                    "NA" if row["max_delta_tau_abs_err_ms"] is None else f"{row['max_delta_tau_abs_err_ms']:.3f}",
                    "NA" if row["hard_case_win_rate"] is None else f"{row['hard_case_win_rate']:.3f}",
                    "NA" if row["block6_selected_delta_tau_ms"] is None else f"{row['block6_selected_delta_tau_ms']:.3f}",
                    "NA" if row["block7_selected_delta_tau_ms"] is None else f"{row['block7_selected_delta_tau_ms']:.3f}",
                    "PASS" if row["setting_pass"] else "FAIL",
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 6 frozen-baseline robustness validation for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round6_baseline_robustness_0223_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    full_signals = {
        case.case_id: lane2.lane1.load_full_case_signals(case, args.data_root)
        for case in base.CASES
    }

    rows = []
    for window_sec in WINDOW_SECS:
        for handoff_ratio in HANDOFF_RATIOS:
            for offset_sec in WINDOW_OFFSETS_SEC:
                for case in base.CASES:
                    rows.append(
                        evaluate_window(
                            case,
                            full_signals[case.case_id],
                            window_sec=window_sec,
                            offset_sec=offset_sec,
                            handoff_ratio=handoff_ratio,
                        )
                    )

    central_rows = [
        row
        for row in rows
        if abs(row["window_sec"] - 5.0) < 1e-9
        and abs(row["handoff_ratio"] - DEPLOYED_HANDOFF_RATIO) < 1e-9
        and abs(row["offset_sec"]) < 1e-9
    ]
    central_summary = summarize_setting(central_rows)
    central_case_errors = {
        row["case_id"]: row["selected"]["best"]["delta_tau_abs_err_ms"]
        for row in central_rows
        if row["selected"] is not None
    }
    window_summary = compute_window_stability(rows, window_sec=5.0, handoff_ratio=DEPLOYED_HANDOFF_RATIO)
    perturbation_summary = summarize_perturbations(rows, central_case_errors)
    gates = build_gate_summary(central_summary, central_rows, window_summary, perturbation_summary)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "promoted_reference": PROMOTED_REFERENCE,
        "window_secs": list(WINDOW_SECS),
        "window_offsets_sec": list(WINDOW_OFFSETS_SEC),
        "handoff_ratios": list(HANDOFF_RATIOS),
        "results": rows,
        "central_summary": central_summary,
        "window_summary": window_summary,
        "perturbation_summary": perturbation_summary,
        "gates": gates,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir / "report.md", payload)
    print(f"[ok] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
