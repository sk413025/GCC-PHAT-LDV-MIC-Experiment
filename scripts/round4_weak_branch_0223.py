#!/usr/bin/env python3
"""
Round 4 lane 2: weak-branch candidate rescue for 0223.

This lane keeps the stable front-end and scorer, but only augments the weaker
branch candidate list. The weak branch is detected blindly from the base
candidate pool using branch peak strength.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import filter_family_time_domain_0223 as td
import round3_blind_proxy_0223 as lane2
import round4_candidate_generation_0223 as lane4a
import filter_sweep_0223_delta_tau as base


TOP_K_WEAK_EXTRA = 6
SUPPRESS_RADIUS_MS = 0.30


VARIANTS: list[dict[str, Any]] = [
    {"name": "base_only", "mode": "none"},
    {"name": "weak_local_zscore_mic", "mode": "local_zscore_mic"},
    {"name": "weak_clip_rms_mic", "mode": "clip_rms_mic"},
    {"name": "weak_diff2_union", "mode": "diff2_union"},
    {"name": "weak_residual_second_pass", "mode": "residual_second_pass"},
    {"name": "weak_bundle", "mode": "bundle"},
]


def branch_strength(candidates: list[dict[str, float]]) -> float:
    if not candidates:
        return 0.0
    top = candidates[0]
    return float(top["amp"] * max(top["psr_like"], 1e-9))


def detect_weak_branch(base_candidates_vl: list[dict[str, float]], base_candidates_vr: list[dict[str, float]]) -> str:
    return "vl" if branch_strength(base_candidates_vl) <= branch_strength(base_candidates_vr) else "vr"


def mic_zscore_proposal(ldv_base: np.ndarray, mic: np.ndarray) -> np.ndarray:
    return td.apply_chain(np.asarray(mic, dtype=np.float64), ["zscore"])


def mic_clip_rms_proposal(mic: np.ndarray) -> np.ndarray:
    return lane2.lane1.process_signal(np.asarray(mic, dtype=np.float64), {"kind": "rms_gain", "window_ms": 40.0, "clip_gain": (0.5, 2.0)}, apply_diff=False)


def residual_second_pass_taus(lag_ms: np.ndarray, abs_cc: np.ndarray) -> list[float]:
    if abs_cc.size == 0:
        return []
    cc = np.asarray(abs_cc, dtype=np.float64).copy()
    idx0 = int(np.argmax(cc))
    radius = int(round(SUPPRESS_RADIUS_MS / max((lag_ms[1] - lag_ms[0]), 1e-9))) if lag_ms.size > 1 else 1
    lo = max(0, idx0 - radius)
    hi = min(len(cc), idx0 + radius + 1)
    cc[lo:hi] = 0.0
    rows = lane2.extract_candidates_with_features(lag_ms, cc, lag_min_ms=lane2.LAG_MIN_MS, lag_max_ms=lane2.LAG_MAX_MS, top_k=TOP_K_WEAK_EXTRA)
    return [row["tau_ms"] for row in rows]


def proposal_taus_for_branch(
    variant_mode: str,
    *,
    weak_branch: str,
    signals: dict[str, np.ndarray],
    base_lag_ms: np.ndarray,
    base_abs_cc: np.ndarray,
) -> list[float]:
    if variant_mode == "none":
        return []

    proposal_taus: list[float] = []
    base_ldv, base_mic_l, base_mic_r = lane4a.base_frontend(signals)
    mic = base_mic_l if weak_branch == "vl" else base_mic_r

    def add_from_curve(ldv_sig: np.ndarray, mic_sig: np.ndarray, bandpass: tuple[float, float]) -> None:
        lag_p, cc_p = lane2.lane1.pair.gcc_curve(ldv_sig, mic_sig, lane2.FS, max_lag_ms=lane2.MAX_LAG_MS, bandpass=bandpass)
        rows = lane2.extract_candidates_with_features(lag_p, cc_p, lag_min_ms=lane2.LAG_MIN_MS, lag_max_ms=lane2.LAG_MAX_MS, top_k=TOP_K_WEAK_EXTRA)
        proposal_taus.extend([row["tau_ms"] for row in rows])

    if variant_mode in {"local_zscore_mic", "bundle"}:
        add_from_curve(base_ldv, mic_zscore_proposal(base_ldv, mic), lane2.BANDPASS)
    if variant_mode in {"clip_rms_mic", "bundle"}:
        add_from_curve(base_ldv, mic_clip_rms_proposal(mic), lane2.BANDPASS)
    if variant_mode in {"diff2_union", "bundle"}:
        ldv_diff2 = td.apply_chain(np.asarray(signals["ldv"], dtype=np.float64), ["diff2"])
        add_from_curve(ldv_diff2, signals["mic_l"] if weak_branch == "vl" else signals["mic_r"], (500.0, 2000.0))
    if variant_mode in {"residual_second_pass", "bundle"}:
        proposal_taus.extend(residual_second_pass_taus(base_lag_ms, base_abs_cc))

    deduped: list[float] = []
    for tau in proposal_taus:
        if all(abs(tau - existing) > lane4a.TAU_MERGE_MS for existing in deduped):
            deduped.append(tau)
    return deduped[: TOP_K_WEAK_EXTRA * 2]


def evaluate_case_windows(case: base.CaseRef, full_signals: dict[str, np.ndarray], *, window_sec: float) -> list[dict[str, Any]]:
    return lane4a.evaluate_case_windows(case, full_signals, window_sec=window_sec)


def evaluate_variant(case_windows: list[dict[str, Any]], variant: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in case_windows:
        weak_branch = detect_weak_branch(item["base_candidates_vl"], item["base_candidates_vr"])
        extra_taus = proposal_taus_for_branch(
            variant["mode"],
            weak_branch=weak_branch,
            signals=item["signals"],
            base_lag_ms=item["base_lag_vl"] if weak_branch == "vl" else item["base_lag_vr"],
            base_abs_cc=item["base_cc_vl"] if weak_branch == "vl" else item["base_cc_vr"],
        )
        if weak_branch == "vl":
            union_vl = lane4a.build_union_candidates(item["base_candidates_vl"], [[{"tau_ms": tau, "amp": 0.0, "prom": 0.0, "prom_ratio": 0.0, "psr_like": 0.0, "amp_ratio_to_top": 0.0} for tau in extra_taus]], item["base_lag_vl"], item["base_cc_vl"])
            union_vr = item["base_candidates_vr"]
        else:
            union_vl = item["base_candidates_vl"]
            union_vr = lane4a.build_union_candidates(item["base_candidates_vr"], [[{"tau_ms": tau, "amp": 0.0, "prom": 0.0, "prom_ratio": 0.0, "psr_like": 0.0, "amp_ratio_to_top": 0.0} for tau in extra_taus]], item["base_lag_vr"], item["base_cc_vr"])

        pair_rows = lane2.build_pair_rows(union_vl, union_vr)
        selected = lane2.select_best_pair(pair_rows, lane4a.CURRENT_SCORE_VARIANT)
        ref = item["reference"]
        oracle_vl_hit = any(abs(c["tau_ms"] - ref["tau_vl_ms"]) <= lane4a.ORACLE_RADIUS_MS for c in union_vl)
        oracle_vr_hit = any(abs(c["tau_ms"] - ref["tau_vr_ms"]) <= lane4a.ORACLE_RADIUS_MS for c in union_vr)
        row = {
            "case_id": item["case_id"],
            "variant": variant["name"],
            "offset_sec": item["offset_sec"],
            "reference": ref,
            "weak_branch": weak_branch,
            "selected": selected,
            "oracle_vl_hit": bool(oracle_vl_hit),
            "oracle_vr_hit": bool(oracle_vr_hit),
            "oracle_pair_recall": bool(oracle_vl_hit and oracle_vr_hit),
            "num_candidates_vl": len(union_vl),
            "num_candidates_vr": len(union_vr),
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
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    summary = []
    for variant, items in grouped.items():
        central = [it for it in items if abs(it["offset_sec"]) < 1e-9]
        central_valid = [it["selected"]["best"] for it in central if it["selected"] is not None]
        if not central_valid:
            summary.append({"variant": variant, "central_delta_tau_mae_ms": None})
            continue
        dt = np.array([it["delta_tau_abs_err_ms"] for it in central_valid], dtype=np.float64)
        hard_dt = np.array(
            [it["selected"]["best"]["delta_tau_abs_err_ms"] for it in central if it["case_id"] in {"block6_n04_19", "block7_n08_20"} and it["selected"] is not None],
            dtype=np.float64,
        )
        summary.append(
            {
                "variant": variant,
                "central_delta_tau_mae_ms": float(np.mean(dt)),
                "central_max_delta_tau_abs_err_ms": float(np.max(dt)),
                "hard_case_delta_tau_mae_ms": float(np.mean(hard_dt)) if hard_dt.size else None,
                "oracle_vl_hit_cases": sum(1 for it in central if it["oracle_vl_hit"]),
                "oracle_vr_hit_cases": sum(1 for it in central if it["oracle_vr_hit"]),
                "oracle_pair_recall_cases": sum(1 for it in central if it["oracle_pair_recall"]),
                "weak_branch_vl_cases": sum(1 for it in central if it["weak_branch"] == "vl"),
                "weak_branch_vr_cases": sum(1 for it in central if it["weak_branch"] == "vr"),
            }
        )
    summary.sort(
        key=lambda x: (
            1 if x["hard_case_delta_tau_mae_ms"] is None else 0,
            float("inf") if x["hard_case_delta_tau_mae_ms"] is None else x["hard_case_delta_tau_mae_ms"],
            float("inf") if x["central_delta_tau_mae_ms"] is None else x["central_delta_tau_mae_ms"],
        )
    )
    return {"variants": summary}


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 4 Weak-Branch Rescue Sweep",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        f"- Fixed scorer: `len80_current_score`",
        "",
        "## Summary",
        "",
        "| variant | central_dt_mae_ms | hard_dt_mae_ms | central_max_dt_ms | oracle_vl_hits | oracle_vr_hits | oracle_pair_recall | weak_vl_cases | weak_vr_cases |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["variants"]:
        def fmt(name: str) -> str:
            value = row.get(name)
            if value is None:
                return "NA"
            return f"{float(value):.3f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    fmt("central_delta_tau_mae_ms"),
                    fmt("hard_case_delta_tau_mae_ms"),
                    fmt("central_max_delta_tau_abs_err_ms"),
                    str(row["oracle_vl_hit_cases"]),
                    str(row["oracle_vr_hit_cases"]),
                    str(row["oracle_pair_recall_cases"]),
                    str(row["weak_branch_vl_cases"]),
                    str(row["weak_branch_vr_cases"]),
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 4 weak-branch rescue sweep for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--window_sec", type=float, default=5.0)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "results" / f"round4_weak_branch_0223_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    full_signals = {case.case_id: lane2.lane1.load_full_case_signals(case, args.data_root) for case in base.CASES}
    case_windows = {case.case_id: evaluate_case_windows(case, full_signals[case.case_id], window_sec=args.window_sec) for case in base.CASES}

    rows = []
    for variant in VARIANTS:
        for case in base.CASES:
            rows.extend(evaluate_variant(case_windows[case.case_id], variant))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "window_sec": float(args.window_sec),
        "variants": VARIANTS,
        "results": rows,
        "summary": summarize(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir / "report.md", payload)
    print(f"[ok] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
