#!/usr/bin/env python3
"""
Round 4 lane 1: candidate-generation search for 0223.

This round keeps the stable scoring side fixed and only changes how candidate
taus are proposed. The main idea is to borrow proposal candidates from
teacher-style or branch-cleaned front-ends, but still score those candidates on
the stable `diff_len80_ldvonly_bp700_1800` base curve.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import filter_family_time_domain_0223 as td
import filter_sweep_0223_delta_tau as base
import round3_blind_proxy_0223 as lane2


BASE_BANDPASS = (700.0, 1800.0)
ORACLE_RADIUS_MS = 0.60
TAU_MERGE_MS = 0.125
TOP_K_BASE = 8
TOP_K_EXTRA = 4


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "base_only",
        "proposal_specs": [],
    },
    {
        "name": "base_plus_diff2_700_1800",
        "proposal_specs": [{"name": "diff2_700_1800", "ldv_ops": ["diff2"], "mic_ops": [], "bandpass": (700.0, 1800.0)}],
    },
    {
        "name": "base_plus_diff2_500_2000",
        "proposal_specs": [{"name": "diff2_500_2000", "ldv_ops": ["diff2"], "mic_ops": [], "bandpass": (500.0, 2000.0)}],
    },
    {
        "name": "base_plus_mic_flatten_700_1800",
        "proposal_specs": [{"name": "mic_flatten_700_1800", "ldv_ops": ["diff", "rms_norm"], "mic_ops": ["flatten"], "bandpass": (700.0, 1800.0)}],
    },
    {
        "name": "base_plus_diff2_700_1800_and_mic_flatten",
        "proposal_specs": [
            {"name": "diff2_700_1800", "ldv_ops": ["diff2"], "mic_ops": [], "bandpass": (700.0, 1800.0)},
            {"name": "mic_flatten_700_1800", "ldv_ops": ["diff", "rms_norm"], "mic_ops": ["flatten"], "bandpass": (700.0, 1800.0)},
        ],
    },
    {
        "name": "base_plus_oracle_bundle",
        "proposal_specs": [
            {"name": "diff2_700_1800", "ldv_ops": ["diff2"], "mic_ops": [], "bandpass": (700.0, 1800.0)},
            {"name": "diff2_500_2000", "ldv_ops": ["diff2"], "mic_ops": [], "bandpass": (500.0, 2000.0)},
            {"name": "mic_flatten_700_1800", "ldv_ops": ["diff", "rms_norm"], "mic_ops": ["flatten"], "bandpass": (700.0, 1800.0)},
        ],
    },
]


CURRENT_SCORE_VARIANT = next(v for v in lane2.VARIANTS if v["name"] == "len80_current_score")


def base_frontend(signals: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ldv = lane2.lane1.process_signal(signals["ldv"], {"kind": "rms_gain", "window_ms": 80.0}, apply_diff=True)
    mic_l = lane2.lane1.process_signal(signals["mic_l"], {"kind": "none"}, apply_diff=False)
    mic_r = lane2.lane1.process_signal(signals["mic_r"], {"kind": "none"}, apply_diff=False)
    return ldv, mic_l, mic_r


def apply_ops(signal: np.ndarray, ops: list[str], *, is_ldv: bool) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    for op in ops:
        if op in {"zscore", "preemphasis", "diff", "diff2", "signed_sqrt", "envelope", "rms_norm"}:
            x = td.apply_chain(x, [op])
        elif op == "flatten":
            x = base.spectral_flatten(x)
        else:
            raise ValueError(f"Unknown op: {op}")
    if not is_ldv and "diff" in ops:
        return x
    return x


def proposal_frontend(signals: dict[str, np.ndarray], spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ldv = apply_ops(signals["ldv"], list(spec["ldv_ops"]), is_ldv=True)
    mic_l = apply_ops(signals["mic_l"], list(spec["mic_ops"]), is_ldv=False)
    mic_r = apply_ops(signals["mic_r"], list(spec["mic_ops"]), is_ldv=False)
    return ldv, mic_l, mic_r


def unique_taus(rows: list[dict[str, float]], *, top_k: int) -> list[float]:
    selected: list[float] = []
    for row in rows:
        tau = float(row["tau_ms"])
        if all(abs(tau - existing) > TAU_MERGE_MS for existing in selected):
            selected.append(tau)
        if len(selected) >= top_k:
            break
    return selected


def materialize_candidates_on_base(
    lag_ms: np.ndarray,
    abs_cc: np.ndarray,
    taus_ms: list[float],
) -> list[dict[str, float]]:
    max_amp = float(np.max(abs_cc)) if abs_cc.size else 1.0
    candidates = []
    for tau in taus_ms:
        idx = int(np.argmin(np.abs(lag_ms - tau)))
        amp = float(abs_cc[idx])
        lo = max(0, idx - 12)
        hi = min(len(abs_cc), idx + 13)
        side = np.concatenate((abs_cc[:lo], abs_cc[hi:]))
        side_max = float(np.max(side)) if side.size else 0.0
        candidates.append(
            {
                "tau_ms": float(lag_ms[idx]),
                "amp": amp,
                "prom": 0.0,
                "prom_ratio": 0.0,
                "psr_like": float(amp / max(side_max, 1e-9)),
                "amp_ratio_to_top": float(amp / max(max_amp, 1e-9)),
            }
        )
    candidates.sort(key=lambda x: x["amp"], reverse=True)
    ranked = []
    for rank, cand in enumerate(candidates, start=1):
        row = dict(cand)
        row["rank"] = rank
        ranked.append(row)
    return ranked


def build_union_candidates(
    base_candidates: list[dict[str, float]],
    proposer_candidates: list[list[dict[str, float]]],
    base_lag_ms: np.ndarray,
    base_abs_cc: np.ndarray,
) -> list[dict[str, float]]:
    tau_pool = unique_taus(base_candidates, top_k=TOP_K_BASE)
    for rows in proposer_candidates:
        tau_pool.extend(unique_taus(rows, top_k=TOP_K_EXTRA))
    deduped: list[float] = []
    for tau in tau_pool:
        if all(abs(tau - existing) > TAU_MERGE_MS for existing in deduped):
            deduped.append(tau)
    return materialize_candidates_on_base(base_lag_ms, base_abs_cc, deduped)


def evaluate_case_windows(case: base.CaseRef, full_signals: dict[str, np.ndarray], *, window_sec: float) -> list[dict[str, Any]]:
    ref = base.compute_reference(case)
    payloads = []
    for offset_sec in lane2.WINDOW_OFFSETS_SEC:
        sliced = lane2.lane1.slice_signals(full_signals, window_sec=window_sec, offset_sec=offset_sec)
        base_ldv, base_mic_l, base_mic_r = base_frontend(sliced)
        lag_vl, cc_vl = lane2.lane1.pair.gcc_curve(base_ldv, base_mic_l, lane2.FS, max_lag_ms=lane2.MAX_LAG_MS, bandpass=BASE_BANDPASS)
        lag_vr, cc_vr = lane2.lane1.pair.gcc_curve(base_ldv, base_mic_r, lane2.FS, max_lag_ms=lane2.MAX_LAG_MS, bandpass=BASE_BANDPASS)
        base_cand_vl = lane2.extract_candidates_with_features(lag_vl, cc_vl, lag_min_ms=lane2.LAG_MIN_MS, lag_max_ms=lane2.LAG_MAX_MS, top_k=lane2.TOP_K)
        base_cand_vr = lane2.extract_candidates_with_features(lag_vr, cc_vr, lag_min_ms=lane2.LAG_MIN_MS, lag_max_ms=lane2.LAG_MAX_MS, top_k=lane2.TOP_K)
        payloads.append(
            {
                "case_id": case.case_id,
                "offset_sec": float(offset_sec),
                "reference": ref,
                "signals": sliced,
                "base_lag_vl": lag_vl,
                "base_cc_vl": cc_vl,
                "base_lag_vr": lag_vr,
                "base_cc_vr": cc_vr,
                "base_candidates_vl": base_cand_vl,
                "base_candidates_vr": base_cand_vr,
            }
        )
    return payloads


def evaluate_variant(case_windows: list[dict[str, Any]], variant: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in case_windows:
        proposal_vl = []
        proposal_vr = []
        for spec in variant["proposal_specs"]:
            ldv_p, mic_l_p, mic_r_p = proposal_frontend(item["signals"], spec)
            lag_vl_p, cc_vl_p = lane2.lane1.pair.gcc_curve(ldv_p, mic_l_p, lane2.FS, max_lag_ms=lane2.MAX_LAG_MS, bandpass=spec["bandpass"])
            lag_vr_p, cc_vr_p = lane2.lane1.pair.gcc_curve(ldv_p, mic_r_p, lane2.FS, max_lag_ms=lane2.MAX_LAG_MS, bandpass=spec["bandpass"])
            proposal_vl.append(
                lane2.extract_candidates_with_features(lag_vl_p, cc_vl_p, lag_min_ms=lane2.LAG_MIN_MS, lag_max_ms=lane2.LAG_MAX_MS, top_k=lane2.TOP_K)
            )
            proposal_vr.append(
                lane2.extract_candidates_with_features(lag_vr_p, cc_vr_p, lag_min_ms=lane2.LAG_MIN_MS, lag_max_ms=lane2.LAG_MAX_MS, top_k=lane2.TOP_K)
            )

        union_vl = build_union_candidates(item["base_candidates_vl"], proposal_vl, item["base_lag_vl"], item["base_cc_vl"])
        union_vr = build_union_candidates(item["base_candidates_vr"], proposal_vr, item["base_lag_vr"], item["base_cc_vr"])
        pair_rows = lane2.build_pair_rows(union_vl, union_vr)
        selected = lane2.select_best_pair(pair_rows, CURRENT_SCORE_VARIANT)
        ref = item["reference"]
        oracle_vl_hit = any(abs(c["tau_ms"] - ref["tau_vl_ms"]) <= ORACLE_RADIUS_MS for c in union_vl)
        oracle_vr_hit = any(abs(c["tau_ms"] - ref["tau_vr_ms"]) <= ORACLE_RADIUS_MS for c in union_vr)
        row = {
            "case_id": item["case_id"],
            "variant": variant["name"],
            "offset_sec": item["offset_sec"],
            "reference": ref,
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
            summary.append(
                {
                    "variant": variant,
                    "central_valid_cases": 0,
                    "central_delta_tau_mae_ms": None,
                    "central_max_delta_tau_abs_err_ms": None,
                    "hard_case_delta_tau_mae_ms": None,
                    "window_delta_tau_mae_ms": None,
                    "oracle_vl_hit_cases": 0,
                    "oracle_vr_hit_cases": 0,
                    "oracle_pair_recall_cases": 0,
                }
            )
            continue
        central_dt = np.array([it["delta_tau_abs_err_ms"] for it in central_valid], dtype=np.float64)
        hard_dt = np.array(
            [it["selected"]["best"]["delta_tau_abs_err_ms"] for it in central if it["case_id"] in {"block6_n04_19", "block7_n08_20"} and it["selected"] is not None],
            dtype=np.float64,
        )
        window_dt = np.array([it["selected"]["best"]["delta_tau_abs_err_ms"] for it in items if it["selected"] is not None], dtype=np.float64)
        summary.append(
            {
                "variant": variant,
                "central_valid_cases": len(central_valid),
                "central_delta_tau_mae_ms": float(np.mean(central_dt)),
                "central_max_delta_tau_abs_err_ms": float(np.max(central_dt)),
                "hard_case_delta_tau_mae_ms": float(np.mean(hard_dt)) if hard_dt.size else None,
                "window_delta_tau_mae_ms": float(np.mean(window_dt)),
                "oracle_vl_hit_cases": sum(1 for it in central if it["oracle_vl_hit"]),
                "oracle_vr_hit_cases": sum(1 for it in central if it["oracle_vr_hit"]),
                "oracle_pair_recall_cases": sum(1 for it in central if it["oracle_pair_recall"]),
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
        "# Round 4 Candidate Generation Sweep",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        f"- Fixed scorer: `len80_current_score` on base curve",
        "",
        "## Summary",
        "",
        "| variant | central_dt_mae_ms | hard_dt_mae_ms | central_max_dt_ms | window_dt_mae_ms | oracle_vl_hits | oracle_vr_hits | oracle_pair_recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
                    fmt("window_delta_tau_mae_ms"),
                    str(row["oracle_vl_hit_cases"]),
                    str(row["oracle_vr_hit_cases"]),
                    str(row["oracle_pair_recall_cases"]),
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 4 candidate-generation sweep for 0223.")
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
        Path(__file__).resolve().parent.parent / "results" / f"round4_candidate_generation_0223_{timestamp}"
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
