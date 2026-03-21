#!/usr/bin/env python3
"""
Round 3 lane 2: blind proxy scoring on top of the provisional len80 front-end.

This lane keeps the Round 3 lane 1 front-end fixed:

- LDV preprocessing: diff1 + 80 ms local RMS normalization
- microphones: unchanged
- band: 700-1800 Hz

The search target is no longer a new filter family. It is a better blind
pair-scoring rule that converts the improved front-end landscape into lower
delta-tau error and better window stability.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks, peak_prominences

import filter_sweep_0223_delta_tau as base
import round3_diff1_local_norm_0223 as lane1


FS = 48000
BANDPASS = (700.0, 1800.0)
TOP_K = 12
LAG_MIN_MS = 4.4
LAG_MAX_MS = 6.5
MAX_LAG_MS = 10.0
DELTA_LIMIT_MS = 1.0
DELTA_SCALE_MS = 0.6
MEAN_TAU_CENTER_MS = 4.8
ORACLE_RADIUS_MS = 0.60
WINDOW_OFFSETS_SEC = lane1.WINDOW_OFFSETS_SEC
SUPPORT_SIGMA_MS = 0.20


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "len80_current_score",
        "mean_tau_scale_ms": 0.45,
        "balance_pow": 0.0,
        "psr_boost": 0.0,
        "support_boost": 0.0,
        "rank_discount_pow": 0.0,
    },
    {
        "name": "len80_tight_mean_tau",
        "mean_tau_scale_ms": 0.35,
        "balance_pow": 0.0,
        "psr_boost": 0.0,
        "support_boost": 0.0,
        "rank_discount_pow": 0.0,
    },
    {
        "name": "len80_balance_guard",
        "mean_tau_scale_ms": 0.45,
        "balance_pow": 0.5,
        "psr_boost": 0.0,
        "support_boost": 0.0,
        "rank_discount_pow": 0.0,
    },
    {
        "name": "len80_psr_like_boost",
        "mean_tau_scale_ms": 0.45,
        "balance_pow": 0.0,
        "psr_boost": 0.6,
        "support_boost": 0.0,
        "rank_discount_pow": 0.0,
    },
    {
        "name": "len80_window_support",
        "mean_tau_scale_ms": 0.45,
        "balance_pow": 0.0,
        "psr_boost": 0.0,
        "support_boost": 1.0,
        "rank_discount_pow": 0.0,
    },
    {
        "name": "len80_window_support_balance",
        "mean_tau_scale_ms": 0.45,
        "balance_pow": 0.5,
        "psr_boost": 0.0,
        "support_boost": 1.0,
        "rank_discount_pow": 0.0,
    },
    {
        "name": "len80_window_support_psr",
        "mean_tau_scale_ms": 0.45,
        "balance_pow": 0.0,
        "psr_boost": 0.6,
        "support_boost": 1.0,
        "rank_discount_pow": 0.0,
    },
    {
        "name": "len80_window_support_rank_guard",
        "mean_tau_scale_ms": 0.45,
        "balance_pow": 0.0,
        "psr_boost": 0.0,
        "support_boost": 1.0,
        "rank_discount_pow": 0.25,
    },
    {
        "name": "len80_hybrid_proxy",
        "mean_tau_scale_ms": 0.40,
        "balance_pow": 0.35,
        "psr_boost": 0.40,
        "support_boost": 0.75,
        "rank_discount_pow": 0.20,
    },
]


def extract_candidates_with_features(
    lag_ms: np.ndarray,
    abs_cc: np.ndarray,
    *,
    lag_min_ms: float,
    lag_max_ms: float,
    top_k: int,
) -> list[dict[str, float]]:
    mask = (lag_ms >= lag_min_ms) & (lag_ms <= lag_max_ms)
    if not np.any(mask):
        return []
    lag_win = lag_ms[mask]
    cc_win = abs_cc[mask]
    peak_idx, _ = find_peaks(cc_win)
    if peak_idx.size == 0:
        peak_idx = np.array([int(np.argmax(cc_win))], dtype=np.int64)
        prominences = np.array([0.0], dtype=np.float64)
    else:
        prominences = peak_prominences(cc_win, peak_idx)[0]

    max_amp = float(np.max(cc_win))
    candidates = []
    for i, idx in enumerate(peak_idx.tolist()):
        lo = max(0, idx - 12)
        hi = min(len(cc_win), idx + 13)
        side = np.concatenate((cc_win[:lo], cc_win[hi:]))
        side_max = float(np.max(side)) if side.size else 0.0
        amp = float(cc_win[idx])
        prom = float(prominences[i]) if i < len(prominences) else 0.0
        candidates.append(
            {
                "tau_ms": float(lag_win[idx]),
                "amp": amp,
                "prom": prom,
                "prom_ratio": float(prom / max(amp, 1e-9)),
                "psr_like": float(amp / max(side_max, 1e-9)),
                "amp_ratio_to_top": float(amp / max(max_amp, 1e-9)),
            }
        )
    candidates.sort(key=lambda x: (x["amp"], x["prom"]), reverse=True)
    ranked = []
    for rank, cand in enumerate(candidates[:top_k], start=1):
        row = dict(cand)
        row["rank"] = rank
        ranked.append(row)
    return ranked


def build_pair_rows(cand_vl: list[dict[str, float]], cand_vr: list[dict[str, float]]) -> list[dict[str, float]]:
    rows = []
    for vl in cand_vl:
        for vr in cand_vr:
            delta_tau_ms = vr["tau_ms"] - vl["tau_ms"]
            if abs(delta_tau_ms) > DELTA_LIMIT_MS:
                continue
            mean_tau_ms = 0.5 * (vl["tau_ms"] + vr["tau_ms"])
            prod = vl["amp"] * vr["amp"]
            balance = min(vl["amp"], vr["amp"]) / max(max(vl["amp"], vr["amp"]), 1e-9)
            rows.append(
                {
                    "tau_vl_ms": float(vl["tau_ms"]),
                    "tau_vr_ms": float(vr["tau_ms"]),
                    "delta_tau_ms": float(delta_tau_ms),
                    "mean_tau_ms": float(mean_tau_ms),
                    "prod": float(prod),
                    "balance": float(balance),
                    "prom_geom": float(np.sqrt(max(vl["prom_ratio"], 0.0) * max(vr["prom_ratio"], 0.0))),
                    "psr_geom": float(np.sqrt(max(vl["psr_like"], 0.0) * max(vr["psr_like"], 0.0))),
                    "amp_ratio_geom": float(np.sqrt(max(vl["amp_ratio_to_top"], 0.0) * max(vr["amp_ratio_to_top"], 0.0))),
                    "vl_rank": int(vl["rank"]),
                    "vr_rank": int(vr["rank"]),
                }
            )
    rows.sort(key=lambda x: x["prod"], reverse=True)
    return rows


def base_score(pair_row: dict[str, float], *, mean_tau_scale_ms: float) -> float:
    delta_quad = math.exp(-((pair_row["delta_tau_ms"] / max(DELTA_SCALE_MS, 1e-6)) ** 2))
    mean_tau_penalty = math.exp(
        -(((pair_row["mean_tau_ms"] - MEAN_TAU_CENTER_MS) / max(mean_tau_scale_ms, 1e-6)) ** 2)
    )
    return float(pair_row["prod"] * delta_quad * mean_tau_penalty)


def fill_window_support(window_pairs: list[list[dict[str, float]]]) -> None:
    normalized_by_window = []
    for pair_rows in window_pairs:
        if not pair_rows:
            normalized_by_window.append([])
            continue
        max_score = max(base_score(row, mean_tau_scale_ms=0.45) for row in pair_rows)
        rows = []
        for row in pair_rows:
            new_row = dict(row)
            new_row["support_base_score"] = base_score(row, mean_tau_scale_ms=0.45)
            new_row["support_norm"] = float(new_row["support_base_score"] / max(max_score, 1e-9))
            rows.append(new_row)
        normalized_by_window.append(rows)

    for idx, pair_rows in enumerate(normalized_by_window):
        neighbors = [rows for j, rows in enumerate(normalized_by_window) if j != idx and rows]
        for row in pair_rows:
            support_values = []
            for neighbor_rows in neighbors:
                best = 0.0
                for other in neighbor_rows:
                    dist2 = (row["tau_vl_ms"] - other["tau_vl_ms"]) ** 2 + (row["tau_vr_ms"] - other["tau_vr_ms"]) ** 2
                    proximity = math.exp(-dist2 / max(2.0 * SUPPORT_SIGMA_MS * SUPPORT_SIGMA_MS, 1e-9))
                    best = max(best, proximity * other["support_norm"])
                support_values.append(best)
            row["window_support"] = float(np.mean(support_values)) if support_values else 0.0

    for idx, pair_rows in enumerate(window_pairs):
        pair_rows.clear()
        pair_rows.extend(normalized_by_window[idx])


def score_pair_variant(pair_row: dict[str, float], variant: dict[str, Any]) -> float:
    score = base_score(pair_row, mean_tau_scale_ms=float(variant["mean_tau_scale_ms"]))
    balance_pow = float(variant["balance_pow"])
    psr_boost = float(variant["psr_boost"])
    support_boost = float(variant["support_boost"])
    rank_discount_pow = float(variant["rank_discount_pow"])

    if balance_pow > 0.0:
        score *= pair_row["balance"] ** balance_pow
    if psr_boost > 0.0:
        score *= 1.0 + psr_boost * max(pair_row["psr_geom"] - 1.0, 0.0)
    if support_boost > 0.0:
        score *= 1.0 + support_boost * pair_row.get("window_support", 0.0)
    if rank_discount_pow > 0.0:
        score /= float((pair_row["vl_rank"] * pair_row["vr_rank"]) ** rank_discount_pow)
    return float(score)


def oracle_match_rank(pair_rows: list[dict[str, float]], ref: dict[str, float], variant: dict[str, Any]) -> dict[str, float] | None:
    scored = []
    for row in pair_rows:
        item = dict(row)
        item["score"] = score_pair_variant(item, variant)
        scored.append(item)
    scored.sort(key=lambda x: x["score"], reverse=True)

    match = None
    match_rank = None
    for idx, row in enumerate(scored, start=1):
        if (
            abs(row["tau_vl_ms"] - ref["tau_vl_ms"]) <= ORACLE_RADIUS_MS
            and abs(row["tau_vr_ms"] - ref["tau_vr_ms"]) <= ORACLE_RADIUS_MS
        ):
            match = row
            match_rank = idx
            break
    if match is None:
        return None
    best_wrong = max((row["score"] for row in scored if row is not match), default=0.0)
    margin = float(match["score"] / max(best_wrong, 1e-9))
    return {
        "correct_pair_rank": int(match_rank),
        "correct_pair_margin": margin,
    }


def select_best_pair(pair_rows: list[dict[str, float]], variant: dict[str, Any]) -> dict[str, Any] | None:
    if not pair_rows:
        return None
    scored = []
    for row in pair_rows:
        item = dict(row)
        item["score"] = score_pair_variant(item, variant)
        scored.append(item)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"best": scored[0], "top_pairs": scored[:10], "num_pairs": len(scored)}


def evaluate_case_windows(case: base.CaseRef, full_signals: dict[str, np.ndarray], *, window_sec: float) -> list[dict[str, Any]]:
    ref = base.compute_reference(case)
    window_payloads = []
    raw_pair_rows = []

    for offset_sec in WINDOW_OFFSETS_SEC:
        sliced = lane1.slice_signals(full_signals, window_sec=window_sec, offset_sec=offset_sec)
        ldv = lane1.process_signal(sliced["ldv"], {"kind": "rms_gain", "window_ms": 80.0}, apply_diff=True)
        mic_l = lane1.process_signal(sliced["mic_l"], {"kind": "none"}, apply_diff=False)
        mic_r = lane1.process_signal(sliced["mic_r"], {"kind": "none"}, apply_diff=False)
        lag_vl, cc_vl = lane1.pair.gcc_curve(ldv, mic_l, FS, max_lag_ms=MAX_LAG_MS, bandpass=BANDPASS)
        lag_vr, cc_vr = lane1.pair.gcc_curve(ldv, mic_r, FS, max_lag_ms=MAX_LAG_MS, bandpass=BANDPASS)
        cand_vl = extract_candidates_with_features(lag_vl, cc_vl, lag_min_ms=LAG_MIN_MS, lag_max_ms=LAG_MAX_MS, top_k=TOP_K)
        cand_vr = extract_candidates_with_features(lag_vr, cc_vr, lag_min_ms=LAG_MIN_MS, lag_max_ms=LAG_MAX_MS, top_k=TOP_K)
        pair_rows = build_pair_rows(cand_vl, cand_vr)
        raw_pair_rows.append(pair_rows)
        window_payloads.append(
            {
                "case_id": case.case_id,
                "offset_sec": float(offset_sec),
                "reference": ref,
                "num_candidates_vl": len(cand_vl),
                "num_candidates_vr": len(cand_vr),
                "pair_rows": pair_rows,
            }
        )

    fill_window_support(raw_pair_rows)
    return window_payloads


def evaluate_variant(case_windows: list[dict[str, Any]], variant: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in case_windows:
        selected = select_best_pair(item["pair_rows"], variant)
        ref = item["reference"]
        row = {
            "case_id": item["case_id"],
            "variant": variant["name"],
            "offset_sec": item["offset_sec"],
            "reference": ref,
            "selected": selected,
            "num_candidates_vl": item["num_candidates_vl"],
            "num_candidates_vr": item["num_candidates_vr"],
        }
        if selected is not None:
            best = selected["best"]
            theta_v = float(np.degrees(np.arcsin(np.clip((best["delta_tau_ms"] / 1000.0) * base.C / 1.4, -1.0, 1.0))))
            best["theta_v_deg"] = theta_v
            best["delta_tau_abs_err_ms"] = abs(best["delta_tau_ms"] - ref["delta_tau_ms"])
            best["theta_v_abs_err_deg"] = abs(theta_v - ref["theta_v_deg"])
            best["physical_positive_lags"] = bool(best["tau_vl_ms"] > 0.0 and best["tau_vr_ms"] > 0.0)
            best["physical_small_delta"] = bool(abs(best["delta_tau_ms"]) <= DELTA_LIMIT_MS)
            best["physical_valid"] = bool(best["physical_positive_lags"] and best["physical_small_delta"])
        if abs(item["offset_sec"]) < 1e-9:
            row["oracle_match"] = oracle_match_rank(item["pair_rows"], ref, variant)
        else:
            row["oracle_match"] = None
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
                    "central_physical_count": 0,
                    "central_delta_tau_mae_ms": None,
                    "central_theta_v_mae_deg": None,
                    "central_max_delta_tau_abs_err_ms": None,
                    "central_max_theta_v_abs_err_deg": None,
                    "central_correct_pair_rank_mean": None,
                    "central_correct_pair_margin_mean": None,
                    "central_oracle_match_cases": 0,
                    "window_valid_rows": 0,
                    "window_delta_tau_mae_ms": None,
                    "window_stability_mean_std_ms": None,
                    "window_stability_max_std_ms": None,
                }
            )
            continue

        dt = np.array([it["delta_tau_abs_err_ms"] for it in central_valid], dtype=np.float64)
        th = np.array([it["theta_v_abs_err_deg"] for it in central_valid], dtype=np.float64)
        physical_count = sum(1 for it in central_valid if it["physical_valid"])

        oracle_rows = [it["oracle_match"] for it in central if it.get("oracle_match") is not None]
        rank_mean = float(np.mean([it["correct_pair_rank"] for it in oracle_rows])) if oracle_rows else None
        margin_mean = float(np.mean([it["correct_pair_margin"] for it in oracle_rows])) if oracle_rows else None

        window_valid = [it for it in items if it["selected"] is not None]
        window_dt = np.array([it["selected"]["best"]["delta_tau_abs_err_ms"] for it in window_valid], dtype=np.float64)
        case_std = []
        for case_id in sorted({it["case_id"] for it in window_valid}):
            vals = [it["selected"]["best"]["delta_tau_ms"] for it in window_valid if it["case_id"] == case_id]
            if len(vals) >= 2:
                case_std.append(float(np.std(np.asarray(vals, dtype=np.float64))))

        summary.append(
            {
                "variant": variant,
                "central_valid_cases": len(central_valid),
                "central_physical_count": physical_count,
                "central_delta_tau_mae_ms": float(np.mean(dt)),
                "central_theta_v_mae_deg": float(np.mean(th)),
                "central_max_delta_tau_abs_err_ms": float(np.max(dt)),
                "central_max_theta_v_abs_err_deg": float(np.max(th)),
                "central_correct_pair_rank_mean": rank_mean,
                "central_correct_pair_margin_mean": margin_mean,
                "central_oracle_match_cases": len(oracle_rows),
                "window_valid_rows": len(window_valid),
                "window_delta_tau_mae_ms": float(np.mean(window_dt)),
                "window_stability_mean_std_ms": float(np.mean(case_std)) if case_std else None,
                "window_stability_max_std_ms": float(np.max(case_std)) if case_std else None,
            }
        )

    summary.sort(
        key=lambda x: (
            1 if x["central_delta_tau_mae_ms"] is None else 0,
            float("inf") if x["central_delta_tau_mae_ms"] is None else x["central_delta_tau_mae_ms"],
            float("inf") if x["window_stability_mean_std_ms"] is None else x["window_stability_mean_std_ms"],
        )
    )
    return {"variants": summary}


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 3 Blind Proxy Scoring Sweep",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        f"- Window sec: `{payload['window_sec']}`",
        f"- Fixed front-end: `diff_len80_ldvonly_bp700_1800`",
        "",
        "## Summary",
        "",
        "| variant | central_valid | physical | central_dt_mae_ms | central_theta_mae_deg | central_max_dt_ms | correct_rank_mean | correct_margin_mean | window_dt_mae_ms | window_std_mean_ms | window_std_max_ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["variants"]:
        def fmt(name: str) -> str:
            value = row.get(name)
            if value is None:
                return "NA"
            if isinstance(value, int):
                return str(value)
            return f"{float(value):.3f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    str(row["central_valid_cases"]),
                    str(row["central_physical_count"]),
                    fmt("central_delta_tau_mae_ms"),
                    fmt("central_theta_v_mae_deg"),
                    fmt("central_max_delta_tau_abs_err_ms"),
                    fmt("central_correct_pair_rank_mean"),
                    fmt("central_correct_pair_margin_mean"),
                    fmt("window_delta_tau_mae_ms"),
                    fmt("window_stability_mean_std_ms"),
                    fmt("window_stability_max_std_ms"),
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 3 blind proxy scoring sweep for 0223.")
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
        Path(__file__).resolve().parent.parent / "results" / f"round3_blind_proxy_0223_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    full_signals = {case.case_id: lane1.load_full_case_signals(case, args.data_root) for case in base.CASES}
    case_windows = {case.case_id: evaluate_case_windows(case, full_signals[case.case_id], window_sec=args.window_sec) for case in base.CASES}

    rows = []
    for variant in VARIANTS:
        for case in base.CASES:
            rows.extend(evaluate_variant(case_windows[case.case_id], variant))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "window_sec": float(args.window_sec),
        "window_offsets_sec": list(WINDOW_OFFSETS_SEC),
        "bandpass": list(BANDPASS),
        "variants": VARIANTS,
        "results": rows,
        "summary": summarize(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir / "report.md", payload)
    print(f"[ok] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
