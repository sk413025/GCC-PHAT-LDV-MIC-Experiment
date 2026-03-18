#!/usr/bin/env python3
"""
Oracle-guided filter search for 0223 LDV-MIC delta-tau recovery.

This round is intentionally answer-conditioned: it uses the known reference
taus to measure how well each filter family makes the correct lag peaks visible.
Shortlisted candidates are then evaluated again with the non-oracle pairing
rule to judge generalizability.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import filter_sweep_0223_delta_tau as base
import filter_family_time_domain_0223 as td
import filter_family_spectral_0223 as sp
import peak_pair_sweep_0223_delta_tau as pair


ORACLE_RADIUS_MS = 0.60
POS_LAG_MIN_MS = 4.0
POS_LAG_MAX_MS = 6.5
NON_ORACLE_LAG_MIN_MS = 4.4


VARIANTS: list[dict[str, Any]] = [
    {"name": "baseline_diff_bp700_1800", "bandpass": (700.0, 1800.0), "ldv_ops": ["diff"], "mic_ops": []},
    {"name": "diff_bp650_1750", "bandpass": (650.0, 1750.0), "ldv_ops": ["diff"], "mic_ops": []},
    {"name": "diff_bp750_1850", "bandpass": (750.0, 1850.0), "ldv_ops": ["diff"], "mic_ops": []},
    {"name": "diff_bp800_2000", "bandpass": (800.0, 2000.0), "ldv_ops": ["diff"], "mic_ops": []},
    {"name": "diff2_bp500_2000", "bandpass": (500.0, 2000.0), "ldv_ops": ["diff2"], "mic_ops": []},
    {"name": "diff2_bp650_1750", "bandpass": (650.0, 1750.0), "ldv_ops": ["diff2"], "mic_ops": []},
    {"name": "diff2_bp700_1800", "bandpass": (700.0, 1800.0), "ldv_ops": ["diff2"], "mic_ops": []},
    {"name": "diff2_rms_bp700_1800", "bandpass": (700.0, 1800.0), "ldv_ops": ["diff2", "rms_norm"], "mic_ops": []},
    {"name": "diff_bp700_1800_both_flatten", "bandpass": (700.0, 1800.0), "ldv_ops": ["diff", "flatten"], "mic_ops": ["flatten"]},
    {"name": "diff2_bp700_1800_both_flatten", "bandpass": (700.0, 1800.0), "ldv_ops": ["diff2", "flatten"], "mic_ops": ["flatten"]},
    {"name": "diff_bp700_1800_mic_flatten", "bandpass": (700.0, 1800.0), "ldv_ops": ["diff"], "mic_ops": ["flatten"]},
]


def apply_variant(signal: np.ndarray, ops: list[str], *, is_ldv: bool) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    for op in ops:
        if op in {"zscore", "preemphasis", "diff", "diff2", "signed_sqrt", "envelope", "rms_norm"}:
            x = td.apply_chain(x, [op])
        elif op == "flatten":
            x = base.spectral_flatten(x)
        elif op == "whiten":
            x = sp.spectral_whiten(x)
        else:
            raise ValueError(f"Unknown op: {op}")
    return x


def candidate_list(
    ldv: np.ndarray,
    mic: np.ndarray,
    *,
    bandpass: tuple[float, float],
    lag_min_ms: float,
    lag_max_ms: float,
    top_k: int = 12,
) -> list[dict[str, float]]:
    lag_ms, cc = pair.gcc_curve(ldv, mic, 48000, max_lag_ms=10.0, bandpass=bandpass)
    return pair.extract_candidates(lag_ms, cc, lag_min_ms=lag_min_ms, lag_max_ms=lag_max_ms, top_k=top_k)


def oracle_pick(candidates: list[dict[str, float]], ref_tau_ms: float, radius_ms: float) -> dict[str, Any] | None:
    if not candidates:
        return None
    filtered = []
    max_amp = max(c["amp"] for c in candidates)
    for rank, cand in enumerate(sorted(candidates, key=lambda x: (x["amp"], x["prom"]), reverse=True), start=1):
        tau_err = abs(cand["tau_ms"] - ref_tau_ms)
        if tau_err <= radius_ms:
            filtered.append(
                {
                    "rank": rank,
                    "tau_ms": float(cand["tau_ms"]),
                    "amp": float(cand["amp"]),
                    "prom": float(cand["prom"]),
                    "tau_abs_err_ms": float(tau_err),
                    "amp_ratio_to_top": float(cand["amp"] / max(max_amp, 1e-9)),
                }
            )
    if not filtered:
        return None
    filtered.sort(key=lambda x: (x["tau_abs_err_ms"], x["rank"]))
    return filtered[0]


def non_oracle_pair(
    cand_vl: list[dict[str, float]],
    cand_vr: list[dict[str, float]],
) -> dict[str, Any] | None:
    selected = pair.select_best_pair(
        cand_vl,
        cand_vr,
        strategy="amp_product_mean_tau_delta_quad",
        delta_limit_ms=1.0,
        delta_scale_ms=0.6,
        mean_tau_center_ms=4.8,
        mean_tau_scale_ms=0.45,
    )
    if selected is None:
        return None
    return selected["best"]


def evaluate_variant(case: base.CaseRef, signals: dict[str, np.ndarray], variant: dict[str, Any]) -> dict[str, Any]:
    ref = base.compute_reference(case)
    ldv = apply_variant(signals["ldv"], list(variant["ldv_ops"]), is_ldv=True)
    mic_l = apply_variant(signals["mic_l"], list(variant["mic_ops"]), is_ldv=False)
    mic_r = apply_variant(signals["mic_r"], list(variant["mic_ops"]), is_ldv=False)

    cand_vl_oracle = candidate_list(
        ldv, mic_l, bandpass=variant["bandpass"], lag_min_ms=POS_LAG_MIN_MS, lag_max_ms=POS_LAG_MAX_MS
    )
    cand_vr_oracle = candidate_list(
        ldv, mic_r, bandpass=variant["bandpass"], lag_min_ms=POS_LAG_MIN_MS, lag_max_ms=POS_LAG_MAX_MS
    )
    cand_vl_non = candidate_list(
        ldv, mic_l, bandpass=variant["bandpass"], lag_min_ms=NON_ORACLE_LAG_MIN_MS, lag_max_ms=POS_LAG_MAX_MS
    )
    cand_vr_non = candidate_list(
        ldv, mic_r, bandpass=variant["bandpass"], lag_min_ms=NON_ORACLE_LAG_MIN_MS, lag_max_ms=POS_LAG_MAX_MS
    )

    oracle_vl = oracle_pick(cand_vl_oracle, ref["tau_vl_ms"], ORACLE_RADIUS_MS)
    oracle_vr = oracle_pick(cand_vr_oracle, ref["tau_vr_ms"], ORACLE_RADIUS_MS)

    oracle_pair = None
    if oracle_vl is not None and oracle_vr is not None:
        oracle_delta = oracle_vr["tau_ms"] - oracle_vl["tau_ms"]
        oracle_theta = float(np.degrees(np.arcsin(np.clip((oracle_delta / 1000.0) * base.C / 1.4, -1.0, 1.0))))
        oracle_pair = {
            "tau_vl_ms": float(oracle_vl["tau_ms"]),
            "tau_vr_ms": float(oracle_vr["tau_ms"]),
            "delta_tau_ms": float(oracle_delta),
            "theta_v_deg": float(oracle_theta),
            "delta_tau_abs_err_ms": abs(oracle_delta - ref["delta_tau_ms"]),
            "theta_v_abs_err_deg": abs(oracle_theta - ref["theta_v_deg"]),
            "rank_sum": int(oracle_vl["rank"] + oracle_vr["rank"]),
            "tau_abs_err_sum_ms": float(oracle_vl["tau_abs_err_ms"] + oracle_vr["tau_abs_err_ms"]),
            "amp_ratio_mean": float(0.5 * (oracle_vl["amp_ratio_to_top"] + oracle_vr["amp_ratio_to_top"])),
        }

    non_oracle = non_oracle_pair(cand_vl_non, cand_vr_non)
    non_oracle_pair_row = None
    if non_oracle is not None:
        non_delta = non_oracle["delta_tau_ms"]
        non_theta = float(np.degrees(np.arcsin(np.clip((non_delta / 1000.0) * base.C / 1.4, -1.0, 1.0))))
        non_oracle_pair_row = {
            "tau_vl_ms": float(non_oracle["tau_vl_ms"]),
            "tau_vr_ms": float(non_oracle["tau_vr_ms"]),
            "delta_tau_ms": float(non_delta),
            "theta_v_deg": float(non_theta),
            "delta_tau_abs_err_ms": abs(non_delta - ref["delta_tau_ms"]),
            "theta_v_abs_err_deg": abs(non_theta - ref["theta_v_deg"]),
            "physical_positive_lags": bool(non_oracle["tau_vl_ms"] > 0.0 and non_oracle["tau_vr_ms"] > 0.0),
            "physical_small_delta": bool(abs(non_delta) <= 1.0),
        }

    return {
        "case_id": case.case_id,
        "variant": variant["name"],
        "reference": ref,
        "oracle_vl": oracle_vl,
        "oracle_vr": oracle_vr,
        "oracle_pair": oracle_pair,
        "non_oracle_pair": non_oracle_pair_row,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)

    summary = []
    for variant, items in grouped.items():
        oracle_valid = [it for it in items if it["oracle_pair"] is not None]
        non_valid = [it for it in items if it["non_oracle_pair"] is not None]
        oracle_hit_cases = len(oracle_valid)
        oracle_rank_sum_mean = None
        oracle_tau_abs_err_mean = None
        oracle_amp_ratio_mean = None
        if oracle_valid:
            oracle_rank_sum_mean = float(np.mean([it["oracle_pair"]["rank_sum"] for it in oracle_valid]))
            oracle_tau_abs_err_mean = float(np.mean([it["oracle_pair"]["tau_abs_err_sum_ms"] for it in oracle_valid]))
            oracle_amp_ratio_mean = float(np.mean([it["oracle_pair"]["amp_ratio_mean"] for it in oracle_valid]))

        if not non_valid:
            summary.append(
                {
                    "variant": variant,
                    "oracle_hit_cases": oracle_hit_cases,
                    "oracle_rank_sum_mean": oracle_rank_sum_mean,
                    "oracle_tau_abs_err_sum_mean_ms": oracle_tau_abs_err_mean,
                    "oracle_amp_ratio_mean": oracle_amp_ratio_mean,
                    "non_oracle_valid_cases": 0,
                    "non_oracle_physical_count": 0,
                    "non_oracle_delta_tau_mae_ms": None,
                    "non_oracle_theta_v_mae_deg": None,
                }
            )
            continue

        non_dt = np.array([it["non_oracle_pair"]["delta_tau_abs_err_ms"] for it in non_valid], dtype=np.float64)
        non_th = np.array([it["non_oracle_pair"]["theta_v_abs_err_deg"] for it in non_valid], dtype=np.float64)
        physical_count = sum(
            1
            for it in non_valid
            if it["non_oracle_pair"]["physical_positive_lags"] and it["non_oracle_pair"]["physical_small_delta"]
        )
        summary.append(
            {
                "variant": variant,
                "oracle_hit_cases": oracle_hit_cases,
                "oracle_rank_sum_mean": oracle_rank_sum_mean,
                "oracle_tau_abs_err_sum_mean_ms": oracle_tau_abs_err_mean,
                "oracle_amp_ratio_mean": oracle_amp_ratio_mean,
                "non_oracle_valid_cases": len(non_valid),
                "non_oracle_physical_count": physical_count,
                "non_oracle_delta_tau_mae_ms": float(np.mean(non_dt)),
                "non_oracle_theta_v_mae_deg": float(np.mean(non_th)),
                "non_oracle_max_delta_tau_abs_err_ms": float(np.max(non_dt)),
                "non_oracle_max_theta_v_abs_err_deg": float(np.max(non_th)),
            }
        )

    summary.sort(
        key=lambda x: (
            -(x["oracle_hit_cases"]),
            float("inf") if x["oracle_rank_sum_mean"] is None else x["oracle_rank_sum_mean"],
            float("inf") if x["oracle_tau_abs_err_sum_mean_ms"] is None else x["oracle_tau_abs_err_sum_mean_ms"],
            float("inf") if x["non_oracle_delta_tau_mae_ms"] is None else x["non_oracle_delta_tau_mae_ms"],
        )
    )
    return {"variants": summary}


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# 0223 Oracle-Guided Filter Search",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        f"- Slice sec: `{payload['slice_sec']}`",
        f"- Oracle radius: `{payload['oracle_radius_ms']}` ms",
        "",
        "## Summary",
        "",
        "| variant | oracle_hit_cases | oracle_rank_sum_mean | oracle_tau_abs_err_sum_mean_ms | oracle_amp_ratio_mean | non_oracle_valid_cases | non_oracle_physical_count | non_oracle_delta_tau_mae_ms | non_oracle_theta_v_mae_deg |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["variants"]:
        oracle_rank_txt = "NA" if row["oracle_rank_sum_mean"] is None else f"{row['oracle_rank_sum_mean']:.3f}"
        oracle_tau_txt = (
            "NA" if row["oracle_tau_abs_err_sum_mean_ms"] is None else f"{row['oracle_tau_abs_err_sum_mean_ms']:.3f}"
        )
        oracle_amp_txt = "NA" if row["oracle_amp_ratio_mean"] is None else f"{row['oracle_amp_ratio_mean']:.3f}"
        non_dt_txt = "NA" if row["non_oracle_delta_tau_mae_ms"] is None else f"{row['non_oracle_delta_tau_mae_ms']:.3f}"
        non_th_txt = "NA" if row["non_oracle_theta_v_mae_deg"] is None else f"{row['non_oracle_theta_v_mae_deg']:.3f}"
        lines.append(
            f"| {row['variant']} | {row['oracle_hit_cases']} | "
            f"{oracle_rank_txt} | "
            f"{oracle_tau_txt} | "
            f"{oracle_amp_txt} | "
            f"{row['non_oracle_valid_cases']} | {row['non_oracle_physical_count']} | "
            f"{non_dt_txt} | "
            f"{non_th_txt} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle-guided filter search for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--slice_sec", type=float, default=5.0)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (Path(__file__).resolve().parent.parent / "results" / f"oracle_guided_filter_search_0223_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    signals_cache = {case.case_id: base.load_case_signals(case, args.data_root, args.slice_sec, 48000) for case in base.CASES}
    rows = []
    for variant in VARIANTS:
        for case in base.CASES:
            rows.append(evaluate_variant(case, signals_cache[case.case_id], variant))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "slice_sec": float(args.slice_sec),
        "oracle_radius_ms": ORACLE_RADIUS_MS,
        "variants": VARIANTS,
        "results": rows,
        "summary": summarize(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir / "report.md", payload)
    print(f"[ok] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
