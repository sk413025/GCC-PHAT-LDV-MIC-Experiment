#!/usr/bin/env python3
"""
Spectral filter family sweep for 0223 LDV-MIC delta-tau recovery.

This script fixes the currently best peak-pair rule and compares spectral
families: alternative sub-bands plus flattening or whitening style variants.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import filter_sweep_0223_delta_tau as base
import peak_pair_sweep_0223_delta_tau as pair


def zscore(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    std = float(np.std(x))
    if std < 1e-9:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def spectral_whiten(signal: np.ndarray, smooth_bins: int = 33) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return x
    x = x - np.mean(x)
    n_fft = 1 << int(np.ceil(np.log2(max(8, x.size))))
    X = np.fft.rfft(x, n=n_fft)
    mag = np.abs(X)
    kernel = np.ones(smooth_bins, dtype=np.float64)
    kernel /= np.sum(kernel)
    smooth = np.convolve(mag, kernel, mode="same")
    Y = X / (smooth + 1e-8)
    y = np.fft.irfft(Y, n=n_fft)[: x.size]
    return y


def apply_ops(signal: np.ndarray, ops: list[str]) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    for op in ops:
        if op == "diff":
            x = base.diff_signal(x)
        elif op == "flatten":
            x = base.spectral_flatten(x)
        elif op == "whiten":
            x = spectral_whiten(x)
        elif op == "zscore":
            x = zscore(x)
        else:
            raise ValueError(f"Unknown op: {op}")
    return x


VARIANTS: list[dict[str, Any]] = [
    {"name": "baseline_ldv_diff_bp500_2000", "bandpass": (500.0, 2000.0), "ldv_ops": ["diff"], "mic_ops": []},
    {"name": "ldv_diff_bp300_1500", "bandpass": (300.0, 1500.0), "ldv_ops": ["diff"], "mic_ops": []},
    {"name": "ldv_diff_bp400_1600", "bandpass": (400.0, 1600.0), "ldv_ops": ["diff"], "mic_ops": []},
    {"name": "ldv_diff_bp700_1800", "bandpass": (700.0, 1800.0), "ldv_ops": ["diff"], "mic_ops": []},
    {"name": "ldv_diff_bp800_2200", "bandpass": (800.0, 2200.0), "ldv_ops": ["diff"], "mic_ops": []},
    {"name": "ldv_diff_bp1000_2600", "bandpass": (1000.0, 2600.0), "ldv_ops": ["diff"], "mic_ops": []},
    {"name": "ldv_diff_bp1200_3000", "bandpass": (1200.0, 3000.0), "ldv_ops": ["diff"], "mic_ops": []},
    {"name": "ldv_diff_ldv_flatten_bp500_2000", "bandpass": (500.0, 2000.0), "ldv_ops": ["diff", "flatten"], "mic_ops": []},
    {"name": "ldv_diff_mic_flatten_bp500_2000", "bandpass": (500.0, 2000.0), "ldv_ops": ["diff"], "mic_ops": ["flatten"]},
    {"name": "ldv_diff_both_flatten_bp500_2000", "bandpass": (500.0, 2000.0), "ldv_ops": ["diff", "flatten"], "mic_ops": ["flatten"]},
    {"name": "ldv_diff_ldv_whiten_bp500_2000", "bandpass": (500.0, 2000.0), "ldv_ops": ["diff", "whiten"], "mic_ops": []},
    {"name": "ldv_diff_both_whiten_bp500_2000", "bandpass": (500.0, 2000.0), "ldv_ops": ["diff", "whiten"], "mic_ops": ["whiten"]},
]


def evaluate_variant(case: base.CaseRef, signals: dict[str, np.ndarray], variant: dict[str, Any]) -> dict[str, Any]:
    ref = base.compute_reference(case)
    ldv = apply_ops(signals["ldv"], list(variant["ldv_ops"]))
    mic_l = apply_ops(signals["mic_l"], list(variant["mic_ops"]))
    mic_r = apply_ops(signals["mic_r"], list(variant["mic_ops"]))

    lag_vl, cc_vl = pair.gcc_curve(ldv, mic_l, 48000, max_lag_ms=10.0, bandpass=variant["bandpass"])
    lag_vr, cc_vr = pair.gcc_curve(ldv, mic_r, 48000, max_lag_ms=10.0, bandpass=variant["bandpass"])
    cand_vl = pair.extract_candidates(lag_vl, cc_vl, lag_min_ms=4.4, lag_max_ms=6.5, top_k=8)
    cand_vr = pair.extract_candidates(lag_vr, cc_vr, lag_min_ms=4.4, lag_max_ms=6.5, top_k=8)
    selected = pair.select_best_pair(
        cand_vl,
        cand_vr,
        strategy="amp_product_mean_tau_delta_quad",
        delta_limit_ms=1.0,
        delta_scale_ms=0.6,
        mean_tau_center_ms=4.8,
        mean_tau_scale_ms=0.45,
    )
    row = {"case_id": case.case_id, "variant": variant["name"], "reference": ref, "selected": selected}
    if selected is None:
        return row
    best = selected["best"]
    theta_v = float(np.degrees(np.arcsin(np.clip((best["delta_tau_ms"] / 1000.0) * base.C / 1.4, -1.0, 1.0))))
    best["theta_v_deg"] = theta_v
    best["delta_tau_abs_err_ms"] = abs(best["delta_tau_ms"] - ref["delta_tau_ms"])
    best["theta_v_abs_err_deg"] = abs(theta_v - ref["theta_v_deg"])
    best["physical_positive_lags"] = bool(best["tau_vl_ms"] > 0.0 and best["tau_vr_ms"] > 0.0)
    best["physical_small_delta"] = bool(abs(best["delta_tau_ms"]) <= 1.0)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    summary = []
    for variant, items in grouped.items():
        valid = [it for it in items if it["selected"] is not None]
        if not valid:
            summary.append({"variant": variant, "valid_cases": 0, "physical_count": 0, "delta_tau_mae_ms": None, "theta_v_mae_deg": None})
            continue
        delta_err = np.array([it["selected"]["best"]["delta_tau_abs_err_ms"] for it in valid], dtype=np.float64)
        theta_err = np.array([it["selected"]["best"]["theta_v_abs_err_deg"] for it in valid], dtype=np.float64)
        physical_count = sum(
            1 for it in valid if it["selected"]["best"]["physical_positive_lags"] and it["selected"]["best"]["physical_small_delta"]
        )
        summary.append(
            {
                "variant": variant,
                "valid_cases": len(valid),
                "physical_count": physical_count,
                "delta_tau_mae_ms": float(np.mean(delta_err)),
                "theta_v_mae_deg": float(np.mean(theta_err)),
                "max_delta_tau_abs_err_ms": float(np.max(delta_err)),
                "max_theta_v_abs_err_deg": float(np.max(theta_err)),
            }
        )
    summary.sort(
        key=lambda x: (
            1 if x["delta_tau_mae_ms"] is None else 0,
            float("inf") if x["delta_tau_mae_ms"] is None else x["delta_tau_mae_ms"],
            float("inf") if x["theta_v_mae_deg"] is None else x["theta_v_mae_deg"],
        )
    )
    return {"variants": summary}


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# 0223 Spectral Filter Family Sweep",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        f"- Slice sec: `{payload['slice_sec']}`",
        "",
        "## Summary",
        "",
        "| variant | valid_cases | physical_count | delta_tau_mae_ms | theta_v_mae_deg | max_delta_tau_err_ms | max_theta_v_err_deg |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["variants"]:
        dt = "NA" if row["delta_tau_mae_ms"] is None else f"{row['delta_tau_mae_ms']:.3f}"
        th = "NA" if row["theta_v_mae_deg"] is None else f"{row['theta_v_mae_deg']:.3f}"
        max_dt = "NA" if row.get("max_delta_tau_abs_err_ms") is None else f"{row['max_delta_tau_abs_err_ms']:.3f}"
        max_th = "NA" if row.get("max_theta_v_abs_err_deg") is None else f"{row['max_theta_v_abs_err_deg']:.3f}"
        lines.append(f"| {row['variant']} | {row['valid_cases']} | {row['physical_count']} | {dt} | {th} | {max_dt} | {max_th} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep spectral preprocessing families for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--slice_sec", type=float, default=5.0)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (Path(__file__).resolve().parent.parent / "results" / f"filter_family_spectral_0223_{timestamp}")
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
        "variants": VARIANTS,
        "results": rows,
        "summary": summarize(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir / "report.md", payload)
    print(f"[ok] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
