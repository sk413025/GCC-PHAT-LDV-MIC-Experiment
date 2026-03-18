#!/usr/bin/env python3
"""
Round 3 lane 1: diff1 plus local normalization search for 0223 delta-tau.

This round starts from the accepted blind baseline `ldv_diff_bp700_1800` and
tests whether local normalization can improve correct-pair visibility and blind
pair selection without introducing the fragility seen in oracle-only winners.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import median_filter

import filter_sweep_0223_delta_tau as base
import peak_pair_sweep_0223_delta_tau as pair
import stage4_doa_ldv_vs_mic_comparison as stage4


FS = 48000
ORACLE_RADIUS_MS = 0.60
ORACLE_LAG_MIN_MS = 4.0
NON_ORACLE_LAG_MIN_MS = 4.4
LAG_MAX_MS = 6.5
MAX_LAG_MS = 10.0
TOP_K = 12
WINDOW_OFFSETS_SEC = (-1.0, -0.5, 0.0, 0.5, 1.0)
PAIR_STRATEGY = "amp_product_mean_tau_delta_quad"
DELTA_LIMIT_MS = 1.0
DELTA_SCALE_MS = 0.6
MEAN_TAU_CENTER_MS = 4.8
MEAN_TAU_SCALE_MS = 0.45


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "baseline_diff_bp700_1800",
        "bandpass": (700.0, 1800.0),
        "ldv_proc": {"kind": "baseline"},
        "mic_proc": {"kind": "none"},
    },
    {
        "name": "diff_len20_ldvonly_bp700_1800",
        "bandpass": (700.0, 1800.0),
        "ldv_proc": {"kind": "rms_gain", "window_ms": 20.0},
        "mic_proc": {"kind": "none"},
    },
    {
        "name": "diff_len40_ldvonly_bp700_1800",
        "bandpass": (700.0, 1800.0),
        "ldv_proc": {"kind": "rms_gain", "window_ms": 40.0},
        "mic_proc": {"kind": "none"},
    },
    {
        "name": "diff_len80_ldvonly_bp700_1800",
        "bandpass": (700.0, 1800.0),
        "ldv_proc": {"kind": "rms_gain", "window_ms": 80.0},
        "mic_proc": {"kind": "none"},
    },
    {
        "name": "diff_len40_both_bp700_1800",
        "bandpass": (700.0, 1800.0),
        "ldv_proc": {"kind": "rms_gain", "window_ms": 40.0},
        "mic_proc": {"kind": "rms_gain", "window_ms": 40.0},
    },
    {
        "name": "diff_len40_ldvonly_clip0p5_2p0_bp700_1800",
        "bandpass": (700.0, 1800.0),
        "ldv_proc": {"kind": "rms_gain", "window_ms": 40.0, "clip_gain": (0.5, 2.0)},
        "mic_proc": {"kind": "none"},
    },
    {
        "name": "diff_len40_ldvonly_sqrtgain_bp700_1800",
        "bandpass": (700.0, 1800.0),
        "ldv_proc": {"kind": "rms_gain", "window_ms": 40.0, "sqrt_gain": True},
        "mic_proc": {"kind": "none"},
    },
    {
        "name": "diff_len40_ldvonly_robustmedian_bp700_1800",
        "bandpass": (700.0, 1800.0),
        "ldv_proc": {"kind": "robust_median", "window_ms": 40.0, "clip_gain": (0.5, 2.5)},
        "mic_proc": {"kind": "none"},
    },
]


def smooth_same(x: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size, dtype=np.float64) / float(kernel_size)
    return np.convolve(x, kernel, mode="same")


def window_to_samples(window_ms: float) -> int:
    return max(3, int(round(window_ms * FS / 1000.0)))


def running_rms(signal: np.ndarray, window_ms: float) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    win = window_to_samples(window_ms)
    power = smooth_same(np.square(x), win)
    return np.sqrt(np.maximum(power, 1e-12))


def running_abs_median(signal: np.ndarray, window_ms: float) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    win = window_to_samples(window_ms)
    if win % 2 == 0:
        win += 1
    abs_x = np.abs(x)
    med = median_filter(abs_x, size=win, mode="nearest")
    return np.maximum(med, 1e-6)


def apply_gain_normalization(
    signal: np.ndarray,
    scale: np.ndarray,
    *,
    clip_gain: tuple[float, float] | None = None,
    sqrt_gain: bool = False,
) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    target = float(np.median(scale))
    gain = target / np.maximum(scale, 1e-9)
    if sqrt_gain:
        gain = np.sqrt(np.maximum(gain, 1e-9))
    if clip_gain is not None:
        gain = np.clip(gain, clip_gain[0], clip_gain[1])
    y = x * gain
    return y - np.mean(y)


def process_signal(signal: np.ndarray, proc: dict[str, Any], *, apply_diff: bool) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if apply_diff:
        x = base.diff_signal(x)
    kind = proc["kind"]
    if kind == "none":
        return x - np.mean(x)
    if kind == "baseline":
        return x - np.mean(x)
    if kind == "rms_gain":
        scale = running_rms(x, float(proc["window_ms"]))
        return apply_gain_normalization(
            x,
            scale,
            clip_gain=proc.get("clip_gain"),
            sqrt_gain=bool(proc.get("sqrt_gain", False)),
        )
    if kind == "robust_median":
        scale = running_abs_median(x, float(proc["window_ms"]))
        return apply_gain_normalization(
            x,
            scale,
            clip_gain=proc.get("clip_gain"),
            sqrt_gain=bool(proc.get("sqrt_gain", False)),
        )
    raise ValueError(f"Unknown proc kind: {kind}")


def load_full_case_signals(case: base.CaseRef, data_root: Path) -> dict[str, np.ndarray]:
    case_dir = data_root / case.rel_dir
    sr_ldv, ldv = stage4.load_wav(str(case_dir / case.ldv_name))
    sr_l, mic_l = stage4.load_wav(str(case_dir / case.mic_left_name))
    sr_r, mic_r = stage4.load_wav(str(case_dir / case.mic_right_name))
    if sr_ldv != FS or sr_l != FS or sr_r != FS:
        raise ValueError(f"Sample-rate mismatch for {case.case_id}: {sr_ldv}, {sr_l}, {sr_r}")
    min_len = min(len(ldv), len(mic_l), len(mic_r))
    return {
        "ldv": np.asarray(ldv[:min_len], dtype=np.float64),
        "mic_l": np.asarray(mic_l[:min_len], dtype=np.float64),
        "mic_r": np.asarray(mic_r[:min_len], dtype=np.float64),
    }


def slice_signals(signals: dict[str, np.ndarray], *, window_sec: float, offset_sec: float) -> dict[str, np.ndarray]:
    slice_samples = int(round(window_sec * FS))
    min_len = min(len(signals["ldv"]), len(signals["mic_l"]), len(signals["mic_r"]))
    center_sample = min_len // 2 + int(round(offset_sec * FS))
    start, end = stage4.extract_centered_slice(
        [signals["ldv"], signals["mic_l"], signals["mic_r"]],
        center_sample=center_sample,
        slice_samples=slice_samples,
    )
    return {
        "ldv": signals["ldv"][start:end],
        "mic_l": signals["mic_l"][start:end],
        "mic_r": signals["mic_r"][start:end],
        "start_sample": start,
        "end_sample": end,
    }


def guided_psr(sig1: np.ndarray, sig2: np.ndarray, *, bandpass: tuple[float, float], tau_ms: float) -> float:
    _, psr = stage4.gcc_phat_full_analysis(
        sig1,
        sig2,
        FS,
        max_tau=MAX_LAG_MS / 1000.0,
        bandpass=bandpass,
        guided_tau=tau_ms / 1000.0,
        guided_radius=ORACLE_RADIUS_MS / 1000.0,
    )
    return float(psr)


def candidate_list(
    ldv: np.ndarray,
    mic: np.ndarray,
    *,
    bandpass: tuple[float, float],
    lag_min_ms: float,
    top_k: int = TOP_K,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    lag_ms, cc = pair.gcc_curve(ldv, mic, FS, max_lag_ms=MAX_LAG_MS, bandpass=bandpass)
    candidates = pair.extract_candidates(lag_ms, cc, lag_min_ms=lag_min_ms, lag_max_ms=LAG_MAX_MS, top_k=top_k)
    return lag_ms, cc, candidates


def oracle_pick(candidates: list[dict[str, float]], ref_tau_ms: float) -> dict[str, Any] | None:
    if not candidates:
        return None
    max_amp = max(c["amp"] for c in candidates)
    filtered = []
    for rank, cand in enumerate(sorted(candidates, key=lambda x: (x["amp"], x["prom"]), reverse=True), start=1):
        tau_err = abs(cand["tau_ms"] - ref_tau_ms)
        if tau_err <= ORACLE_RADIUS_MS:
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


def build_pairs(cand_vl: list[dict[str, float]], cand_vr: list[dict[str, float]]) -> list[dict[str, float]]:
    pairs = []
    for vl in cand_vl:
        for vr in cand_vr:
            dt = vr["tau_ms"] - vl["tau_ms"]
            if abs(dt) > DELTA_LIMIT_MS:
                continue
            score = pair.score_pair(
                vl,
                vr,
                PAIR_STRATEGY,
                DELTA_SCALE_MS,
                MEAN_TAU_CENTER_MS,
                MEAN_TAU_SCALE_MS,
            )
            pairs.append(
                {
                    "tau_vl_ms": float(vl["tau_ms"]),
                    "tau_vr_ms": float(vr["tau_ms"]),
                    "delta_tau_ms": float(dt),
                    "score": float(score),
                }
            )
    pairs.sort(key=lambda x: x["score"], reverse=True)
    return pairs


def oracle_pair_metrics(
    ref: dict[str, float],
    bandpass: tuple[float, float],
    ldv: np.ndarray,
    mic_l: np.ndarray,
    mic_r: np.ndarray,
    cand_vl: list[dict[str, float]],
    cand_vr: list[dict[str, float]],
) -> dict[str, Any] | None:
    oracle_vl = oracle_pick(cand_vl, ref["tau_vl_ms"])
    oracle_vr = oracle_pick(cand_vr, ref["tau_vr_ms"])
    if oracle_vl is None or oracle_vr is None:
        return None

    pairs = build_pairs(cand_vl, cand_vr)
    matched_pair = None
    matched_rank = None
    for idx, pair_row in enumerate(pairs, start=1):
        if (
            abs(pair_row["tau_vl_ms"] - oracle_vl["tau_ms"]) < 1e-9
            and abs(pair_row["tau_vr_ms"] - oracle_vr["tau_ms"]) < 1e-9
        ):
            matched_pair = pair_row
            matched_rank = idx
            break
    if matched_pair is None:
        return None

    wrong_scores = [p["score"] for p in pairs if p is not matched_pair]
    best_wrong_score = max(wrong_scores) if wrong_scores else 0.0
    margin_ratio = (
        float(matched_pair["score"] / best_wrong_score)
        if best_wrong_score > 1e-12
        else float("inf")
    )
    theta_v = float(np.degrees(np.arcsin(np.clip((matched_pair["delta_tau_ms"] / 1000.0) * base.C / 1.4, -1.0, 1.0))))
    return {
        "tau_vl_ms": float(oracle_vl["tau_ms"]),
        "tau_vr_ms": float(oracle_vr["tau_ms"]),
        "delta_tau_ms": float(matched_pair["delta_tau_ms"]),
        "theta_v_deg": float(theta_v),
        "delta_tau_abs_err_ms": abs(matched_pair["delta_tau_ms"] - ref["delta_tau_ms"]),
        "theta_v_abs_err_deg": abs(theta_v - ref["theta_v_deg"]),
        "correct_vl_rank": int(oracle_vl["rank"]),
        "correct_vr_rank": int(oracle_vr["rank"]),
        "correct_pair_rank": int(matched_rank),
        "correct_pair_margin": float(margin_ratio),
        "correct_vl_amp_ratio": float(oracle_vl["amp_ratio_to_top"]),
        "correct_vr_amp_ratio": float(oracle_vr["amp_ratio_to_top"]),
        "correct_vl_psr": guided_psr(ldv, mic_l, bandpass=bandpass, tau_ms=ref["tau_vl_ms"]),
        "correct_vr_psr": guided_psr(ldv, mic_r, bandpass=bandpass, tau_ms=ref["tau_vr_ms"]),
    }


def blind_pair_metrics(cand_vl: list[dict[str, float]], cand_vr: list[dict[str, float]], ref: dict[str, float]) -> dict[str, Any] | None:
    selected = pair.select_best_pair(
        cand_vl,
        cand_vr,
        strategy=PAIR_STRATEGY,
        delta_limit_ms=DELTA_LIMIT_MS,
        delta_scale_ms=DELTA_SCALE_MS,
        mean_tau_center_ms=MEAN_TAU_CENTER_MS,
        mean_tau_scale_ms=MEAN_TAU_SCALE_MS,
    )
    if selected is None:
        return None
    best_pair = dict(selected["best"])
    theta_v = float(np.degrees(np.arcsin(np.clip((best_pair["delta_tau_ms"] / 1000.0) * base.C / 1.4, -1.0, 1.0))))
    best_pair["theta_v_deg"] = theta_v
    best_pair["delta_tau_abs_err_ms"] = abs(best_pair["delta_tau_ms"] - ref["delta_tau_ms"])
    best_pair["theta_v_abs_err_deg"] = abs(theta_v - ref["theta_v_deg"])
    best_pair["physical_positive_lags"] = bool(best_pair["tau_vl_ms"] > 0.0 and best_pair["tau_vr_ms"] > 0.0)
    best_pair["physical_small_delta"] = bool(abs(best_pair["delta_tau_ms"]) <= DELTA_LIMIT_MS)
    best_pair["physical_valid"] = bool(best_pair["physical_positive_lags"] and best_pair["physical_small_delta"])
    return {"best": best_pair, "top_pairs": selected["top_pairs"], "num_pairs": selected["num_pairs"]}


def evaluate_window(case: base.CaseRef, signals: dict[str, np.ndarray], variant: dict[str, Any], *, window_sec: float, offset_sec: float) -> dict[str, Any]:
    ref = base.compute_reference(case)
    sliced = slice_signals(signals, window_sec=window_sec, offset_sec=offset_sec)
    ldv = process_signal(sliced["ldv"], variant["ldv_proc"], apply_diff=True)
    mic_l = process_signal(sliced["mic_l"], variant["mic_proc"], apply_diff=False)
    mic_r = process_signal(sliced["mic_r"], variant["mic_proc"], apply_diff=False)

    _, _, cand_vl_oracle = candidate_list(ldv, mic_l, bandpass=variant["bandpass"], lag_min_ms=ORACLE_LAG_MIN_MS)
    _, _, cand_vr_oracle = candidate_list(ldv, mic_r, bandpass=variant["bandpass"], lag_min_ms=ORACLE_LAG_MIN_MS)
    _, _, cand_vl_non = candidate_list(ldv, mic_l, bandpass=variant["bandpass"], lag_min_ms=NON_ORACLE_LAG_MIN_MS)
    _, _, cand_vr_non = candidate_list(ldv, mic_r, bandpass=variant["bandpass"], lag_min_ms=NON_ORACLE_LAG_MIN_MS)

    return {
        "case_id": case.case_id,
        "variant": variant["name"],
        "window_sec": float(window_sec),
        "offset_sec": float(offset_sec),
        "start_sample": int(sliced["start_sample"]),
        "end_sample": int(sliced["end_sample"]),
        "reference": ref,
        "oracle_pair": oracle_pair_metrics(ref, variant["bandpass"], ldv, mic_l, mic_r, cand_vl_oracle, cand_vr_oracle),
        "blind_pair": blind_pair_metrics(cand_vl_non, cand_vr_non, ref),
        "num_candidates_vl_oracle": len(cand_vl_oracle),
        "num_candidates_vr_oracle": len(cand_vr_oracle),
        "num_candidates_vl_non": len(cand_vl_non),
        "num_candidates_vr_non": len(cand_vr_non),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)

    summary_rows = []
    for variant, items in grouped.items():
        central = [it for it in items if abs(it["offset_sec"]) < 1e-9]
        central_blind = [it["blind_pair"]["best"] for it in central if it["blind_pair"] is not None]
        central_oracle = [it["oracle_pair"] for it in central if it["oracle_pair"] is not None]
        if not central_blind:
            summary_rows.append(
                {
                    "variant": variant,
                    "central_valid_cases": 0,
                    "central_physical_count": 0,
                    "central_delta_tau_mae_ms": None,
                    "central_theta_v_mae_deg": None,
                    "central_max_delta_tau_abs_err_ms": None,
                    "central_max_theta_v_abs_err_deg": None,
                    "oracle_hit_cases": len(central_oracle),
                    "oracle_correct_pair_rank_mean": None,
                    "oracle_correct_pair_margin_mean": None,
                    "oracle_correct_psr_mean": None,
                    "window_valid_rows": 0,
                    "window_delta_tau_mae_ms": None,
                    "window_stability_mean_std_ms": None,
                    "window_stability_max_std_ms": None,
                }
            )
            continue

        central_delta = np.array([it["delta_tau_abs_err_ms"] for it in central_blind], dtype=np.float64)
        central_theta = np.array([it["theta_v_abs_err_deg"] for it in central_blind], dtype=np.float64)
        central_physical = sum(1 for it in central_blind if it["physical_valid"])

        blind_rows = [it for it in items if it["blind_pair"] is not None]
        blind_delta = np.array([it["blind_pair"]["best"]["delta_tau_abs_err_ms"] for it in blind_rows], dtype=np.float64)

        case_std = []
        for case_id in {it["case_id"] for it in blind_rows}:
            case_vals = [
                it["blind_pair"]["best"]["delta_tau_ms"]
                for it in blind_rows
                if it["case_id"] == case_id
            ]
            if len(case_vals) >= 2:
                case_std.append(float(np.std(np.asarray(case_vals, dtype=np.float64))))

        oracle_rank_mean = None
        oracle_margin_mean = None
        oracle_psr_mean = None
        if central_oracle:
            oracle_rank_mean = float(np.mean([it["correct_pair_rank"] for it in central_oracle]))
            oracle_margin_mean = float(np.mean([it["correct_pair_margin"] for it in central_oracle]))
            oracle_psr_mean = float(
                np.mean(
                    [
                        0.5 * (it["correct_vl_psr"] + it["correct_vr_psr"])
                        for it in central_oracle
                    ]
                )
            )

        summary_rows.append(
            {
                "variant": variant,
                "central_valid_cases": len(central_blind),
                "central_physical_count": central_physical,
                "central_delta_tau_mae_ms": float(np.mean(central_delta)),
                "central_theta_v_mae_deg": float(np.mean(central_theta)),
                "central_max_delta_tau_abs_err_ms": float(np.max(central_delta)),
                "central_max_theta_v_abs_err_deg": float(np.max(central_theta)),
                "oracle_hit_cases": len(central_oracle),
                "oracle_correct_pair_rank_mean": oracle_rank_mean,
                "oracle_correct_pair_margin_mean": oracle_margin_mean,
                "oracle_correct_psr_mean": oracle_psr_mean,
                "window_valid_rows": len(blind_rows),
                "window_delta_tau_mae_ms": float(np.mean(blind_delta)) if blind_rows else None,
                "window_stability_mean_std_ms": float(np.mean(case_std)) if case_std else None,
                "window_stability_max_std_ms": float(np.max(case_std)) if case_std else None,
            }
        )

    summary_rows.sort(
        key=lambda x: (
            1 if x["central_delta_tau_mae_ms"] is None else 0,
            float("inf") if x["central_delta_tau_mae_ms"] is None else x["central_delta_tau_mae_ms"],
            float("inf") if x["window_stability_mean_std_ms"] is None else x["window_stability_mean_std_ms"],
        )
    )
    return {"variants": summary_rows}


def write_report(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 3 Diff1 Local Normalization Sweep",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        f"- Window sec: `{payload['window_sec']}`",
        f"- Window offsets sec: `{payload['window_offsets_sec']}`",
        "",
        "## Summary",
        "",
        "| variant | central_valid | physical | central_dt_mae_ms | central_theta_mae_deg | central_max_dt_ms | oracle_rank_mean | oracle_margin_mean | oracle_psr_mean | window_dt_mae_ms | window_std_mean_ms | window_std_max_ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
                    fmt("oracle_correct_pair_rank_mean"),
                    fmt("oracle_correct_pair_margin_mean"),
                    fmt("oracle_correct_psr_mean"),
                    fmt("window_delta_tau_mae_ms"),
                    fmt("window_stability_mean_std_ms"),
                    fmt("window_stability_max_std_ms"),
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 3 diff1 plus local normalization sweep for 0223.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
    )
    parser.add_argument("--window_sec", type=float, default=5.0)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (Path(__file__).resolve().parent.parent / "results" / f"round3_diff1_local_norm_0223_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    full_signals = {case.case_id: load_full_case_signals(case, args.data_root) for case in base.CASES}
    rows = []
    for variant in VARIANTS:
        for case in base.CASES:
            for offset_sec in WINDOW_OFFSETS_SEC:
                rows.append(
                    evaluate_window(
                        case,
                        full_signals[case.case_id],
                        variant,
                        window_sec=args.window_sec,
                        offset_sec=offset_sec,
                    )
                )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "window_sec": float(args.window_sec),
        "window_offsets_sec": list(WINDOW_OFFSETS_SEC),
        "variants": VARIANTS,
        "results": rows,
        "summary": summarize(rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir / "report.md", payload)
    print(f"[ok] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
