#!/usr/bin/env python3
"""
Minimal preprocessing sweep for 0223 LDV-MIC delta-tau accuracy.

The goal is not to reproduce the full paper pipeline. This script tests whether
simple preprocessing can improve GCC-PHAT estimates of:

  tau_VL, tau_VR, delta_tau = tau_VR - tau_VL

on a small set of 0223 blocked cases whose v-point references were already
backed out in the historical report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import stage4_doa_ldv_vs_mic_comparison as stage4


C = 343.0
MIC_LEFT = (-0.7, 2.0)
MIC_RIGHT = (0.7, 2.0)
V_Y = 0.50


@dataclass(frozen=True)
class CaseRef:
    case_id: str
    rel_dir: str
    ldv_name: str
    mic_left_name: str
    mic_right_name: str
    speaker_x_m: float
    v_x_m: float
    note: str


CASES: list[CaseRef] = [
    CaseRef(
        case_id="block4_p08_17",
        rel_dir="0223-block/0223-block-4(high)",
        ldv_name="0223-LDV-40-boy(+0.8m)-17-block.wav",
        mic_left_name="0223-MIC-LEFT-40-boy(+0.8m)-17-block.wav",
        mic_right_name="0223-MIC-RIGHT-40-boy(+0.8m)-17-block.wav",
        speaker_x_m=0.8,
        v_x_m=0.18,
        note="High-point session; report uses v=(+0.18,0.50).",
    ),
    CaseRef(
        case_id="block5_p00_18",
        rel_dir="0223-block/0223-block-5(high)",
        ldv_name="0223-LDV-40-boy(+0.0m)-18-block.wav",
        mic_left_name="0223-MIC-LEFT-40-boy(+0.0m)-18-block.wav",
        mic_right_name="0223-MIC-RIGHT-40-boy(+0.0m)-18-block.wav",
        speaker_x_m=0.0,
        v_x_m=0.18,
        note="High-point session; report uses v=(+0.18,0.50).",
    ),
    CaseRef(
        case_id="block6_n04_19",
        rel_dir="0223-block-6(high)",
        ldv_name="0223-LDV-40-boy(-0.4m)-19-block.wav",
        mic_left_name="0223-MIC-LEFT-40-boy(-0.4m)-19-block.wav",
        mic_right_name="0223-MIC-RIGHT-40-boy(-0.4m)-19-block.wav",
        speaker_x_m=-0.4,
        v_x_m=-0.21,
        note="Report scan-best v_x for block-6 is -0.21.",
    ),
    CaseRef(
        case_id="block7_n08_20",
        rel_dir="0223-block/0223-block-7(high)",
        ldv_name="0223-LDV-40-boy(-0.8m)-20-block.wav",
        mic_left_name="0223-MIC-LEFT-40-boy(-0.8m)-20-block.wav",
        mic_right_name="0223-MIC-RIGHT-40-boy(-0.8m)-20-block.wav",
        speaker_x_m=-0.8,
        v_x_m=0.12,
        note="Report scan-best v_x for block-7 is +0.12.",
    ),
]


VARIANTS: list[dict[str, Any]] = [
    {"name": "raw_fullband", "bandpass": None, "ldv_ops": [], "mic_ops": []},
    {"name": "bp_300_3000", "bandpass": (300.0, 3000.0), "ldv_ops": [], "mic_ops": []},
    {"name": "bp_500_2000", "bandpass": (500.0, 2000.0), "ldv_ops": [], "mic_ops": []},
    {"name": "bp_1000_3000", "bandpass": (1000.0, 3000.0), "ldv_ops": [], "mic_ops": []},
    {
        "name": "ldv_preemph_bp500_2000",
        "bandpass": (500.0, 2000.0),
        "ldv_ops": ["preemphasis"],
        "mic_ops": [],
    },
    {
        "name": "ldv_diff_bp500_2000",
        "bandpass": (500.0, 2000.0),
        "ldv_ops": ["diff"],
        "mic_ops": [],
    },
    {
        "name": "both_flatten_bp500_2000",
        "bandpass": (500.0, 2000.0),
        "ldv_ops": ["spectral_flatten"],
        "mic_ops": ["spectral_flatten"],
    },
]


def compute_reference(case: CaseRef) -> dict[str, float]:
    v = (case.v_x_m, V_Y)
    d_vl = float(np.hypot(v[0] - MIC_LEFT[0], v[1] - MIC_LEFT[1]))
    d_vr = float(np.hypot(v[0] - MIC_RIGHT[0], v[1] - MIC_RIGHT[1]))
    tau_vl_ms = d_vl / C * 1000.0
    tau_vr_ms = d_vr / C * 1000.0
    delta_tau_ms = tau_vr_ms - tau_vl_ms
    theta_v_deg = float(np.degrees(np.arcsin(np.clip((delta_tau_ms / 1000.0) * C / 1.4, -1.0, 1.0))))
    return {
        "tau_vl_ms": tau_vl_ms,
        "tau_vr_ms": tau_vr_ms,
        "delta_tau_ms": delta_tau_ms,
        "theta_v_deg": theta_v_deg,
    }


def spectral_flatten(signal: np.ndarray, smooth_bins: int = 65) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return x
    x = x - np.mean(x)
    n_fft = 1 << int(np.ceil(np.log2(max(8, x.size))))
    X = np.fft.rfft(x, n=n_fft)
    mag = np.abs(X)
    kernel = np.ones(int(max(3, smooth_bins)), dtype=np.float64)
    kernel /= np.sum(kernel)
    smooth = np.convolve(mag, kernel, mode="same")
    Xw = X / (smooth + 1e-8)
    yw = np.fft.irfft(Xw, n=n_fft)[: x.size]
    return yw.astype(np.float64, copy=False)


def preemphasis(signal: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    return y


def diff_signal(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = 0.0
    y[1:] = np.diff(x)
    return y


def apply_ops(signal: np.ndarray, ops: list[str]) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    for op in ops:
        if op == "preemphasis":
            x = preemphasis(x)
        elif op == "diff":
            x = diff_signal(x)
        elif op == "spectral_flatten":
            x = spectral_flatten(x)
        else:
            raise ValueError(f"Unknown op: {op}")
    return x


def load_case_signals(case: CaseRef, data_root: Path, slice_sec: float, fs: int) -> dict[str, np.ndarray]:
    case_dir = data_root / case.rel_dir
    sr_ldv, ldv = stage4.load_wav(str(case_dir / case.ldv_name))
    sr_l, mic_l = stage4.load_wav(str(case_dir / case.mic_left_name))
    sr_r, mic_r = stage4.load_wav(str(case_dir / case.mic_right_name))
    if sr_ldv != fs or sr_l != fs or sr_r != fs:
        raise ValueError(f"Sample-rate mismatch for {case.case_id}: {sr_ldv}, {sr_l}, {sr_r}")

    min_len = min(len(ldv), len(mic_l), len(mic_r))
    ldv = ldv[:min_len]
    mic_l = mic_l[:min_len]
    mic_r = mic_r[:min_len]

    if slice_sec > 0:
        slice_samples = int(round(slice_sec * fs))
        center = min_len // 2
        start, end = stage4.extract_centered_slice([ldv, mic_l, mic_r], center_sample=center, slice_samples=slice_samples)
        ldv = ldv[start:end]
        mic_l = mic_l[start:end]
        mic_r = mic_r[start:end]

    return {"ldv": ldv, "mic_l": mic_l, "mic_r": mic_r}


def run_measurement(
    ldv: np.ndarray,
    mic: np.ndarray,
    fs: int,
    *,
    bandpass: tuple[float, float] | None,
    guided_tau_ms: float,
    guided_radius_ms: float,
    max_lag_ms: float,
) -> dict[str, Any]:
    tau_global_sec, psr_global = stage4.gcc_phat_full_analysis(
        ldv,
        mic,
        fs,
        max_tau=max_lag_ms / 1000.0,
        bandpass=bandpass,
    )
    tau_guided_sec, psr_guided = stage4.gcc_phat_full_analysis(
        ldv,
        mic,
        fs,
        max_tau=max_lag_ms / 1000.0,
        bandpass=bandpass,
        guided_tau=guided_tau_ms / 1000.0,
        guided_radius=guided_radius_ms / 1000.0,
    )
    return {
        "global_tau_ms": float(tau_global_sec * 1000.0),
        "global_psr": float(psr_global),
        "guided_tau_ms": float(tau_guided_sec * 1000.0),
        "guided_psr": float(psr_guided),
    }


def build_case_result(case: CaseRef, variant: dict[str, Any], signals: dict[str, np.ndarray], guided_radius_ms: float, max_lag_ms: float) -> dict[str, Any]:
    ref = compute_reference(case)
    ldv = apply_ops(signals["ldv"], list(variant["ldv_ops"]))
    mic_l = apply_ops(signals["mic_l"], list(variant["mic_ops"]))
    mic_r = apply_ops(signals["mic_r"], list(variant["mic_ops"]))
    bandpass = variant["bandpass"]

    vl = run_measurement(
        ldv,
        mic_l,
        fs=48000,
        bandpass=bandpass,
        guided_tau_ms=ref["tau_vl_ms"],
        guided_radius_ms=guided_radius_ms,
        max_lag_ms=max_lag_ms,
    )
    vr = run_measurement(
        ldv,
        mic_r,
        fs=48000,
        bandpass=bandpass,
        guided_tau_ms=ref["tau_vr_ms"],
        guided_radius_ms=guided_radius_ms,
        max_lag_ms=max_lag_ms,
    )

    def compose(prefix: str) -> dict[str, float]:
        tau_vl = vl[f"{prefix}_tau_ms"]
        tau_vr = vr[f"{prefix}_tau_ms"]
        delta_tau = tau_vr - tau_vl
        theta_v = float(np.degrees(np.arcsin(np.clip((delta_tau / 1000.0) * C / 1.4, -1.0, 1.0))))
        return {
            "tau_vl_ms": tau_vl,
            "tau_vr_ms": tau_vr,
            "delta_tau_ms": delta_tau,
            "theta_v_deg": theta_v,
            "tau_vl_abs_err_ms": abs(tau_vl - ref["tau_vl_ms"]),
            "tau_vr_abs_err_ms": abs(tau_vr - ref["tau_vr_ms"]),
            "delta_tau_abs_err_ms": abs(delta_tau - ref["delta_tau_ms"]),
            "theta_v_abs_err_deg": abs(theta_v - ref["theta_v_deg"]),
            "physical_positive_lags": bool((tau_vl > 0.0) and (tau_vr > 0.0)),
            "physical_small_delta": bool(abs(delta_tau) <= 1.0),
        }

    return {
        "case_id": case.case_id,
        "case_note": case.note,
        "variant": variant["name"],
        "bandpass": bandpass,
        "ldv_ops": list(variant["ldv_ops"]),
        "mic_ops": list(variant["mic_ops"]),
        "reference": ref,
        "global": {
            **compose("global"),
            "vl_psr": vl["global_psr"],
            "vr_psr": vr["global_psr"],
        },
        "guided": {
            **compose("guided"),
            "vl_psr": vl["guided_psr"],
            "vr_psr": vr["guided_psr"],
        },
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_variant.setdefault(row["variant"], []).append(row)

    summary_rows = []
    for variant, rows in by_variant.items():
        g_delta = np.array([r["global"]["delta_tau_abs_err_ms"] for r in rows], dtype=np.float64)
        g_theta = np.array([r["global"]["theta_v_abs_err_deg"] for r in rows], dtype=np.float64)
        gd_delta = np.array([r["guided"]["delta_tau_abs_err_ms"] for r in rows], dtype=np.float64)
        gd_theta = np.array([r["guided"]["theta_v_abs_err_deg"] for r in rows], dtype=np.float64)
        physical = sum(1 for r in rows if r["global"]["physical_positive_lags"] and r["global"]["physical_small_delta"])
        summary_rows.append(
            {
                "variant": variant,
                "n_cases": len(rows),
                "global_delta_tau_mae_ms": float(np.mean(g_delta)),
                "global_theta_mae_deg": float(np.mean(g_theta)),
                "guided_delta_tau_mae_ms": float(np.mean(gd_delta)),
                "guided_theta_mae_deg": float(np.mean(gd_theta)),
                "global_physical_count": int(physical),
            }
        )
    summary_rows.sort(key=lambda x: (x["global_delta_tau_mae_ms"], x["guided_delta_tau_mae_ms"]))
    return {"variants": summary_rows}


def write_markdown(out_path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# 0223 Filter Sweep Delta-Tau Report",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Data root: `{payload['data_root']}`",
        f"- Slice sec: `{payload['slice_sec']}`",
        f"- Guided radius ms: `{payload['guided_radius_ms']}`",
        f"- Max lag ms: `{payload['max_lag_ms']}`",
        "",
        "## Variant Summary",
        "",
        "| variant | global Δτ MAE (ms) | global θ_v MAE (deg) | guided Δτ MAE (ms) | guided θ_v MAE (deg) | physical global count |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]["variants"]:
        lines.append(
            f"| {row['variant']} | {row['global_delta_tau_mae_ms']:.3f} | {row['global_theta_mae_deg']:.3f} | "
            f"{row['guided_delta_tau_mae_ms']:.3f} | {row['guided_theta_mae_deg']:.3f} | {row['global_physical_count']} |"
        )

    lines.extend(["", "## Per-Case Highlights", ""])
    for row in payload["results"]:
        lines.extend(
            [
                f"### {row['case_id']} / {row['variant']}",
                "",
                f"- note: {row['case_note']}",
                f"- reference: tau_VL={row['reference']['tau_vl_ms']:.3f} ms, tau_VR={row['reference']['tau_vr_ms']:.3f} ms, "
                f"Δτ={row['reference']['delta_tau_ms']:.3f} ms, θ_v={row['reference']['theta_v_deg']:.3f}°",
                f"- global: tau_VL={row['global']['tau_vl_ms']:.3f} ms, tau_VR={row['global']['tau_vr_ms']:.3f} ms, "
                f"Δτ={row['global']['delta_tau_ms']:.3f} ms, θ_v={row['global']['theta_v_deg']:.3f}°, "
                f"Δτ err={row['global']['delta_tau_abs_err_ms']:.3f} ms, θ_v err={row['global']['theta_v_abs_err_deg']:.3f}°",
                f"- guided: tau_VL={row['guided']['tau_vl_ms']:.3f} ms, tau_VR={row['guided']['tau_vr_ms']:.3f} ms, "
                f"Δτ={row['guided']['delta_tau_ms']:.3f} ms, θ_v={row['guided']['theta_v_deg']:.3f}°, "
                f"Δτ err={row['guided']['delta_tau_abs_err_ms']:.3f} ms, θ_v err={row['guided']['theta_v_abs_err_deg']:.3f}°",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep simple preprocessors for 0223 LDV delta-tau accuracy.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"C:\Users\Jenner\Documents\SBP Lab\LDVReorientation\worktree\doc-interspeech-2026-repro\dataset\0223"),
        help="Root of the 0223 dataset.",
    )
    parser.add_argument("--slice_sec", type=float, default=0.0, help="Centered slice duration in seconds; 0 means full clip.")
    parser.add_argument("--guided_radius_ms", type=float, default=0.75, help="Guided GCC search radius around reference tau.")
    parser.add_argument("--max_lag_ms", type=float, default=10.0, help="Global GCC lag search range.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory. Defaults to results/filter_sweep_0223_<timestamp>.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (Path(__file__).resolve().parent.parent / "results" / f"filter_sweep_0223_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    signals_cache = {case.case_id: load_case_signals(case, args.data_root, args.slice_sec, 48000) for case in CASES}
    for variant in VARIANTS:
        for case in CASES:
            results.append(
                build_case_result(
                    case,
                    variant,
                    signals_cache[case.case_id],
                    guided_radius_ms=args.guided_radius_ms,
                    max_lag_ms=args.max_lag_ms,
                )
            )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "slice_sec": float(args.slice_sec),
        "guided_radius_ms": float(args.guided_radius_ms),
        "max_lag_ms": float(args.max_lag_ms),
        "cases": [asdict(case) for case in CASES],
        "variants": VARIANTS,
        "results": results,
        "summary": summarize(results),
    }

    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(out_dir / "report.md", payload)
    print(f"[ok] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
