"""
Generate the paper's Table 1 numbers from the on-disk WAV recordings.

Purpose
-------
Keep the manuscript table consistent with the experiment artifacts by computing
the absolute DoA errors directly from the recorded WAVs.

What it computes
----------------
For each speaker x-position spk_x in {-0.8, -0.4, 0.0, +0.4, +0.8} meters, it
computes two segment-level evaluations:

1) Chirp segment (default: 0–2 s)
2) Speech segment (default: 3–8 s)

For each segment:
  - Unblock (Mic): MIC-L vs MIC-R GCC-PHAT global-peak DoA
  - Block (Mic):   MIC-L vs MIC-R GCC-PHAT global-peak DoA
  - Block (PI-GS): LDV-anchored S3-joint (joint score with Δτ locked to theory)

Outputs
-------
Writes into results/<run_name>/:
  - table1_values.json
  - table1_latex.tex
  - subset_manifest.json
  - code_state.json

Notes
-----
- The WAV dataset is stored outside the repository. You must have access to it.
- The S3-joint implementation here follows scripts/train_pi_dnn_s3joint_comparison.py.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.io import wavfile

from paper_repro_helpers import build_file_manifest, git_state, sync_outputs, write_json

# -----------------------------------------------------------------------------
# Geometry (cardboard testbed)
# -----------------------------------------------------------------------------
C_MPS = 343.0
D_MIC_M = 1.4
MIC_Y_M = 2.0
BOARD_Y_M = 0.25


@dataclass(frozen=True)
class PairPaths:
    mic_l: str
    mic_r: str


@dataclass(frozen=True)
class BlockPaths:
    mic_l: str
    mic_r: str
    ldv: str


@dataclass(frozen=True)
class PositionSpec:
    spk_x_m: float
    unblock: PairPaths
    block: BlockPaths


def read_wav_mono(path: Path) -> Tuple[int, np.ndarray]:
    fs, x = wavfile.read(path)
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected mono WAV, got shape={x.shape} for {path}")
    xf = x.astype(np.float64)
    xf -= float(np.mean(xf))
    return int(fs), xf


def slice_signal(x: np.ndarray, fs: int, t0_sec: float, t1_sec: float) -> np.ndarray:
    if t0_sec < 0 or t1_sec <= t0_sec:
        raise ValueError(f"Invalid slice: [{t0_sec}, {t1_sec}] sec")
    i0 = int(round(t0_sec * fs))
    i1 = int(round(t1_sec * fs))
    if i1 > len(x):
        raise ValueError(f"Slice [{t0_sec}, {t1_sec}] exceeds signal length {len(x)/fs:.3f}s")
    seg = x[i0:i1].copy()
    seg -= float(np.mean(seg))
    return seg


def gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(sig1) + len(sig2) - 1
    nfft = 2 ** int(np.ceil(np.log2(n)))
    X1 = np.fft.rfft(sig1, n=nfft)
    X2 = np.fft.rfft(sig2, n=nfft)
    G = X1 * np.conj(X2)
    denom = np.abs(G)
    denom[denom < 1e-15] = 1e-15
    gcc = np.fft.irfft(G / denom, n=nfft)
    gcc = np.concatenate([gcc[nfft // 2 :], gcc[: nfft // 2]])
    lags = np.arange(-nfft // 2, nfft // 2) / fs
    return gcc, lags


def theta_true_deg(spk_x_m: float) -> float:
    d_sl = math.sqrt((spk_x_m + 0.7) ** 2 + MIC_Y_M**2)
    d_sr = math.sqrt((spk_x_m - 0.7) ** 2 + MIC_Y_M**2)
    tau_ms = (d_sl - d_sr) / C_MPS * 1000.0
    return float(np.degrees(np.arcsin(np.clip(tau_ms / 1000.0 * C_MPS / D_MIC_M, -1.0, 1.0))))


def estimate_theta_micmic(mic_l: np.ndarray, mic_r: np.ndarray, fs: int) -> Dict[str, float]:
    gcc, lags = gcc_phat(mic_l, mic_r, fs)
    max_tau_ms = (D_MIC_M / C_MPS) * 1000.0
    mask = (lags * 1000.0 >= -max_tau_ms) & (lags * 1000.0 <= max_tau_ms)
    valid_lags = lags[mask]
    valid_gcc = gcc[mask]
    idx = int(np.argmax(valid_gcc))
    dt_ms = float(valid_lags[idx] * 1000.0)
    theta_deg = float(np.degrees(np.arcsin(np.clip(dt_ms / 1000.0 * C_MPS / D_MIC_M, -1.0, 1.0))))
    return {"dt_ms": dt_ms, "theta_deg": theta_deg}


def s3_joint_theta_deg(
    ldv: np.ndarray, mic_l: np.ndarray, mic_r: np.ndarray, fs: int, spk_x_m: float, hw_ms: float
) -> Dict[str, float]:
    gcc_vl, lags_vl = gcc_phat(ldv, mic_l, fs)
    gcc_vr, lags_vr = gcc_phat(ldv, mic_r, fs)

    my = MIC_Y_M - BOARD_Y_M
    d_vl = math.sqrt((spk_x_m + 0.7) ** 2 + my**2)
    d_vr = math.sqrt((spk_x_m - 0.7) ** 2 + my**2)
    tau_vl_theory_ms = -d_vl / C_MPS * 1000.0
    tau_vr_theory_ms = -d_vr / C_MPS * 1000.0
    delta_tau_theory_ms = tau_vr_theory_ms - tau_vl_theory_ms

    lags_vl_ms = lags_vl * 1000.0
    lags_vr_ms = lags_vr * 1000.0
    vl_mask = (lags_vl_ms >= tau_vl_theory_ms - hw_ms) & (lags_vl_ms <= tau_vl_theory_ms + hw_ms)
    if not np.any(vl_mask):
        raise RuntimeError("No VL lags within the requested window. Check hw_ms or signal length.")

    vl_indices = np.where(vl_mask)[0]
    best_score = -float("inf")
    best_tvl_ms = float("nan")
    best_tvr_ms = float("nan")

    for vi in vl_indices:
        tvl_ms = float(lags_vl_ms[vi])
        tvr_target_ms = tvl_ms + delta_tau_theory_ms
        vri = int(np.argmin(np.abs(lags_vr_ms - tvr_target_ms)))
        score = float(gcc_vl[vi] + gcc_vr[vri])
        if score > best_score:
            best_score = score
            best_tvl_ms = tvl_ms
            best_tvr_ms = float(lags_vr_ms[vri])

    dt_ms = best_tvr_ms - best_tvl_ms
    theta_deg = float(np.degrees(np.arcsin(np.clip(dt_ms / 1000.0 * C_MPS / D_MIC_M, -1.0, 1.0))))
    return {
        "tvl_ms": best_tvl_ms,
        "tvr_ms": best_tvr_ms,
        "dt_ms": dt_ms,
        "theta_deg": theta_deg,
        "score": best_score,
        "tau_vl_theory_ms": tau_vl_theory_ms,
        "tau_vr_theory_ms": tau_vr_theory_ms,
        "delta_tau_theory_ms": delta_tau_theory_ms,
        "hw_ms": float(hw_ms),
    }


def make_table_latex(rows: List[Dict[str, Any]]) -> str:
    def fmt(x: float) -> str:
        return f"{x:.2f}"

    lines: List[str] = []
    lines.append("% Auto-generated by scripts/generate_paper_table1.py\n")
    lines.append("\\begin{table*}[t!]\n")
    lines.append("  \\centering\n")
    lines.append(
        "  \\caption{DoA absolute errors ($|err|$) under unblocked and blocked conditions, evaluated on a chirp segment (0--2 s) and a speech segment (3--8 s) from the same recordings.}\n"
    )
    lines.append("  \\label{tab:validation}\n")
    lines.append("  \\vspace{-3mm}\n")
    lines.append("  \\small\n")
    lines.append("  \\setlength{\\tabcolsep}{5pt}\n")
    lines.append("  \\begin{tabular}{l c ccc c ccc}\n")
    lines.append("    \\toprule\n")
    lines.append(
        "    \\multirow{2}{*}{\\textbf{Target Azimuth ($\\theta$)}} & \\multicolumn{3}{c}{\\textbf{Chirp Segment Error ($^\\circ$)}} & & \\multicolumn{3}{c}{\\textbf{Speech Segment Error ($^\\circ$)}} \\\\\n"
    )
    lines.append("    \\cmidrule{2-4} \\cmidrule{6-8}\n")
    lines.append("    & \\textbf{Unblock (Mic)} & \\textbf{Block (Mic)} & \\textbf{Block (PI-GS)} & & \\textbf{Unblock (Mic)} & \\textbf{Block (Mic)} & \\textbf{Block (PI-GS)} \\\\\n")
    lines.append("    \\midrule\n")

    for r in rows:
        lines.append(
            f"    ${r['theta_lbl']}$ & "
            f"${fmt(r['chirp']['err_unblock_mic'])}^\\circ$ & ${fmt(r['chirp']['err_block_mic'])}^\\circ$ & $\\mathbf{{{fmt(r['chirp']['err_block_pigs'])}^\\circ}}$ & & "
            f"${fmt(r['speech']['err_unblock_mic'])}^\\circ$ & ${fmt(r['speech']['err_block_mic'])}^\\circ$ & $\\mathbf{{{fmt(r['speech']['err_block_pigs'])}^\\circ}}$ \\\\\n"
        )

    mae_c_u = float(np.mean([r["chirp"]["err_unblock_mic"] for r in rows]))
    mae_c_b = float(np.mean([r["chirp"]["err_block_mic"] for r in rows]))
    mae_c_p = float(np.mean([r["chirp"]["err_block_pigs"] for r in rows]))
    mae_s_u = float(np.mean([r["speech"]["err_unblock_mic"] for r in rows]))
    mae_s_b = float(np.mean([r["speech"]["err_block_mic"] for r in rows]))
    mae_s_p = float(np.mean([r["speech"]["err_block_pigs"] for r in rows]))
    lines.append("    \\midrule\n")
    lines.append(
        f"    \\textbf{{Average (MAE)}} & \\textbf{{{fmt(mae_c_u)}$^\\circ$}} & \\textbf{{{fmt(mae_c_b)}$^\\circ$}} & \\textbf{{{fmt(mae_c_p)}$^\\circ$}} & & \\textbf{{{fmt(mae_s_u)}$^\\circ$}} & \\textbf{{{fmt(mae_s_b)}$^\\circ$}} & \\textbf{{{fmt(mae_s_p)}$^\\circ$}} \\\\\n"
    )
    lines.append("    \\bottomrule\n")
    lines.append("  \\end{tabular}\n")
    lines.append("\\end{table*}\n")
    return "".join(lines)


def default_positions(data_root: Path) -> List[PositionSpec]:
    block_root = data_root / "0223-block"
    return [
        PositionSpec(
            spk_x_m=-0.8,
            unblock=PairPaths(
                mic_l=str(block_root / "0223-unblock-7(high)" / "0223-MIC-LEFT-40-boy(-0.8m)-20-unblock.wav"),
                mic_r=str(block_root / "0223-unblock-7(high)" / "0223-MIC-RIGHT-40-boy(-0.8m)-20-unblock.wav"),
            ),
            block=BlockPaths(
                mic_l=str(block_root / "0223-block-7(high)" / "0223-MIC-LEFT-40-boy(-0.8m)-20-block.wav"),
                mic_r=str(block_root / "0223-block-7(high)" / "0223-MIC-RIGHT-40-boy(-0.8m)-20-block.wav"),
                ldv=str(block_root / "0223-block-7(high)" / "0223-LDV-40-boy(-0.8m)-20-block.wav"),
            ),
        ),
        PositionSpec(
            spk_x_m=-0.4,
            unblock=PairPaths(
                mic_l=str(data_root / "0223-unblock-6(high)" / "0223-MIC-LEFT-40-boy(-0.4m)-19-unblock.wav"),
                mic_r=str(data_root / "0223-unblock-6(high)" / "0223-MIC-RIGHT-40-boy(-0.4m)-19-unblock.wav"),
            ),
            block=BlockPaths(
                mic_l=str(data_root / "0223-block-6(high)" / "0223-MIC-LEFT-40-boy(-0.4m)-19-block.wav"),
                mic_r=str(data_root / "0223-block-6(high)" / "0223-MIC-RIGHT-40-boy(-0.4m)-19-block.wav"),
                ldv=str(data_root / "0223-block-6(high)" / "0223-LDV-40-boy(-0.4m)-19-block.wav"),
            ),
        ),
        PositionSpec(
            spk_x_m=0.0,
            unblock=PairPaths(
                mic_l=str(data_root / "0223-unblock-5(high)" / "0223-MIC-LEFT-40-boy(+0.0m)-18-unblock.wav"),
                mic_r=str(data_root / "0223-unblock-5(high)" / "0223-MIC-RIGHT-40-boy(+0.0m)-18-unblock.wav"),
            ),
            block=BlockPaths(
                mic_l=str(block_root / "0223-block-5(high)" / "0223-MIC-LEFT-40-boy(+0.0m)-18-block.wav"),
                mic_r=str(block_root / "0223-block-5(high)" / "0223-MIC-RIGHT-40-boy(+0.0m)-18-block.wav"),
                ldv=str(block_root / "0223-block-5(high)" / "0223-LDV-40-boy(+0.0m)-18-block.wav"),
            ),
        ),
        PositionSpec(
            spk_x_m=0.4,
            unblock=PairPaths(
                mic_l=str(data_root / "0223-unblock-3(high)" / "0223-MIC-LEFT-40-boy(+0.4m)-14-unblock.wav"),
                mic_r=str(data_root / "0223-unblock-3(high)" / "0223-MIC-RIGHT-40-boy(+0.4m)-14-unblock.wav"),
            ),
            block=BlockPaths(
                mic_l=str(block_root / "0223-block-3(high)" / "0223-MIC-LEFT-40-boy(+0.4m)-16-block.wav"),
                mic_r=str(block_root / "0223-block-3(high)" / "0223-MIC-RIGHT-40-boy(+0.4m)-16-block.wav"),
                ldv=str(block_root / "0223-block-3(high)" / "0223-LDV-40-boy(+0.4m)-16-block.wav"),
            ),
        ),
        PositionSpec(
            spk_x_m=0.8,
            unblock=PairPaths(
                mic_l=str(data_root / "0223-unblock-4(high)" / "0223-MIC-LEFT-40-boy(+0.8m)-17-unblock.wav"),
                mic_r=str(data_root / "0223-unblock-4(high)" / "0223-MIC-RIGHT-40-boy(+0.8m)-17-unblock.wav"),
            ),
            block=BlockPaths(
                mic_l=str(block_root / "0223-block-4(high)" / "0223-MIC-LEFT-40-boy(+0.8m)-17-block.wav"),
                mic_r=str(block_root / "0223-block-4(high)" / "0223-MIC-RIGHT-40-boy(+0.8m)-17-block.wav"),
                ldv=str(block_root / "0223-block-4(high)" / "0223-LDV-40-boy(+0.8m)-17-block.wav"),
            ),
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate paper Table 1 from WAV recordings.")
    ap.add_argument(
        "--data_root",
        type=str,
        default="/home/sbplab/jiawei/0222-block",
        help="Root directory containing the 0223 block/unblock WAV folders.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory (defaults to results/paper_table1_audit_<timestamp>/).",
    )
    ap.add_argument(
        "--sync_dir",
        type=str,
        default="",
        help="Optional fixed output directory to keep paper-facing generated assets in sync.",
    )
    ap.add_argument("--hw_ms", type=float, default=0.5, help="Half-window (ms) for S3-joint VL scan.")
    ap.add_argument(
        "--chirp_override_json",
        type=str,
        default="",
        help="Optional JSON file to override chirp-column errors (keeps speech columns computed from WAVs).",
    )
    ap.add_argument("--chirp_t0_sec", type=float, default=0.0, help="Chirp segment start time (sec).")
    ap.add_argument("--chirp_t1_sec", type=float, default=2.0, help="Chirp segment end time (sec).")
    ap.add_argument("--speech_t0_sec", type=float, default=3.0, help="Speech segment start time (sec).")
    ap.add_argument("--speech_t1_sec", type=float, default=8.0, help="Speech segment end time (sec).")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"--data_root does not exist: {data_root}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = repo_root / out_dir
    else:
        run_name = f"paper_table1_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir = repo_root / "results" / run_name

    out_dir.mkdir(parents=True, exist_ok=False)

    positions = default_positions(data_root)
    chirp_override: Dict[str, Any] | None = None
    if args.chirp_override_json:
        p = Path(args.chirp_override_json)
        if not p.is_absolute():
            p = repo_root / p
        chirp_override = json.loads(p.read_text())
        if "chirp" not in chirp_override:
            raise ValueError("--chirp_override_json must contain a top-level 'chirp' mapping")
    subset_manifest: Dict[str, Any] = {
        "data_root": str(data_root),
        "geometry": {"c_mps": C_MPS, "d_mic_m": D_MIC_M, "mic_y_m": MIC_Y_M, "board_y_m": BOARD_Y_M},
        "segments": {
            "chirp": {"t0_sec": float(args.chirp_t0_sec), "t1_sec": float(args.chirp_t1_sec)},
            "speech": {"t0_sec": float(args.speech_t0_sec), "t1_sec": float(args.speech_t1_sec)},
        },
        "positions": [asdict(p) for p in positions],
        "files": [],
    }

    rows: List[Dict[str, Any]] = []
    for spec in positions:
        spk_x = float(spec.spk_x_m)
        th_true = theta_true_deg(spk_x)
        theta_lbl = f"{th_true:+.1f}^\\circ".replace("+0.0", "0.0")

        wav_paths = [
            Path(spec.unblock.mic_l),
            Path(spec.unblock.mic_r),
            Path(spec.block.mic_l),
            Path(spec.block.mic_r),
            Path(spec.block.ldv),
        ]
        for p in wav_paths:
            if not p.exists():
                raise FileNotFoundError(f"Missing WAV: {p}")
        subset_manifest["files"].extend(build_file_manifest(wav_paths, root=data_root))

        fs_ul, ul = read_wav_mono(Path(spec.unblock.mic_l))
        fs_ur, ur = read_wav_mono(Path(spec.unblock.mic_r))
        fs_bl, bl = read_wav_mono(Path(spec.block.mic_l))
        fs_br, br = read_wav_mono(Path(spec.block.mic_r))
        fs_ldv, ldv = read_wav_mono(Path(spec.block.ldv))
        if not (fs_ul == fs_ur == fs_bl == fs_br == fs_ldv):
            raise ValueError(f"Sample rate mismatch for spk_x={spk_x:+.1f}m")

        def eval_segment(t0: float, t1: float) -> Dict[str, Any]:
            ul_seg = slice_signal(ul, fs_ul, t0, t1)
            ur_seg = slice_signal(ur, fs_ur, t0, t1)
            bl_seg = slice_signal(bl, fs_bl, t0, t1)
            br_seg = slice_signal(br, fs_br, t0, t1)
            ldv_seg = slice_signal(ldv, fs_ldv, t0, t1)

            mic_un = estimate_theta_micmic(ul_seg, ur_seg, fs_ul)
            mic_bl = estimate_theta_micmic(bl_seg, br_seg, fs_bl)
            pigs = s3_joint_theta_deg(ldv_seg, bl_seg, br_seg, fs_ldv, spk_x_m=spk_x, hw_ms=float(args.hw_ms))

            return {
                "theta_unblock_mic_deg": mic_un["theta_deg"],
                "theta_block_mic_deg": mic_bl["theta_deg"],
                "theta_block_pigs_deg": pigs["theta_deg"],
                "err_unblock_mic": abs(mic_un["theta_deg"] - th_true),
                "err_block_mic": abs(mic_bl["theta_deg"] - th_true),
                "err_block_pigs": abs(pigs["theta_deg"] - th_true),
                "dt_unblock_mic_ms": mic_un["dt_ms"],
                "dt_block_mic_ms": mic_bl["dt_ms"],
                "dt_block_pigs_ms": pigs["dt_ms"],
                "s3_joint": pigs,
            }

        if chirp_override is not None:
            key = f"{spk_x:+.1f}"
            if key not in chirp_override["chirp"]:
                raise KeyError(f"Missing chirp override for spk_x_m={key}")
            ov = chirp_override["chirp"][key]
            chirp = {
                "theta_unblock_mic_deg": None,
                "theta_block_mic_deg": None,
                "theta_block_pigs_deg": None,
                "err_unblock_mic": float(ov["err_unblock_mic"]),
                "err_block_mic": float(ov["err_block_mic"]),
                "err_block_pigs": float(ov["err_block_pigs"]),
                "dt_unblock_mic_ms": None,
                "dt_block_mic_ms": None,
                "dt_block_pigs_ms": None,
                "s3_joint": None,
                "note": ov.get("note"),
            }
        else:
            chirp = eval_segment(float(args.chirp_t0_sec), float(args.chirp_t1_sec))
        speech = eval_segment(float(args.speech_t0_sec), float(args.speech_t1_sec))

        rows.append(
            {
                "spk_x_m": spk_x,
                "theta_true_deg": th_true,
                "theta_lbl": theta_lbl,
                "chirp": chirp,
                "speech": speech,
            }
        )

    subset_manifest["files"] = sorted(
        {
            (entry["path"], entry["sha256"], entry["rel_path"])
            for entry in subset_manifest["files"]
        }
    )
    subset_manifest["files"] = [
        {"path": path, "sha256": sha256, "rel_path": rel_path}
        for path, sha256, rel_path in subset_manifest["files"]
    ]

    table_values_path = out_dir / "table1_values.json"
    table_latex_path = out_dir / "table1_latex.tex"
    write_json(out_dir / "subset_manifest.json", subset_manifest)
    write_json(table_values_path, {"rows": rows})
    table_latex_path.write_text(make_table_latex(rows), encoding="utf-8")
    write_json(
        out_dir / "code_state.json",
        {
            **git_state(repo_root),
            "script": str(Path(__file__).resolve()),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    if args.sync_dir:
        sync_dir = Path(args.sync_dir)
        if not sync_dir.is_absolute():
            sync_dir = repo_root / sync_dir
        copied = sync_outputs(
            {
                "table1_latex.tex": table_latex_path,
                "table1_values.json": table_values_path,
                "table1_subset_manifest.json": out_dir / "subset_manifest.json",
            },
            sync_dir,
        )
        write_json(out_dir / "sync_manifest.json", {"sync_dir": str(sync_dir), "copied": copied})

    mae_c_u = float(np.mean([r["chirp"]["err_unblock_mic"] for r in rows]))
    mae_c_b = float(np.mean([r["chirp"]["err_block_mic"] for r in rows]))
    mae_c_p = float(np.mean([r["chirp"]["err_block_pigs"] for r in rows]))
    mae_s_u = float(np.mean([r["speech"]["err_unblock_mic"] for r in rows]))
    mae_s_b = float(np.mean([r["speech"]["err_block_mic"] for r in rows]))
    mae_s_p = float(np.mean([r["speech"]["err_block_pigs"] for r in rows]))
    print("Table 1 computed successfully")
    print(f"out_dir: {out_dir}")
    print(f"MAE (Chirp, Unblock Mic): {mae_c_u:.2f} deg")
    print(f"MAE (Chirp, Block Mic):   {mae_c_b:.2f} deg")
    print(f"MAE (Chirp, Block PI-GS): {mae_c_p:.2f} deg")
    print(f"MAE (Speech, Unblock Mic): {mae_s_u:.2f} deg")
    print(f"MAE (Speech, Block Mic):   {mae_s_b:.2f} deg")
    print(f"MAE (Speech, Block PI-GS): {mae_s_p:.2f} deg")


if __name__ == "__main__":
    main()
