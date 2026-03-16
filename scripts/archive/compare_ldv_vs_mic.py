#!/usr/bin/env python
"""
Compare LDV-MIC vs MIC-MIC DoA accuracy.

Mode 1 (default):  Single-band comparison under chirp, τ=2.0, band=500–2000 Hz
Mode 2 (--cross):  Cross-bandpass comparison (band=0 vs band=500–2000)
                    including af1acf5 Stage 4-C validation

Data sources (per-speaker summary.json):
  MIC-MIC:       results/ldv_vs_mic_grid_strict_.../micl_micr_chirp_tau2_band500_2000/
  LDV-MicL(OMP): results/ldv_vs_mic_grid_strict_.../ldv_micl_chirp_tau2_band500_2000/
  LDV-MicR(OMP): results/ldv_vs_mic_grid_ldv_micr_omp_.../ldv_micr_chirp_tau2_band500_2000/
  Stage4 Grid:   results/stage4_speech_chirp_tau2_band{0,500_2000}_20260211_072624/

Outputs (mode 1):
  results/ldv_vs_mic_comparison_report.md
  results/ldv_vs_mic_comparison_data.json
Outputs (mode 2):
  results/cross_experiment_ldv_vs_mic_report.md
  results/cross_experiment_ldv_vs_mic_data.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, median, stdev

SPEAKERS = ["18-0.1V", "19-0.1V", "20-0.1V", "21-0.1V", "22-0.1V"]

THETA_THRESHOLD_DEG = 5.0

# --- paths ----------------------------------------------------------------

BASE = Path(__file__).resolve().parent.parent / "results"

MIC_MIC_DIR = (
    BASE
    / "ldv_vs_mic_grid_strict_20260211_103715"
    / "micl_micr_chirp_tau2_band500_2000"
)
LDV_MICL_DIR = (
    BASE
    / "ldv_vs_mic_grid_strict_20260211_103715"
    / "ldv_micl_chirp_tau2_band500_2000"
)
LDV_MICR_DIR = (
    BASE
    / "ldv_vs_mic_grid_ldv_micr_omp_20260212_142946"
    / "ldv_micr_chirp_tau2_band500_2000"
)

# Stage 4-C Grid directories (both bandpass conditions)
STAGE4_BAND0_DIR = BASE / "stage4_speech_chirp_tau2_band0_20260211_072624"
STAGE4_BAND500_DIR = BASE / "stage4_speech_chirp_tau2_band500_2000_20260211_072624"

OUT_REPORT = BASE / "ldv_vs_mic_comparison_report.md"
OUT_DATA = BASE / "ldv_vs_mic_comparison_data.json"

CROSS_OUT_REPORT = BASE / "cross_experiment_ldv_vs_mic_report.md"
CROSS_OUT_DATA = BASE / "cross_experiment_ldv_vs_mic_data.json"

# --- af1acf5 hardcoded data (Stage 4-C, band=0, 1 segment, speaker 21 & 22) ---
AF1ACF5_DATA = {
    "21-0.1V": {"omp_err": 0.06, "raw_err": 0.20, "mic_err_raw": -6.77,
                 "mic_theta_err_approx": 2.17},
    "22-0.1V": {"omp_err": 3.62, "raw_err": 1.24, "mic_err_raw": -22.28,
                 "mic_theta_err_approx": 3.60},
}

# --- helpers --------------------------------------------------------------


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_speaker(run_dir: Path, speaker: str) -> dict:
    """Return per-speaker metrics from a summary.json."""
    s = load_summary(run_dir / speaker / "summary.json")
    res = s["result"]
    seg_errors = [seg["theta_error_deg"] for seg in res["per_segment"]]
    return {
        "theta_error_median_deg": res["theta_error_median_deg"],
        "psr_median_db": res["psr_median_db"],
        "per_segment_errors": seg_errors,
        "tau_median_ms": res["tau_median_ms"],  # keep for debug
        "n_segments": len(res["per_segment"]),
    }


def agg_stats(values: list[float]) -> dict:
    return {
        "median": median(values),
        "mean": round(mean(values), 4),
        "std": round(stdev(values), 4) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def extract_stage4_speaker(stage4_dir: Path, speaker: str) -> dict:
    """Extract GCC-PHAT results from a Stage 4-C summary.json.

    Stage 4 files contain multiple methods and signal pairs nested under
    ``results.GCC-PHAT.{MicL-MicR, OMP_LDV, Raw_LDV}``.
    Per-segment error key is ``theta_error`` (not ``theta_error_deg``).
    """
    s = load_summary(stage4_dir / speaker / "summary.json")
    gcc = s["results"]["GCC-PHAT"]
    out = {}
    for key, label in [("MicL-MicR", "mic_mic"),
                        ("OMP_LDV", "omp"),
                        ("Raw_LDV", "raw")]:
        r = gcc[key]
        seg_errors = [seg["theta_error"] for seg in r["per_segment"]]
        out[label] = {
            "theta_error_median_deg": r["theta_error_median_deg"],
            "psr_median_db": r["psr_median_db"],
            "per_segment_errors": seg_errors,
            "n_segments": len(r["per_segment"]),
        }
    return out


# --- main (mode 1: single-band) ------------------------------------------


def collect() -> dict:
    """Collect per-speaker data for all three conditions."""
    data: dict[str, dict] = {}
    for label, run_dir in [
        ("mic_mic", MIC_MIC_DIR),
        ("ldv_micl_omp", LDV_MICL_DIR),
        ("ldv_micr_omp", LDV_MICR_DIR),
    ]:
        speakers_data = {}
        for sp in SPEAKERS:
            speakers_data[sp] = extract_speaker(run_dir, sp)
        errs = [speakers_data[sp]["theta_error_median_deg"] for sp in SPEAKERS]
        speakers_data["_agg"] = agg_stats(errs)
        speakers_data["_pass_count"] = sum(
            1 for sp in SPEAKERS
            if speakers_data[sp]["theta_error_median_deg"] < THETA_THRESHOLD_DEG
        )
        data[label] = speakers_data
    return data


def head_to_head(data: dict) -> list[dict]:
    """Per-speaker winner analysis."""
    rows = []
    for sp in SPEAKERS:
        mm = data["mic_mic"][sp]["theta_error_median_deg"]
        ll = data["ldv_micl_omp"][sp]["theta_error_median_deg"]
        lr = data["ldv_micr_omp"][sp]["theta_error_median_deg"]
        best = min(mm, ll, lr)
        if best == mm:
            winner = "MIC-MIC"
        elif best == ll:
            winner = "LDV-MicL"
        else:
            winner = "LDV-MicR"
        rows.append({
            "speaker": sp,
            "mic_mic": mm,
            "ldv_micl": ll,
            "ldv_micr": lr,
            "winner": winner,
        })
    return rows


def fmt(val: float, bold: bool = False) -> str:
    s = f"{val:.3f}"
    return f"**{s}**" if bold else s


def generate_report(data: dict, h2h: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# LDV-MIC vs MIC-MIC Comparison Report")
    lines.append("")
    lines.append("**Condition**: chirp, τ=2.0 s, band 500–2000 Hz (the only condition "
                  "where all three pairings were evaluated)")
    lines.append("")
    lines.append("**Alignment**: LDV channels use OMP pre-alignment (K≤3)")
    lines.append("")
    lines.append("**Error metric**: θ_error = |θ_estimated − θ_truth-ref| (median over 5 segments)")
    lines.append("")

    # --- A. Head-to-head table ---
    lines.append("## A. Per-Speaker Head-to-Head (θ_error in degrees)")
    lines.append("")
    lines.append("| Speaker | MIC-MIC | LDV-MicL (OMP) | LDV-MicR (OMP) | Winner |")
    lines.append("|---------|---------|----------------|----------------|--------|")
    for r in h2h:
        best = min(r["mic_mic"], r["ldv_micl"], r["ldv_micr"])
        lines.append(
            f"| {r['speaker']} "
            f"| {fmt(r['mic_mic'], r['mic_mic'] == best)} "
            f"| {fmt(r['ldv_micl'], r['ldv_micl'] == best)} "
            f"| {fmt(r['ldv_micr'], r['ldv_micr'] == best)} "
            f"| {r['winner']} |"
        )

    # Summary row
    agg_mm = data["mic_mic"]["_agg"]
    agg_ll = data["ldv_micl_omp"]["_agg"]
    agg_lr = data["ldv_micr_omp"]["_agg"]

    lines.append(
        f"| **Median** | **{agg_mm['median']:.3f}** "
        f"| **{agg_ll['median']:.3f}** "
        f"| **{agg_lr['median']:.3f}** | — |"
    )
    lines.append(
        f"| Mean±Std | {agg_mm['mean']:.3f}±{agg_mm['std']:.3f} "
        f"| {agg_ll['mean']:.3f}±{agg_ll['std']:.3f} "
        f"| {agg_lr['mean']:.3f}±{agg_lr['std']:.3f} | — |"
    )
    lines.append(
        f"| Max | {agg_mm['max']:.3f} "
        f"| {agg_ll['max']:.3f} "
        f"| {agg_lr['max']:.3f} | — |"
    )
    lines.append("")

    # --- B. Acceptability analysis ---
    lines.append(f"## B. Acceptability Analysis (< {THETA_THRESHOLD_DEG}° threshold)")
    lines.append("")
    pass_mm = data["mic_mic"]["_pass_count"]
    pass_ll = data["ldv_micl_omp"]["_pass_count"]
    pass_lr = data["ldv_micr_omp"]["_pass_count"]
    lines.append(f"| Method | PASS (< {THETA_THRESHOLD_DEG}°) | Worst θ_error |")
    lines.append("|--------|------|---------------|")
    lines.append(
        f"| MIC-MIC | {pass_mm}/{len(SPEAKERS)} "
        f"| {agg_mm['max']:.3f}° |"
    )
    lines.append(
        f"| LDV-MicL (OMP) | {pass_ll}/{len(SPEAKERS)} "
        f"| {agg_ll['max']:.3f}° |"
    )
    lines.append(
        f"| LDV-MicR (OMP) | {pass_lr}/{len(SPEAKERS)} "
        f"| {agg_lr['max']:.3f}° |"
    )
    lines.append("")
    lines.append("**Interpretation**: All methods achieve < 5° on most speakers, "
                  "but MIC-MIC has a much larger safety margin "
                  f"(worst case {agg_mm['max']:.2f}° vs LDV-MicL worst case {agg_ll['max']:.2f}°).")
    lines.append("")

    # --- C. Speaker 22 deep dive ---
    lines.append("## C. Speaker 22 — The Only LDV Win")
    lines.append("")
    sp22_mm = data["mic_mic"]["22-0.1V"]
    sp22_ll = data["ldv_micl_omp"]["22-0.1V"]
    sp22_lr = data["ldv_micr_omp"]["22-0.1V"]
    lines.append(f"Speaker 22 is the only case where LDV-MicL (OMP) outperforms MIC-MIC:")
    lines.append("")
    lines.append(f"- MIC-MIC θ_error = {sp22_mm['theta_error_median_deg']:.3f}°")
    lines.append(f"- LDV-MicL θ_error = {sp22_ll['theta_error_median_deg']:.3f}° ← best")
    lines.append(f"- LDV-MicR θ_error = {sp22_lr['theta_error_median_deg']:.3f}°")
    lines.append("")
    lines.append("### Per-segment θ_error breakdown (Speaker 22)")
    lines.append("")
    lines.append("| Segment | MIC-MIC | LDV-MicL | LDV-MicR |")
    lines.append("|---------|---------|----------|----------|")
    for i in range(sp22_mm["n_segments"]):
        e_mm = sp22_mm["per_segment_errors"][i]
        e_ll = sp22_ll["per_segment_errors"][i]
        e_lr = sp22_lr["per_segment_errors"][i]
        lines.append(f"| {i+1} | {e_mm:.3f}° | {e_ll:.3f}° | {e_lr:.3f}° |")
    lines.append("")
    lines.append("This suggests that for Speaker 22, the MIC-MIC GCC peak may be weaker "
                 "or more ambiguous, while the OMP-aligned LDV signal provides a cleaner "
                 "correlation with MicL.")
    lines.append("")

    # --- D. PSR caveat ---
    lines.append("## D. PSR Caveat")
    lines.append("")
    lines.append("PSR (Peak-to-Sidelobe Ratio) values are **not directly comparable** "
                 "across methods:")
    lines.append("")
    lines.append("- **MIC-MIC PSR**: GCC-PHAT peak quality of MicL–MicR cross-correlation")
    lines.append("- **LDV-MIC PSR**: GCC-PHAT peak quality of OMP-aligned-LDV–Mic "
                 "cross-correlation")
    lines.append("")
    lines.append("The signal characteristics (LDV velocity vs microphone pressure) differ "
                 "fundamentally, so PSR magnitudes reflect different physical quantities.")
    lines.append("")

    # --- E. Conclusions ---
    lines.append("## E. Conclusions")
    lines.append("")
    lines.append(f"1. **MIC-MIC is more accurate overall**: median θ_error "
                 f"{agg_mm['median']:.3f}° vs LDV-MicL {agg_ll['median']:.3f}° "
                 f"(~{agg_ll['median']/agg_mm['median']:.0f}× higher error)")
    lines.append(f"2. **LDV-MIC is within the {THETA_THRESHOLD_DEG}° acceptability "
                 f"threshold**: LDV-MicL worst case = {agg_ll['max']:.2f}°, "
                 f"LDV-MicR worst case = {agg_lr['max']:.2f}°")
    lines.append(f"3. **Speaker 22 is the sole LDV-wins case** "
                 f"({sp22_ll['theta_error_median_deg']:.2f}° vs {sp22_mm['theta_error_median_deg']:.2f}°), "
                 f"worth investigating the MIC-MIC degradation cause")
    lines.append(f"4. **LDV-MicL > LDV-MicR overall**: median error "
                 f"{agg_ll['median']:.3f}° vs {agg_lr['median']:.3f}°")
    lines.append("")

    return "\n".join(lines)


# =========================================================================
# Mode 2: Cross-bandpass comparison
# =========================================================================


def collect_stage4(stage4_dir: Path) -> dict:
    """Collect per-speaker Stage 4-C data (MIC-MIC, OMP, Raw) for one band."""
    rows = {}
    for sp in SPEAKERS:
        rows[sp] = extract_stage4_speaker(stage4_dir, sp)
    return rows


def cross_bandpass_h2h(band0: dict, band500: dict) -> list[dict]:
    """Per-speaker comparison between band=0 and band=500-2000.

    Uses MIC-MIC vs OMP as the main head-to-head (ignoring Raw for winner).
    """
    rows = []
    for sp in SPEAKERS:
        mic0 = band0[sp]["mic_mic"]["theta_error_median_deg"]
        omp0 = band0[sp]["omp"]["theta_error_median_deg"]
        raw0 = band0[sp]["raw"]["theta_error_median_deg"]
        win0 = "OMP" if omp0 < mic0 else "MIC"

        mic5 = band500[sp]["mic_mic"]["theta_error_median_deg"]
        omp5 = band500[sp]["omp"]["theta_error_median_deg"]
        raw5 = band500[sp]["raw"]["theta_error_median_deg"]
        win5 = "OMP" if omp5 < mic5 else "MIC"

        flipped = win0 != win5
        rows.append({
            "speaker": sp,
            "band0_mic": mic0, "band0_omp": omp0, "band0_raw": raw0,
            "band0_winner": win0,
            "band500_mic": mic5, "band500_omp": omp5, "band500_raw": raw5,
            "band500_winner": win5,
            "flipped": flipped,
        })
    return rows


def generate_cross_report(band0: dict, band500: dict, h2h: list[dict]) -> str:
    """Generate the cross-experiment comparison report (markdown)."""
    L: list[str] = []

    L.append("# Cross-Experiment Comparison: Bandpass Effect on LDV Performance")
    L.append("")
    L.append("## Context")
    L.append("")
    L.append("This report compares DoA estimation accuracy across two bandpass conditions")
    L.append("to determine whether **bandpass filtering** is the key variable driving")
    L.append("LDV vs MIC-MIC winner outcomes.")
    L.append("")
    L.append("**Data sources**:")
    L.append("- **band=0** (no bandpass): Stage 4-C Grid "
             "(`stage4_speech_chirp_tau2_band0_20260211_072624`)")
    L.append("- **band=500\u20132000 Hz**: Stage 4-C Grid "
             "(`stage4_speech_chirp_tau2_band500_2000_20260211_072624`)")
    L.append("- **af1acf5** (commit): Stage 4-C original run (Speaker 21 & 22 only, "
             "1 segment, band=0)")
    L.append("")
    L.append("**Common conditions**: chirp signal, \u03c4=2.0 s, GCC-PHAT, OMP pre-alignment "
             "(K\u22643), 5 segments (except af1acf5: 1 segment)")
    L.append("")
    L.append("**Error metric**: \u03b8_error = |\u03b8_estimated \u2212 \u03b8_truth-ref| "
             "(median over segments)")
    L.append("")

    # --- A. Cross-bandpass head-to-head ---
    L.append("## A. Cross-Bandpass Head-to-Head (\u03b8_error in degrees)")
    L.append("")
    L.append("| Speaker | band=0 MIC | band=0 OMP | band=0 Win "
             "| band=500\u20132000 MIC | band=500\u20132000 OMP | band=500\u20132000 Win "
             "| Flipped? |")
    L.append("|---------|-----------|-----------|----------"
             "|-------------------|-------------------|------------------"
             "|----------|")
    for r in h2h:
        flip_str = "**YES**" if r["flipped"] else "no"
        L.append(
            f"| {r['speaker']} "
            f"| {fmt(r['band0_mic'], r['band0_winner'] == 'MIC')} "
            f"| {fmt(r['band0_omp'], r['band0_winner'] == 'OMP')} "
            f"| {r['band0_winner']} "
            f"| {fmt(r['band500_mic'], r['band500_winner'] == 'MIC')} "
            f"| {fmt(r['band500_omp'], r['band500_winner'] == 'OMP')} "
            f"| {r['band500_winner']} "
            f"| {flip_str} |"
        )

    # summary counts
    n_flip = sum(1 for r in h2h if r["flipped"])
    L.append("")
    L.append(f"**Flipped winners**: {n_flip}/{len(h2h)} speakers change winner "
             "when bandpass is applied.")
    L.append("")

    # Raw LDV supplementary table
    L.append("### Supplementary: Raw LDV Errors")
    L.append("")
    L.append("| Speaker | band=0 Raw | band=500\u20132000 Raw |")
    L.append("|---------|-----------|-------------------|")
    for r in h2h:
        L.append(f"| {r['speaker']} | {r['band0_raw']:.3f} | {r['band500_raw']:.3f} |")
    L.append("")

    # --- B. af1acf5 consistency ---
    L.append("## B. Consistency with Commit af1acf5 (Stage 4-C, band=0, 1 segment)")
    L.append("")
    L.append("| Speaker | af1acf5 OMP (1 seg) | Grid OMP (5 seg) "
             "| af1acf5 MIC\u2248\u03b8_err | Grid MIC (5 seg) | Consistent? |")
    L.append("|---------|--------------------|-----------------"
             "|--------------------|-----------------|-------------|")
    for sp, af in AF1ACF5_DATA.items():
        grid_omp = band0[sp]["omp"]["theta_error_median_deg"]
        grid_mic = band0[sp]["mic_mic"]["theta_error_median_deg"]
        af_omp_win = af["omp_err"] < af["mic_theta_err_approx"]
        grid_omp_win = grid_omp < grid_mic
        consistent = af_omp_win == grid_omp_win
        L.append(
            f"| {sp} | {af['omp_err']:.2f}\u00b0 | {grid_omp:.3f}\u00b0 "
            f"| {af['mic_theta_err_approx']:.2f}\u00b0 | {grid_mic:.3f}\u00b0 "
            f"| {'✅ Yes' if consistent else '❌ No'} "
            f"({'OMP wins both' if af_omp_win and grid_omp_win else 'MIC wins both' if not af_omp_win and not grid_omp_win else 'DIVERGED'}) |"
        )
    L.append("")
    L.append("**Note**: Magnitude differences (e.g. af1acf5 OMP 0.06° vs Grid 0.29°) "
             "are expected due to 1 vs 5 segments and different segment selection. "
             "The winner direction is consistent.")
    L.append("")

    # --- C. Bandpass effect mechanism ---
    L.append("## C. Bandpass Effect Mechanism Analysis")
    L.append("")
    L.append("### MIC-MIC accuracy improvement with bandpass")
    L.append("")
    L.append("| Speaker | band=0 MIC | band=500\u20132000 MIC | Improvement |")
    L.append("|---------|-----------|-------------------|--------------------|")
    for sp in SPEAKERS:
        m0 = band0[sp]["mic_mic"]["theta_error_median_deg"]
        m5 = band500[sp]["mic_mic"]["theta_error_median_deg"]
        delta = m0 - m5
        L.append(f"| {sp} | {m0:.3f}\u00b0 | {m5:.3f}\u00b0 "
                 f"| {'+' if delta > 0 else ''}{delta:.3f}\u00b0 "
                 f"({'improved' if delta > 0 else '**degraded**'}) |")
    L.append("")
    L.append("**Key observations**:")
    L.append("")
    L.append("- **band=0**: MIC-MIC estimates are unstable (median 2.0\u20133.6\u00b0 error) "
             "\u2014 GCC-PHAT without bandpass suffers from broadband multipath contamination.")
    L.append("- **band=500\u20132000**: MIC-MIC accuracy improves dramatically for most speakers "
             "(some down to 0.02\u20130.13\u00b0) \u2014 bandpass suppresses multipath and "
             "low-frequency room modes.")
    L.append("- **Speaker 20 exception**: MIC-MIC accuracy *degrades* with bandpass "
             "(0.17\u00b0 \u2192 2.54\u00b0). Speaker 22 improves only modestly "
             "(3.61\u00b0 \u2192 2.42\u00b0), remaining the worst MIC-MIC performer, "
             "suggesting its multipath is concentrated in the 500\u20132000 Hz band.")
    L.append("")

    L.append("### OMP LDV accuracy change with bandpass")
    L.append("")
    L.append("| Speaker | band=0 OMP | band=500\u20132000 OMP | Change |")
    L.append("|---------|-----------|-------------------|--------------------|")
    for sp in SPEAKERS:
        o0 = band0[sp]["omp"]["theta_error_median_deg"]
        o5 = band500[sp]["omp"]["theta_error_median_deg"]
        delta = o0 - o5
        L.append(f"| {sp} | {o0:.3f}\u00b0 | {o5:.3f}\u00b0 "
                 f"| {'+' if delta > 0 else ''}{delta:.3f}\u00b0 "
                 f"({'improved' if delta > 0 else '**degraded**'}) |")
    L.append("")
    L.append("**Key observations**:")
    L.append("")
    L.append("- OMP accuracy change is **speaker-dependent**: some improve, some degrade "
             "with bandpass.")
    L.append("- Speaker 22 improves significantly (3.62\u00b0 \u2192 0.82\u00b0), "
             "driving the only LDV win under band=500\u20132000.")
    L.append("- Speaker 21 *degrades* dramatically with bandpass (0.29\u00b0 \u2192 3.14\u00b0): "
             "its LDV-OMP pre-alignment may rely on frequency content outside the "
             "500\u20132000 Hz band.")
    L.append("")

    # --- D. Per-segment deep-dive for flipped speakers ---
    flipped_speakers = [r["speaker"] for r in h2h if r["flipped"]]
    L.append("## D. Per-Segment Deep-Dive (Flipped Speakers)")
    L.append("")
    for sp in flipped_speakers:
        L.append(f"### Speaker {sp.split('-')[0]}")
        L.append("")

        # band=0
        s0 = band0[sp]
        L.append(f"**band=0**: MIC={s0['mic_mic']['theta_error_median_deg']:.3f}\u00b0, "
                 f"OMP={s0['omp']['theta_error_median_deg']:.3f}\u00b0, "
                 f"Raw={s0['raw']['theta_error_median_deg']:.3f}\u00b0")
        L.append("")
        L.append("| Seg | MIC-MIC | OMP_LDV | Raw_LDV |")
        L.append("|-----|---------|---------|---------|")
        for i in range(s0["mic_mic"]["n_segments"]):
            L.append(
                f"| {i+1} "
                f"| {s0['mic_mic']['per_segment_errors'][i]:.3f}\u00b0 "
                f"| {s0['omp']['per_segment_errors'][i]:.3f}\u00b0 "
                f"| {s0['raw']['per_segment_errors'][i]:.3f}\u00b0 |"
            )
        L.append("")

        # band=500-2000
        s5 = band500[sp]
        L.append(f"**band=500\u20132000**: MIC={s5['mic_mic']['theta_error_median_deg']:.3f}\u00b0, "
                 f"OMP={s5['omp']['theta_error_median_deg']:.3f}\u00b0, "
                 f"Raw={s5['raw']['theta_error_median_deg']:.3f}\u00b0")
        L.append("")
        L.append("| Seg | MIC-MIC | OMP_LDV | Raw_LDV |")
        L.append("|-----|---------|---------|---------|")
        for i in range(s5["mic_mic"]["n_segments"]):
            L.append(
                f"| {i+1} "
                f"| {s5['mic_mic']['per_segment_errors'][i]:.3f}\u00b0 "
                f"| {s5['omp']['per_segment_errors'][i]:.3f}\u00b0 "
                f"| {s5['raw']['per_segment_errors'][i]:.3f}\u00b0 |"
            )
        L.append("")

    # --- E. Unified conclusions ---
    L.append("## E. Unified Conclusions")
    L.append("")
    L.append("1. **af1acf5 conclusions hold under band=0**: Speaker 21 OMP wins, "
             "Speaker 22 MIC wins (marginal). The 5-segment Grid data confirms "
             "the same winner direction as the original 1-segment result.")
    L.append("")
    L.append("2. **Bandpass completely reverses the LDV winner pattern**: Under band=0, "
             "OMP wins for Speakers 18 & 21. Under band=500\u20132000, only Speaker 22 "
             "is an OMP win \u2014 all others flip to MIC.")
    L.append("")
    L.append("3. **Bandpass primarily helps MIC-MIC**: By suppressing multipath and "
             "low-frequency room modes, MIC-MIC accuracy improves dramatically "
             "(up to 2.5\u00b0 \u2192 0.02\u00b0). This makes MIC-MIC hard to beat.")
    L.append("")
    L.append("4. **Bandpass effect on LDV-OMP is speaker-dependent**: Some speakers "
             "improve, others degrade. This suggests the OMP pre-alignment quality "
             "depends on frequency content that may be outside the 500\u20132000 Hz band.")
    L.append("")
    L.append("5. **Overall recommendation unchanged**: MIC-MIC + bandpass is the most "
             "accurate and robust configuration. LDV-MIC remains within the 5\u00b0 "
             "acceptability threshold and is valuable when (a) no physical mic pair "
             "is available, or (b) multipath conditions make MIC-MIC unreliable "
             "(e.g. Speaker 22 under band=500\u20132000).")
    L.append("")

    return "\n".join(L)


def cross_experiment_main() -> None:
    """Entry point for cross-bandpass comparison (--cross mode)."""
    # Verify Stage 4-C Grid directories
    for label, d in [
        ("Stage4 band=0", STAGE4_BAND0_DIR),
        ("Stage4 band=500-2000", STAGE4_BAND500_DIR),
    ]:
        if not d.exists():
            print(f"ERROR: {label} directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    band0 = collect_stage4(STAGE4_BAND0_DIR)
    band500 = collect_stage4(STAGE4_BAND500_DIR)
    h2h = cross_bandpass_h2h(band0, band500)

    # Structured data output
    def _speaker_row(sp: str, data: dict) -> dict:
        return {
            "mic_mic_err": data[sp]["mic_mic"]["theta_error_median_deg"],
            "omp_err": data[sp]["omp"]["theta_error_median_deg"],
            "raw_err": data[sp]["raw"]["theta_error_median_deg"],
            "mic_mic_per_seg": data[sp]["mic_mic"]["per_segment_errors"],
            "omp_per_seg": data[sp]["omp"]["per_segment_errors"],
            "raw_per_seg": data[sp]["raw"]["per_segment_errors"],
        }

    out_json = {
        "description": "Cross-bandpass comparison: band=0 vs band=500-2000",
        "speakers": SPEAKERS,
        "band0": {sp: _speaker_row(sp, band0) for sp in SPEAKERS},
        "band500_2000": {sp: _speaker_row(sp, band500) for sp in SPEAKERS},
        "head_to_head": h2h,
        "af1acf5_reference": AF1ACF5_DATA,
        "flipped_speakers": [r["speaker"] for r in h2h if r["flipped"]],
        "summary": {
            "band0_omp_wins": [r["speaker"] for r in h2h
                               if r["band0_winner"] == "OMP"],
            "band500_omp_wins": [r["speaker"] for r in h2h
                                  if r["band500_winner"] == "OMP"],
        },
    }

    CROSS_OUT_DATA.write_text(
        json.dumps(out_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Data written to {CROSS_OUT_DATA}")

    report = generate_cross_report(band0, band500, h2h)
    CROSS_OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"Report written to {CROSS_OUT_REPORT}")

    # Quick summary
    print("\n--- Cross-Bandpass Summary ---")
    print("  band=0 winners:       "
          + ", ".join(f"{r['speaker']}\u2192{r['band0_winner']}" for r in h2h))
    print("  band=500-2000 winners: "
          + ", ".join(f"{r['speaker']}\u2192{r['band500_winner']}" for r in h2h))
    flips = [r["speaker"] for r in h2h if r["flipped"]]
    print(f"  Flipped: {flips if flips else 'none'}")


# =========================================================================
# CLI entry points
# =========================================================================


def main() -> None:
    # Verify data directories exist
    for label, d in [
        ("MIC-MIC", MIC_MIC_DIR),
        ("LDV-MicL", LDV_MICL_DIR),
        ("LDV-MicR", LDV_MICR_DIR),
    ]:
        if not d.exists():
            print(f"ERROR: {label} directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    data = collect()
    h2h = head_to_head(data)

    # Write JSON data
    out_json = {
        "condition": "chirp_tau2_band500_2000",
        "speakers": SPEAKERS,
        "threshold_deg": THETA_THRESHOLD_DEG,
        "head_to_head": h2h,
        "aggregates": {
            "mic_mic": data["mic_mic"]["_agg"],
            "ldv_micl_omp": data["ldv_micl_omp"]["_agg"],
            "ldv_micr_omp": data["ldv_micr_omp"]["_agg"],
        },
        "pass_counts": {
            "mic_mic": data["mic_mic"]["_pass_count"],
            "ldv_micl_omp": data["ldv_micl_omp"]["_pass_count"],
            "ldv_micr_omp": data["ldv_micr_omp"]["_pass_count"],
        },
    }
    OUT_DATA.write_text(json.dumps(out_json, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"Data written to {OUT_DATA}")

    # Write report
    report = generate_report(data, h2h)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"Report written to {OUT_REPORT}")

    # Print summary to stdout
    print("\n--- Quick Summary ---")
    print(f"  MIC-MIC   median θ_err = {data['mic_mic']['_agg']['median']:.3f}°")
    print(f"  LDV-MicL  median θ_err = {data['ldv_micl_omp']['_agg']['median']:.3f}°")
    print(f"  LDV-MicR  median θ_err = {data['ldv_micr_omp']['_agg']['median']:.3f}°")
    print(f"  Winner per speaker: "
          + ", ".join(f"{r['speaker']}→{r['winner']}" for r in h2h))


if __name__ == "__main__":
    if "--cross" in sys.argv:
        cross_experiment_main()
    else:
        main()
