#!/usr/bin/env python
"""
Compare two student runs for mic-only vs LDV+MIC observation ablation.

This script is intentionally lightweight:
- It reads each run's test_windows.jsonl and computes pooled + per-speaker metrics.
- It reports whether LDV+MIC student provides a measurable benefit over mic-only student.

CLI (plan-locked)
-----------------
python -u scripts/compare_ablation_runs.py \\
  --run_a results/ablation_student_ldv_mic_<ts> \\
  --run_b results/ablation_student_mic_only_<ts> \\
  --out_dir results/ablation_compare_<YYYYMMDD_HHMMSS>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def configure_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head_and_dirty() -> tuple[str | None, bool | None]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() != ""
        return head, dirty
    except Exception:
        return None, None


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_code_state(out_dir: Path, script_path: Path) -> None:
    head, dirty = git_head_and_dirty()
    payload = {
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "git_head": head,
        "dirty": dirty,
        "timestamp": datetime.now().isoformat(),
    }
    write_json(out_dir / "code_state.json", payload)


def summarize(vals: list[float]) -> dict[str, float]:
    x = np.asarray(vals, dtype=np.float64)
    if x.size == 0:
        return {"count": 0, "median": float("nan"), "p90": float("nan"), "p95": float("nan"), "fail_rate_gt5deg": float("nan")}
    return {
        "count": int(x.size),
        "median": float(np.quantile(x, 0.5)),
        "p90": float(np.quantile(x, 0.9)),
        "p95": float(np.quantile(x, 0.95)),
        "fail_rate_gt5deg": float(np.mean(x > 5.0)),
    }


def frac_improve(old: float, new: float) -> float:
    if not np.isfinite(old) or old == 0.0:
        return float("nan")
    return float((old - new) / old)


def load_test_windows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "test_windows.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing test_windows.jsonl: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty test_windows.jsonl: {path}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", type=str, required=True, help="LDV+MIC student run directory")
    ap.add_argument("--run_b", type=str, required=True, help="MIC-only student run directory")
    ap.add_argument("--teacher_a", type=str, default=None, help="Optional teacher_trajectories.npz for run_a")
    ap.add_argument("--teacher_b", type=str, default=None, help="Optional teacher_trajectories.npz for run_b")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    run_a = Path(args.run_a).expanduser().resolve()
    run_b = Path(args.run_b).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    teacher_a = None if args.teacher_a is None else Path(args.teacher_a).expanduser().resolve()
    teacher_b = None if args.teacher_b is None else Path(args.teacher_b).expanduser().resolve()

    configure_logging(out_dir)
    write_code_state(out_dir, Path(__file__).resolve())

    rows_a = load_test_windows(run_a)
    rows_b = load_test_windows(run_b)

    # Join by (speaker_id, center_sec)
    key_a = {(str(r["speaker_id"]), float(r["center_sec"])): r for r in rows_a}
    key_b = {(str(r["speaker_id"]), float(r["center_sec"])): r for r in rows_b}
    keys = sorted(set(key_a.keys()) & set(key_b.keys()))
    if not keys:
        raise ValueError("No overlapping test windows between run_a and run_b")

    errs_a = [float(key_a[k]["student"]["theta_error_ref_deg"]) for k in keys]
    errs_b = [float(key_b[k]["student"]["theta_error_ref_deg"]) for k in keys]

    pooled_a = summarize(errs_a)
    pooled_b = summarize(errs_b)

    # Per-speaker
    per_speaker: dict[str, Any] = {}
    for sp in sorted({k[0] for k in keys}):
        ks = [k for k in keys if k[0] == sp]
        per_speaker[sp] = {
            "run_a": summarize([float(key_a[k]["student"]["theta_error_ref_deg"]) for k in ks]),
            "run_b": summarize([float(key_b[k]["student"]["theta_error_ref_deg"]) for k in ks]),
        }

    p95_imp = frac_improve(float(pooled_b["p95"]), float(pooled_a["p95"]))
    fail_imp = frac_improve(float(pooled_b["fail_rate_gt5deg"]), float(pooled_a["fail_rate_gt5deg"]))

    acceptance = {
        "p95_ldv_mic_better_by_ge_10pct": bool(p95_imp >= 0.10),
        "fail_rate_ldv_mic_better_by_ge_20pct": bool(fail_imp >= 0.20),
        "overall_pass": bool((p95_imp >= 0.10) and (fail_imp >= 0.20)),
        "computed": {"p95_improvement_frac": p95_imp, "fail_rate_improvement_frac": fail_imp},
    }

    teacher_actions_check = None
    if (teacher_a is not None) or (teacher_b is not None):
        if teacher_a is None or teacher_b is None:
            raise ValueError("Provide both --teacher_a and --teacher_b, or neither")
        if not teacher_a.exists():
            raise FileNotFoundError(f"Missing teacher_a: {teacher_a}")
        if not teacher_b.exists():
            raise FileNotFoundError(f"Missing teacher_b: {teacher_b}")
        a1 = np.load(str(teacher_a), allow_pickle=True)["actions"]
        a2 = np.load(str(teacher_b), allow_pickle=True)["actions"]
        same = bool(np.array_equal(a1, a2))
        teacher_actions_check = {
            "teacher_a": str(teacher_a),
            "teacher_b": str(teacher_b),
            "actions_identical": same,
            "diff_count": int(np.sum(a1 != a2)),
        }

    summary = {
        "generated": datetime.now().isoformat(),
        "run_dir": str(out_dir),
        "inputs": {
            "run_a": str(run_a),
            "run_b": str(run_b),
            "sha256": {
                "run_a/test_windows.jsonl": sha256_file(run_a / "test_windows.jsonl"),
                "run_b/test_windows.jsonl": sha256_file(run_b / "test_windows.jsonl"),
            },
        },
        "pooled": {"run_a": pooled_a, "run_b": pooled_b},
        "per_speaker": per_speaker,
        "acceptance": acceptance,
        "teacher_actions_check": teacher_actions_check,
    }
    write_json(out_dir / "summary.json", summary)

    # Markdown report
    lines: list[str] = []
    lines.append("# Mic-only Ablation Comparison\n\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n\n")
    lines.append(f"Run A (LDV+MIC student): `{run_a.as_posix()}`\n\n")
    lines.append(f"Run B (MIC-only student): `{run_b.as_posix()}`\n\n")
    lines.append("## Acceptance (Claim 2)\n\n")
    lines.append(f"- p95 improvement (A vs B): {p95_imp:.3f} (>= 0.10)\n")
    lines.append(f"- fail-rate improvement (A vs B): {fail_imp:.3f} (>= 0.20)\n")
    lines.append(f"- OVERALL: {'PASS' if acceptance['overall_pass'] else 'FAIL'}\n\n")
    if teacher_actions_check is not None:
        lines.append("## Teacher action identity check\n\n")
        lines.append(f"- actions_identical: {teacher_actions_check['actions_identical']}\n")
        lines.append(f"- diff_count: {teacher_actions_check['diff_count']}\n\n")
    lines.append("## Pooled test metrics (vs chirp reference)\n\n")
    lines.append("| Run | count | median | p90 | p95 | fail_rate(>5°) |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |\n")
    lines.append(
        f"| A (LDV+MIC) | {pooled_a['count']} | {pooled_a['median']:.3f} | {pooled_a['p90']:.3f} | {pooled_a['p95']:.3f} | {pooled_a['fail_rate_gt5deg']:.3f} |\n"
    )
    lines.append(
        f"| B (MIC-only) | {pooled_b['count']} | {pooled_b['median']:.3f} | {pooled_b['p90']:.3f} | {pooled_b['p95']:.3f} | {pooled_b['fail_rate_gt5deg']:.3f} |\n"
    )
    lines.append("\n## Per-speaker p95 / fail-rate (A vs B)\n\n")
    lines.append("| speaker | A p95 | A fail | B p95 | B fail |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: |\n")
    for sp, m in per_speaker.items():
        a = m["run_a"]
        b = m["run_b"]
        lines.append(f"| {sp} | {a['p95']:.3f} | {a['fail_rate_gt5deg']:.3f} | {b['p95']:.3f} | {b['fail_rate_gt5deg']:.3f} |\n")

    (out_dir / "ablation_report.md").write_text("".join(lines), encoding="utf-8")
    logger.info("Wrote ablation_report.md")
    logger.info("Acceptance OVERALL: %s", "PASS" if acceptance["overall_pass"] else "FAIL")


if __name__ == "__main__":
    main()
