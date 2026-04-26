#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from paper_repro_helpers import build_file_manifest, git_state, write_json


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Reproduce the current paper bundle end-to-end.")
    ap.add_argument(
        "--data_root",
        type=str,
        default=str(repo_root / "dataset" / "0223"),
        help="Dataset root containing the canonical 0223 WAV folders.",
    )
    ap.add_argument("--out_dir", type=str, default="", help="Output directory (defaults to results/paper_repro_<timestamp>/).")
    ap.add_argument(
        "--only",
        type=str,
        default="all",
        choices=("all", "table1", "spatial_score", "jammer"),
        help="Only regenerate a single paper asset.",
    )
    ap.add_argument("--skip_pdf", action="store_true", help="Generate data assets only; skip LaTeX compilation.")
    return ap.parse_args()


def _run_logged(cmd: list[str], *, cwd: Path, log_path: Path, commands: list[str]) -> None:
    commands.append(f"(cd {shlex.quote(str(cwd))} && {shlex.join(cmd)})")
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {shlex.join(cmd)}\nSee log: {log_path}")


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _asset_paths(data_root: Path, manifest: dict[str, Any], asset_key: str) -> list[Path]:
    return [(data_root / rel).resolve() for rel in manifest["assets"][asset_key].get("data_files", [])]


def _validate_files(files: list[Path], *, label: str) -> None:
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {label} files:\n- " + "\n- ".join(missing))


def _compile_paper(paper_dir: Path, run_dir: Path, commands: list[str]) -> Path:
    log_path = run_dir / "build.log"
    steps = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ]
    for i, cmd in enumerate(steps, start=1):
        _run_logged(cmd, cwd=paper_dir, log_path=run_dir / f"build_step_{i}.log", commands=commands)
    # Concatenate logs into a single build log for easier auditing.
    combined = []
    for i in range(1, len(steps) + 1):
        combined.append((run_dir / f"build_step_{i}.log").read_text(encoding="utf-8"))
    log_path.write_text("\n\n".join(combined), encoding="utf-8")

    pdf_path = paper_dir / "main.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Expected compiled PDF at {pdf_path}")
    target = run_dir / "paper_main.pdf"
    shutil.copy2(pdf_path, target)
    return target


def main() -> None:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    paper_dir = repo_root / "paper"
    generated_dir = paper_dir / "generated"
    manifest_path = paper_dir / "repro_asset_manifest.json"
    manifest = _load_manifest(manifest_path)

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"--data_root does not exist: {data_root}")

    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "results" / f"paper_repro_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    generated_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    selected_assets = ["table1", "spatial_score", "jammer_curve"] if args.only == "all" else [args.only if args.only != "jammer" else "jammer_curve"]

    repo_files = [repo_root / rel for asset in selected_assets for rel in manifest["assets"][asset].get("repo_files", [])]
    _validate_files(repo_files, label="repo")

    data_files = [p for asset in selected_assets for p in _asset_paths(data_root, manifest, asset)]
    _validate_files(data_files, label="data")

    commands: list[str] = ["#!/usr/bin/env bash", "set -euo pipefail"]
    artifact_index: dict[str, Any] = {}

    if "table1" in selected_assets:
        table_out = out_dir / "table1"
        cmd = [
            "python",
            "scripts/generate_paper_table1.py",
            "--data_root",
            str(data_root),
            "--out_dir",
            str(table_out),
            "--sync_dir",
            str(generated_dir),
            "--chirp_override_json",
            str(repo_root / "paper" / "table1_chirp_override.json"),
        ]
        _run_logged(cmd, cwd=repo_root, log_path=out_dir / "logs" / "table1.log", commands=commands)
        artifact_index["table1"] = {
            "run_dir": str(table_out),
            "generated_files": [
                str(generated_dir / "table1_latex.tex"),
                str(generated_dir / "table1_values.json"),
                str(generated_dir / "table1_subset_manifest.json"),
            ],
        }

    if "spatial_score" in selected_assets:
        spatial_files = _asset_paths(data_root, manifest, "spatial_score")
        spatial_out = out_dir / "spatial_score"
        cmd = [
            "python",
            "scripts/generate_spatial_score_figure.py",
            "--data_root",
            str(data_root),
            "--ldv_wav",
            str(spatial_files[0]),
            "--micl_wav",
            str(spatial_files[1]),
            "--micr_wav",
            str(spatial_files[2]),
            "--out_dir",
            str(spatial_out),
            "--sync_dir",
            str(generated_dir),
        ]
        _run_logged(cmd, cwd=repo_root, log_path=out_dir / "logs" / "spatial_score.log", commands=commands)
        artifact_index["spatial_score"] = {
            "run_dir": str(spatial_out),
            "generated_files": [
                str(generated_dir / "spatial_score_curves.dat"),
                str(generated_dir / "spatial_score_meta.json"),
            ],
        }

    if "jammer_curve" in selected_assets:
        jammer_files = _asset_paths(data_root, manifest, "jammer_curve")
        jammer_params = manifest["assets"]["jammer_curve"].get("params", {})
        jammer_out = out_dir / "jammer_curve"
        cmd = [
            "python",
            "scripts/generate_jammer_curve_sim.py",
            "--data_root",
            str(data_root),
            "--target_ldv_wav",
            str(jammer_files[0]),
            "--target_micl_wav",
            str(jammer_files[1]),
            "--target_micr_wav",
            str(jammer_files[2]),
            "--jammer_micl_wav",
            str(jammer_files[3]),
            "--jammer_micr_wav",
            str(jammer_files[4]),
            "--speaker_key",
            str(jammer_params.get("speaker_key", "19")),
            "--out_dir",
            str(jammer_out),
            "--sync_dir",
            str(generated_dir),
        ]
        _run_logged(cmd, cwd=repo_root, log_path=out_dir / "logs" / "jammer_curve.log", commands=commands)
        artifact_index["jammer_curve"] = {
            "run_dir": str(jammer_out),
            "generated_files": [
                str(generated_dir / "jammer_resilience_curve_sim.dat"),
                str(generated_dir / "jammer_resilience_curve_sim_meta.json"),
            ],
        }

    generated_requirements = [
        generated_dir / "table1_latex.tex",
        generated_dir / "spatial_score_curves.dat",
        generated_dir / "jammer_resilience_curve_sim.dat",
    ]
    pdf_path = None
    if not args.skip_pdf:
        _validate_files(generated_requirements, label="generated paper asset")
        pdf_path = _compile_paper(paper_dir, out_dir, commands)

    dataset_manifest = {
        "data_root": str(data_root),
        "selected_assets": selected_assets,
        "files": build_file_manifest(data_files, root=data_root),
    }
    run_manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selected_assets": selected_assets,
        "data_root": str(data_root),
        "repo_manifest": str(manifest_path),
        "git": git_state(repo_root),
        "pdf_path": str(pdf_path) if pdf_path else None,
    }

    write_json(out_dir / "dataset_manifest.json", dataset_manifest)
    write_json(out_dir / "run_manifest.json", run_manifest)
    write_json(out_dir / "artifact_index.json", artifact_index)
    write_json(
        out_dir / "code_state.json",
        {
            **git_state(repo_root),
            "script": str(Path(__file__).resolve()),
        },
    )
    (out_dir / "commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")

    print(f"Paper reproduction completed: {out_dir}")
    if pdf_path:
        print(f"Compiled PDF: {pdf_path}")


if __name__ == "__main__":
    main()
