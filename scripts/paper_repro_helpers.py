from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def git_state(repo_root: Path) -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()
        except Exception:
            return None

    status = _run("status", "--short")
    return {
        "git_head": _run("rev-parse", "HEAD"),
        "git_branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
        "captured_at": datetime.now().isoformat(timespec="seconds"),
    }


def build_file_manifest(paths: Iterable[Path], *, root: Path | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted({p.resolve() for p in paths}):
        rel_path = path.relative_to(root).as_posix() if root and path.is_relative_to(root) else str(path)
        items.append(
            {
                "path": str(path),
                "rel_path": rel_path,
                "sha256": sha256_file(path),
            }
        )
    return items


def sync_outputs(outputs: dict[str, Path], sync_dir: Path) -> list[dict[str, str]]:
    sync_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    for dest_name, src_path in outputs.items():
        dest_path = sync_dir / dest_name
        shutil.copy2(src_path, dest_path)
        copied.append({"src": str(src_path), "dest": str(dest_path)})
    return copied
