"""Structured run artifacts and diagnostic bundles."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import zipfile
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


class RunArtifacts:
    def __init__(self, root: Path, name: str) -> None:
        self.path = root / f"{timestamp()}-{name}"
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "plots").mkdir()
        (self.path / "videos").mkdir()

    def write_json(self, name: str, data: Any) -> Path:
        path = self.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")
        return path

    def append_jsonl(self, name: str, data: Any) -> None:
        with (self.path / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, default=json_default) + "\n")

    def write_csv(self, name: str, rows: list[dict[str, Any]]) -> Path:
        path = self.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def metadata(self) -> None:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
        self.write_json(
            "metadata.json",
            {
                "created_at": datetime.now(UTC).isoformat(),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "git_commit": commit,
                "command": sys.argv,
            },
        )

    def bundle(self) -> Path:
        bundle_path = self.path / "diagnostic-bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in self.path.rglob("*"):
                if path.is_file() and path != bundle_path and path.suffix != ".pt":
                    archive.write(path, path.relative_to(self.path))
        return bundle_path
