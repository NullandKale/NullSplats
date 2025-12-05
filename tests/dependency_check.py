"""Dependency gate for tests/run_checks.bat."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys


def _require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        sys.stderr.write(f"Missing dependency: {name}\n")
        sys.exit(1)


def _find_executable(name: str, extra_dirs: list[Path]) -> str | None:
    candidates = []
    for directory in extra_dirs:
        candidates.append(directory / name)
        candidates.append(directory / f"{name}.exe")
        candidates.append(directory / f"{name}.bat")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    return None


def _require_executable(name: str, *, hint: str = "", extra_dirs: list[Path] | None = None) -> None:
    found = _find_executable(name, extra_dirs or [])
    if found is None:
        extra = f" ({hint})" if hint else ""
        sys.stderr.write(f"Missing executable: {name}{extra}\n")
        sys.exit(1)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    tool_dirs = [
        repo_root / "tools" / "COLMAP-3.7-windows-cuda" / "bin",
        repo_root / "tools" / "colmap",
        repo_root / "tools" / "colmap" / "bin",
    ]
    _require_module("numpy")
    _require_module("PIL")
    _require_module("torch")
    _require_module("packaging")
    _require_module("ninja")
    import torch  # noqa: WPS433

    if not torch.cuda.is_available():
        sys.stderr.write("CUDA not available for torch; training requires a CUDA build of PyTorch.\n")
        sys.exit(1)
    _require_executable("ffmpeg")
    _require_executable("ffprobe")
    _require_executable(
        "colmap",
        hint="Place colmap.exe under tools\\COLMAP-3.7-windows-cuda\\bin or provide another real colmap binary.",
        extra_dirs=tool_dirs,
    )
    print("Dependencies present: numpy, pillow, torch (CUDA), ffmpeg, ffprobe, colmap")


if __name__ == "__main__":
    main()
