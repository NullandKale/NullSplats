"""Shared path helpers for bundled tools and CUDA runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def app_root() -> Path:
    """Resolve application root (frozen dist or repo)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def default_colmap_path() -> str:
    """Prefer a bundled COLMAP; fall back to repo-local tools copy."""
    root = app_root()
    bundled = root / "tools" / "colmap" / "COLMAP.bat"
    if bundled.exists():
        return str(bundled)
    repo_default = Path(r"C:\Users\alec\source\python\NullSplats\tools\colmap\COLMAP.bat")
    return str(repo_default)


def default_cuda_path() -> str:
    """Prefer bundled CUDA runtime, then env, then the system install."""
    bundled = app_root() / "cuda"
    if bundled.exists():
        return str(bundled)
    env_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if env_path:
        return env_path
    preferred = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
    return str(preferred) if preferred.exists() else ""
