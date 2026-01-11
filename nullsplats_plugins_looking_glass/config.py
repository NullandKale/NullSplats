"""Environment-driven configuration for Looking Glass preview."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _get_env_float(name: str, default: float, *, min_val: float | None = None, max_val: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except Exception:
        return default
    if min_val is not None:
        val = max(min_val, val)
    if max_val is not None:
        val = min(max_val, val)
    return val


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class LKGConfig:
    display_index: int
    quilt_scale: float
    max_fps: float
    depthiness: float
    focus: float
    fov: float
    viewcone: float
    zoom: float
    views_override: Optional[int]
    debug_quilt: bool


def load_config() -> LKGConfig:
    baseline = None
    raw_baseline = os.getenv("NULLSPLATS_LKG_BASELINE")
    if raw_baseline is not None:
        try:
            baseline = float(raw_baseline)
        except Exception:
            baseline = None
    if baseline is not None:
        baseline = max(0.0, min(5.0, baseline))
    depthiness = _get_env_float("NULLSPLATS_LKG_DEPTHINESS", 1.0, min_val=0.0, max_val=5.0)
    if baseline is not None:
        depthiness = baseline
    return LKGConfig(
        display_index=_get_env_int("NULLSPLATS_LKG_DISPLAY_INDEX", -1),
        quilt_scale=_get_env_float("NULLSPLATS_LKG_QUILT_SCALE", 1.0, min_val=0.25, max_val=2.0),
        max_fps=_get_env_float("NULLSPLATS_LKG_MAX_FPS", 15.0, min_val=1.0, max_val=120.0),
        depthiness=depthiness,
        focus=_get_env_float("NULLSPLATS_LKG_FOCUS", 2.0, min_val=-10.0, max_val=10.0),
        fov=_get_env_float("NULLSPLATS_LKG_FOV", 14.0, min_val=5.0, max_val=120.0),
        viewcone=_get_env_float("NULLSPLATS_LKG_VIEWCONE", 40.0, min_val=0.0, max_val=89.0),
        zoom=_get_env_float("NULLSPLATS_LKG_ZOOM", 1.0, min_val=0.1, max_val=4.0),
        views_override=_get_env_int("NULLSPLATS_LKG_VIEWS", 0) or None,
        debug_quilt=False,
    )
