"""Shared types for splat training backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nullsplats.backend.colmap_io import ColmapData
from nullsplats.backend.io_cache import ScenePaths
from nullsplats.util.scene_id import SceneId


@dataclass(frozen=True)
class TrainingInput:
    scene_id: SceneId
    scene_paths: ScenePaths
    frames_dir: Path
    colmap_dir: Path
    images: list[Path]
    colmap: ColmapData
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrainingOutput:
    primary_path: Path
    method: str
    timestamp: str
    export_format: str
    metrics: dict[str, Any]
    extra_files: list[Path]


@dataclass(frozen=True)
class TrainerCapabilities:
    live_preview: bool
    supports_unconstrained: bool
    supports_constrained: bool
    requires_colmap: bool
