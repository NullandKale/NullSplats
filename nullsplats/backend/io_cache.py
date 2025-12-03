"""Cache path helpers and metadata utilities for NullSplats.

All paths are derived from a validated ``SceneId`` so callers cannot
accidentally create unexpected directories. No environment variables are
consulted; callers must pass explicit paths when deviating from defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from nullsplats.util.scene_id import SceneId


def _coerce_scene_id(value: str | SceneId) -> SceneId:
    """Normalize user input into a SceneId."""
    if isinstance(value, SceneId):
        return value
    return SceneId(str(value))


@dataclass(frozen=True)
class ScenePaths:
    """Compute all cache locations for a given scene."""

    scene_id: SceneId
    cache_root: Path = Path("cache")

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_id", _coerce_scene_id(self.scene_id))
        object.__setattr__(self, "cache_root", Path(self.cache_root))

    @property
    def inputs_root(self) -> Path:
        return self.cache_root / "inputs" / str(self.scene_id)

    @property
    def source_dir(self) -> Path:
        return self.inputs_root / "source"

    @property
    def frames_all_dir(self) -> Path:
        return self.inputs_root / "frames_all"

    @property
    def frames_selected_dir(self) -> Path:
        return self.inputs_root / "frames_selected"

    @property
    def metadata_path(self) -> Path:
        return self.inputs_root / "metadata.json"

    @property
    def outputs_root(self) -> Path:
        return self.cache_root / "outputs" / str(self.scene_id)

    @property
    def sfm_dir(self) -> Path:
        return self.outputs_root / "sfm"

    @property
    def splats_dir(self) -> Path:
        return self.outputs_root / "splats"

    @property
    def renders_dir(self) -> Path:
        return self.outputs_root / "renders"

    def iter_required_dirs(self) -> Iterable[Path]:
        """Return directories that must exist for the scene."""
        return (
            self.inputs_root,
            self.source_dir,
            self.frames_all_dir,
            self.frames_selected_dir,
            self.outputs_root,
            self.sfm_dir,
            self.splats_dir,
            self.renders_dir,
        )

    def __repr__(self) -> str:
        return (
            f"ScenePaths(scene_id={self.scene_id!r}, "
            f"cache_root={str(self.cache_root)!r})"
        )


def ensure_scene_dirs(scene_id: str | SceneId, cache_root: str | Path = "cache") -> ScenePaths:
    """Create cache directories for the provided scene."""
    paths = ScenePaths(scene_id=_coerce_scene_id(scene_id), cache_root=Path(cache_root))
    for directory in paths.iter_required_dirs():
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def save_metadata(
    scene_id: str | SceneId, data: Dict[str, Any], cache_root: str | Path = "cache"
) -> Path:
    """Persist metadata.json for the scene.

    The function writes full JSON content without truncation and refuses to
    accept non-dictionary payloads to avoid ambiguous metadata structures.
    """
    if not isinstance(data, dict):
        raise TypeError("Metadata payload must be a dictionary.")
    paths = ensure_scene_dirs(scene_id, cache_root)
    metadata_json = json.dumps(data, indent=2, sort_keys=True)
    metadata_file = paths.metadata_path
    metadata_file.write_text(metadata_json + "\n", encoding="utf-8")
    return metadata_file


def load_metadata(scene_id: str | SceneId, cache_root: str | Path = "cache") -> Dict[str, Any]:
    """Load metadata.json for the scene, raising if it is missing."""
    paths = ScenePaths(scene_id=_coerce_scene_id(scene_id), cache_root=Path(cache_root))
    metadata_file = paths.metadata_path
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_file}")
    with metadata_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "ScenePaths",
    "SceneId",
    "ensure_scene_dirs",
    "save_metadata",
    "load_metadata",
]
