"""Application state container for NullSplats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

from nullsplats.backend.io_cache import ScenePaths
from nullsplats.util.config import AppConfig
from nullsplats.util.scene_id import SceneId


@dataclass(frozen=True)
class SceneStatus:
    """Snapshot of what data exists for a scene."""

    scene_id: SceneId
    has_inputs: bool
    has_sfm: bool
    has_splats: bool
    has_renders: bool


class SceneRegistry:
    """Discover scenes on disk and report which assets are present."""

    def __init__(self, cache_root: str | Path = "cache") -> None:
        self.cache_root = Path(cache_root)

    def list_scenes(self) -> List[SceneStatus]:
        """Return all scenes found under cache/inputs or cache/outputs."""
        scene_ids = self._discover_scene_ids()
        return [self._build_status(scene_id) for scene_id in sorted(scene_ids, key=str)]

    def _discover_scene_ids(self) -> Set[SceneId]:
        scenes: Set[SceneId] = set()
        for candidate in self._dirnames(self.cache_root / "inputs"):
            scenes.add(SceneId(candidate))
        for candidate in self._dirnames(self.cache_root / "outputs"):
            scenes.add(SceneId(candidate))
        return scenes

    def _build_status(self, scene_id: SceneId) -> SceneStatus:
        paths = ScenePaths(scene_id, cache_root=self.cache_root)
        has_inputs = (
            self._has_files(paths.frames_all_dir)
            or self._has_files(paths.frames_selected_dir)
            or paths.metadata_path.exists()
        )
        has_sfm = self._has_files(paths.sfm_dir)
        has_splats = self._has_files(paths.splats_dir)
        has_renders = self._has_files(paths.renders_dir)
        return SceneStatus(
            scene_id=scene_id,
            has_inputs=has_inputs,
            has_sfm=has_sfm,
            has_splats=has_splats,
            has_renders=has_renders,
        )

    @staticmethod
    def _dirnames(path: Path) -> Set[str]:
        if not path.exists():
            return set()
        return {item.name for item in path.iterdir() if item.is_dir()}

    @staticmethod
    def _has_files(path: Path) -> bool:
        return path.exists() and any(path.iterdir())


class AppState:
    """Global state shared across the UI."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.scene_registry = SceneRegistry(cache_root=self.config.cache_root)
        self.current_scene_id: Optional[SceneId] = None

    def refresh_scene_status(self) -> List[SceneStatus]:
        """Re-scan the cache and return all scene statuses."""
        return self.scene_registry.list_scenes()

    def set_current_scene(self, scene_id: Optional[str | SceneId]) -> Optional[SceneId]:
        """Update the active scene, returning the normalized SceneId or None."""
        if scene_id is None:
            self.current_scene_id = None
            return None
        self.current_scene_id = SceneId(str(scene_id))
        return self.current_scene_id


__all__ = ["AppState", "SceneRegistry", "SceneStatus"]
