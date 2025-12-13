"""Application state container for NullSplats."""

from __future__ import annotations

from typing import List, Optional

from nullsplats.backend.scene_manager import SceneManager, SceneRegistry, SceneStatus
from nullsplats.util.config import AppConfig
from nullsplats.util.scene_id import SceneId


class AppState:
    """Global state shared across the UI."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.scene_manager = SceneManager(cache_root=self.config.cache_root)
        # Keep a direct handle for compatibility (tests/UI)
        self.scene_registry = self.scene_manager.registry
        # Global training image target resolution (smallest side, in px) used across tabs.
        self.training_image_target_px: int = 1080
        self.training_image_resample: str = "lanczos"

    @property
    def current_scene_id(self) -> Optional[SceneId]:
        return self.scene_manager.current_scene

    @current_scene_id.setter
    def current_scene_id(self, value: Optional[SceneId | str]) -> None:
        self.scene_manager.current_scene = SceneId(str(value)) if value is not None else None

    def refresh_scene_status(self) -> List[SceneStatus]:
        """Re-scan the cache and return all scene statuses."""
        return self.scene_manager.list_scenes()

    def set_current_scene(self, scene_id: Optional[str | SceneId]) -> Optional[SceneId]:
        """Update the active scene, returning the normalized SceneId or None."""
        return self.scene_manager.set_current_scene(scene_id)


__all__ = ["AppState", "SceneManager", "SceneRegistry", "SceneStatus"]
