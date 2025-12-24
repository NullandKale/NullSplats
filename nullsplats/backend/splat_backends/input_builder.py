"""Build shared training inputs for splat backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nullsplats.backend.io_cache import ScenePaths, ensure_scene_dirs, load_metadata
from nullsplats.backend.splat_backends.types import ColmapData, TrainingInput
from nullsplats.backend.colmap_io import load_colmap_data
from nullsplats.util.scene_id import SceneId


def build_training_input(scene_id: str | SceneId, *, cache_root: str | Path = "cache") -> TrainingInput:
    normalized_scene = SceneId(str(scene_id))
    paths = ensure_scene_dirs(normalized_scene, cache_root=cache_root)
    colmap = load_colmap_data(paths)
    frames_dir = paths.frames_selected_dir
    images = _resolve_frame_paths(frames_dir, colmap)
    metadata: dict[str, Any]
    try:
        metadata = load_metadata(normalized_scene, cache_root=cache_root)
    except FileNotFoundError:
        metadata = {}
    return TrainingInput(
        scene_id=normalized_scene,
        scene_paths=paths,
        frames_dir=frames_dir,
        colmap_dir=paths.sfm_dir,
        images=images,
        colmap=colmap,
        metadata=metadata,
    )


def _resolve_frame_paths(frames_dir: Path, colmap: ColmapData) -> list[Path]:
    ordered_ids = sorted(colmap.images.keys())
    images = []
    for image_id in ordered_ids:
        entry = colmap.images[image_id]
        image_path = frames_dir / entry.name
        if not image_path.exists():
            image_path = frames_dir / Path(entry.name).name
        if not image_path.exists():
            raise FileNotFoundError(f"Image {entry.name} not found under {frames_dir}")
        images.append(image_path)
    if not images:
        raise FileNotFoundError(f"No frames found under {frames_dir}")
    return images
