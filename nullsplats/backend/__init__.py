"""Backend helpers for NullSplats."""

from nullsplats.backend.io_cache import (
    SceneId,
    ScenePaths,
    ensure_scene_dirs,
    delete_scene,
    load_metadata,
    save_metadata,
)
from nullsplats.backend.video_frames import (
    FrameScore,
    ExtractionResult,
    FFMPEGVideoReader,
    auto_select_best,
    extract_frames,
    load_cached_frames,
    persist_selection,
)
from nullsplats.backend.sfm_pipeline import SfmConfig, SfmResult, run_sfm
from nullsplats.backend.splat_train import SplatTrainingConfig, TrainingResult, train_scene

__all__ = [
    "SceneId",
    "ScenePaths",
    "ensure_scene_dirs",
    "delete_scene",
    "save_metadata",
    "load_metadata",
    "FrameScore",
    "ExtractionResult",
    "FFMPEGVideoReader",
    "auto_select_best",
    "extract_frames",
    "load_cached_frames",
    "persist_selection",
    "SfmConfig",
    "SfmResult",
    "run_sfm",
    "SplatTrainingConfig",
    "TrainingResult",
    "train_scene",
]
