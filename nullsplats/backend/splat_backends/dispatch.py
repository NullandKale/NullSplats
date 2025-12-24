"""Dispatch helper for training using a selected backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from nullsplats.backend.splat_backends.base import CheckpointCallback, ProgressCallback
from nullsplats.backend.splat_backends.input_builder import build_training_input
from nullsplats.backend.splat_backends.registry import get_trainer
from nullsplats.backend.splat_backends.types import TrainingOutput
from nullsplats.backend.splat_train_config import PreviewPayload
from nullsplats.util.scene_id import SceneId


def train_with_trainer(
    scene_id: str | SceneId,
    trainer_name: str,
    config: dict[str, Any],
    *,
    cache_root: str | Path = "cache",
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    preview_callback: Callable[[PreviewPayload], None] | None = None,
) -> TrainingOutput:
    trainer = get_trainer(trainer_name)
    inputs = build_training_input(scene_id, cache_root=cache_root)
    trainer.prepare(inputs, config)
    return trainer.train(
        inputs,
        config,
        on_progress=progress_callback,
        on_checkpoint=checkpoint_callback,
        on_preview=preview_callback,
    )
