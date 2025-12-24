"""Gsplat backend wrapper for the unified trainer interface."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from nullsplats.backend.splat_backends.base import CheckpointCallback, ProgressCallback
from nullsplats.backend.splat_backends.types import TrainerCapabilities, TrainingInput, TrainingOutput
from nullsplats.backend.splat_train import train_scene
from nullsplats.backend.splat_train_config import PreviewPayload, SplatTrainingConfig


class GsplatTrainer:
    name = "gsplat"
    capabilities = TrainerCapabilities(
        live_preview=True,
        supports_unconstrained=True,
        supports_constrained=True,
        requires_colmap=True,
    )

    def prepare(self, inputs: TrainingInput, config: dict[str, Any]) -> None:
        _ = inputs
        _coerce_config(config)

    def train(
        self,
        inputs: TrainingInput,
        config: dict[str, Any],
        *,
        on_progress: ProgressCallback | None = None,
        on_checkpoint: CheckpointCallback | None = None,
        on_preview: Callable[[PreviewPayload], None] | None = None,
    ) -> TrainingOutput:
        gs_config = _coerce_config(config)
        result = train_scene(
            inputs.scene_id,
            gs_config,
            cache_root=inputs.scene_paths.cache_root,
            progress_callback=on_progress,
            checkpoint_callback=on_checkpoint,
            preview_callback=on_preview,
        )
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
        return TrainingOutput(
            primary_path=result.last_checkpoint,
            method=self.name,
            timestamp=timestamp,
            export_format=result.export_format,
            metrics={},
            extra_files=[result.log_path, result.config_path],
        )


def _coerce_config(config: dict | SplatTrainingConfig) -> SplatTrainingConfig:
    if isinstance(config, SplatTrainingConfig):
        return config
    if isinstance(config, dict):
        return SplatTrainingConfig(**config)
    raise TypeError(f"Unsupported config type: {type(config)}")
