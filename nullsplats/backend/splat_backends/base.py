"""Backend interface for splat training methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from nullsplats.backend.splat_train_config import PreviewPayload
from nullsplats.backend.splat_backends.types import TrainerCapabilities, TrainingInput, TrainingOutput


ProgressCallback = Callable[[int, int, float], None]
CheckpointCallback = Callable[[int, Path], None]


class SplatTrainer(Protocol):
    name: str
    capabilities: TrainerCapabilities

    def prepare(self, inputs: TrainingInput, config: dict[str, Any]) -> None:
        """Validate inputs and configuration before running training."""

    def train(
        self,
        inputs: TrainingInput,
        config: dict[str, Any],
        *,
        on_progress: ProgressCallback | None = None,
        on_checkpoint: CheckpointCallback | None = None,
        on_preview: Callable[[PreviewPayload], None] | None = None,
    ) -> TrainingOutput:
        """Run training/inference and return the primary output file."""
