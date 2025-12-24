"""Splat training backends."""

from nullsplats.backend.splat_backends.base import SplatTrainer
from nullsplats.backend.splat_backends.dispatch import train_with_trainer
from nullsplats.backend.splat_backends.registry import get_trainer, list_trainers
from nullsplats.backend.splat_backends.types import TrainerCapabilities, TrainingInput, TrainingOutput

__all__ = [
    "SplatTrainer",
    "train_with_trainer",
    "get_trainer",
    "list_trainers",
    "TrainerCapabilities",
    "TrainingInput",
    "TrainingOutput",
]
