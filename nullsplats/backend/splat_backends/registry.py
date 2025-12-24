"""Registry of available splat training backends."""

from __future__ import annotations

from nullsplats.backend.splat_backends.base import SplatTrainer
from nullsplats.backend.splat_backends.depth_anything3_trainer import DepthAnything3Trainer
from nullsplats.backend.splat_backends.gsplat_trainer import GsplatTrainer


_TRAINERS: dict[str, SplatTrainer] = {
    "gsplat": GsplatTrainer(),
    "depth_anything_3": DepthAnything3Trainer(),
}


def get_trainer(name: str) -> SplatTrainer:
    key = name.strip().lower()
    if key not in _TRAINERS:
        raise KeyError(f"Unknown trainer: {name}")
    return _TRAINERS[key]


def list_trainers() -> list[SplatTrainer]:
    return list(_TRAINERS.values())
