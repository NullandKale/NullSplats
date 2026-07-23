"""Registry of elementwise loss functions for color/depth targets -- a "method" hyperparameter
independent of model architecture. Each returns a per-element tensor (same shape as its inputs,
no reduction applied); train._composite_loss applies the shared masked-mean reduction on top, so
masking/cascade-padding behavior is identical no matter which loss is chosen here.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

ELEMENTWISE_LOSSES: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "mse": lambda pred, target: (pred - target) ** 2,
    "l1": lambda pred, target: (pred - target).abs(),
    "huber": lambda pred, target: F.huber_loss(pred, target, reduction="none"),
}


def get_loss(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    key = name.strip().lower()
    if key not in ELEMENTWISE_LOSSES:
        raise KeyError(f"Unknown loss: {name!r}. Available: {list_losses()}")
    return ELEMENTWISE_LOSSES[key]


def list_losses() -> list[str]:
    return sorted(ELEMENTWISE_LOSSES)
