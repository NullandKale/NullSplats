"""Registry of hidden-layer activation functions shared by every MLP-family TileModel."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "leaky_relu": lambda x: F.leaky_relu(x, negative_slope=0.01),
    "gelu": F.gelu,
    "silu": F.silu,
    "tanh": torch.tanh,
    "sine": torch.sin,
    "identity": lambda x: x,
}


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    key = name.strip().lower()
    if key not in ACTIVATIONS:
        raise KeyError(f"Unknown activation: {name!r}. Available: {list_activations()}")
    return ACTIVATIONS[key]


def list_activations() -> list[str]:
    return sorted(ACTIVATIONS)
