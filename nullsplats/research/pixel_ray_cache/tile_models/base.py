"""Abstract interface every per-tile network architecture must implement."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from nullsplats.research.pixel_ray_cache.config import TargetSpec


@dataclass(frozen=True)
class HyperparamSpec:
    """Self-describing schema for one constructor keyword argument of a TileModel subclass
    (beyond n_tiles/target_spec, which every architecture takes implicitly and which aren't
    user-facing hyperparameters). Lets a generic caller -- the CLI's --model-kwargs JSON merge,
    or the GUI's dynamically-built per-architecture widget panel -- construct correct controls
    and validation without hardcoding anything per architecture.
    """

    name: str
    kind: str  # "int" | "float" | "choice" | "bool"
    default: Any
    choices: tuple[str, ...] = field(default_factory=tuple)  # only meaningful for kind="choice"
    help: str = ""


class TileModel(nn.Module, abc.ABC):
    """N_tiles independent per-tile sub-networks, addressed by a single ray input.

    Concrete subclasses are constructed as `Cls(n_tiles, target_spec, **model_kwargs)` and must
    hold their own vectorized/batched weights internally -- nothing else in the pipeline
    (data.py, tiles.py, train.py, preview.py, sweep.py) inspects the internals, only this
    input/output contract. This is what makes the per-tile architecture swappable: a wider/
    deeper MLP, one with a positional encoding, or a fundamentally different architecture only
    needs to satisfy forward()'s shape contract (and declare its own `hyperparams` schema).
    """

    n_tiles: int
    target_spec: tuple[TargetSpec, ...]
    hyperparams: ClassVar[tuple[HyperparamSpec, ...]] = ()
    """Schema for this class's own __init__ kwargs (excluding n_tiles/target_spec/raw_in_dim).
    Every concrete subclass should override this with its actual tunable knobs."""

    def __init__(self, n_tiles: int, target_spec: tuple[TargetSpec, ...]) -> None:
        super().__init__()
        self.n_tiles = n_tiles
        self.target_spec = tuple(target_spec)

    @abc.abstractmethod
    def forward(self, rays: torch.Tensor) -> dict[str, torch.Tensor]:
        """rays: (N_tiles, B, 6) [ox,oy,oz,dx,dy,dz] -> {target_name: (N_tiles, B, channels)},
        each already passed through its target's activation (see config.target_spec_for)."""
        raise NotImplementedError

    @abc.abstractmethod
    def expand(self, parent_index: torch.Tensor) -> "TileModel":
        """Return a new instance of this same architecture/hyperparameters with
        n_tiles = len(parent_index), whose per-group parameters are warm-started by copying this
        model's own parameters at row parent_index[i] for each new group i (parent_index values
        are indices into this model's own n_tiles). Used by cascade.py's coarse-to-fine trainer
        to initialize a finer level's networks from the coarser level they were trained at,
        rather than from scratch."""
        raise NotImplementedError


def split_target_outputs(x: torch.Tensor, target_spec: tuple[TargetSpec, ...]) -> dict[str, torch.Tensor]:
    """Split a model's raw final-layer output (..., sum(channels)) into named per-target
    tensors, each passed through its own activation (sigmoid/softplus/unit-normalize) -- shared
    by every concrete TileModel so the output convention stays identical regardless of
    architecture; only this function needs to know what "sigmoid"/"softplus"/"unit" mean.
    """
    outputs: dict[str, torch.Tensor] = {}
    offset = 0
    for name, channels, activation in target_spec:
        chunk = x[..., offset : offset + channels]
        offset += channels
        if activation == "sigmoid":
            chunk = torch.sigmoid(chunk)
        elif activation == "softplus":
            chunk = F.softplus(chunk)
        elif activation == "unit":
            chunk = F.normalize(torch.tanh(chunk), dim=-1, eps=1e-8)
        else:
            raise ValueError(f"Unknown activation: {activation!r}")
        outputs[name] = chunk
    return outputs
