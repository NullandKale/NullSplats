"""Trivial per-tile model: a single linear layer, no hidden layers. Fast/cheap baseline, and the
minimal example of the TileModel interface (useful as a reference when adding new architectures).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nullsplats.research.pixel_ray_cache.config import TargetSpec
from nullsplats.research.pixel_ray_cache.tile_models.base import HyperparamSpec, TileModel, split_target_outputs
from nullsplats.research.pixel_ray_cache.tile_models.encodings import build_encoding, list_encodings


class BatchedTileLinear(TileModel):
    hyperparams = (
        HyperparamSpec(
            "encoding", "choice", "none", choices=tuple(list_encodings()),
            help="Input feature encoding applied to the raw 6D ray before the linear layer.",
        ),
        HyperparamSpec("encoding_freqs", "int", 4, help="Frequency bands, only used by encoding='fourier'."),
        HyperparamSpec("encoding_features", "int", 32, help="Feature count, only used by encoding='gaussian_fourier'."),
        HyperparamSpec("encoding_scale", "float", 10.0, help="Basis scale, only used by encoding='gaussian_fourier'."),
    )

    def __init__(
        self,
        n_tiles: int,
        target_spec: tuple[TargetSpec, ...],
        *,
        encoding: str = "none",
        encoding_freqs: int = 4,
        encoding_features: int = 32,
        encoding_scale: float = 10.0,
        raw_in_dim: int = 6,
    ) -> None:
        super().__init__(n_tiles, target_spec)
        self.encoding_name = encoding
        self.encoding_freqs = encoding_freqs
        self.encoding_features = encoding_features
        self.encoding_scale = encoding_scale
        self.raw_in_dim = raw_in_dim

        self.encoding_module = build_encoding(
            encoding, raw_in_dim, n_freqs=encoding_freqs, n_features=encoding_features, scale=encoding_scale, seed=0
        )
        in_dim = self.encoding_module.out_dim
        out_dim = sum(channels for _, channels, _ in target_spec)
        w = torch.empty(n_tiles, in_dim, out_dim)
        bound = 1.0 / math.sqrt(in_dim)
        nn.init.uniform_(w, -bound, bound)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_tiles, out_dim))

    def forward(self, rays: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.encoding_module(rays)
        x = torch.einsum("tbi,tio->tbo", x, self.weight) + self.bias[:, None, :]
        return split_target_outputs(x, self.target_spec)

    def expand(self, parent_index: torch.Tensor) -> "BatchedTileLinear":
        n_children = int(parent_index.shape[0])
        child = BatchedTileLinear(
            n_children,
            self.target_spec,
            encoding=self.encoding_name,
            encoding_freqs=self.encoding_freqs,
            encoding_features=self.encoding_features,
            encoding_scale=self.encoding_scale,
            raw_in_dim=self.raw_in_dim,
        )
        device = self.weight.device
        idx = parent_index.to(device=device, dtype=torch.long)
        with torch.no_grad():
            child.weight.copy_(self.weight[idx])
            child.bias.copy_(self.bias[idx])
        return child.to(device)
