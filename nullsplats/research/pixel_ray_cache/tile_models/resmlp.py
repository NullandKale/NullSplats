"""Residual per-tile MLP: same batched-weight-tensor vectorization as BatchedTileMLP, but with
skip connections between hidden layers (x = x + activation(Wx+b)) for better gradient flow at
depth. All hidden layers share hidden_width (required for the residual add to type-check).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nullsplats.research.pixel_ray_cache.config import TargetSpec
from nullsplats.research.pixel_ray_cache.tile_models.activations import get_activation, list_activations
from nullsplats.research.pixel_ray_cache.tile_models.base import HyperparamSpec, TileModel, split_target_outputs
from nullsplats.research.pixel_ray_cache.tile_models.encodings import build_encoding, list_encodings


def _make_layer(n_tiles: int, d_in: int, d_out: int) -> tuple[nn.Parameter, nn.Parameter]:
    w = torch.empty(n_tiles, d_in, d_out)
    bound = 1.0 / math.sqrt(d_in)
    nn.init.uniform_(w, -bound, bound)
    b = torch.zeros(n_tiles, d_out)
    return nn.Parameter(w), nn.Parameter(b)


def _apply_layer(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.einsum("tbi,tio->tbo", x, w) + b[:, None, :]


class BatchedTileResMLP(TileModel):
    hyperparams = (
        HyperparamSpec("hidden_width", "int", 32, help="Width of every hidden layer (shared, needed for residual adds)."),
        HyperparamSpec("hidden_layers", "int", 4, help="Number of residual hidden layers (>=2)."),
        HyperparamSpec(
            "hidden_activation", "choice", "relu", choices=tuple(list_activations()),
            help="Nonlinearity applied inside each residual block.",
        ),
        HyperparamSpec(
            "encoding", "choice", "fourier", choices=tuple(list_encodings()),
            help="Input feature encoding applied to the raw 6D ray before the first layer.",
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
        hidden_width: int = 32,
        hidden_layers: int = 4,
        hidden_activation: str = "relu",
        encoding: str = "fourier",
        encoding_freqs: int = 4,
        encoding_features: int = 32,
        encoding_scale: float = 10.0,
        raw_in_dim: int = 6,
    ) -> None:
        super().__init__(n_tiles, target_spec)
        if hidden_layers < 2:
            raise ValueError("hidden_layers must be >= 2 for a residual MLP")
        self.hidden_width = hidden_width
        self.hidden_layers = hidden_layers
        self.hidden_activation = hidden_activation
        self.encoding_name = encoding
        self.encoding_freqs = encoding_freqs
        self.encoding_features = encoding_features
        self.encoding_scale = encoding_scale
        self.raw_in_dim = raw_in_dim

        self._activation_fn = get_activation(hidden_activation)
        self.encoding_module = build_encoding(
            encoding, raw_in_dim, n_freqs=encoding_freqs, n_features=encoding_features, scale=encoding_scale, seed=0
        )
        in_dim = self.encoding_module.out_dim
        out_dim = sum(channels for _, channels, _ in target_spec)

        self.in_weight, self.in_bias = _make_layer(n_tiles, in_dim, hidden_width)
        self.res_weights = nn.ParameterList()
        self.res_biases = nn.ParameterList()
        for _ in range(hidden_layers - 1):
            w, b = _make_layer(n_tiles, hidden_width, hidden_width)
            self.res_weights.append(w)
            self.res_biases.append(b)
        self.out_weight, self.out_bias = _make_layer(n_tiles, hidden_width, out_dim)

    def forward(self, rays: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.encoding_module(rays)
        x = self._activation_fn(_apply_layer(x, self.in_weight, self.in_bias))
        for w, b in zip(self.res_weights, self.res_biases):
            x = x + self._activation_fn(_apply_layer(x, w, b))
        x = _apply_layer(x, self.out_weight, self.out_bias)
        return split_target_outputs(x, self.target_spec)

    def expand(self, parent_index: torch.Tensor) -> "BatchedTileResMLP":
        n_children = int(parent_index.shape[0])
        child = BatchedTileResMLP(
            n_children,
            self.target_spec,
            hidden_width=self.hidden_width,
            hidden_layers=self.hidden_layers,
            hidden_activation=self.hidden_activation,
            encoding=self.encoding_name,
            encoding_freqs=self.encoding_freqs,
            encoding_features=self.encoding_features,
            encoding_scale=self.encoding_scale,
            raw_in_dim=self.raw_in_dim,
        )
        device = self.in_weight.device
        idx = parent_index.to(device=device, dtype=torch.long)
        with torch.no_grad():
            child.in_weight.copy_(self.in_weight[idx])
            child.in_bias.copy_(self.in_bias[idx])
            for parent_w, child_w in zip(self.res_weights, child.res_weights):
                child_w.copy_(parent_w[idx])
            for parent_b, child_b in zip(self.res_biases, child.res_biases):
                child_b.copy_(parent_b[idx])
            child.out_weight.copy_(self.out_weight[idx])
            child.out_bias.copy_(self.out_bias[idx])
        return child.to(device)
