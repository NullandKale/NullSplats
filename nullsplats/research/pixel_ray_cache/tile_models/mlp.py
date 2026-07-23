"""Default per-tile network: N_tiles independent MLPs, vectorized via batched weight tensors."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nullsplats.research.pixel_ray_cache.config import TargetSpec
from nullsplats.research.pixel_ray_cache.tile_models.activations import get_activation, list_activations
from nullsplats.research.pixel_ray_cache.tile_models.base import HyperparamSpec, TileModel, split_target_outputs
from nullsplats.research.pixel_ray_cache.tile_models.encodings import build_encoding, list_encodings


class BatchedTileMLP(TileModel):
    """N_tiles independent small MLPs implemented as batched per-layer weight tensors, so the
    whole forward/backward pass for every tile runs as a handful of GPU einsum calls instead of
    N_tiles separate nn.Linear instances (infeasible at N_tiles ~ 2,073,600 for a full
    1920x1080 grid).
    """

    hyperparams = (
        HyperparamSpec("hidden_width", "int", 16, help="Width of every hidden layer."),
        HyperparamSpec("hidden_layers", "int", 1, help="Number of hidden layers (>=1)."),
        HyperparamSpec(
            "hidden_activation", "choice", "leaky_relu", choices=tuple(list_activations()),
            help="Nonlinearity between hidden layers.",
        ),
        HyperparamSpec(
            "encoding", "choice", "none", choices=tuple(list_encodings()),
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
        hidden_width: int = 16,
        hidden_layers: int = 1,
        hidden_activation: str = "leaky_relu",
        encoding: str = "none",
        encoding_freqs: int = 4,
        encoding_features: int = 32,
        encoding_scale: float = 10.0,
        raw_in_dim: int = 6,
    ) -> None:
        super().__init__(n_tiles, target_spec)
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
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

        dims = [in_dim] + [hidden_width] * hidden_layers + [out_dim]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            w = torch.empty(n_tiles, d_in, d_out)
            # Kaiming-uniform per independent sub-network: bound derived from this layer's own
            # fan_in only -- default nn.Linear init doesn't apply here since the leading N_tiles
            # dim is not itself part of any single layer's fan-in/out.
            bound = 1.0 / math.sqrt(d_in)
            nn.init.uniform_(w, -bound, bound)
            b = torch.zeros(n_tiles, d_out)
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(b))

    def forward(self, rays: torch.Tensor) -> dict[str, torch.Tensor]:
        """rays: (N_tiles, B, raw_in_dim) -> {target_name: (N_tiles, B, channels)}."""
        x = self.encoding_module(rays)
        n_layers = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.einsum("tbi,tio->tbo", x, w) + b[:, None, :]
            if i < n_layers - 1:
                x = self._activation_fn(x)
        return split_target_outputs(x, self.target_spec)

    def expand(self, parent_index: torch.Tensor) -> "BatchedTileMLP":
        n_children = int(parent_index.shape[0])
        child = BatchedTileMLP(
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
        device = self.weights[0].device
        idx = parent_index.to(device=device, dtype=torch.long)
        with torch.no_grad():
            for parent_w, child_w in zip(self.weights, child.weights):
                child_w.copy_(parent_w[idx])
            for parent_b, child_b in zip(self.biases, child.biases):
                child_b.copy_(parent_b[idx])
        return child.to(device)
