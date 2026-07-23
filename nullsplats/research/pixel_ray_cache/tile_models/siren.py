"""SIREN (Sitzmann et al. 2020): sine-activation MLP with a specialized initialization scheme.

The sine nonlinearity, combined with this init, provides an implicit high-frequency basis
similar in spirit to a Fourier positional encoding -- so unlike BatchedTileMLP, SIREN typically
does not need an explicit `encoding` on its raw ray input to fit high-frequency detail.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nullsplats.research.pixel_ray_cache.config import TargetSpec
from nullsplats.research.pixel_ray_cache.tile_models.base import HyperparamSpec, TileModel, split_target_outputs


class BatchedTileSiren(TileModel):
    hyperparams = (
        HyperparamSpec("hidden_width", "int", 16, help="Width of every hidden layer."),
        HyperparamSpec("hidden_layers", "int", 2, help="Number of hidden layers (>=1)."),
        HyperparamSpec(
            "omega_0", "float", 30.0,
            help="Frequency scale inside sin(omega_0 * (Wx+b)); SIREN paper default is 30.",
        ),
    )

    def __init__(
        self,
        n_tiles: int,
        target_spec: tuple[TargetSpec, ...],
        *,
        hidden_width: int = 16,
        hidden_layers: int = 2,
        omega_0: float = 30.0,
        raw_in_dim: int = 6,
    ) -> None:
        super().__init__(n_tiles, target_spec)
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        self.hidden_width = hidden_width
        self.hidden_layers = hidden_layers
        self.omega_0 = omega_0
        self.raw_in_dim = raw_in_dim

        out_dim = sum(channels for _, channels, _ in target_spec)
        dims = [raw_in_dim] + [hidden_width] * hidden_layers + [out_dim]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            w = torch.empty(n_tiles, d_in, d_out)
            if i == 0:
                # First layer: U(-1/fan_in, 1/fan_in) per the SIREN paper (no omega_0 factor).
                bound = 1.0 / d_in
            else:
                bound = math.sqrt(6.0 / d_in) / omega_0
            nn.init.uniform_(w, -bound, bound)
            b = torch.empty(n_tiles, d_out)
            nn.init.uniform_(b, -bound, bound)
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(b))

    def forward(self, rays: torch.Tensor) -> dict[str, torch.Tensor]:
        x = rays
        n_layers = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.einsum("tbi,tio->tbo", x, w) + b[:, None, :]
            if i < n_layers - 1:
                x = torch.sin(self.omega_0 * x)
        return split_target_outputs(x, self.target_spec)

    def expand(self, parent_index: torch.Tensor) -> "BatchedTileSiren":
        n_children = int(parent_index.shape[0])
        child = BatchedTileSiren(
            n_children,
            self.target_spec,
            hidden_width=self.hidden_width,
            hidden_layers=self.hidden_layers,
            omega_0=self.omega_0,
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
