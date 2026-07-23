"""Registry of input encodings for ray-based TileModels: map the raw 6D ray (or any raw_dim
vector) to a higher-dimensional feature vector before the first layer.

Purely a function of the input -- no per-tile parameters -- so TileModel.expand() implementations
never need special-case logic for these: reconstructing an encoding with the same class + kwargs
(including a fixed seed for the random one) reproduces it exactly, which is what every concrete
model's expand() here does by just re-passing its own stored encoding kwargs to its constructor.
"""

from __future__ import annotations

import abc
import inspect
import math

import torch
import torch.nn as nn


class Encoding(nn.Module, abc.ABC):
    def __init__(self, raw_dim: int) -> None:
        super().__init__()
        self.raw_dim = raw_dim

    @property
    @abc.abstractmethod
    def out_dim(self) -> int: ...

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class IdentityEncoding(Encoding):
    """No-op: out_dim == raw_dim."""

    @property
    def out_dim(self) -> int:
        return self.raw_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class FourierEncoding(Encoding):
    """Raw x concatenated with sin/cos bands at 2^k frequencies per component, k in
    [0, n_freqs) -- the classic NeRF-style positional encoding."""

    def __init__(self, raw_dim: int, n_freqs: int = 4) -> None:
        super().__init__(raw_dim)
        self.n_freqs = n_freqs

    @property
    def out_dim(self) -> int:
        return self.raw_dim * (1 + 2 * self.n_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_freqs <= 0:
            return x
        freqs = 2.0 ** torch.arange(self.n_freqs, device=x.device, dtype=x.dtype)
        scaled = x.unsqueeze(-1) * freqs  # (..., raw_dim, n_freqs)
        bands = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(-2)
        return torch.cat([x, bands], dim=-1)


class GaussianFourierEncoding(Encoding):
    """Random Fourier features (Tancik et al. 2020): a FIXED (non-trainable) random Gaussian
    projection. `seed` is part of the spec (not just a construction-time accident) specifically
    so a cascade child reconstructs the identical basis as its parent when expand() re-invokes
    this same constructor with the same kwargs.
    """

    def __init__(self, raw_dim: int, n_features: int = 32, scale: float = 10.0, seed: int = 0) -> None:
        super().__init__(raw_dim)
        self.n_features = n_features
        self.scale = scale
        self.seed = seed
        generator = torch.Generator().manual_seed(seed)
        basis = torch.randn(raw_dim, n_features, generator=generator) * scale
        self.register_buffer("basis", basis)

    @property
    def out_dim(self) -> int:
        return 2 * self.n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = 2.0 * math.pi * (x @ self.basis)
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


ENCODINGS: dict[str, type[Encoding]] = {
    "none": IdentityEncoding,
    "fourier": FourierEncoding,
    "gaussian_fourier": GaussianFourierEncoding,
}


def build_encoding(name: str, raw_dim: int, **kwargs) -> Encoding:
    """Constructs ENCODINGS[name](raw_dim, **kwargs), silently dropping any kwargs the chosen
    encoding's constructor doesn't accept -- lets a caller (e.g. an MLP model) always pass the
    union of every encoding's possible kwargs without needing to know which ones apply."""
    key = name.strip().lower()
    if key not in ENCODINGS:
        raise KeyError(f"Unknown encoding: {name!r}. Available: {list_encodings()}")
    cls = ENCODINGS[key]
    accepted_params = set(inspect.signature(cls.__init__).parameters) - {"self", "raw_dim"}
    filtered = {k: v for k, v in kwargs.items() if k in accepted_params}
    return cls(raw_dim, **filtered)


def list_encodings() -> list[str]:
    return sorted(ENCODINGS)
