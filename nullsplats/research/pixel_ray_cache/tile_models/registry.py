"""Registry of available per-tile network architectures. Mirrors
nullsplats.backend.splat_backends.registry -- adding a new architecture is one line here."""

from __future__ import annotations

from nullsplats.research.pixel_ray_cache.tile_models.base import HyperparamSpec, TileModel
from nullsplats.research.pixel_ray_cache.tile_models.linear import BatchedTileLinear
from nullsplats.research.pixel_ray_cache.tile_models.mlp import BatchedTileMLP
from nullsplats.research.pixel_ray_cache.tile_models.resmlp import BatchedTileResMLP
from nullsplats.research.pixel_ray_cache.tile_models.siren import BatchedTileSiren

_TILE_MODELS: dict[str, type[TileModel]] = {
    "mlp": BatchedTileMLP,
    "siren": BatchedTileSiren,
    "resmlp": BatchedTileResMLP,
    "linear": BatchedTileLinear,
}


def get_tile_model_cls(name: str) -> type[TileModel]:
    key = name.strip().lower()
    if key not in _TILE_MODELS:
        raise KeyError(f"Unknown tile model: {name!r}. Available: {list_tile_models()}")
    return _TILE_MODELS[key]


def list_tile_models() -> list[str]:
    return list(_TILE_MODELS.keys())


def get_hyperparams(name: str) -> tuple[HyperparamSpec, ...]:
    return get_tile_model_cls(name).hyperparams


def estimate_params_per_tile(name: str, target_spec, **model_kwargs) -> int:
    """Per-tile parameter count for this architecture+hyperparams, without materializing the
    full N-tile model: every architecture allocates its per-layer weight tensors shaped
    [n_tiles, ...], so constructing a throwaway probe with n_tiles=1 gives the exact per-tile
    count in O(1) time/memory regardless of the real N -- lets a caller (e.g. viewer.py) estimate
    total memory for a target tile count (params_per_tile * n_tiles) before ever allocating the
    real model, which is what actually OOMs for an oversized architecture/hyperparameter choice.
    """
    model_cls = get_tile_model_cls(name)
    probe = model_cls(1, target_spec, **model_kwargs)
    return sum(p.numel() for p in probe.parameters())
