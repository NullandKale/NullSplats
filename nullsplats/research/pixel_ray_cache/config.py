"""Single source of truth for one Pixel Ray Cache run's hyperparameters.

Both cli.py (single fixed-hyperparameter run) and sweep.py (Optuna search over many runs) build
on this one dataclass, so there is no parallel hyperparameter plumbing to keep in sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TargetSpec = tuple[str, int, str]
"""(target_name, channels, activation) -- e.g. ("color", 3, "sigmoid")."""

_TARGET_DEFS: dict[str, TargetSpec] = {
    "color": ("color", 3, "sigmoid"),
    "depth": ("depth", 1, "softplus"),
    "normal": ("normal", 3, "unit"),
}
_TARGET_ORDER = ("color", "depth", "normal")


def target_spec_for(targets: tuple[str, ...]) -> tuple[TargetSpec, ...]:
    """Build the ordered target_spec for a set of enabled target names."""
    unknown = set(targets) - _TARGET_DEFS.keys()
    if unknown:
        raise ValueError(f"Unknown target(s): {sorted(unknown)}")
    if "color" not in targets:
        raise ValueError("targets must include 'color'")
    if "normal" in targets and "depth" not in targets:
        raise ValueError("'normal' target requires 'depth' to also be enabled")
    return tuple(_TARGET_DEFS[name] for name in _TARGET_ORDER if name in targets)


@dataclass(frozen=True)
class PixelRayCacheConfig:
    scene: str
    cache_root: Path = Path("cache")
    targets: tuple[str, ...] = ("color",)
    tile_model: str = "mlp"
    tile_model_kwargs: dict[str, Any] = field(default_factory=lambda: {"hidden_width": 16, "hidden_layers": 1})
    lr: float = 1e-3
    epochs: int = 20
    tile_crop: tuple[int, int, int, int] | None = None
    cascade: bool = False
    cascade_levels: int = 32
    optimizer: str = "adam"
    lr_schedule: str = "constant"
    color_loss: str = "mse"
    depth_loss: str = "l1"
    depth_loss_weight: float = 0.1
    normal_loss_weight: float = 0.05
    oob_loss_weight: float = 0.0
    oob_translation_scale: float | None = None
    normalize_poses: bool = False
    images_per_step: int = 1
    da3_pretrained: str = "depth-anything/DA3NESTED-GIANT-LARGE"
    da3_chunk_size: int = 4
    device: str = "cuda:0"
    seed: int = 0
    preview_every: int = 5
    out_dir: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_root", Path(self.cache_root))
        if self.out_dir is not None:
            object.__setattr__(self, "out_dir", Path(self.out_dir))
        target_spec_for(self.targets)  # raises on invalid/unknown/inconsistent targets

    @property
    def target_spec(self) -> tuple[TargetSpec, ...]:
        return target_spec_for(self.targets)

    def resolve_out_dir(self) -> Path:
        if self.out_dir is not None:
            return self.out_dir
        return self.cache_root / "outputs" / self.scene / "pixel_ray_cache"
