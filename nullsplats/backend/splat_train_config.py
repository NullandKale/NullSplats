"""Training configuration models for gsplat optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from nullsplats.backend.io_cache import ScenePaths
from nullsplats.util.scene_id import SceneId

ProgressCallback = Callable[[int, int, float], None]
CheckpointCallback = Callable[[int, Path], None]
PreviewCallback = Callable[["PreviewPayload"], None]


@dataclass(frozen=True)
class SplatTrainingConfig:
    """Training hyperparameters for gsplat optimization."""

    cuda_toolkit_path: str = ""
    iterations: int = 3000
    snapshot_interval: int = 7000
    device: str = "cuda:0"
    export_format: str = "ply"
    max_points: int = 0
    image_downscale: int = 4
    batch_size: int = 1
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh_lr: float = 2.5e-3
    ssim_weight: float = 0.2
    lr_final_scale: float = 0.01
    pose_opt: bool = False
    pose_opt_lr: float = 1e-5
    pose_opt_reg: float = 1e-6
    pose_noise: float = 0.0
    app_opt: bool = False
    app_embed_dim: int = 16
    app_feature_dim: int = 32
    app_opt_lr: float = 1e-3
    app_opt_reg: float = 1e-6
    densify_start: int = 500
    densify_interval: int = 100
    densify_max_points: int = 2_000_000
    densify_opacity_threshold: float = 0.005
    densify_scale_threshold: float = 0.01
    densify_scale_multiplier: float = 0.6
    densify_position_noise: float = 0.5
    prune_opacity_threshold: float = 0.005
    prune_scale_threshold: float = 0.1
    init_scale: float = 1.0
    min_scale: float = 1e-4
    max_scale: float = 0.2
    opacity_bias: float = 0.1
    opacity_reg: float = 0.0
    random_background: bool = False
    loss_l1_weight: float = 1.0
    seed: int = 42
    preview_interval_seconds: float = 1.0
    preview_min_iters: int = 100
    max_preview_points: int = 0


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a training run."""

    scene_id: SceneId
    paths: ScenePaths
    iterations: int
    last_checkpoint: Path
    export_format: str
    log_path: Path
    config_path: Path


@dataclass(frozen=True)
class PreviewPayload:
    """Lightweight splat payload for in-memory previews."""

    iteration: int
    means: torch.Tensor
    scales_log: torch.Tensor
    quats_wxyz: torch.Tensor
    opacities: torch.Tensor
    sh_dc: torch.Tensor


@dataclass(frozen=True)
class FrameRecord:
    """Parsed COLMAP frame with camera pose and intrinsics."""

    index: int
    name: str
    image_path: Path
    camtoworld: torch.Tensor  # (4, 4)
    K: torch.Tensor  # (3, 3)
    width: int
    height: int
    image: torch.Tensor  # (H, W, 3) on CPU
