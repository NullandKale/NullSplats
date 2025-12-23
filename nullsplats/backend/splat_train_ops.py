"""Optimization helpers for splat training."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from nullsplats.backend.splat_train_config import SplatTrainingConfig
from nullsplats.util.tooling_paths import default_cuda_path

gsplat = None  # set after toolkit configuration
rasterization = None  # set after toolkit configuration
_SSIM_AVAILABLE = True

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure  # type: ignore
except Exception:
    StructuralSimilarityIndexMeasure = None  # type: ignore
    _SSIM_AVAILABLE = False

_SSIM_METRICS: dict[str, StructuralSimilarityIndexMeasure] = {}


def ssim_available() -> bool:
    return _SSIM_AVAILABLE


def configure_cuda_toolkit(cuda_path: Optional[str]) -> None:
    """Force the CUDA toolkit used by torch/cpp_extension to the preferred install."""

    preferred = Path(cuda_path) if cuda_path else Path(default_cuda_path())
    cuda_home = preferred if preferred.exists() else Path(os.environ.get("CUDA_HOME", ""))
    if not cuda_home.exists():
        return

    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["CUDA_PATH"] = str(cuda_home)

    # Prepend CUDA bin/lib paths so nvcc and DLLs come from the selected toolkit.
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for extra in [cuda_home / "bin", cuda_home / "lib" / "x64"]:
        if extra.exists():
            extra_str = str(extra)
            if extra_str not in path_parts:
                path_parts.insert(0, extra_str)
    os.environ["PATH"] = os.pathsep.join(path_parts)

    try:
        from torch.utils import cpp_extension as ce
    except Exception:
        return

    ce.CUDA_HOME = str(cuda_home)

    def _fixed_cuda_home() -> str:
        return str(cuda_home)

    # Ensure helper lookups return the configured path without requiring external env configuration.
    ce._find_cuda_home = _fixed_cuda_home  # type: ignore[attr-defined]
    ce._join_cuda_home = lambda *paths: str(Path(cuda_home, *paths))  # type: ignore[attr-defined]


def get_rasterization():
    """Import gsplat after CUDA toolkit selection and return rasterization."""

    global gsplat, rasterization  # noqa: PLW0603
    if gsplat is not None and rasterization is not None:
        return rasterization
    import gsplat as _gsplat  # local import to honor configured CUDA toolkit
    from gsplat.rendering import rasterization as _rasterization

    gsplat = _gsplat
    rasterization = _rasterization
    return rasterization


def build_splat_optimizers(
    splats_param: torch.nn.ParameterDict,
    config: SplatTrainingConfig,
    sh_rest_lr: float,
) -> dict[str, torch.optim.Optimizer]:
    lr_scale = math.sqrt(config.batch_size)
    betas = (1 - config.batch_size * (1 - 0.9), 1 - config.batch_size * (1 - 0.999))
    eps = 1e-15 / lr_scale
    lrs: dict[str, float] = {
        "means": config.means_lr * lr_scale,
        "scales": config.scales_lr * lr_scale,
        "opacities": config.opacities_lr * lr_scale,
        "quats": config.quats_lr * lr_scale,
        "sh0": config.sh_lr * lr_scale,
        "shN": sh_rest_lr * lr_scale,
    }
    if "features" in splats_param:
        lrs["features"] = config.sh_lr * lr_scale
    optimizers: dict[str, torch.optim.Optimizer] = {}
    for name, param in splats_param.items():
        lr = lrs.get(name)
        if lr is None:
            continue
        optimizers[name] = torch.optim.Adam(
            [{"params": param, "lr": lr, "name": name}],
            eps=eps,
            betas=betas,
            fused=True,
        )
    return optimizers


def compute_means_decay_gamma(config: SplatTrainingConfig) -> Optional[float]:
    if config.iterations <= 0 or config.lr_final_scale <= 0 or config.lr_final_scale >= 1.0:
        return None
    return config.lr_final_scale ** (1.0 / float(config.iterations))


def initialize_parameters(
    means: torch.Tensor,
    colors: torch.Tensor,
    config: SplatTrainingConfig,
    *,
    with_features: bool,
    feature_dim: int,
) -> torch.nn.ParameterDict:
    scales = initial_scales(means, config.init_scale, config.min_scale, config.max_scale)
    opacities = torch.full((means.shape[0],), config.opacity_bias, device=means.device, dtype=torch.float32)
    scales_log = torch.log(scales)
    opacities_logit = torch.logit(opacities.clamp(1e-4, 1.0 - 1e-4))
    sh_channels = max(1, (config.sh_degree + 1) ** 2)
    sh0 = torch.zeros((means.shape[0], 1, 3), device=means.device, dtype=torch.float32)
    sh0[:, 0, :] = colors
    shN_count = max(0, sh_channels - 1)
    shN = torch.zeros((means.shape[0], shN_count, 3), device=means.device, dtype=torch.float32)
    quats = identity_quats(means.shape[0], means.device)
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales_log),
            "opacities": torch.nn.Parameter(opacities_logit),
            "sh0": torch.nn.Parameter(sh0),
            "shN": torch.nn.Parameter(shN),
            "quats": torch.nn.Parameter(quats),
            **(
                {"features": torch.nn.Parameter(torch.rand((means.shape[0], feature_dim), device=means.device))}
                if with_features
                else {}
            ),
        }
    )


def initial_scales(means: torch.Tensor, scale_multiplier: float, min_scale: float, max_scale: float) -> torch.Tensor:
    # Match gsplat simple_trainer: set scale to log of mean distance to 3 nearest neighbors.
    if means.numel() == 0:
        return torch.empty_like(means)
    dists2 = torch.cdist(means, means, compute_mode="donot_use_mm_for_euclid_dist") ** 2
    # Ignore self-distance, take 3 nearest neighbors.
    topk = torch.topk(dists2, k=min(4, dists2.shape[1]), largest=False)
    dist_avg = torch.sqrt(topk.values[:, 1:].mean(dim=-1, keepdim=True))  # (N, 1)
    return dist_avg.repeat(1, 3) * scale_multiplier


def identity_quats(count: int, device: torch.device) -> torch.Tensor:
    quats = torch.zeros((count, 4), device=device, dtype=torch.float32)
    quats[:, 0] = 1.0
    return quats


def sample_frames(frames: list, batch_size: int) -> list:
    if batch_size >= len(frames):
        return list(frames)
    indices = np.random.choice(len(frames), size=batch_size, replace=False)
    return [frames[int(i)] for i in indices]


def append_log(log_path: Path, payload: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def export_splats(
    splats_param: torch.nn.ParameterDict,
    export_path: Path,
    *,
    max_points: int,
    fmt: str,
) -> Path:
    get_rasterization()
    fmt_clean = "splat" if fmt.lower().strip() == "splat" else "ply"
    target_path = (
        export_path
        if export_path.suffix.lower().lstrip(".") == fmt_clean
        else export_path.with_suffix(f".{fmt_clean}")
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    scales_log = splats_param["scales"].detach().cpu()
    opacities_logit = splats_param["opacities"].detach().cpu()
    opacities = torch.sigmoid(opacities_logit)
    means = splats_param["means"].detach().cpu()
    colors = torch.cat([splats_param["sh0"], splats_param["shN"]], dim=1).detach().cpu()
    quats_cpu = F.normalize(splats_param["quats"].detach().cpu(), dim=1)

    if max_points > 0 and means.shape[0] > max_points:
        keep = torch.topk(opacities, max_points).indices
        means = means[keep]
        scales_log = scales_log[keep]
        opacities = opacities[keep]
        colors = colors[keep]
        quats_cpu = quats_cpu[keep]

    sh0 = colors[:, :1, :]
    shN = colors[:, 1:, :]
    gsplat.export_splats(
        means,
        scales_log,
        quats_cpu,
        opacities,
        sh0,
        shN,
        format=fmt_clean,
        save_to=str(target_path),
    )
    return target_path


def ssim_loss(img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
    """Compute SSIM using torchmetrics (expects NHWC input)."""

    if not _SSIM_AVAILABLE or StructuralSimilarityIndexMeasure is None:
        raise RuntimeError("SSIM not available; torchmetrics failed to import.")
    device = img_a.device
    key = str(device)
    metric = _SSIM_METRICS.get(key)
    if metric is None:
        metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        _SSIM_METRICS[key] = metric
    a = img_a.permute(0, 3, 1, 2)
    b = img_b.permute(0, 3, 1, 2)
    return metric(a, b)
