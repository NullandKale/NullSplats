"""CUDA-only Gaussian splat training using gsplat.

This module consumes COLMAP outputs (cameras.txt, images.txt, points3D) plus the
selected training frames and optimizes Gaussian parameters with torch + gsplat
rasterization. It exports checkpoints as .ply and .splat without any mock paths
or fallbacks. CUDA is required; CPU execution is rejected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import random
import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure

from nullsplats.backend.io_cache import ScenePaths, ensure_scene_dirs
from nullsplats.backend.gs_utils import AppearanceOptModule, CameraOptModule, rgb_to_sh, set_random_seed
from nullsplats.util.logging import get_logger
from nullsplats.util.scene_id import SceneId
from gsplat.strategy import DefaultStrategy


logger = get_logger("splat_train")
ProgressCallback = Callable[[int, int, float], None]
CheckpointCallback = Callable[[int, Path], None]
gsplat = None  # set after toolkit configuration
rasterization = None  # set after toolkit configuration


def _configure_cuda_toolkit(cuda_path: Optional[str]) -> None:
    """Force the CUDA toolkit used by torch/cpp_extension to the preferred install."""

    preferred = Path(cuda_path) if cuda_path else Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
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


def _import_gsplat() -> None:
    """Import gsplat after CUDA toolkit selection and cache the module globals."""

    global gsplat, rasterization  # noqa: PLW0603
    if gsplat is not None and rasterization is not None:
        return
    import gsplat as _gsplat  # local import to honor configured CUDA toolkit
    from gsplat.rendering import rasterization as _rasterization

    gsplat = _gsplat
    rasterization = _rasterization


@dataclass(frozen=True)
class SplatTrainingConfig:
    """Training hyperparameters for gsplat optimization."""

    cuda_toolkit_path: str = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
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


def train_scene(
    scene_id: str | SceneId,
    config: SplatTrainingConfig,
    *,
    cache_root: str | Path = "cache",
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> TrainingResult:
    """Train Gaussian splats on a scene using real COLMAP outputs and frames."""

    _configure_cuda_toolkit(config.cuda_toolkit_path)
    _import_gsplat()
    if config.iterations <= 0:
        raise ValueError("iterations must be positive.")
    if config.snapshot_interval <= 0:
        raise ValueError("snapshot_interval must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.image_downscale <= 0:
        raise ValueError("image_downscale must be positive.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; install a CUDA build of PyTorch and use a CUDA device.")
    device = torch.device(config.device)

    set_random_seed(config.seed)

    normalized_scene = SceneId(str(scene_id))
    paths = ensure_scene_dirs(normalized_scene, cache_root=cache_root)
    frames = _load_colmap_frames(paths, device, image_downscale=config.image_downscale)
    if not frames:
        raise FileNotFoundError("No COLMAP frames with poses found; ensure images.txt and cameras.txt exist.")

    means, colors = _load_sparse_points(paths)
    if means.numel() == 0:
        raise FileNotFoundError(f"No sparse points found under {paths.sfm_dir}; run COLMAP first.")
    means = means.to(device=device, dtype=torch.float32)
    colors = colors.to(device=device, dtype=torch.float32)
    logger.info("Loaded sparse seeds: %d points", means.shape[0])

    splats_param = _initialize_parameters(
        means,
        colors,
        config,
        with_features=config.app_opt,
        feature_dim=config.app_feature_dim,
    ).to(device)
    sh_rest_lr = config.sh_lr / 20.0
    splat_optimizers = _build_splat_optimizers(splats_param, config, sh_rest_lr)
    means_decay_gamma = _compute_means_decay_gamma(config)
    means_scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(
            splat_optimizers["means"], gamma=means_decay_gamma  # type: ignore[arg-type]
        )
        if means_decay_gamma is not None and "means" in splat_optimizers
        else None
    )
    scene_scale = float(torch.linalg.norm(means, dim=1).mean().item()) if means.numel() > 0 else 1.0
    scene_scale = max(scene_scale, 1e-6)
    strategy = DefaultStrategy(
        prune_opa=config.prune_opacity_threshold,
        grow_scale3d=config.densify_scale_threshold,
        prune_scale3d=config.prune_scale_threshold,
        refine_start_iter=config.densify_start,
        refine_stop_iter=min(15000, config.iterations),
        refine_every=config.densify_interval,
        verbose=True,
    )
    strategy.check_sanity(splats_param, splat_optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    pose_adjust = CameraOptModule(len(frames)).to(device) if config.pose_opt else None
    pose_perturb = CameraOptModule(len(frames)).to(device) if config.pose_noise > 0.0 else None
    if pose_perturb is not None:
        pose_perturb.random_init(config.pose_noise)
    if pose_adjust is not None:
        pose_adjust.zero_init()
    pose_optimizer = (
        torch.optim.Adam(
            pose_adjust.parameters(),
            lr=config.pose_opt_lr * math.sqrt(config.batch_size),
            weight_decay=config.pose_opt_reg,
        )
        if pose_adjust is not None
        else None
    )
    appearance_module = (
        AppearanceOptModule(
            len(frames),
            feature_dim=config.app_feature_dim,
            embed_dim=config.app_embed_dim,
            sh_degree=config.sh_degree,
        ).to(device)
        if config.app_opt
        else None
    )
    if appearance_module is not None:
        torch.nn.init.zeros_(appearance_module.color_head[-1].weight)
        torch.nn.init.zeros_(appearance_module.color_head[-1].bias)
    appearance_optimizer = (
        torch.optim.Adam(
            list(appearance_module.parameters()),
            lr=config.app_opt_lr * math.sqrt(config.batch_size),
            weight_decay=config.app_opt_reg,
        )
        if appearance_module is not None
        else None
    )

    splat_dir = paths.splats_dir
    splat_dir.mkdir(parents=True, exist_ok=True)
    config_path = splat_dir / "config.json"
    log_path = splat_dir / "training_log.jsonl"
    config_path.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    export_format = config.export_format.lower().strip()
    export_format = "splat" if export_format == "splat" else "ply"

    def _checkpoint_path(iteration: int) -> Path:
        return splat_dir / f"iter_{iteration:05d}.{export_format}"

    _append_log(
        log_path,
        {
            "event": "start",
            "scene_id": str(normalized_scene),
            "frames": len(frames),
            "iterations": config.iterations,
            "snapshot_interval": config.snapshot_interval,
            "device": config.device,
            "export_format": export_format,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    )
    logger.info(
        "Training loop start scene=%s frames=%d iterations=%d snapshot_interval=%d device=%s",
        normalized_scene,
        len(frames),
        config.iterations,
        config.snapshot_interval,
        config.device,
    )

    last_checkpoint = _checkpoint_path(0)
    last_checkpoint = _export_splats(
        splats_param,
        last_checkpoint,
        max_points=config.max_points,
        fmt=export_format,
    )
    logger.info("Initial checkpoint written: %s", last_checkpoint)
    if checkpoint_callback is not None:
        checkpoint_callback(0, last_checkpoint)

    for iteration in range(1, config.iterations + 1):
        batch = _sample_frames(frames, config.batch_size)
        embed_ids = torch.tensor([f.index for f in batch], device=device, dtype=torch.long)
        batch_c2w = torch.stack([f.camtoworld for f in batch], dim=0)
        batch_K = torch.stack([f.K for f in batch], dim=0)
        height = batch[0].height
        width = batch[0].width
        batch_images = torch.stack([f.image for f in batch], dim=0).to(device=device)
        if pose_perturb is not None:
            batch_c2w = pose_perturb(batch_c2w, embed_ids)
        if pose_adjust is not None:
            batch_c2w = pose_adjust(batch_c2w, embed_ids)

        active_degree = (
            min(config.sh_degree, iteration // config.sh_degree_interval)
            if config.sh_degree_interval > 0
            else config.sh_degree
        )
        active_channels = (active_degree + 1) ** 2
        colors_full = torch.cat([splats_param["sh0"], splats_param["shN"]], dim=1)
        if appearance_module is not None:
            dirs = splats_param["means"][None, :, :] - batch_c2w[:, None, :3, 3]
            app_rgb = torch.sigmoid(appearance_module(splats_param["features"], embed_ids, dirs, active_degree))
            app_rgb = app_rgb.mean(dim=0)
            band0 = rgb_to_sh(app_rgb).unsqueeze(1)
            colors_full = torch.cat([band0, splats_param["shN"]], dim=1)
        colors = colors_full[:, :active_channels, :]
        scales = torch.exp(splats_param["scales"])
        opacities = torch.sigmoid(splats_param["opacities"])
        quats = F.normalize(splats_param["quats"], dim=1)

        renders, alphas, info = rasterization(
            means=splats_param["means"],
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(batch_c2w),
            Ks=batch_K,
            width=width,
            height=height,
            sh_degree=active_degree,
            render_mode="RGB",
            absgrad=strategy.absgrad if isinstance(strategy, DefaultStrategy) else False,
        )
        if config.random_background:
            bkgd = torch.rand((1, 1, 1, 3), device=device)
            renders = renders + bkgd * (1.0 - alphas)
        renders = torch.clamp(renders, 0.0, 1.0)

        packed_mode = info.get("packed", False) or (
            strategy.key_for_gradient in info and info[strategy.key_for_gradient].dim() == 2
        )
        if not packed_mode:
            info["n_cameras"] = info.get("n_cameras", info["radii"].shape[0])
            info["width"] = info.get("width", width)
            info["height"] = info.get("height", height)

        loss = config.loss_l1_weight * F.l1_loss(renders, batch_images)
        ssim_loss = None
        if config.ssim_weight > 0.0:
            ssim_val = _ssim_loss(renders, batch_images)
            ssim_loss = float(ssim_val.item())
            loss = (1.0 - config.ssim_weight) * loss + config.ssim_weight * (1.0 - ssim_val)
        if config.opacity_reg > 0.0:
            loss = loss + config.opacity_reg * opacities.mean()

        strategy.step_pre_backward(
            params=splats_param,
            optimizers=splat_optimizers,
            state=strategy_state,
            step=iteration,
            info=info,
        )
        loss.backward()
        for opt in splat_optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if means_scheduler is not None:
            means_scheduler.step()
        if pose_optimizer is not None:
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
        if appearance_optimizer is not None:
            appearance_optimizer.step()
            appearance_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            splats_param["quats"].data = F.normalize(splats_param["quats"].data, dim=1)
        strategy.step_post_backward(
            params=splats_param,
            optimizers=splat_optimizers,
            state=strategy_state,
            step=iteration,
            info=info,
            packed=packed_mode,
        )

        mse = F.mse_loss(renders, batch_images)
        psnr = float(-10.0 * torch.log10(mse + 1e-8))
        if progress_callback is not None:
            progress_callback(iteration, config.iterations, float(loss.item()))

        _append_log(
            log_path,
            {
                "event": "iteration",
                "iteration": iteration,
                "loss_l1": float(loss.item()),
                "psnr": psnr,
                "mean_scale": float(scales.mean().item()),
                "min_scale": float(scales.min().item()),
                "max_scale": float(scales.max().item()),
                "mean_opacity": float(opacities.mean().item()),
                "mean_quat_norm": float(torch.linalg.norm(splats_param["quats"], dim=1).mean().item()),
                "ssim": ssim_loss,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )
        logger.info(
            "Iteration %d/%d loss=%.4f psnr=%.2f mean_scale=%.6f mean_opacity=%.4f",
            iteration,
            config.iterations,
            float(loss.item()),
            psnr,
            float(scales.mean().item()),
            float(opacities.mean().item()),
        )

        if iteration % config.snapshot_interval == 0 or iteration == config.iterations:
            last_checkpoint = _checkpoint_path(iteration)
            last_checkpoint = _export_splats(
                splats_param,
                last_checkpoint,
                max_points=config.max_points,
                fmt=export_format,
            )
            logger.info("Wrote checkpoint %s", last_checkpoint)
            if checkpoint_callback is not None:
                checkpoint_callback(iteration, last_checkpoint)

    _append_log(
        log_path,
        {
            "event": "stop",
            "iterations": config.iterations,
            "last_checkpoint": str(last_checkpoint),
            "export_format": export_format,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    )
    logger.info(
        "Training loop stop scene=%s iterations=%d last_checkpoint=%s",
        normalized_scene,
        config.iterations,
        last_checkpoint,
    )
    return TrainingResult(
        scene_id=normalized_scene,
        paths=paths,
        iterations=config.iterations,
        last_checkpoint=last_checkpoint,
        export_format=export_format,
        log_path=log_path,
        config_path=config_path,
    )


def _build_splat_optimizers(
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


def _compute_means_decay_gamma(config: SplatTrainingConfig) -> Optional[float]:
    if config.iterations <= 0 or config.lr_final_scale <= 0 or config.lr_final_scale >= 1.0:
        return None
    return config.lr_final_scale ** (1.0 / float(config.iterations))


def _densify_and_prune(
    splats_param: torch.nn.ParameterDict,
    config: SplatTrainingConfig,
    *,
    iteration: int,
    log_path: Path,
) -> bool:
    """Deprecated legacy hook retained for backward compatibility."""
    raise RuntimeError("DefaultStrategy handles densification; _densify_and_prune is unused.")

def _load_colmap_frames(paths: ScenePaths, device: torch.device, *, image_downscale: int) -> List[FrameRecord]:
    cameras_txt, images_txt = _find_text_model(paths)
    cameras = _parse_cameras(cameras_txt)
    images = _parse_images(images_txt)
    records: List[FrameRecord] = []
    frames_dir = paths.frames_selected_dir
    for idx, image_entry in enumerate(images):
        camera = cameras.get(image_entry["camera_id"])
        if camera is None:
            raise RuntimeError(f"Camera id {image_entry['camera_id']} missing in cameras.txt")
        image_path = frames_dir / image_entry["name"]
        if not image_path.exists():
            image_path = frames_dir / Path(image_entry["name"]).name
        if not image_path.exists():
            raise FileNotFoundError(f"Image {image_entry['name']} not found under {frames_dir}")
        width = camera["width"]
        height = camera["height"]
        fx, fy, cx, cy = camera["params"]
        if image_downscale > 1:
            width = max(1, width // image_downscale)
            height = max(1, height // image_downscale)
            fx = fx / image_downscale
            fy = fy / image_downscale
            cx = cx / image_downscale
            cy = cy / image_downscale
        K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        camtoworld = _cam_to_world_matrix(image_entry["qvec"], image_entry["tvec"], device=device)
        image_tensor = _load_image_tensor(image_path, height, width)
        records.append(
            FrameRecord(
                index=idx,
                name=image_entry["name"],
                image_path=image_path,
                camtoworld=camtoworld,
                K=K,
                width=width,
                height=height,
                image=image_tensor,
            )
        )
    first_size = (records[0].height, records[0].width) if records else None
    for rec in records:
        if (rec.height, rec.width) != first_size:
            raise RuntimeError("All frames must share the same resolution after downscale.")
    return records


def _find_text_model(paths: ScenePaths) -> Tuple[Path, Path]:
    candidates = [
        (paths.sfm_dir / "sparse" / "text" / "cameras.txt", paths.sfm_dir / "sparse" / "text" / "images.txt"),
        (paths.sfm_dir / "sparse" / "0" / "cameras.txt", paths.sfm_dir / "sparse" / "0" / "images.txt"),
        (paths.sfm_dir / "sparse" / "cameras.txt", paths.sfm_dir / "sparse" / "images.txt"),
    ]
    for cams, imgs in candidates:
        if cams.exists() and imgs.exists():
            return cams, imgs
    raise FileNotFoundError(
        f"cameras.txt/images.txt not found under {paths.sfm_dir}. Re-run COLMAP so text models are exported."
    )


def _parse_cameras(path: Path) -> dict[int, dict]:
    cameras: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            if model not in {"PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
                raise ValueError(f"Unsupported COLMAP camera model: {model}")
            if model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            else:
                fx = fy = params[0]
                cx = params[1]
                cy = params[2] if len(params) > 2 else params[1]
            cameras[cam_id] = {"model": model, "width": width, "height": height, "params": (fx, fy, cx, cy)}
    return cameras


def _parse_images(path: Path) -> List[dict]:
    entries: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        lines = iter(handle.readlines())
        for line in lines:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]
            entries.append(
                {
                    "image_id": image_id,
                    "qvec": (qw, qx, qy, qz),
                    "tvec": (tx, ty, tz),
                    "camera_id": camera_id,
                    "name": name,
                }
            )
            next(lines, None)  # skip 2D point line
    return entries


def _cam_to_world_matrix(qvec: Tuple[float, float, float, float], tvec: Tuple[float, float, float], device: torch.device) -> torch.Tensor:
    qw, qx, qy, qz = qvec
    q = torch.tensor([qw, qx, qy, qz], dtype=torch.float64, device=device)
    R = _qvec_to_rotmat(q)
    t = torch.tensor(tvec, dtype=torch.float64, device=device)
    c2w = torch.eye(4, dtype=torch.float64, device=device)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w.float()


def _qvec_to_rotmat(qvec: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = qvec
    return torch.tensor(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=torch.float64,
        device=qvec.device,
    )


def _load_sparse_points(paths: ScenePaths) -> Tuple[torch.Tensor, torch.Tensor]:
    ply_path = paths.sfm_dir / "sparse" / "model.ply"
    if not ply_path.exists():
        ply_path = paths.sfm_dir / "sparse" / "0" / "points3D.ply"
    if not ply_path.exists():
        ply_path = paths.sfm_dir / "sparse" / "0" / "points3D.txt"
    if not ply_path.exists():
        raise FileNotFoundError(f"Sparse model not found under {paths.sfm_dir}")
    if ply_path.suffix.lower() == ".txt":
        means, colors, _tracks = _load_colmap_txt_points(ply_path)
    else:
        means, colors, _tracks = _load_ply_points(ply_path)
    return means, colors


def _load_ply_points(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    type_map = {
        "float": np.float32,
        "float32": np.float32,
        "double": np.float64,
        "uchar": np.uint8,
        "uint8": np.uint8,
        "uint": np.uint32,
        "int": np.int32,
    }
    with path.open("rb") as handle:
        header: list[str] = []
        while True:
            line_bytes = handle.readline()
            if not line_bytes:
                raise ValueError("Invalid PLY: no end_header")
            line = line_bytes.decode("ascii", errors="ignore").strip()
            header.append(line)
            if line == "end_header":
                break
        data = handle.read()

    format_line = next((line for line in header if line.startswith("format ")), "format ascii 1.0")
    ascii_format = "ascii" in format_line
    vertex_count = 0
    properties: list[tuple[str, type]] = []
    collecting_vertex = False
    for line in header:
        if line.startswith("element vertex"):
            parts = line.split()
            vertex_count = int(parts[-1])
            collecting_vertex = True
            continue
        if line.startswith("element ") and not line.startswith("element vertex"):
            collecting_vertex = False
        if collecting_vertex and line.startswith("property"):
            _, typ, name = line.split()[:3]
            if typ not in type_map:
                continue
            properties.append((name, type_map[typ]))

    def _extract_structured_array() -> np.ndarray:
        dtype = np.dtype([(name, typ) for name, typ in properties])
        return np.frombuffer(data, dtype=dtype, count=vertex_count)

    def _extract_from_ascii() -> np.ndarray:
        rows = []
        text = data.decode("ascii", errors="ignore").strip().splitlines()
        for line in text[: vertex_count or None]:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < len(properties):
                continue
            parsed = []
            for value, (_, typ) in zip(parts, properties):
                parsed.append(typ(type(typ)(float(value)) if np.issubdtype(typ, np.floating) else int(float(value))))
            rows.append(tuple(parsed))
        dtype = np.dtype([(name, typ) for name, typ in properties])
        return np.array(rows, dtype=dtype)

    arr = _extract_from_ascii() if ascii_format else _extract_structured_array()
    if arr.size == 0 or vertex_count == 0:
        return torch.empty((0, 3)), torch.empty((0, 3)), torch.empty((0,))

    def _get_field(candidates: list[str]) -> Optional[np.ndarray]:
        for name in candidates:
            if name in arr.dtype.names:
                return arr[name]
        return None

    xs = _get_field(["x"])
    ys = _get_field(["y"])
    zs = _get_field(["z"])
    rs = _get_field(["red", "r"])
    gs_ = _get_field(["green", "g"])
    bs = _get_field(["blue", "b"])
    if xs is None or ys is None or zs is None or rs is None or gs_ is None or bs is None:
        return torch.empty((0, 3)), torch.empty((0, 3)), torch.empty((0,))

    means = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    colors = np.stack([rs, gs_, bs], axis=1).astype(np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0
    tracks = _get_field(["track_length", "tracks", "track", "num_obs", "n_obs"])
    track_tensor = (
        torch.from_numpy(tracks.astype(np.float32))
        if tracks is not None
        else torch.zeros((means.shape[0],), dtype=torch.float32)
    )
    return torch.from_numpy(means), torch.from_numpy(colors), track_tensor


def _load_colmap_txt_points(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means = []
    colors = []
    tracks = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            x, y, z = map(float, parts[1:4])
            r, g, b = [float(int(val)) / 255.0 for val in parts[4:7]]
            track_len = float(len(parts) - 8) / 2 if len(parts) > 8 else 0.0
            means.append((x, y, z))
            colors.append((r, g, b))
            tracks.append(track_len)
    if not means:
        return torch.empty((0, 3)), torch.empty((0, 3)), torch.empty((0,))
    return (
        torch.tensor(means, dtype=torch.float32),
        torch.tensor(colors, dtype=torch.float32),
        torch.tensor(tracks, dtype=torch.float32),
    )


def _initialize_parameters(
    means: torch.Tensor,
    colors: torch.Tensor,
    config: SplatTrainingConfig,
    *,
    with_features: bool,
    feature_dim: int,
) -> torch.nn.ParameterDict:
    scales = _initial_scales(means, config.init_scale, config.min_scale, config.max_scale)
    opacities = torch.full((means.shape[0],), config.opacity_bias, device=means.device, dtype=torch.float32)
    scales_log = torch.log(scales)
    opacities_logit = torch.logit(opacities.clamp(1e-4, 1.0 - 1e-4))
    sh_channels = max(1, (config.sh_degree + 1) ** 2)
    sh0 = torch.zeros((means.shape[0], 1, 3), device=means.device, dtype=torch.float32)
    sh0[:, 0, :] = colors
    shN_count = max(0, sh_channels - 1)
    shN = torch.zeros((means.shape[0], shN_count, 3), device=means.device, dtype=torch.float32)
    quats = _identity_quats(means.shape[0], means.device)
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


def _initial_scales(means: torch.Tensor, scale_multiplier: float, min_scale: float, max_scale: float) -> torch.Tensor:
    # Match gsplat simple_trainer: set scale to log of mean distance to 3 nearest neighbors.
    if means.numel() == 0:
        return torch.empty_like(means)
    dists2 = torch.cdist(means, means, compute_mode="donot_use_mm_for_euclid_dist") ** 2
    # Ignore self-distance, take 3 nearest neighbors.
    topk = torch.topk(dists2, k=min(4, dists2.shape[1]), largest=False)
    dist_avg = torch.sqrt(topk.values[:, 1:].mean(dim=-1, keepdim=True))  # (N, 1)
    return dist_avg.repeat(1, 3) * scale_multiplier


def _neighbor_scale(means: torch.Tensor, sample_size: int = 4000, k: int = 8) -> torch.Tensor:
    n = means.shape[0]
    sample_size = min(sample_size, n)
    indices = torch.randperm(n, device=means.device)[:sample_size]
    sample = means[indices]
    dists = torch.cdist(means, sample, compute_mode="donot_use_mm_for_euclid_dist")
    dists_sorted, _ = torch.topk(dists, k=min(k + 1, sample_size), largest=False)
    neighbor = dists_sorted[:, 1:]
    avg = neighbor.mean(dim=1)
    return avg.unsqueeze(1).repeat(1, 3)


def _identity_quats(count: int, device: torch.device) -> torch.Tensor:
    quats = torch.zeros((count, 4), device=device, dtype=torch.float32)
    quats[:, 0] = 1.0
    return quats


def _sample_frames(frames: Sequence[FrameRecord], batch_size: int) -> List[FrameRecord]:
    if batch_size >= len(frames):
        return list(frames)
    indices = np.random.choice(len(frames), size=batch_size, replace=False)
    return [frames[int(i)] for i in indices]


def _load_image_tensor(path: Path, height: int, width: int) -> torch.Tensor:
    with Image.open(path) as handle:
        img = handle.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), resample=Image.BILINEAR)
        array = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(array)


def _append_log(log_path: Path, payload: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _export_splats(
    splats_param: torch.nn.ParameterDict,
    export_path: Path,
    *,
    max_points: int,
    fmt: str,
) -> Path:
    _import_gsplat()
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
    logger.info(
        "Exporting splats fmt=%s count=%d", fmt_clean, means.shape[0],
    )
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


_SSIM_METRICS: dict[str, StructuralSimilarityIndexMeasure] = {}


def _ssim_loss(img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
    """Compute SSIM using torchmetrics (expects NHWC input)."""

    device = img_a.device
    key = str(device)
    metric = _SSIM_METRICS.get(key)
    if metric is None:
        metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        _SSIM_METRICS[key] = metric
    a = img_a.permute(0, 3, 1, 2)
    b = img_b.permute(0, 3, 1, 2)
    return metric(a, b)


__all__ = ["SplatTrainingConfig", "TrainingResult", "train_scene"]
