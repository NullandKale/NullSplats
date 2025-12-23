"""CUDA-only Gaussian splat training using gsplat.

This module consumes COLMAP outputs (cameras.txt, images.txt, points3D) plus the
selected training frames and optimizes Gaussian parameters with torch + gsplat
rasterization. It exports checkpoints as .ply and .splat without any mock paths
or fallbacks. CUDA is required; CPU execution is rejected.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict
from datetime import datetime
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from nullsplats.backend.io_cache import ensure_scene_dirs
from nullsplats.backend.gs_utils import AppearanceOptModule, CameraOptModule, rgb_to_sh, set_random_seed
from nullsplats.backend.splat_train_config import (
    CheckpointCallback,
    PreviewCallback,
    PreviewPayload,
    ProgressCallback,
    SplatTrainingConfig,
    TrainingResult,
)
from nullsplats.backend.splat_train_io import load_colmap_frames, load_sparse_points
from nullsplats.backend.splat_train_ops import (
    append_log,
    build_splat_optimizers,
    compute_means_decay_gamma,
    configure_cuda_toolkit,
    export_splats,
    get_rasterization,
    initialize_parameters,
    sample_frames,
    ssim_available,
    ssim_loss,
)
from nullsplats.util.logging import get_logger
from nullsplats.util.scene_id import SceneId
from gsplat.strategy import DefaultStrategy


logger = get_logger("splat_train")
LOG_PROGRESS_INTERVAL = 100

def train_scene(
    scene_id: str | SceneId,
    config: SplatTrainingConfig,
    *,
    cache_root: str | Path = "cache",
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    preview_callback: PreviewCallback | None = None,
) -> TrainingResult:
    """Train Gaussian splats on a scene using real COLMAP outputs and frames."""

    configure_cuda_toolkit(config.cuda_toolkit_path)
    if config.ssim_weight > 0.0 and not ssim_available():
        logger.warning("SSIM requested but torchmetrics is unavailable or failed to import; disabling SSIM.")
        config = dataclasses.replace(config, ssim_weight=0.0)
    rasterization = get_rasterization()
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
    frames = load_colmap_frames(paths, device, image_downscale=config.image_downscale)
    if not frames:
        raise FileNotFoundError("No COLMAP frames with poses found; ensure images.txt and cameras.txt exist.")

    means, colors = load_sparse_points(paths)
    if means.numel() == 0:
        raise FileNotFoundError(f"No sparse points found under {paths.sfm_dir}; run COLMAP first.")
    means = means.to(device=device, dtype=torch.float32)
    colors = colors.to(device=device, dtype=torch.float32)
    logger.info("Loaded sparse seeds: %d points", means.shape[0])

    splats_param = initialize_parameters(
        means,
        colors,
        config,
        with_features=config.app_opt,
        feature_dim=config.app_feature_dim,
    ).to(device)
    sh_rest_lr = config.sh_lr / 20.0
    splat_optimizers = build_splat_optimizers(splats_param, config, sh_rest_lr)
    means_decay_gamma = compute_means_decay_gamma(config)
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

    append_log(
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
    last_checkpoint = export_splats(
        splats_param,
        last_checkpoint,
        max_points=config.max_points,
        fmt=export_format,
    )
    logger.info("Initial checkpoint written: %s", last_checkpoint)
    if checkpoint_callback is not None:
        checkpoint_callback(0, last_checkpoint)

    last_preview_time = time.perf_counter()
    last_preview_iter = 0
    for iteration in range(1, config.iterations + 1):
        batch = sample_frames(frames, config.batch_size)
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
        ssim_metric = None
        if config.ssim_weight > 0.0:
            ssim_val = ssim_loss(renders, batch_images)
            ssim_metric = float(ssim_val.item())
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

        if iteration == 1 or iteration == config.iterations or iteration % LOG_PROGRESS_INTERVAL == 0:
            append_log(
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
                    "ssim": ssim_metric,
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

        if preview_callback is not None and config.preview_interval_seconds > 0.0:
            now = time.perf_counter()
            is_forced = iteration in (1, config.iterations)
            ready_by_time = (now - last_preview_time) >= config.preview_interval_seconds
            ready_by_iters = (
                config.preview_min_iters <= 0 or (iteration - last_preview_iter) >= config.preview_min_iters
            )
            if is_forced or (ready_by_time and ready_by_iters):
                last_preview_time = now
                last_preview_iter = iteration
                with torch.no_grad():
                    means_preview = splats_param["means"].detach()
                    scales_log_preview = splats_param["scales"].detach()
                    quats_preview = F.normalize(splats_param["quats"].detach(), dim=1)
                    opacities_preview = torch.sigmoid(splats_param["opacities"].detach())
                    sh_dc_preview = splats_param["sh0"].detach()[:, 0, :]

                    if config.max_preview_points > 0 and means_preview.shape[0] > config.max_preview_points:
                        keep = torch.topk(opacities_preview, config.max_preview_points).indices
                        means_preview = means_preview[keep]
                        scales_log_preview = scales_log_preview[keep]
                        quats_preview = quats_preview[keep]
                        opacities_preview = opacities_preview[keep]
                        sh_dc_preview = sh_dc_preview[keep]

                    payload = PreviewPayload(
                        iteration=iteration,
                        means=means_preview.detach().cpu(),
                        scales_log=scales_log_preview.detach().cpu(),
                        quats_wxyz=quats_preview.detach().cpu(),
                        opacities=opacities_preview.detach().cpu(),
                        sh_dc=sh_dc_preview.detach().cpu(),
                    )
                try:
                    preview_callback(payload)
                except Exception:  # noqa: BLE001
                    logger.exception("Preview callback failed at iteration %d", iteration)

        if iteration % config.snapshot_interval == 0 or iteration == config.iterations:
            last_checkpoint = _checkpoint_path(iteration)
            last_checkpoint = export_splats(
                splats_param,
                last_checkpoint,
                max_points=config.max_points,
                fmt=export_format,
            )
            logger.info("Wrote checkpoint %s", last_checkpoint)
            _prune_checkpoints(splat_dir, last_checkpoint, export_format)
            if checkpoint_callback is not None:
                checkpoint_callback(iteration, last_checkpoint)

    append_log(
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


def _prune_checkpoints(splats_dir: Path, latest: Path, export_format: str) -> None:
    """Keep iter_00000 plus the latest checkpoint, remove older ones."""
    initial_name = f"iter_{0:05d}.{export_format}"
    keep = {initial_name, latest.name}
    for path in splats_dir.glob(f"iter_*.{export_format}"):
        if path.name in keep:
            continue
        try:
            path.unlink()
            logger.info("Removed old checkpoint %s", path)
        except Exception:
            logger.debug("Failed to remove checkpoint %s", path, exc_info=True)


__all__ = ["SplatTrainingConfig", "TrainingResult", "PreviewPayload", "train_scene"]

