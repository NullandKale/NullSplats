"""Coarse-to-fine cascade training for Pixel Ray Cache.

Level 0 trains a single network on every tile of every training image (maximum data, minimum
specialization). Each subsequent level (see cascade.build_cascade_plan) quarters the group size
-- 1 group -> 4 -> 16 -> ... -- warm-starting each child network by copying its parent's trained
weights (TileModel.expand) before continuing training on just that child's own (smaller, more
local) share of the data. Continues until groups reach a single tile (matching the flat
baseline's granularity exactly) or config.cascade_levels is reached, whichever comes first.

The tile grid is padded up to the next power of two in each dimension (see cascade.py) so every
level's block size evenly divides the grid; padded pixels are zero-filled and excluded from every
loss term via a mask (see train._composite_loss's `mask` parameter) rather than contributing
fictitious targets.

Produces a checkpoint in the exact same format as train.run_training's flat baseline (same
save_checkpoint/load_checkpoint, same TileModel n_tiles/target_spec metadata) -- as long as the
cascade actually reaches leaf (single-tile) resolution, the result is a drop-in replacement:
preview.render_preview and cli.py's checkpoint loading work on it unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
import random
import threading
from typing import Callable

import torch

from nullsplats.research.pixel_ray_cache import cascade, oob, rays, tiles
from nullsplats.research.pixel_ray_cache import optim as optim_registry
from nullsplats.research.pixel_ray_cache.cascade import CascadeLevelSpec
from nullsplats.research.pixel_ray_cache.config import PixelRayCacheConfig
from nullsplats.research.pixel_ray_cache.data import FrameImageCache, RayCacheFrame
from nullsplats.research.pixel_ray_cache.losses import get_loss
from nullsplats.research.pixel_ray_cache.normals import depth_to_normals
from nullsplats.research.pixel_ray_cache.tile_models.base import TileModel
from nullsplats.research.pixel_ray_cache.tile_models.registry import get_tile_model_cls
from nullsplats.research.pixel_ray_cache.train import OnBatchProgress, OnStep, TrainingResult, _composite_loss, save_checkpoint
from nullsplats.util.logging import get_logger

logger = get_logger("research.pixel_ray_cache.cascade_train")

OnLevelStart = Callable[[int, CascadeLevelSpec], None]
OnEpochLoss = Callable[[int, int, float], None]  # (level_idx, epoch, loss)
OnLevelPreview = Callable[[int, TileModel, RayCacheFrame, tuple[int, int], int, int], None]
# (level_idx, model, frame, block_size, padded_height, padded_width)


def _pad_grid(grid: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(h,w,c) -> (height,width,c), zero-padded at the bottom/right. No-op if already that size."""
    h, w, c = grid.shape
    if h == height and w == width:
        return grid
    canvas = torch.zeros(height, width, c, device=grid.device, dtype=grid.dtype)
    canvas[:h, :w] = grid
    return canvas


def run_cascade_training(
    config: PixelRayCacheConfig,
    frames: list[RayCacheFrame],
    *,
    on_level_start: OnLevelStart | None = None,
    on_epoch_loss: OnEpochLoss | None = None,
    on_level_preview: OnLevelPreview | None = None,
    on_step: OnStep | None = None,
    on_batch_progress: OnBatchProgress | None = None,
    stop_event: threading.Event | None = None,
) -> TrainingResult:
    """stop_event, if given, is checked after every image; when set, the cascade ends early
    (checkpointing whatever level it was mid-training, same "stop keeps progress" semantics as
    train.run_training) instead of raising."""
    if not frames:
        raise ValueError("run_cascade_training requires at least one frame")
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)

    height, width = frames[0].height, frames[0].width
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"cascade requires even frame dimensions, got {height}x{width}")
    height_tiles, width_tiles = height // 2, width // 2

    plan = cascade.build_cascade_plan(height_tiles, width_tiles, max_levels=config.cascade_levels)
    ht_pad = cascade.padded_size(height_tiles)
    wt_pad = cascade.padded_size(width_tiles)
    pad_h, pad_w = ht_pad * 2, wt_pad * 2
    if not plan[-1].is_leaf:
        logger.warning(
            "cascade_levels=%d is not enough to reach full per-tile resolution (native tile grid "
            "%dx%d needs up to %d levels); final level groups still cover %dx%d tiles each -- "
            "the resulting checkpoint will NOT be directly compatible with preview.render_preview's "
            "default tile=2.",
            config.cascade_levels, height_tiles, width_tiles, len(cascade.build_cascade_plan(height_tiles, width_tiles, 999)),
            plan[-1].region_h, plan[-1].region_w,
        )
    logger.info(
        "Cascade plan: %d levels, native=%dx%d tile_grid=%dx%d padded_tile_grid=%dx%d",
        len(plan), height, width, height_tiles, width_tiles, ht_pad, wt_pad,
    )

    valid_mask = torch.zeros(pad_h, pad_w, 1, device=device)
    valid_mask[:height, :width, :] = 1.0

    depth_by_frame: dict[str, torch.Tensor] | None = None
    if "depth" in config.targets:
        from nullsplats.research.pixel_ray_cache.depth_targets import compute_or_load_depth_for_scene

        cache_dir = config.cache_root / "outputs" / config.scene / "pixel_ray_cache" / "depth_cache"
        depth_by_frame = compute_or_load_depth_for_scene(
            frames,
            cache_dir=cache_dir,
            pretrained_id=config.da3_pretrained,
            device=device,
            chunk_size=config.da3_chunk_size,
        )

    oob_pose_scale: float | None = None
    if config.oob_loss_weight > 0:
        oob_pose_scale = (
            config.oob_translation_scale if config.oob_translation_scale is not None else oob.estimate_pose_scale(frames)
        )
        logger.info(
            "Out-of-distribution black regularization enabled: weight=%.4f translation_scale=%.4f",
            config.oob_loss_weight, oob_pose_scale,
        )
    color_loss_fn = get_loss(config.color_loss)
    image_cache = FrameImageCache()
    batch_size = max(1, config.images_per_step)

    model: TileModel | None = None
    last_loss = float("nan")
    last_epoch_order: list[int] = []

    for level_idx, spec in enumerate(plan):
        block = (spec.region_h * 2, spec.region_w * 2)  # raw-pixel block size for this level
        mask_tiles = tiles.image_to_tiles(valid_mask, tile=block)  # (n_groups, block_area, 1)

        if model is None:
            model_cls = get_tile_model_cls(config.tile_model)
            model = model_cls(spec.n_groups, config.target_spec, **config.tile_model_kwargs).to(device)
        else:
            parent_index = cascade.parent_index_for(spec, plan[level_idx - 1])
            model = model.expand(parent_index).to(device)

        optimizer = optim_registry.build_optimizer(config.optimizer, model.parameters(), config.lr)
        scheduler = optim_registry.build_scheduler(config.lr_schedule, optimizer, total_steps=config.epochs)
        logger.info(
            "Level %d/%d: n_groups=%d block=%dx%d tiles/group=%d",
            level_idx + 1,
            len(plan),
            spec.n_groups,
            block[0],
            block[1],
            spec.region_h * spec.region_w,
        )
        if on_level_start is not None:
            on_level_start(level_idx, spec)

        order = list(range(len(frames)))
        n_batches = math.ceil(len(order) / batch_size)
        stopped = False
        for epoch in range(config.epochs):
            rng.shuffle(order)
            epoch_loss_sum = 0.0
            processed = 0
            for batch_idx, batch_start in enumerate(range(0, len(order), batch_size)):
                batch_indices = order[batch_start : batch_start + batch_size]
                batch_ray_inputs: list[torch.Tensor] = []
                batch_targets: dict[str, list[torch.Tensor]] = {}
                batch_frame = frames[batch_indices[-1]]  # used for the (at most one) oob sample below
                for idx in batch_indices:
                    frame = frames[idx]
                    origins, directions = rays.pixel_rays(frame.K, frame.camtoworld, pad_h, pad_w, device=device)
                    o_tiles, d_tiles = tiles.rays_to_tiles(origins, directions, tile=block)
                    batch_ray_inputs.append(torch.cat([o_tiles, d_tiles], dim=-1))

                    color_native = image_cache.get(frame, device=device)
                    grids = {"color": _pad_grid(color_native, pad_h, pad_w)}
                    if "depth" in config.targets:
                        assert depth_by_frame is not None
                        depth_native = depth_by_frame[frame.name].to(device)
                        if "normal" in config.targets:
                            native_o, native_d = rays.pixel_rays(
                                frame.K, frame.camtoworld, frame.height, frame.width, device=device
                            )
                            normal_native = depth_to_normals(depth_native, native_o, native_d)
                            grids["normal"] = _pad_grid(normal_native, pad_h, pad_w)
                        grids["depth"] = _pad_grid(depth_native, pad_h, pad_w)

                    for name, grid in grids.items():
                        batch_targets.setdefault(name, []).append(tiles.image_to_tiles(grid, tile=block))

                # See train.run_training's identical comment: concatenating along the sample dim
                # is exactly what the batched-weight-tensor forward already handles generically;
                # batch_size=1 (default) is a no-op concat, byte-for-byte identical to before.
                ray_inputs = torch.cat(batch_ray_inputs, dim=1)
                target_tiles = {name: torch.cat(tensors, dim=1) for name, tensors in batch_targets.items()}
                batch_mask = mask_tiles.repeat(1, len(batch_indices), 1)

                pred = model(ray_inputs)
                loss = _composite_loss(pred, target_tiles, config, mask=batch_mask)

                if oob_pose_scale is not None:
                    # One oob sample per batch, same rationale as train.run_training.
                    perturbed_c2w = oob.random_perturbed_pose(
                        batch_frame.camtoworld, translation_scale=oob_pose_scale, device=device
                    )
                    oob_o, oob_d = rays.pixel_rays(batch_frame.K, perturbed_c2w, pad_h, pad_w, device=device)
                    oob_o_tiles, oob_d_tiles = tiles.rays_to_tiles(oob_o, oob_d, tile=block)
                    oob_inputs = torch.cat([oob_o_tiles, oob_d_tiles], dim=-1)
                    oob_pred = model(oob_inputs)
                    oob_target = torch.zeros_like(oob_pred["color"])
                    oob_per_elem = color_loss_fn(oob_pred["color"], oob_target)
                    oob_denom = mask_tiles.expand_as(oob_per_elem).sum().clamp_min(1.0)
                    oob_loss = (oob_per_elem * mask_tiles).sum() / oob_denom
                    loss = loss + config.oob_loss_weight * oob_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_loss = float(loss.detach())
                epoch_loss_sum += batch_loss * len(batch_indices)
                processed += len(batch_indices)

                if on_step is not None:
                    on_step(model, block, pad_h, pad_w)
                if on_batch_progress is not None:
                    on_batch_progress(level_idx, epoch, batch_idx, n_batches, batch_loss)
                if stop_event is not None and stop_event.is_set():
                    stopped = True
                    break

            last_loss = epoch_loss_sum / max(1, processed)
            logger.info(
                "level %d/%d epoch %d/%d loss=%.6f%s",
                level_idx + 1,
                len(plan),
                epoch + 1,
                config.epochs,
                last_loss,
                " (stopped early)" if stopped else "",
            )
            if on_epoch_loss is not None:
                on_epoch_loss(level_idx, epoch, last_loss)
            if scheduler is not None:
                scheduler.step()
            if stopped:
                break
        last_epoch_order = order

        if on_level_preview is not None:
            on_level_preview(level_idx, model, frames[last_epoch_order[0]], block, pad_h, pad_w)

        if stopped:
            logger.info("Stop requested; ending cascade early at level %d/%d", level_idx + 1, len(plan))
            break

    assert model is not None
    checkpoint_path = save_checkpoint(model, config, render_tile=block, render_height=pad_h, render_width=pad_w)
    return TrainingResult(
        final_loss=last_loss, epochs_run=config.epochs * len(plan), checkpoint_path=checkpoint_path
    )
