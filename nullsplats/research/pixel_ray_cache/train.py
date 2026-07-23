"""Training loop, checkpointing, and epoch-loss reporting for Pixel Ray Cache."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import threading
from typing import Callable

import torch

from nullsplats.research.pixel_ray_cache import oob
from nullsplats.research.pixel_ray_cache import optim as optim_registry
from nullsplats.research.pixel_ray_cache import rays, tiles
from nullsplats.research.pixel_ray_cache.config import PixelRayCacheConfig
from nullsplats.research.pixel_ray_cache.data import FrameImageCache, RayCacheFrame
from nullsplats.research.pixel_ray_cache.losses import get_loss
from nullsplats.research.pixel_ray_cache.normals import depth_to_normals
from nullsplats.research.pixel_ray_cache.tile_models.base import TileModel
from nullsplats.research.pixel_ray_cache.tiles import BlockSize
from nullsplats.util.logging import get_logger

logger = get_logger("research.pixel_ray_cache.train")

OnEpochLoss = Callable[[int, float], None]
OnPreview = Callable[[int, RayCacheFrame], None]
OnStep = Callable[[TileModel, BlockSize, "int | None", "int | None"], None]
"""Called after every optimizer step with (model, render_tile, render_height, render_width) --
render_height/render_width are None for the flat baseline (meaning "use whatever frame you're
previewing's own native size", matching LoadedCheckpoint's convention) and always-explicit padded
dims for a cascade level. Intended for a caller (e.g. viewer.py) that wants to render a live
preview from the model's CURRENT weights mid-training; the training loop itself does nothing with
it beyond calling it -- any throttling (e.g. "only actually render every 0.3s") is the caller's
responsibility, since every image in every epoch would otherwise call this."""
OnBatchProgress = Callable[..., None]
"""Called after every batch with (epoch, batch_idx, n_batches, loss) for the flat baseline, or
(level_idx, epoch, batch_idx, n_batches, loss) for the cascade trainer -- mirrors OnEpochLoss's
variable-arity convention (a caller distinguishes the two by len(args)) so one callback shape
serves both trainers. Meant to drive a progress bar (tqdm in cli.py, a ttk.Progressbar in
viewer.py) -- purely observational, the training loop does nothing with the return value."""

TILE = 2


@dataclass(frozen=True)
class TrainingResult:
    final_loss: float
    epochs_run: int
    checkpoint_path: Path | None


def _load_depth_by_frame(
    frames: list[RayCacheFrame], config: PixelRayCacheConfig, *, device: torch.device
) -> dict[str, torch.Tensor]:
    """Depth is only meaningful once per scene, not once per epoch -- computed/cached up front
    via chunked DA3 calls (see depth_targets.compute_or_load_depth_for_scene) rather than inside
    the per-frame loop below.

    Cache dir is keyed on (cache_root, scene) only, NOT on config.out_dir -- so multiple staged
    runs against the same scene (e.g. color-only, then color+depth, then color+depth+normal, each
    writing checkpoints to their own --out-dir) share one DA3 depth cache instead of paying for
    DA3 inference again every time depth is requested.
    """
    from nullsplats.research.pixel_ray_cache.depth_targets import compute_or_load_depth_for_scene

    cache_dir = config.cache_root / "outputs" / config.scene / "pixel_ray_cache" / "depth_cache"
    return compute_or_load_depth_for_scene(
        frames,
        cache_dir=cache_dir,
        pretrained_id=config.da3_pretrained,
        device=device,
        chunk_size=config.da3_chunk_size,
    )


def _build_target_grids(
    frame: RayCacheFrame,
    origins: torch.Tensor,
    directions: torch.Tensor,
    config: PixelRayCacheConfig,
    *,
    image_cache: FrameImageCache,
    depth_by_frame: dict[str, torch.Tensor] | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """(H,W,*) target grids for every enabled target on this frame. image_cache decodes each
    frame's PNG from disk once and reuses it for every subsequent epoch; depth_by_frame's
    tensors live on CPU (see depth_targets.compute_or_load_depth_for_scene) and are moved to
    `device` here, just-in-time, rather than sitting resident in VRAM for the whole run."""
    grids: dict[str, torch.Tensor] = {"color": image_cache.get(frame, device=device)}
    if "depth" in config.targets:
        assert depth_by_frame is not None
        depth = depth_by_frame[frame.name].to(device)
        grids["depth"] = depth
        if "normal" in config.targets:
            grids["normal"] = depth_to_normals(depth, origins, directions)
    return grids


def _composite_loss(
    pred: dict[str, torch.Tensor],
    target_tiles: dict[str, torch.Tensor],
    config: PixelRayCacheConfig,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """mask, if given, is (N, B, 1) broadcastable, 1.0 for real samples and 0.0 for padding
    (see cascade.py / cascade_train.py, whose padded-to-power-of-two grid has fictitious samples
    outside the original image that must not influence gradients) -- excluded via a masked mean
    instead of a plain torch.mean. None (the flat baseline's default) means every sample counts.
    config.color_loss/config.depth_loss select the elementwise loss (see losses.py registry) --
    a "method" hyperparameter independent of model architecture."""

    def masked_mean(per_element: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return per_element.mean()
        denom = mask.expand_as(per_element).sum().clamp_min(1.0)
        return (per_element * mask).sum() / denom

    color_loss_fn = get_loss(config.color_loss)
    loss = masked_mean(color_loss_fn(pred["color"], target_tiles["color"]))
    if "depth" in config.targets:
        depth_loss_fn = get_loss(config.depth_loss)
        loss = loss + config.depth_loss_weight * masked_mean(depth_loss_fn(pred["depth"], target_tiles["depth"]))
    if "normal" in config.targets:
        cos = (pred["normal"] * target_tiles["normal"]).sum(dim=-1, keepdim=True)  # (N,B,1)
        loss = loss + config.normal_loss_weight * masked_mean(1.0 - cos)
    return loss


def run_training(
    config: PixelRayCacheConfig,
    frames: list[RayCacheFrame],
    model: TileModel,
    *,
    on_epoch_loss: OnEpochLoss | None = None,
    on_preview: OnPreview | None = None,
    on_step: OnStep | None = None,
    on_batch_progress: OnBatchProgress | None = None,
    stop_event: threading.Event | None = None,
) -> TrainingResult:
    """stop_event, if given, is checked after every batch (see images_per_step below); when set,
    training ends early (checkpointing whatever's been trained so far -- "stop" keeps progress,
    it doesn't discard it) rather than raising. Lets a caller running this on a background
    thread (e.g. viewer.py) cancel cooperatively without killing the thread."""
    if not frames:
        raise ValueError("run_training requires at least one frame")
    device = torch.device(config.device)
    model = model.to(device)
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)

    image_cache = FrameImageCache()
    depth_by_frame = _load_depth_by_frame(frames, config, device=device) if "depth" in config.targets else None

    optimizer = optim_registry.build_optimizer(config.optimizer, model.parameters(), config.lr)
    scheduler = optim_registry.build_scheduler(config.lr_schedule, optimizer, total_steps=config.epochs)
    crop = config.tile_crop
    batch_size = max(1, config.images_per_step)

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

    last_loss = float("nan")
    epochs_run = 0
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
                origins, directions = rays.pixel_rays(
                    frame.K, frame.camtoworld, frame.height, frame.width, device=device
                )
                o_tiles, d_tiles = tiles.rays_to_tiles(origins, directions, tile=TILE, crop=crop)
                batch_ray_inputs.append(torch.cat([o_tiles, d_tiles], dim=-1))  # (N,4,6)

                target_grids = _build_target_grids(
                    frame, origins, directions, config, image_cache=image_cache, depth_by_frame=depth_by_frame,
                    device=device,
                )
                for name, grid in target_grids.items():
                    batch_targets.setdefault(name, []).append(tiles.image_to_tiles(grid, tile=TILE, crop=crop))

            # Concatenating along the sample dim (not the tile dim) is exactly what
            # BatchedTileMLP-family forward()/einsum already handles generically -- K images
            # become one (N, 4*K, C) call instead of K separate (N, 4, C) calls, cutting the
            # number of forward/backward/optimizer.step() Python+CUDA-dispatch round trips by
            # ~1/K for the same total sample count. batch_size=1 (the default) concatenates a
            # single tensor with itself, i.e. a no-op -- byte-for-byte identical to the
            # unbatched loop this replaced.
            ray_inputs = torch.cat(batch_ray_inputs, dim=1)
            target_tiles = {name: torch.cat(tensors, dim=1) for name, tensors in batch_targets.items()}

            pred = model(ray_inputs)
            loss = _composite_loss(pred, target_tiles, config)

            if oob_pose_scale is not None:
                # One oob sample per BATCH (not per image) -- keeps the extra forward/backward
                # cost independent of batch_size rather than scaling with it. Weaker signal per
                # epoch at batch_size>1, but batch_size=1 (default) is unaffected: exactly one
                # image per batch, so this is identical to the original per-image behavior.
                perturbed_c2w = oob.random_perturbed_pose(
                    batch_frame.camtoworld, translation_scale=oob_pose_scale, device=device
                )
                oob_o, oob_d = rays.pixel_rays(
                    batch_frame.K, perturbed_c2w, batch_frame.height, batch_frame.width, device=device
                )
                oob_o_tiles, oob_d_tiles = tiles.rays_to_tiles(oob_o, oob_d, tile=TILE, crop=crop)
                oob_inputs = torch.cat([oob_o_tiles, oob_d_tiles], dim=-1)
                oob_pred = model(oob_inputs)
                oob_target = torch.zeros_like(oob_pred["color"])
                loss = loss + config.oob_loss_weight * color_loss_fn(oob_pred["color"], oob_target).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_loss = float(loss.detach())
            epoch_loss_sum += batch_loss * len(batch_indices)
            processed += len(batch_indices)

            if on_step is not None:
                on_step(model, TILE, None, None)
            if on_batch_progress is not None:
                on_batch_progress(epoch, batch_idx, n_batches, batch_loss)
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break

        last_loss = epoch_loss_sum / max(1, processed)
        epochs_run = epoch + 1
        logger.info("epoch %d/%d loss=%.6f%s", epoch + 1, config.epochs, last_loss, " (stopped early)" if stopped else "")
        if on_epoch_loss is not None:
            on_epoch_loss(epoch, last_loss)
        if on_preview is not None and config.preview_every > 0 and (epoch + 1) % config.preview_every == 0:
            on_preview(epoch, frames[order[0]])
        if scheduler is not None:
            scheduler.step()
        if stopped:
            break

    checkpoint_path = save_checkpoint(model, config)
    return TrainingResult(final_loss=last_loss, epochs_run=epochs_run, checkpoint_path=checkpoint_path)


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: TileModel
    render_tile: tuple[int, int]
    render_height: int | None  # None -> use the target RayCacheFrame's own native height
    render_width: int | None  # None -> use the target RayCacheFrame's own native width


def save_checkpoint(
    model: TileModel,
    config: PixelRayCacheConfig,
    *,
    render_tile: tuple[int, int] = (2, 2),
    render_height: int | None = None,
    render_width: int | None = None,
) -> Path:
    """render_tile/height/width record what preview.render_preview needs to reconstruct a full
    image from this checkpoint. The flat baseline's defaults (tile=(2,2), height/width=None,
    meaning "use whatever frame you're previewing's own native size") are correct because its
    model.n_tiles always equals that frame's native leaf tile count. A cascade checkpoint (see
    cascade_train.py) is different: if the tile grid wasn't a power of two, the finest level
    trained on a PADDED grid (n_tiles = padded leaf count, not native), so render_height/width
    must be recorded explicitly -- without them, reloading the checkpoint later and calling
    render_preview with native dimensions would raise a shape mismatch in the model's forward
    einsum (n_tiles wouldn't match the tile count implied by the native size).
    """
    out_dir = config.resolve_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"
    meta_path = out_dir / "checkpoint.json"
    torch.save(model.state_dict(), checkpoint_path)
    meta_path.write_text(
        json.dumps(
            {
                "n_tiles": model.n_tiles,
                "tile_model": config.tile_model,
                "tile_model_kwargs": config.tile_model_kwargs,
                "target_spec": list(model.target_spec),
                "tile_crop": config.tile_crop,
                "render_tile": list(render_tile),
                "render_height": render_height,
                "render_width": render_width,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return checkpoint_path


def load_checkpoint(out_dir: Path, *, device: torch.device) -> LoadedCheckpoint:
    """Reconstruct a TileModel (+ the render_tile/height/width preview.render_preview needs)
    from a checkpoint written by save_checkpoint."""
    from nullsplats.research.pixel_ray_cache.tile_models.registry import get_tile_model_cls

    meta = json.loads((out_dir / "checkpoint.json").read_text(encoding="utf-8"))
    target_spec = tuple(tuple(entry) for entry in meta["target_spec"])
    model_cls = get_tile_model_cls(meta["tile_model"])
    model = model_cls(meta["n_tiles"], target_spec, **meta["tile_model_kwargs"])
    state = torch.load(out_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(state)
    render_tile = tuple(meta.get("render_tile", [2, 2]))
    return LoadedCheckpoint(
        model=model.to(device),
        render_tile=render_tile,
        render_height=meta.get("render_height"),
        render_width=meta.get("render_width"),
    )
