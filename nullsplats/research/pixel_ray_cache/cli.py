"""Standalone headless entry point for a single Pixel Ray Cache training run.

Mirrors test.py's --run-case ergonomics: explicit flags, real exceptions (not swallowed), a
final `OK ...` summary line. Not wired into the production UI/registry/dispatch -- run directly:

    python -m nullsplats.research.pixel_ray_cache.cli --scene <id> --epochs 20
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import torch
from tqdm import tqdm

from nullsplats.backend.io_cache import ScenePaths
from nullsplats.research.pixel_ray_cache import data, optim as optim_registry, preview, tiles, train
from nullsplats.research.pixel_ray_cache.config import PixelRayCacheConfig
from nullsplats.research.pixel_ray_cache.losses import list_losses
from nullsplats.research.pixel_ray_cache.tile_models.registry import get_hyperparams, get_tile_model_cls, list_tile_models
from nullsplats.util.logging import get_logger, setup_logging

_TILE_CROP_RE = re.compile(r"(\d+)x(\d+)\+(\d+)\+(\d+)")


def _resolve_tile_model_kwargs(
    model_name: str, *, hidden_width: int, hidden_layers: int, extra_json: str | None
) -> dict[str, Any]:
    """--hidden-width/--hidden-layers stay as direct flags for convenience (the two most-tuned
    knobs, and the only ones the original CLI exposed) but only apply if the chosen architecture
    actually has them; --model-kwargs is the generic escape hatch for everything else any
    registered architecture declares (encoding, activation, omega_0, ...) without needing a
    dedicated CLI flag per hyperparameter per architecture."""
    spec_names = {spec.name for spec in get_hyperparams(model_name)}
    kwargs: dict[str, Any] = {}
    if "hidden_width" in spec_names:
        kwargs["hidden_width"] = hidden_width
    if "hidden_layers" in spec_names:
        kwargs["hidden_layers"] = hidden_layers
    if extra_json:
        extra = json.loads(extra_json)
        kwargs.update({k: v for k, v in extra.items() if k in spec_names})
    return kwargs


def _parse_tile_crop(value: str | None) -> tuple[int, int, int, int] | None:
    """'WxH+X+Y' (tile units) -> (x, y, w, h); None passthrough."""
    if value is None:
        return None
    match = _TILE_CROP_RE.fullmatch(value)
    if not match:
        raise argparse.ArgumentTypeError(f"--tile-crop must look like WxH+X+Y, got {value!r}")
    w, h, x, y = (int(g) for g in match.groups())
    return (x, y, w, h)


def _parse_targets(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Pixel Ray Cache: one independent small network per screen-space tile, "
            "mapping a ray (camera position + pixel direction) to color (and optionally "
            "depth/normal)."
        ),
        epilog=(
            "Limitation: tile identity is a fixed screen-space index, not a 3D location, so this "
            "assumes a small-baseline / near-fixed-viewpoint capture. A full object orbit will "
            "ask one tile's network to fit inconsistent content across views -- use the "
            "production gsplat/DA3/SHARP pipeline for that instead."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scene", required=True, help="Scene id already processed by extract_frames + run_sfm.")
    parser.add_argument("--cache-root", default="cache")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--tile-model", default="mlp", choices=list_tile_models())
    parser.add_argument("--hidden-width", type=int, default=16)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument(
        "--model-kwargs",
        default=None,
        help="JSON object of extra architecture-specific kwargs, e.g. "
        "'{\"encoding\": \"fourier\", \"hidden_activation\": \"gelu\"}'. Keys not recognized by "
        "the chosen --tile-model are ignored. Run --tile-model <name> with an invalid "
        "--model-kwargs value to see that architecture's available keys in the resulting error.",
    )
    parser.add_argument("--optimizer", default="adam", choices=optim_registry.list_optimizers())
    parser.add_argument("--lr-schedule", default="constant", choices=optim_registry.list_schedulers())
    parser.add_argument("--color-loss", default="mse", choices=list_losses())
    parser.add_argument("--depth-loss", default="l1", choices=list_losses())
    parser.add_argument(
        "--normalize-poses",
        action="store_true",
        help="Recenter+isotropically rescale camera positions into [-1,1] before training (see "
        "data.compute_pose_normalization). Off by default -- changes the coordinate convention "
        "a checkpoint is trained against, so a checkpoint trained with this on is not compatible "
        "with frames loaded with it off (or vice versa).",
    )
    parser.add_argument(
        "--images-per-step",
        type=int,
        default=1,
        help="Concatenate this many training images into one forward/backward/optimizer.step() "
        "call instead of one call per image -- cuts Python/CUDA-dispatch overhead roughly by "
        "this factor for the same total sample count. 1 (default) is byte-for-byte identical to "
        "the original per-image loop. Raises peak VRAM roughly proportionally; increase "
        "gradually and watch the memory estimate.",
    )
    parser.add_argument("--tile-crop", default=None, help="WxH+X+Y in tile units; omitted = full grid.")
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="Coarse-to-fine training: 1 network trained on the whole image, then repeatedly "
        "split into quadrants (each child warm-started from its trained parent) down to full "
        "per-tile resolution. Not compatible with --tile-crop.",
    )
    parser.add_argument(
        "--cascade-levels",
        type=int,
        default=32,
        help="Max cascade levels (level 0 = whole image). Generously large by default -- the "
        "cascade stops on its own once it reaches full per-tile resolution, so this is really an "
        "upper bound, not a target.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preview-every", type=int, default=5)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--targets", default="color", help="Comma-separated: color[,depth[,normal]].")
    parser.add_argument("--da3-pretrained", default="depth-anything/DA3NESTED-GIANT-LARGE")
    parser.add_argument(
        "--da3-chunk-size",
        type=int,
        default=4,
        help="Frames per DA3 inference call when computing depth (keeps DA3's peak GPU memory "
        "bounded regardless of scene size; lower it if depth extraction still OOMs).",
    )
    parser.add_argument("--depth-loss-weight", type=float, default=0.1)
    parser.add_argument("--normal-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--oob-loss-weight",
        type=float,
        default=0.0,
        help="Weight for out-of-distribution black regularization (see oob.py): each step, also "
        "supervises a randomly perturbed camera pose's rays as pure black, so tiles learn to "
        "output black for rays unlike anything they were trained on (e.g. a viewer camera flown "
        "away from the capture) instead of regressing toward an averaged color. 0 = disabled "
        "(default, zero extra cost).",
    )
    parser.add_argument(
        "--oob-translation-scale",
        type=float,
        default=None,
        help="World-unit magnitude for the out-of-distribution pose jitter. Omit to "
        "auto-estimate from the scene's own camera pose spread (largest pairwise distance "
        "between training camera positions).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar (per-epoch loss is still logged either way).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    logger = get_logger("research.pixel_ray_cache.cli")
    args = _build_arg_parser().parse_args(argv)
    crop = _parse_tile_crop(args.tile_crop)
    targets = _parse_targets(args.targets)
    if args.cascade and crop is not None:
        raise ValueError("--tile-crop is not supported together with --cascade")

    tile_model_kwargs = _resolve_tile_model_kwargs(
        args.tile_model, hidden_width=args.hidden_width, hidden_layers=args.hidden_layers, extra_json=args.model_kwargs
    )

    config = PixelRayCacheConfig(
        scene=args.scene,
        cache_root=Path(args.cache_root),
        targets=targets,
        tile_model=args.tile_model,
        tile_model_kwargs=tile_model_kwargs,
        lr=args.lr,
        optimizer=args.optimizer,
        lr_schedule=args.lr_schedule,
        color_loss=args.color_loss,
        depth_loss=args.depth_loss,
        epochs=args.epochs,
        tile_crop=crop,
        cascade=args.cascade,
        cascade_levels=args.cascade_levels,
        depth_loss_weight=args.depth_loss_weight,
        normal_loss_weight=args.normal_loss_weight,
        oob_loss_weight=args.oob_loss_weight,
        oob_translation_scale=args.oob_translation_scale,
        normalize_poses=args.normalize_poses,
        images_per_step=args.images_per_step,
        da3_pretrained=args.da3_pretrained,
        da3_chunk_size=args.da3_chunk_size,
        device=args.device,
        preview_every=args.preview_every,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )

    device = torch.device(config.device)
    paths = ScenePaths(scene_id=config.scene, cache_root=config.cache_root)
    logger.info("Loading scene frames for %s from %s (normalize_poses=%s)", config.scene, config.cache_root, config.normalize_poses)
    frames = data.load_scene_frames(paths, device=device, normalize_poses=config.normalize_poses)
    logger.info("Loaded %d frames", len(frames))

    out_dir = config.resolve_out_dir()
    n_batches_per_epoch = math.ceil(len(frames) / max(1, config.images_per_step))

    if config.cascade:
        from nullsplats.research.pixel_ray_cache import cascade, cascade_train

        height_tiles, width_tiles = frames[0].height // 2, frames[0].width // 2
        plan = cascade.build_cascade_plan(height_tiles, width_tiles, max_levels=config.cascade_levels)
        logger.info(
            "cascade_levels=%d tile_model=%s targets=%s plan=%d levels (leaf=%s)",
            config.cascade_levels, config.tile_model, config.targets, len(plan), plan[-1].is_leaf,
        )

        def _on_level_preview(level_idx: int, model, frame, block, pad_h: int, pad_w: int) -> None:
            preview.render_preview(
                model,
                frame,
                device=device,
                out_path=out_dir / "previews" / f"level_{level_idx:02d}.png",
                tile=block,
                render_height=pad_h,
                render_width=pad_w,
            )

        progress = None if args.no_progress else tqdm(
            total=len(plan) * config.epochs * n_batches_per_epoch, unit="batch", desc="cascade"
        )

        def _on_batch_progress(level_idx: int, epoch: int, batch_idx: int, n_batches: int, loss: float) -> None:
            if progress is None:
                return
            progress.set_description(f"level {level_idx + 1}/{len(plan)} epoch {epoch + 1}/{config.epochs}")
            progress.set_postfix(loss=f"{loss:.4f}")
            progress.update(1)

        try:
            result = cascade_train.run_cascade_training(
                config, frames, on_level_preview=_on_level_preview, on_batch_progress=_on_batch_progress
            )
        finally:
            if progress is not None:
                progress.close()
        final_n_tiles = plan[-1].n_groups
        print(
            f"OK scene={config.scene} tile_model={config.tile_model} targets={','.join(config.targets)} "
            f"cascade_levels={len(plan)} leaf_reached={plan[-1].is_leaf} final_n_groups={final_n_tiles} "
            f"final_loss={result.final_loss:.6f} checkpoint={result.checkpoint_path}"
        )
        return 0

    n_tiles = tiles.n_tiles_for(frames[0].height, frames[0].width, crop=config.tile_crop)
    logger.info("n_tiles=%d tile_model=%s targets=%s", n_tiles, config.tile_model, config.targets)
    model_cls = get_tile_model_cls(config.tile_model)
    model = model_cls(n_tiles, config.target_spec, **config.tile_model_kwargs)

    def _on_preview(epoch: int, frame) -> None:
        preview.render_preview(
            model, frame, device=device, out_path=out_dir / "previews" / f"epoch_{epoch + 1:04d}.png"
        )

    progress = None if args.no_progress else tqdm(total=config.epochs * n_batches_per_epoch, unit="batch", desc="train")

    def _on_batch_progress(epoch: int, batch_idx: int, n_batches: int, loss: float) -> None:
        if progress is None:
            return
        progress.set_description(f"epoch {epoch + 1}/{config.epochs}")
        progress.set_postfix(loss=f"{loss:.4f}")
        progress.update(1)

    try:
        result = train.run_training(config, frames, model, on_preview=_on_preview, on_batch_progress=_on_batch_progress)
    finally:
        if progress is not None:
            progress.close()

    print(
        f"OK scene={config.scene} tile_model={config.tile_model} targets={','.join(config.targets)} "
        f"epochs={result.epochs_run} n_tiles={n_tiles} final_loss={result.final_loss:.6f} "
        f"checkpoint={result.checkpoint_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
