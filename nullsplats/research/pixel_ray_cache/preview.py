"""Render a trained TileModel from a camera pose, either to a PNG or a raw numpy array.

The array-producing functions (render_color_array/render_depth_array/render_normal_array) are
the shared core -- render_preview/render_depth_preview/render_normal_preview are thin PNG-saving
wrappers around them, and viewer.py's interactive window calls the array functions directly on
every frame instead of round-tripping through disk.

The production Gaussian-splat OpenGL viewer/export pipeline (nullsplats/ui/gl_canvas.py etc.) is
hard-wired to the 3D-Gaussian vertex schema and cannot render this representation, so neither
this module nor viewer.py integrate with it -- both are standalone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nullsplats.research.pixel_ray_cache import rays, tiles
from nullsplats.research.pixel_ray_cache.data import RayCacheFrame
from nullsplats.research.pixel_ray_cache.tile_models.base import TileModel
from nullsplats.research.pixel_ray_cache.tiles import BlockSize


def _eval_grid(
    model: TileModel,
    frame: RayCacheFrame,
    *,
    device: torch.device,
    tile: BlockSize = 2,
    render_height: int | None = None,
    render_width: int | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate model over frame's own camera pose at the given block size.

    render_height/render_width default to frame.height/frame.width (the flat baseline's leaf
    case); pass explicit padded dimensions for a cascade level whose block size only evenly
    divides the padded grid, not the native one (see cascade_train.py / train.LoadedCheckpoint).

    For a genuinely novel pose, build a synthetic RayCacheFrame with the target K/camtoworld and
    call this the same way -- see the package docstring for why that's a diagnostic curiosity
    rather than the primary use case (tile identity is fixed by screen-space index, not world
    location); viewer.py does exactly this to let you fly a camera around interactively.
    """
    height = render_height if render_height is not None else frame.height
    width = render_width if render_width is not None else frame.width
    origins, directions = rays.pixel_rays(frame.K, frame.camtoworld, height, width, device=device)
    o_tiles, d_tiles = tiles.rays_to_tiles(origins, directions, tile=tile)
    ray_inputs = torch.cat([o_tiles, d_tiles], dim=-1)
    with torch.no_grad():
        pred = model(ray_inputs)
    return {"pred": pred, "height": height, "width": width}


def _crop(image: torch.Tensor, crop_to: tuple[int, int] | None) -> torch.Tensor:
    if crop_to is None:
        return image
    crop_h, crop_w = crop_to
    return image[:crop_h, :crop_w]


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    return (image.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def render_color_array(
    model: TileModel,
    frame: RayCacheFrame,
    *,
    device: torch.device,
    tile: BlockSize = 2,
    render_height: int | None = None,
    render_width: int | None = None,
    crop_to: tuple[int, int] | None = None,
) -> np.ndarray:
    """(H,W,3) uint8 RGB array. crop_to defaults to frame's native (height,width) -- pass None
    explicitly to keep the full (possibly padded) render instead."""
    result = _eval_grid(model, frame, device=device, tile=tile, render_height=render_height, render_width=render_width)
    image = tiles.tiles_to_image(result["pred"]["color"], result["height"], result["width"], tile=tile)
    image = _crop(image, (frame.height, frame.width) if crop_to is None else crop_to)
    return _to_uint8(image)


def render_depth_array(
    model: TileModel,
    frame: RayCacheFrame,
    *,
    device: torch.device,
    tile: BlockSize = 2,
    render_height: int | None = None,
    render_width: int | None = None,
    crop_to: tuple[int, int] | None = None,
) -> np.ndarray:
    """(H,W) uint8 grayscale array, normalized to this render's own min/max (within the crop).
    Only valid if model.target_spec includes a 'depth' entry."""
    if not any(name == "depth" for name, _, _ in model.target_spec):
        raise ValueError("model has no 'depth' target -- train with --targets color,depth first")
    result = _eval_grid(model, frame, device=device, tile=tile, render_height=render_height, render_width=render_width)
    depth_image = tiles.tiles_to_image(result["pred"]["depth"], result["height"], result["width"], tile=tile)
    depth_image = _crop(depth_image, (frame.height, frame.width) if crop_to is None else crop_to)[..., 0]
    lo, hi = float(depth_image.min()), float(depth_image.max())
    normalized = (depth_image - lo) / max(hi - lo, 1e-6)
    return (normalized.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def render_normal_array(
    model: TileModel,
    frame: RayCacheFrame,
    *,
    device: torch.device,
    tile: BlockSize = 2,
    render_height: int | None = None,
    render_width: int | None = None,
    crop_to: tuple[int, int] | None = None,
) -> np.ndarray:
    """(H,W,3) uint8 RGB-encoded normal map ((n+1)/2). Only valid if model.target_spec includes
    a 'normal' entry."""
    if not any(name == "normal" for name, _, _ in model.target_spec):
        raise ValueError("model has no 'normal' target -- train with --targets color,depth,normal first")
    result = _eval_grid(model, frame, device=device, tile=tile, render_height=render_height, render_width=render_width)
    normal_image = tiles.tiles_to_image(result["pred"]["normal"], result["height"], result["width"], tile=tile)
    normal_image = _crop(normal_image, (frame.height, frame.width) if crop_to is None else crop_to)
    return _to_uint8((normal_image + 1.0) * 0.5)


def render_preview(
    model: TileModel,
    frame: RayCacheFrame,
    *,
    device: torch.device,
    out_path: Path,
    tile: BlockSize = 2,
    render_height: int | None = None,
    render_width: int | None = None,
) -> None:
    """Save the reconstructed color channel as an RGB PNG, cropped to frame's native size (in
    case render_height/render_width were padded larger, e.g. a cascade level)."""
    arr = render_color_array(model, frame, device=device, tile=tile, render_height=render_height, render_width=render_width)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(out_path)


def render_depth_preview(
    model: TileModel,
    frame: RayCacheFrame,
    *,
    device: torch.device,
    out_path: Path,
    tile: BlockSize = 2,
    render_height: int | None = None,
    render_width: int | None = None,
) -> None:
    """Save the predicted depth channel as a grayscale PNG. Only valid if model.target_spec
    includes a 'depth' entry."""
    arr = render_depth_array(model, frame, device=device, tile=tile, render_height=render_height, render_width=render_width)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(out_path)


def render_normal_preview(
    model: TileModel,
    frame: RayCacheFrame,
    *,
    device: torch.device,
    out_path: Path,
    tile: BlockSize = 2,
    render_height: int | None = None,
    render_width: int | None = None,
) -> None:
    """Save the predicted normal channel as an RGB-encoded normal map ((n+1)/2). Only valid if
    model.target_spec includes a 'normal' entry."""
    arr = render_normal_array(model, frame, device=device, tile=tile, render_height=render_height, render_width=render_width)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(out_path)
