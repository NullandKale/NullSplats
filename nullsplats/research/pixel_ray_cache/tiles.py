"""Reshape helpers mapping a full-resolution (H,W,C) grid to/from a fixed block-size layout.

Shared by color, depth, and normal target tensors alike, and by the (origin, direction) ray
grids -- any (H,W,C) tensor goes through the same block <-> image mapping so group index N lines
up 1:1 across every tensor fed into or read out of a TileModel.

Block size can be a single int (square blocks, e.g. the leaf-level 2x2 pixel tile used by the
flat baseline) or a (block_h, block_w) tuple (rectangular blocks, used by cascade.py's coarser
levels where height and width don't necessarily halve in lockstep on a non-square grid).
"""

from __future__ import annotations

import torch

Crop = tuple[int, int, int, int]
"""(x, y, w, h) in BLOCK units -- offset and size of a block-grid sub-window."""

BlockSize = int | tuple[int, int]


def _normalize_block(block: BlockSize) -> tuple[int, int]:
    if isinstance(block, tuple):
        return block
    return (block, block)


def image_to_tiles(image: torch.Tensor, tile: BlockSize = 2, crop: Crop | None = None) -> torch.Tensor:
    """(H, W, C) -> (N_blocks, block_h*block_w, C), grouping each block_h x block_w block of
    adjacent pixels (row-major) into one group. N_blocks = (H/block_h)*(W/block_w), or the
    cropped subset if `crop` is given."""
    tile_h, tile_w = _normalize_block(tile)
    h, w, c = image.shape
    if h % tile_h != 0 or w % tile_w != 0:
        raise ValueError(f"image size ({h},{w}) not divisible by block=({tile_h},{tile_w})")
    x = image.view(h // tile_h, tile_h, w // tile_w, tile_w, c)  # (Ht, ty, Wt, tx, C)
    x = x.permute(0, 2, 1, 3, 4).contiguous()  # (Ht, Wt, ty, tx, C) -- block index order
    if crop is not None:
        cx, cy, cw, ch = crop
        x = x[cy : cy + ch, cx : cx + cw]
    ht, wt = x.shape[0], x.shape[1]
    return x.reshape(ht * wt, tile_h * tile_w, c)


def rays_to_tiles(
    origins: torch.Tensor, directions: torch.Tensor, tile: BlockSize = 2, crop: Crop | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """image_to_tiles applied identically to the (H,W,3) origin/direction grids."""
    return image_to_tiles(origins, tile, crop), image_to_tiles(directions, tile, crop)


def tiles_to_image(tiles: torch.Tensor, height: int, width: int, tile: BlockSize = 2) -> torch.Tensor:
    """Inverse of image_to_tiles (uncropped only -- a cropped block batch has no full-image
    reconstruction, since the pixels outside the crop were never produced)."""
    tile_h, tile_w = _normalize_block(tile)
    ht, wt = height // tile_h, width // tile_w
    n, s, c = tiles.shape
    if n != ht * wt or s != tile_h * tile_w:
        raise ValueError(f"tiles shape {tuple(tiles.shape)} incompatible with {height}x{width}, block=({tile_h},{tile_w})")
    x = tiles.view(ht, wt, tile_h, tile_w, c)  # (Ht, Wt, ty, tx, C)
    x = x.permute(0, 2, 1, 3, 4).contiguous()  # (Ht, ty, Wt, tx, C)
    return x.reshape(height, width, c)


def n_tiles_for(height: int, width: int, tile: BlockSize = 2, crop: Crop | None = None) -> int:
    """Number of blocks image_to_tiles will produce for this (height,width,tile,crop)."""
    tile_h, tile_w = _normalize_block(tile)
    if crop is not None:
        _cx, _cy, cw, ch = crop
        return cw * ch
    ht, wt = height // tile_h, width // tile_w
    return ht * wt
