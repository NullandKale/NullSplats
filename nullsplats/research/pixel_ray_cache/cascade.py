"""Coarse-to-fine quadtree structure over the tile grid.

Level 0 covers the whole tile grid with a single group (one network trained on every tile in
every image). Each subsequent level halves group size in both dimensions (quartering the group's
area, hence "quadtree"), stopping once groups reach a single tile -- full per-tile resolution,
matching the flat baseline's granularity exactly -- or a configured max level count is reached,
whichever comes first.

To keep every level's block size an exact, regular divisor of the grid (needed so tiles.py's
single reshape trick works at every level, not just power-of-two-friendly ones), the tile grid is
conceptually padded up to the next power of two in each dimension independently. Padding is
handled by the caller (cascade_train.py): this module only computes the level geometry and the
parent-to-child index mapping used to warm-start each level's networks from the previous level's
trained weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def padded_size(size: int) -> int:
    """Next power of two >= size (size must be >= 1)."""
    if size < 1:
        raise ValueError("size must be >= 1")
    if size == 1:
        return 1
    return 1 << (size - 1).bit_length()


@dataclass(frozen=True)
class CascadeLevelSpec:
    level: int
    region_h: int  # tiles per group, height axis, at this level
    region_w: int  # tiles per group, width axis, at this level
    regions_h: int  # groups along the height axis (= height_tiles_padded / region_h)
    regions_w: int  # groups along the width axis

    @property
    def n_groups(self) -> int:
        return self.regions_h * self.regions_w

    @property
    def is_leaf(self) -> bool:
        return self.region_h == 1 and self.region_w == 1


def build_cascade_plan(height_tiles: int, width_tiles: int, max_levels: int) -> list[CascadeLevelSpec]:
    """Level 0 = one group covering the padded grid; each level halves region_h/region_w until
    both reach 1 (full per-tile resolution) or len(plan) == max_levels, whichever comes first.
    """
    if max_levels < 1:
        raise ValueError("max_levels must be >= 1")
    ht_pad = padded_size(height_tiles)
    wt_pad = padded_size(width_tiles)

    plan: list[CascadeLevelSpec] = []
    level = 0
    while True:
        region_h = max(1, ht_pad >> level)
        region_w = max(1, wt_pad >> level)
        spec = CascadeLevelSpec(
            level=level,
            region_h=region_h,
            region_w=region_w,
            regions_h=ht_pad // region_h,
            regions_w=wt_pad // region_w,
        )
        plan.append(spec)
        if spec.is_leaf or len(plan) >= max_levels:
            break
        level += 1
    return plan


def parent_index_for(child: CascadeLevelSpec, parent: CascadeLevelSpec) -> torch.Tensor:
    """(child.n_groups,) long tensor mapping each child group's flat index to its parent
    group's flat index in the previous level. Flat index order is row-major over
    (regions_h, regions_w), matching tiles.image_to_tiles's block ordering.
    """
    ci = torch.arange(child.regions_h)
    cj = torch.arange(child.regions_w)
    # Row/col of the padded-grid tile that this child group's top-left tile sits at, mapped back
    # into the parent's (regions_h, regions_w) grid via integer division of region sizes.
    pi = (ci * child.region_h) // parent.region_h  # (regions_h,)
    pj = (cj * child.region_w) // parent.region_w  # (regions_w,)
    parent_row = pi.view(-1, 1).expand(child.regions_h, child.regions_w)  # (regions_h, regions_w)
    parent_col = pj.view(1, -1).expand(child.regions_h, child.regions_w)
    parent_flat = parent_row * parent.regions_w + parent_col
    return parent_flat.reshape(-1).to(torch.long)
