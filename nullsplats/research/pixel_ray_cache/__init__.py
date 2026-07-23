"""Pixel Ray Cache: one independent small network per screen-space tile, trained via backprop
to map a ray (camera position + pixel-derived direction) to color, and optionally depth/normal.

Assumption/limitation: tile identity is a fixed screen-space pixel index, not a 3D world
location. This requires every training frame to share one resolution/intrinsics grid (enforced
in data.load_scene_frames), and works best for a small-baseline / near-fixed-viewpoint capture
(a locked-down camera with slight jitter, or a light-field-style rig) rather than a full object
orbit -- a large pose change reprojects different world content onto the same screen tile,
asking one independent tiny per-tile network to fit inconsistent targets, which it cannot do.
The existing gsplat/DA3/SHARP pipeline (nullsplats.backend.splat_backends) remains the right
tool for full-orbit captures.

Not wired into nullsplats.backend.splat_backends.dispatch, the training UI, or test.py -- run
directly via `python -m nullsplats.research.pixel_ray_cache.cli`.
"""

from __future__ import annotations
