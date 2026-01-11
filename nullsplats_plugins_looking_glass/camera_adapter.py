"""Camera helpers for Looking Glass quilt rendering."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch

from nullsplats.ui.gl_canvas import CameraView, _camera_to_world


def base_pose_from_view(base_view: CameraView) -> Tuple[np.ndarray, np.ndarray]:
    camtoworld = _camera_to_world(base_view)
    pos = camtoworld[:3, 3]
    target = base_view.target
    return pos.detach().cpu().numpy(), target.detach().cpu().numpy()


def generate_view_offsets(
    base_view: CameraView,
    view_count: int,
    baseline_scale: float,
    *,
    aspect: float,
    fov_deg: float = 45.0,
    viewcone_deg: float = 40.0,
    focus: float = 0.0,
) -> List[Tuple[float, float]]:
    if view_count <= 0:
        return []
    camera_distance = float(base_view.distance)
    camera_offset = camera_distance * math.tan(math.radians(viewcone_deg))
    # Mirror LKGCamera: size is the rendered volume size, camera_distance = size / tan(fov).
    size = camera_distance * math.tan(math.radians(fov_deg))
    depthiness = float(baseline_scale)
    offsets: List[Tuple[float, float]] = []
    for idx in range(view_count):
        normalized_view = 0.5 if view_count == 1 else (idx / (view_count - 1))
        offset = -(normalized_view - 0.5) * depthiness * camera_offset
        frustum_shift = (normalized_view - 0.5) * focus
        proj_shift = 0.0
        if size > 1e-6:
            proj_shift = (offset * 2.0 / (size * aspect)) + frustum_shift
        offsets.append((float(offset), float(proj_shift)))
    return offsets


def generate_view_poses(base_view: CameraView, view_count: int, baseline_scale: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create per-view camera poses for a quilt render."""
    if view_count <= 0:
        return []
    camtoworld = _camera_to_world(base_view)
    base_pos = camtoworld[:3, 3]
    # Bridge expects view index 0 to be the left-most eye; negate right to align parallax
    right = -camtoworld[:3, 0]
    target = base_view.target
    forward = target - base_pos
    forward = forward / (torch.linalg.norm(forward) + 1e-8)
    camera_distance = float(base_view.distance)
    center = (view_count - 1) / 2.0
    poses: List[Tuple[np.ndarray, np.ndarray]] = []
    baseline = float(baseline_scale) * camera_distance
    for idx in range(view_count):
        normalized_view = 0.5 if view_count == 1 else (idx / (view_count - 1))
        offset = -(normalized_view - 0.5) * baseline
        pos = base_pos + right * offset
        # Match Looking Glass's parallel camera shift: move target by the same offset.
        tgt = target + right * offset
        poses.append((pos.detach().cpu().numpy(), tgt.detach().cpu().numpy()))
    return poses
