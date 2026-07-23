"""Scene/frame loading for Pixel Ray Cache -- COLMAP-native, no gsplat coupling.

Deliberately does not reuse nullsplats.backend.splat_train_io.load_colmap_frames (forces a
uniform image_downscale, eagerly loads every image to GPU as part of FrameRecord); this loader
returns lightweight pose/intrinsics records only, with image pixels loaded lazily per-frame by
load_image_tensor inside the training loop, so a 4K scene is never fully materialized in memory
at once.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nullsplats.backend.colmap_io import ColmapData, load_colmap_data
from nullsplats.backend.io_cache import ScenePaths
from nullsplats.research.pixel_ray_cache.rays import qvec_tvec_to_camtoworld


@dataclass(frozen=True)
class RayCacheFrame:
    name: str
    image_path: Path
    camtoworld: torch.Tensor  # (4,4)
    K: torch.Tensor  # (3,3), native full resolution
    width: int
    height: int


@dataclass(frozen=True)
class PoseNormalization:
    center: torch.Tensor  # (3,)
    scale: float


def compute_pose_normalization(frames: list["RayCacheFrame"]) -> PoseNormalization:
    """Isotropic (single-scalar) recentering that fits every camera position into [-1,1] on
    every axis: center = centroid of camera positions, scale = largest per-axis distance from
    that centroid. A single shared scale (not fit per-axis) is required -- fitting each axis to
    exactly [-1,1] independently would be an anisotropic transform that distorts the trajectory's
    shape and breaks the geometric relationship between position and direction (ray directions
    are unit vectors and only stay consistent with positions under a uniform scale).
    """
    positions = torch.stack([f.camtoworld[:3, 3] for f in frames]).cpu()
    center = positions.mean(dim=0)
    scale = float((positions - center).abs().max().clamp_min(1e-6))
    return PoseNormalization(center=center, scale=scale)


def apply_pose_normalization(frame: "RayCacheFrame", norm: PoseNormalization) -> "RayCacheFrame":
    """Only the translation is affected -- a pure isotropic scale+recenter of the world frame
    never changes camera orientation, and ray directions (already unit vectors) are scale
    invariant, so K/rotation/width/height all stay exactly as they were.

    Known limitation, not solved here: if a checkpoint's normalize_poses setting doesn't match
    the frames it's evaluated against (e.g. reloading a checkpoint trained without normalization
    but loading frames with it on, or vice versa), ray origins will be silently wrong -- treat
    normalize_poses as fixed per scene/experiment, not something to toggle between train and
    eval of the same checkpoint. Also not yet reconciled with depth_targets.py: DA3 depth values
    are in DA3's own (unnormalized) scale, so combining depth targets with normalize_poses would
    unproject points using mismatched scales (origin normalized, depth not) -- fine for
    color-only training, a follow-up if depth+normalize_poses together are needed.
    """
    camtoworld = frame.camtoworld.clone()
    camtoworld[:3, 3] = (frame.camtoworld[:3, 3] - norm.center.to(frame.camtoworld.device)) / norm.scale
    return dataclasses.replace(frame, camtoworld=camtoworld)


def load_scene_frames(
    paths: ScenePaths, *, device: torch.device, normalize_poses: bool = False
) -> list[RayCacheFrame]:
    """Load every posed COLMAP frame at its native (un-cropped, un-downscaled) resolution.

    Raises ValueError if frames don't share one resolution/intrinsics grid -- Pixel Ray Cache
    assigns tile identity by fixed screen-space pixel index, so every training frame must sit
    on the same pixel grid (see the package docstring for the small-baseline-capture assumption
    this implies).

    normalize_poses=True recenters+rescales every camera position into [-1,1] (see
    compute_pose_normalization/apply_pose_normalization) -- off by default since it changes the
    coordinate convention a model is trained against (a checkpoint trained with it on is not
    compatible with frames loaded with it off, or vice versa).
    """
    colmap: ColmapData = load_colmap_data(paths)
    frames_dir = paths.frames_selected_dir
    records: list[RayCacheFrame] = []
    for image in colmap.images.values():
        camera = colmap.cameras[image.camera_id]
        image_path = frames_dir / image.name
        if not image_path.exists():
            image_path = frames_dir / Path(image.name).name
        if not image_path.exists():
            raise FileNotFoundError(f"Image {image.name} not found under {frames_dir}")
        fx, fy, cx, cy = camera.params
        K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device
        )
        c2w = qvec_tvec_to_camtoworld(image.qvec, image.tvec, device=device)
        records.append(
            RayCacheFrame(
                name=image.name,
                image_path=image_path,
                camtoworld=c2w,
                K=K,
                width=camera.width,
                height=camera.height,
            )
        )
    if not records:
        raise ValueError(f"No posed frames found for scene at {paths.sfm_dir}")
    first = records[0]
    if any((r.width, r.height) != (first.width, first.height) for r in records):
        raise ValueError(
            "pixel_ray_cache requires every training frame to share one resolution/intrinsics "
            "grid (tile identity is a fixed screen-space index, not a 3D location)."
        )
    if normalize_poses:
        norm = compute_pose_normalization(records)
        records = [apply_pose_normalization(r, norm) for r in records]
    return records


def load_image_tensor(path: Path, height: int, width: int, *, device: torch.device) -> torch.Tensor:
    """(H, W, 3) float32 in [0,1]; resized only if the file's size disagrees with COLMAP's camera size."""
    with Image.open(path) as handle:
        img = handle.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device)


class FrameImageCache:
    """Decoded-image cache for repeated-access training loops (a single training run visits
    every frame once per epoch, so re-reading + re-decoding the same PNG from disk N_epochs
    times is pure waste for a 500+-frame scene). Each frame's pixels are read+decoded from disk
    exactly once (as a compact CPU uint8 tensor, ~4x smaller than the float32 load_image_tensor
    produces) and kept in RAM for the rest of the process; every subsequent .get() call is just a
    cheap CPU->GPU copy + normalize, not a disk read. Bounded to O(1) images on the GPU at a time
    -- cached tensors live on CPU, never accumulate in VRAM.
    """

    def __init__(self) -> None:
        self._cache: dict[str, torch.Tensor] = {}

    def get(self, frame: "RayCacheFrame", *, device: torch.device) -> torch.Tensor:
        """(H, W, 3) float32 in [0,1] on `device`."""
        cached = self._cache.get(frame.name)
        if cached is None:
            with Image.open(frame.image_path) as handle:
                img = handle.convert("RGB")
                if img.size != (frame.width, frame.height):
                    img = img.resize((frame.width, frame.height), resample=Image.BILINEAR)
                cached = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy())
            self._cache[frame.name] = cached
        return cached.to(device=device, dtype=torch.float32) / 255.0

    def __len__(self) -> int:
        return len(self._cache)

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._cache.values())
