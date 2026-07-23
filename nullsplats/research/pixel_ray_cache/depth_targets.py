"""Optional dense per-pixel depth source for Pixel Ray Cache, using Depth Anything 3.

Depth Anything 3 is already vendored as a submodule (tools/depth-anything-3) and already used by
nullsplats/backend/splat_backends/depth_anything3_trainer.py, which runs one batched
model.inference(...) call across every view in a scene and reads back prediction.depth (shape
[V, H, W]) -- but only to prune/mask Gaussians before PLY export, then discards it. This module
is the first place in the repo that persists that depth for downstream use.

Import of the DA3 package is lazy -- only touched when a caller actually requests a "depth"
target, so the color-only Pixel Ray Cache path has zero DA3 dependency at runtime.

Inference is chunked (a handful of frames per forward pass, not the whole scene at once): DA3's
backbone is a large ViT, and pushing hundreds of frames through one forward pass allocates
activation memory that scales with the full batch (observed: a 589-frame batch tried to allocate
15.65 GiB in a single op and OOM'd on an 8 GiB GPU). A small chunk_size keeps DA3's peak memory
bounded regardless of scene size, and matters even more here because the Pixel Ray Cache tile
model + optimizer are usually already resident on the same GPU during training.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch

from nullsplats.research.pixel_ray_cache.data import RayCacheFrame

DEFAULT_CHUNK_SIZE = 4


def compute_or_load_depth_for_scene(
    frames: list[RayCacheFrame],
    *,
    cache_dir: Path,
    pretrained_id: str,
    device: torch.device,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, torch.Tensor]:
    """{frame.name: (H,W,1) depth} for every frame, using a per-scene disk cache. Any frames
    missing from cache are computed via chunked DA3 inference (see _run_da3_depth_chunked) --
    so a multi-epoch training loop never re-runs DA3 inference at all after the first call for a
    given scene, and even that first call never holds more than chunk_size frames' worth of
    activations at once.

    Returned tensors are on CPU, not `device` -- for a 500+-frame scene, holding every frame's
    depth map resident on the GPU for the whole training run adds up fast (2+ GiB for a modest
    scene) on top of the tile model + optimizer that also need to fit. Callers should move a
    specific frame's depth to GPU just-in-time inside their per-image training step, the same way
    color images are already loaded lazily.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, torch.Tensor] = {}
    missing: list[RayCacheFrame] = []
    for frame in frames:
        cache_path = cache_dir / f"{Path(frame.name).stem}.npy"
        if cache_path.exists():
            depth = np.load(cache_path)
            result[frame.name] = torch.from_numpy(depth).to(dtype=torch.float32).unsqueeze(-1)
        else:
            missing.append(frame)

    if missing:
        depths = _run_da3_depth_chunked(missing, pretrained_id=pretrained_id, device=device, chunk_size=chunk_size)
        for frame, depth_map in zip(missing, depths):
            cache_path = cache_dir / f"{Path(frame.name).stem}.npy"
            np.save(cache_path, depth_map.astype(np.float32))
            result[frame.name] = torch.from_numpy(depth_map).to(dtype=torch.float32).unsqueeze(-1)
    return result


def _run_da3_depth_chunked(
    frames: list[RayCacheFrame], *, pretrained_id: str, device: torch.device, chunk_size: int
) -> list[np.ndarray]:
    """Loads the DA3 model once, then runs inference over `frames` in groups of `chunk_size`
    (last chunk may be smaller), returning one (H,W) depth array per frame in input order.
    extrinsics/intrinsics are built directly from RayCacheFrame poses (rather than re-parsing
    ColmapImage/ColmapCamera like depth_anything3_trainer.py does) since we already have them;
    use_input_resolution-equivalent process_res since load_scene_frames already enforces one
    shared resolution across frames. Explicitly frees the (multi-GB) DA3 model and clears the
    CUDA cache before returning, since the caller still needs GPU room for the much smaller
    Pixel Ray Cache tile model + optimizer to keep training.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    from depth_anything_3.api import DepthAnything3  # vendored under tools/depth-anything-3

    model = DepthAnything3.from_pretrained(pretrained_id).to(device).eval()
    process_res = max(frames[0].width, frames[0].height)

    depths: list[np.ndarray] = []
    try:
        for start in range(0, len(frames), chunk_size):
            chunk = frames[start : start + chunk_size]
            image_paths = [str(frame.image_path) for frame in chunk]
            extrinsics = np.stack(
                [torch.linalg.inv(frame.camtoworld).cpu().numpy() for frame in chunk], axis=0
            ).astype(np.float32)
            intrinsics = np.stack([frame.K.cpu().numpy() for frame in chunk], axis=0).astype(np.float32)

            with torch.no_grad():
                prediction = model.inference(
                    image=image_paths,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    align_to_input_ext_scale=False,
                    infer_gs=False,
                    use_ray_pose=False,
                    ref_view_strategy="saddle_balanced",
                    process_res=process_res,
                    process_res_method="upper_bound_crop",
                    export_dir=None,
                    export_format="mini_npz",
                )
            chunk_depth = np.asarray(prediction.depth)  # (chunk_len, H, W)
            if chunk_depth.shape[0] != len(chunk):
                raise ValueError(
                    f"DA3 returned {chunk_depth.shape[0]} depth maps for a chunk of {len(chunk)} frames."
                )
            depths.extend(chunk_depth[i] for i in range(chunk_depth.shape[0]))
            del prediction, chunk_depth
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(depths) != len(frames):
        raise ValueError(f"DA3 produced {len(depths)} depth maps for {len(frames)} input frames.")
    return depths
