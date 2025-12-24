"""Depth Anything 3 backend for the unified trainer interface."""

from __future__ import annotations

from datetime import datetime
import gc
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from nullsplats.backend.colmap_io import ColmapCamera, ColmapImage
from nullsplats.backend.splat_backends.types import TrainerCapabilities, TrainingInput, TrainingOutput


class DepthAnything3Trainer:
    name = "depth_anything_3"
    capabilities = TrainerCapabilities(
        live_preview=False,
        supports_unconstrained=False,
        supports_constrained=True,
        requires_colmap=True,
    )

    def prepare(self, inputs: TrainingInput, config: dict[str, Any]) -> None:
        _ = inputs
        _validate_config(config)

    def train(self, inputs: TrainingInput, config: dict[str, Any], **_: Any) -> TrainingOutput:
        cfg = _normalize_config(config)
        _validate_config(cfg)
        _ensure_depth_anything_available()
        _cleanup_torch(cfg.get("device"))

        from depth_anything_3.api import DepthAnything3

        _install_da3_cleanup_patch()
        device = str(cfg["device"]).strip() or "cuda"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested for Depth Anything 3 but CUDA is not available.")
        if device.startswith("cuda") and ":" in device:
            try:
                torch.cuda.set_device(device)
            except Exception:
                pass
        model = _load_model(cfg, DepthAnything3)
        model = model.to(device)
        try:
            model.device = torch.device(device)
        except Exception:
            pass

        image_paths = [str(path) for path in inputs.images]
        extrinsics, intrinsics = _build_camera_matrices(inputs, inputs.images)

        image_paths, extrinsics, intrinsics = _maybe_subsample_views(
            image_paths, extrinsics, intrinsics, cfg, inputs
        )
        process_res, process_res_method = _resolve_process_res(cfg, [Path(p) for p in image_paths])

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
        output_name = f"splat_{self.name}_{timestamp}.ply"
        output_path = inputs.scene_paths.splats_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prediction = None
        try:
            prediction = model.inference(
                image=image_paths,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                align_to_input_ext_scale=cfg["align_to_input_ext_scale"],
                infer_gs=cfg["infer_gs"],
                use_ray_pose=cfg["use_ray_pose"],
                ref_view_strategy=cfg["ref_view_strategy"],
                process_res=process_res,
                process_res_method=process_res_method,
                export_dir=None,
                export_format="mini_npz",
            )
            _move_prediction_to_cpu(prediction)
        finally:
            try:
                del model
            except Exception:
                pass
            _cleanup_torch(cfg.get("device"))

        if prediction is None:
            raise RuntimeError("Depth Anything 3 inference failed without returning a prediction.")
        _export_gaussian_ply(prediction, output_path, cfg)
        _cleanup_torch(cfg.get("device"))

        return TrainingOutput(
            primary_path=output_path,
            method=self.name,
            timestamp=timestamp,
            export_format="ply",
            metrics={},
            extra_files=[],
        )


def _ensure_depth_anything_available() -> None:
    try:
        import depth_anything_3  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "Depth Anything 3 is not installed. Install it with:\n"
            "  pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3"
        ) from exc


def _cleanup_torch(device: str | None) -> None:
    try:
        torch.set_grad_enabled(False)
    except Exception:
        pass
    gc.collect()
    if device and str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _install_da3_cleanup_patch() -> None:
    try:
        from depth_anything_3.api import DepthAnything3
        from depth_anything_3.utils.io.input_processor import InputProcessor
    except Exception:
        return
    if getattr(DepthAnything3, "_nullsplats_cleanup_patch", False):
        return

    original_inference = DepthAnything3.inference

    def _patched_inference(self, *args, **kwargs):  # type: ignore[no-redef]
        prediction = original_inference(self, *args, **kwargs)
        try:
            _move_prediction_to_cpu(prediction)
        finally:
            device = None
            try:
                device = str(self._get_model_device())
            except Exception:
                device = None
            _cleanup_torch(device)
        return prediction

    DepthAnything3.inference = _patched_inference  # type: ignore[assignment]
    DepthAnything3._nullsplats_cleanup_patch = True  # type: ignore[attr-defined]
    DepthAnything3._nullsplats_orig_inference = original_inference  # type: ignore[attr-defined]

    if not getattr(InputProcessor, "_nullsplats_resize_patch", False):
        original_unify = InputProcessor._unify_batch_shapes

        def _resize_unify(self, processed_images, out_sizes, out_intrinsics):  # type: ignore[no-redef]
            if len(set(out_sizes)) <= 1:
                return processed_images, out_sizes, out_intrinsics
            max_h = max(h for h, _ in out_sizes)
            max_w = max(w for _, w in out_sizes)
            new_imgs, new_sizes, new_ixts = [], [], []
            for img_t, (H, W), K in zip(processed_images, out_sizes, out_intrinsics):
                if (H, W) != (max_h, max_w):
                    img_t = torch.nn.functional.interpolate(
                        img_t.unsqueeze(0),
                        size=(max_h, max_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                new_imgs.append(img_t)
                new_sizes.append((max_h, max_w))
                if K is None:
                    new_ixts.append(None)
                else:
                    K_adj = K.copy()
                    K_adj[0, :] *= max_w / float(W)
                    K_adj[1, :] *= max_h / float(H)
                    new_ixts.append(K_adj)
            return new_imgs, new_sizes, new_ixts

        InputProcessor._unify_batch_shapes = _resize_unify  # type: ignore[assignment]
        InputProcessor._nullsplats_resize_patch = True  # type: ignore[attr-defined]
        InputProcessor._nullsplats_orig_unify = original_unify  # type: ignore[attr-defined]


def _move_prediction_to_cpu(prediction: Any) -> None:
    gaussians = getattr(prediction, "gaussians", None)
    if gaussians is None:
        return
    for field in ("means", "harmonics", "rotations", "scales", "opacities"):
        tensor = getattr(gaussians, field, None)
        if hasattr(tensor, "detach"):
            try:
                setattr(gaussians, field, tensor.detach().cpu())
            except Exception:
                continue


def _load_model(config: dict[str, Any], da3_cls: Any) -> Any:
    pretrained_id = config.get("pretrained_id")
    weights_path = config.get("weights_path")
    model_name = config.get("model_name")
    if pretrained_id:
        return da3_cls.from_pretrained(pretrained_id)
    if weights_path:
        weights_dir = Path(weights_path)
        if weights_dir.is_dir():
            return da3_cls.from_pretrained(str(weights_dir))
        raise ValueError("Depth Anything 3 weights_path must be a directory with config.json/model.safetensors")
    if model_name:
        return da3_cls(model_name=model_name)
    raise ValueError("Depth Anything 3 requires pretrained_id, weights_path, or model_name")


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError(f"Depth Anything 3 config must be a dict, got {type(config)}")
    normalized = dict(config)
    normalized.setdefault("device", "cuda")
    normalized.setdefault("process_res", 504)
    normalized.setdefault("process_res_method", "upper_bound_resize")
    normalized.setdefault("use_input_resolution", True)
    normalized.setdefault("align_to_input_ext_scale", True)
    normalized.setdefault("infer_gs", True)
    normalized.setdefault("use_ray_pose", False)
    normalized.setdefault("ref_view_strategy", "saddle_balanced")
    normalized.setdefault("gs_views_interval", 1)
    normalized.setdefault("max_views", 0)
    normalized.setdefault("view_stride", 1)
    return normalized


def _validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise TypeError(f"Depth Anything 3 config must be a dict, got {type(config)}")
    if not (config.get("pretrained_id") or config.get("weights_path") or config.get("model_name")):
        raise ValueError("Depth Anything 3 requires pretrained_id, weights_path, or model_name")
    if not config.get("device"):
        raise ValueError("Depth Anything 3 config missing device")
    if not config.get("infer_gs", True):
        raise ValueError("Depth Anything 3 requires infer_gs=True for gs_ply export")
    if not isinstance(config.get("use_ray_pose", False), bool):
        raise TypeError("Depth Anything 3 config 'use_ray_pose' must be a bool")
    if not isinstance(config.get("align_to_input_ext_scale", True), bool):
        raise TypeError("Depth Anything 3 config 'align_to_input_ext_scale' must be a bool")


def _export_gaussian_ply(prediction: Any, output_path: Path, config: dict[str, Any]) -> None:
    from depth_anything_3.utils.gsply_helpers import export_ply, inverse_sigmoid

    gaussians = prediction.gaussians
    ctx_depth = torch.from_numpy(prediction.depth).unsqueeze(-1).to(gaussians.means)
    b = gaussians.means.shape[0]
    if b != 1:
        raise ValueError("Depth Anything 3 export expects batch_size=1.")
    src_v, out_h, out_w, _ = ctx_depth.shape
    gs_views_interval = max(1, int(config.get("gs_views_interval", 1)))

    world_means = gaussians.means
    world_shs = gaussians.harmonics
    world_rotations = gaussians.rotations
    gs_scales = gaussians.scales
    gs_opacities = inverse_sigmoid(gaussians.opacities)

    mask = torch.ones_like(ctx_depth, dtype=torch.bool)
    gstrim_h = int(8 / 256 * out_h)
    gstrim_w = int(8 / 256 * out_w)
    if gstrim_h > 0 and gstrim_w > 0:
        mask[:, :gstrim_h, :, :] = 0
        mask[:, -gstrim_h:, :, :] = 0
        mask[:, :, :gstrim_w, :] = 0
        mask[:, :, -gstrim_w:, :] = 0

    prune_by_depth_percent = 0.9
    if prune_by_depth_percent < 1.0:
        in_depths = ctx_depth
        d_percentile = torch.quantile(
            in_depths.view(in_depths.shape[0], -1), q=prune_by_depth_percent, dim=1
        ).view(-1, 1, 1, 1)
        d_mask = in_depths <= d_percentile
        mask = mask & d_mask
    mask = mask.squeeze(-1)

    def _reshape_select(element: torch.Tensor) -> torch.Tensor:
        reshaped = element[0].reshape(src_v, out_h, out_w, *element.shape[2:])
        selected = reshaped[::gs_views_interval]
        selected_mask = mask[::gs_views_interval]
        return selected[selected_mask]

    means = _reshape_select(world_means)
    scales = _reshape_select(gs_scales)
    rotations = _reshape_select(world_rotations)
    harmonics = _reshape_select(world_shs)
    opacities = _reshape_select(gs_opacities).squeeze(-1)


    export_ply(
        means=means,
        scales=scales,
        rotations=rotations,
        harmonics=harmonics,
        opacities=opacities,
        path=output_path,
        shift_and_scale=False,
        save_sh_dc_only=True,
        match_3dgs_mcmc_dev=False,
    )


def _build_camera_matrices(inputs: TrainingInput, images: Iterable[Path]) -> tuple[np.ndarray, np.ndarray]:
    ordered = _match_colmap_entries(inputs, images)
    extrinsics = np.stack([_colmap_to_extrinsics(entry) for entry in ordered], axis=0).astype(np.float32)
    intrinsics = np.stack(
        [_colmap_camera_to_intrinsics(inputs.colmap.cameras[entry.camera_id]) for entry in ordered], axis=0
    ).astype(np.float32)
    return extrinsics, intrinsics


def _resolve_process_res(config: dict[str, Any], images: Iterable[Path]) -> tuple[int, str]:
    if config.get("use_input_resolution", True):
        sizes = [_read_image_size(path) for path in images]
        first = sizes[0]
        if any(size != first for size in sizes):
            raise ValueError("Depth Anything 3 requires all images to share the same resolution for no-resize mode.")
        process_res = max(first)
        return process_res, "upper_bound_crop"
    return int(config.get("process_res", 504)), str(config.get("process_res_method", "upper_bound_resize"))


def _read_image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as handle:
        return handle.size


def _maybe_subsample_views(
    image_paths: list[str],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray | None,
    config: dict[str, Any],
    inputs: TrainingInput,
) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    stride = max(1, int(config.get("view_stride", 1) or 1))
    max_views = int(config.get("max_views", 0) or 0)
    if stride <= 1 and max_views <= 0:
        return image_paths, extrinsics, intrinsics

    indices = list(range(0, len(image_paths), stride))
    if max_views > 0:
        quality_indices = _select_views_by_colmap(indices, image_paths, inputs, max_views)
        indices = quality_indices if quality_indices is not None else _evenly_spaced(indices, max_views)
    if not indices:
        raise ValueError("View subsampling produced an empty image list.")

    image_paths = [image_paths[i] for i in indices]
    extrinsics = extrinsics[indices]
    if intrinsics is not None:
        intrinsics = intrinsics[indices]
    return image_paths, extrinsics, intrinsics


def _select_views_by_colmap(
    indices: list[int],
    image_paths: list[str],
    inputs: TrainingInput,
    max_views: int,
) -> list[int] | None:
    if max_views <= 0:
        return indices
    try:
        entries = _match_colmap_entries(inputs, [Path(image_paths[i]) for i in indices])
    except Exception:
        return None

    points = inputs.colmap.points3D
    scored: list[tuple[float, int, int]] = []
    for local_idx, entry in enumerate(entries):
        errors = []
        for point_id in entry.point3D_ids:
            if point_id < 0:
                continue
            point = points.get(point_id)
            if point is not None:
                errors.append(point.error)
        if not errors:
            continue
        mean_err = float(sum(errors) / len(errors))
        scored.append((mean_err, -len(errors), local_idx))

    if not scored:
        return None

    scored.sort()
    chosen_local = [idx for _err, _count, idx in scored[:max_views]]
    chosen_local.sort()
    return [indices[i] for i in chosen_local]


def _evenly_spaced(indices: list[int], max_views: int) -> list[int]:
    if max_views <= 0 or max_views >= len(indices):
        return indices
    step = (len(indices) - 1) / float(max_views - 1) if max_views > 1 else 1.0
    chosen = [indices[int(round(i * step))] for i in range(max_views)]
    return sorted(set(chosen))


def _match_colmap_entries(inputs: TrainingInput, images: Iterable[Path]) -> list[ColmapImage]:
    images_by_name = {entry.name: entry for entry in inputs.colmap.images.values()}
    images_by_basename = {Path(entry.name).name: entry for entry in inputs.colmap.images.values()}
    ordered: list[ColmapImage] = []
    for path in images:
        entry = images_by_name.get(path.name) or images_by_basename.get(path.name)
        if entry is None:
            raise FileNotFoundError(f"COLMAP image entry not found for {path.name}")
        ordered.append(entry)
    return ordered


def _colmap_camera_to_intrinsics(camera: ColmapCamera) -> np.ndarray:
    fx, fy, cx, cy = camera.params
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _colmap_to_extrinsics(image: ColmapImage) -> np.ndarray:
    qw, qx, qy, qz = image.qvec
    R = np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float32,
    )
    t = np.array(image.tvec, dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = t
    return extrinsics
