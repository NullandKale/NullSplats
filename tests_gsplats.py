"""Gsplat hyperparameter tuning runner for low/medium/high quality profiles.

This runner is designed around common 3DGS/gsplat densification defaults (start ~500 iters, refine every ~100, prune at opacity ~0.005),
while exposing reproducible low/medium/high presets and producing a machine-readable summary.

References (for preset intuition):
- graphdeco-inria gaussian-splatting defaults (densify_from_iter=500, densification_interval=100, opacity_reset_interval=3000, densify_grad_threshold=2e-4, lambda_dssim=0.2).
- gsplat DefaultStrategy defaults (refine_start_iter=500, refine_every=100, refine_stop_iter=25000, min_opacity=0.005).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from nullsplats.backend.io_cache import ScenePaths, ensure_scene_dirs
from nullsplats.backend.colmap_io import find_text_model, parse_cameras, parse_images, find_points3d
from nullsplats.backend.splat_train import train_scene
from nullsplats.backend.splat_train_config import SplatTrainingConfig


DEFAULT_SCENE_ID = "20260101_131646"
DEFAULT_CACHE_ROOT = "cache"
DEFAULT_PROFILES = "low,medium,high"
DEFAULT_OUTPUT_PREFIX = "gsplat_tune"
DEFAULT_RUN_ID = ""
DEFAULT_DEVICE = "cuda:0"
DEFAULT_SEED = 42
DEFAULT_OVERWRITE = False
DEFAULT_REUSE_BASE = False
DEFAULT_DISABLE_SNAPSHOTS = True
DEFAULT_CLONE_MODE = "copy"
DEFAULT_SUMMARY_DIRNAME = "tuning_runs"
DEFAULT_FAIL_FAST = False

# Baseline defaults for SplatTrainingConfig, aligned with common 3DGS/gsplat practice where possible.
GSPLAT_DEFAULTS: dict[str, object] = {
    "cuda_toolkit_path": "",
    "iterations": 30_000,
    "snapshot_interval": 7_000,
    "device": "cuda:0",
    "export_format": "ply",
    "max_points": 0,
    "image_downscale": 1,
    "batch_size": 1,
    "sh_degree": 3,
    "sh_degree_interval": 1000,
    "means_lr": 1.6e-4,
    "scales_lr": 5e-3,
    "opacities_lr": 5e-2,
    "quats_lr": 1e-3,
    "sh_lr": 2.5e-3,
    "ssim_weight": 0.2,
    "lr_final_scale": 0.01,
    "pose_opt": False,
    "pose_opt_lr": 1e-5,
    "pose_opt_reg": 1e-6,
    "pose_noise": 0.0,
    "app_opt": False,
    "app_embed_dim": 16,
    "app_feature_dim": 32,
    "app_opt_lr": 1e-3,
    "app_opt_reg": 1e-6,
    "densify_start": 500,
    "densify_interval": 100,
    "densify_max_points": 5_000_000,
    "densify_opacity_threshold": 0.005,
    "densify_scale_threshold": 0.01,
    "densify_scale_multiplier": 0.6,
    "densify_position_noise": 0.5,
    "prune_opacity_threshold": 0.005,
    "prune_scale_threshold": 0.12,
    "init_scale": 1.0,
    "min_scale": 1e-4,
    "max_scale": 0.2,
    "opacity_bias": 0.05,
    "opacity_reg": 0.0,
    "random_background": False,
    "loss_l1_weight": 1.0,
    "seed": 42,
    "preview_interval_seconds": 0.0,
    "preview_min_iters": 100,
    "max_preview_points": 0,
}

# Presets focus on quality-per-budget:
# - low: strong quality with constrained points/time (good for iteration cycles).
# - medium: near "paper-like" training length but capped points for stability.
# - high: maximal detail with longer point budget, conservative pruning.
TUNED_PRESETS: dict[str, dict[str, object]] = {
    "low": {
        "iterations": 12_000,
        "max_points": 1_000_000,
        "densify_max_points": 1_000_000,
        "image_downscale": 1,
        "batch_size": 1,
        "sh_degree": 2,
        "ssim_weight": 0.2,
        "opacity_reg": 0.0002,
        "opacity_bias": 0.05,
        "densify_start": 300,
        "densify_interval": 150,
        "densify_opacity_threshold": 0.0045,
        "densify_scale_threshold": 0.011,
        "prune_opacity_threshold": 0.0045,
        "prune_scale_threshold": 0.13,
        "max_scale": 0.2,
    },
    "medium": {
        "iterations": 30_000,
        "max_points": 2_500_000,
        "densify_max_points": 2_500_000,
        "image_downscale": 1,
        "batch_size": 1,
        "sh_degree": 3,
        "ssim_weight": 0.2,
        "opacity_reg": 0.0001,
        "opacity_bias": 0.05,
        "densify_start": 500,
        "densify_interval": 100,
        "densify_opacity_threshold": 0.0045,
        "densify_scale_threshold": 0.01,
        "prune_opacity_threshold": 0.0045,
        "prune_scale_threshold": 0.12,
        "max_scale": 0.22,
    },
    "high": {
        "iterations": 40_000,
        "max_points": 4_000_000,
        "densify_max_points": 4_000_000,
        "image_downscale": 1,
        "batch_size": 1,
        "sh_degree": 3,
        "ssim_weight": 0.2,
        "opacity_reg": 0.00005,
        "opacity_bias": 0.05,
        "densify_start": 500,
        "densify_interval": 100,
        "densify_opacity_threshold": 0.004,
        "densify_scale_threshold": 0.009,
        "prune_opacity_threshold": 0.004,
        "prune_scale_threshold": 0.1,
        "max_scale": 0.25,
    },
}


@dataclass(frozen=True)
class ProfileResult:
    profile: str
    scene_id: str
    seconds: float
    output: str
    config_json: str
    error: str


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gsplat training with tuned low/medium/high presets.")
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID, help="Base scene id containing inputs/outputs.")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT, help="Cache root for inputs/outputs.")
    parser.add_argument("--profiles", default=DEFAULT_PROFILES, help="Comma-separated profiles to run (low,medium,high).")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX, help="Prefix for generated scene ids.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Optional run id suffix (defaults to timestamp).")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="CUDA device id for training.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for training.")
    parser.add_argument("--overwrite", action="store_true", default=DEFAULT_OVERWRITE, help="Delete existing output scenes if they already exist.")
    parser.add_argument("--reuse-base", action="store_true", default=DEFAULT_REUSE_BASE, help="Train directly on the base scene id (overwrites its splats).")
    parser.add_argument("--disable-snapshots", action="store_true", default=DEFAULT_DISABLE_SNAPSHOTS, help="Skip intermediate checkpoints (keep final only).")
    parser.add_argument("--clone-mode", choices=["copy", "symlink"], default=DEFAULT_CLONE_MODE, help="How to clone inputs to new scene ids.")
    parser.add_argument("--summary-dirname", default=DEFAULT_SUMMARY_DIRNAME, help="Directory under cache root to write JSON summaries.")
    parser.add_argument("--fail-fast", action="store_true", default=DEFAULT_FAIL_FAST, help="Stop after first failed profile.")
    parser.add_argument("--print-presets", action="store_true", default=False, help="Print resolved preset configs and exit.")
    parser.add_argument("--override-json", default="", help="Path to JSON dict of config overrides applied to all profiles.")
    return parser.parse_args(argv)


def _split_profiles(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _require_base_scene(scene_id: str, cache_root: Path) -> ScenePaths:
    base_paths = ScenePaths(scene_id, cache_root=cache_root)
    if not base_paths.frames_selected_dir.exists():
        raise FileNotFoundError(f"Missing frames_selected at {base_paths.frames_selected_dir}")
    if not base_paths.sfm_dir.exists():
        raise FileNotFoundError(f"Missing sfm data at {base_paths.sfm_dir}")
    return base_paths


def _read_frame_dir_summary(frames_dir: Path) -> dict[str, object]:
    files = sorted([p for p in frames_dir.iterdir() if p.is_file()])
    total = len(files)
    resolution = ""
    if files:
        try:
            from PIL import Image

            with Image.open(files[0]) as handle:
                resolution = f"{handle.width}x{handle.height}"
        except Exception:
            resolution = ""
    return {
        "frames_selected_dir": str(frames_dir),
        "selected_count": total,
        "resolution": resolution,
        "sample_name": files[0].name if files else "",
    }


def _count_sparse_points(paths: ScenePaths) -> dict[str, object]:
    candidates = [
        paths.sfm_dir / "sparse" / "model.ply",
        paths.sfm_dir / "sparse" / "0" / "points3D.ply",
        paths.sfm_dir / "sparse" / "0" / "points3D.txt",
        paths.sfm_dir / "sparse" / "points3D.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".txt":
            count = 0
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line or line.startswith("#"):
                        continue
                    count += 1
            return {"sparse_points_count": count, "sparse_points_path": str(path)}
        if path.suffix.lower() == ".ply":
            with path.open("rb") as handle:
                vertex_count = 0
                while True:
                    line = handle.readline()
                    if not line:
                        break
                    text = line.decode("ascii", errors="ignore").strip()
                    if text.startswith("element vertex"):
                        parts = text.split()
                        if len(parts) >= 3:
                            try:
                                vertex_count = int(parts[2])
                            except ValueError:
                                vertex_count = 0
                    if text == "end_header":
                        break
            return {"sparse_points_count": vertex_count, "sparse_points_path": str(path)}
    return {"sparse_points_count": 0, "sparse_points_path": ""}


def _read_colmap_summary(paths: ScenePaths) -> dict[str, object]:
    summary: dict[str, object] = {
        "cameras_count": 0,
        "images_count": 0,
        "cameras_path": "",
        "images_path": "",
        "points3d_path": "",
    }
    try:
        cameras_txt, images_txt = find_text_model(paths)
        cameras = parse_cameras(cameras_txt)
        images = parse_images(images_txt)
        summary.update(
            {
                "cameras_count": len(cameras),
                "images_count": len(images),
                "cameras_path": str(cameras_txt),
                "images_path": str(images_txt),
            }
        )
        points_path = find_points3d(paths, cameras_txt.parent)
        if points_path is not None:
            summary["points3d_path"] = str(points_path)
    except Exception:
        pass
    return summary


def _read_input_summary(paths: ScenePaths) -> dict[str, object]:
    frames_selected = _read_frame_dir_summary(paths.frames_selected_dir)
    frames_selected["frames_selected_dir"] = str(paths.frames_selected_dir)
    frames_all = _read_frame_dir_summary(paths.frames_all_dir) if paths.frames_all_dir.exists() else {}
    if frames_all:
        frames_all["frames_all_dir"] = str(paths.frames_all_dir)
    metadata: dict[str, object] = {}
    try:
        from nullsplats.backend.io_cache import load_metadata

        metadata = load_metadata(paths.scene_id, cache_root=paths.cache_root)
    except Exception:
        metadata = {}
    sparse_summary = _count_sparse_points(paths)
    colmap_summary = _read_colmap_summary(paths)
    return {
        "frames_selected": frames_selected,
        "frames_all": frames_all,
        "metadata_source_path": str(metadata.get("source_path", "")),
        "metadata_selected_count": len(metadata.get("selected_frames", []) or []),
        "colmap": colmap_summary,
        "sparse": sparse_summary,
    }


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to find unique path for {path}")


def _symlink_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
            os.symlink(path, target)


def _clone_inputs(base_paths: ScenePaths, target_scene_id: str, cache_root: Path, *, overwrite: bool, clone_mode: str) -> ScenePaths:
    target_paths = ScenePaths(target_scene_id, cache_root=cache_root)
    if target_paths.inputs_root.exists() or target_paths.outputs_root.exists():
        if overwrite:
            shutil.rmtree(target_paths.inputs_root, ignore_errors=True)
            shutil.rmtree(target_paths.outputs_root, ignore_errors=True)
        else:
            raise FileExistsError(f"Output scene already exists: {target_scene_id}")
    target_paths = ensure_scene_dirs(target_scene_id, cache_root=cache_root)
    if clone_mode == "copy":
        shutil.copytree(base_paths.frames_selected_dir, target_paths.frames_selected_dir, dirs_exist_ok=True)
        shutil.copytree(base_paths.sfm_dir, target_paths.sfm_dir, dirs_exist_ok=True)
    if clone_mode == "symlink":
        _symlink_tree(base_paths.frames_selected_dir, target_paths.frames_selected_dir)
        _symlink_tree(base_paths.sfm_dir, target_paths.sfm_dir)
    if base_paths.metadata_path.exists():
        target_paths.inputs_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_paths.metadata_path, target_paths.metadata_path)
    return target_paths


def _load_overrides(path: str) -> dict[str, object]:
    if not path.strip():
        return {}
    override_path = Path(path).expanduser()
    data = json.loads(override_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"override-json must be a JSON object, got {type(data).__name__}")
    return {str(k): v for k, v in data.items()}


def _build_config(profile: str, device: str, seed: int, *, disable_snapshots: bool, overrides: dict[str, object]) -> SplatTrainingConfig:
    if profile not in TUNED_PRESETS:
        raise ValueError(f"Unknown profile: {profile}")
    base: dict[str, object] = dict(GSPLAT_DEFAULTS)
    base.update(TUNED_PRESETS[profile])
    base.update(overrides)
    iterations = int(base["iterations"])
    if disable_snapshots:
        base["snapshot_interval"] = iterations
    if not disable_snapshots:
        base["snapshot_interval"] = max(1, iterations // 5)
    base["device"] = device
    base["export_format"] = "ply"
    base["seed"] = seed
    base["preview_interval_seconds"] = 0.0
    return SplatTrainingConfig(**base)


def _format_label(profile: str, config: SplatTrainingConfig) -> str:
    return f"{profile}_it{config.iterations}_pts{config.max_points}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_profiles(
    profiles: Iterable[str],
    *,
    base_scene_id: str,
    cache_root: Path,
    output_prefix: str,
    run_id: str,
    device: str,
    seed: int,
    overwrite: bool,
    reuse_base: bool,
    disable_snapshots: bool,
    clone_mode: str,
    summary_dirname: str,
    fail_fast: bool,
    overrides: dict[str, object],
) -> int:
    base_paths = _require_base_scene(base_scene_id, cache_root)
    base_summary = _read_input_summary(base_paths)
    results: list[ProfileResult] = []
    total_start = time.perf_counter()
    exit_code = 0

    for profile in profiles:
        start = time.perf_counter()
        scene_id = ""
        output_path: Path | None = None
        config_path: Path | None = None
        error = ""
        try:
            config = _build_config(profile, device, seed, disable_snapshots=disable_snapshots, overrides=overrides)
            if reuse_base:
                scene_id = base_scene_id
                scene_paths = base_paths
            if not reuse_base:
                scene_id = f"{output_prefix}_{profile}_{run_id}"
                scene_paths = _clone_inputs(base_paths, scene_id, cache_root, overwrite=overwrite, clone_mode=clone_mode)
            input_summary = _read_input_summary(scene_paths)
            frames_summary = input_summary.get("frames_selected", {})
            print(
                "Starting profile={profile} scene={scene_id} device={device} iters={iters} max_points={max_points} "
                "frames={frames} res={res}".format(
                    profile=profile,
                    scene_id=scene_id,
                    device=config.device,
                    iters=config.iterations,
                    max_points=config.max_points,
                    frames=frames_summary.get("selected_count", 0),
                    res=frames_summary.get("resolution", ""),
                )
            )
            result = train_scene(scene_id, config, cache_root=cache_root)
            label = _format_label(profile, config)
            output_path = _unique_path(scene_paths.splats_dir / f"{label}.ply")
            shutil.copy2(result.last_checkpoint, output_path)
            config_path = _unique_path(scene_paths.splats_dir / f"{label}_config.json")
            _write_json(config_path, asdict(config))
            print(f"Finished profile={profile} output={output_path}")
        except Exception as exc:
            error = str(exc)
            exit_code = 1
            print(f"[error] profile={profile} scene={scene_id} {exc}")
            if fail_fast:
                duration = time.perf_counter() - start
                results.append(ProfileResult(profile=profile, scene_id=scene_id, seconds=duration, output=str(output_path or ""), config_json=str(config_path or ""), error=error))
                break
        duration = time.perf_counter() - start
        results.append(ProfileResult(profile=profile, scene_id=scene_id, seconds=duration, output=str(output_path or ""), config_json=str(config_path or ""), error=error))

    total_seconds = time.perf_counter() - total_start
    failures = [item for item in results if item.error]
    print("Summary:")
    for item in results:
        status = "OK" if not item.error else "FAIL"
        detail = item.output if status == "OK" else item.error
        print(" - {profile}: {status} ({seconds:.1f}s) {detail}".format(profile=item.profile, status=status, seconds=float(item.seconds), detail=detail))
    print(
        "Input: selected={sel} res={res} all={all_count} source={source}".format(
            sel=base_summary.get("frames_selected", {}).get("selected_count", 0),
            res=base_summary.get("frames_selected", {}).get("resolution", ""),
            all_count=base_summary.get("frames_all", {}).get("selected_count", 0),
            source=base_summary.get("metadata_source_path", ""),
        )
    )
    colmap_info = base_summary.get("colmap", {})
    sparse_info = base_summary.get("sparse", {})
    print(
        "COLMAP: cameras={cams} images={imgs} points={pts}".format(
            cams=colmap_info.get("cameras_count", 0),
            imgs=colmap_info.get("images_count", 0),
            pts=sparse_info.get("sparse_points_count", 0),
        )
    )
    print(f"Total time: {total_seconds:.1f}s profiles={len(results)} failures={len(failures)}")

    summary_root = cache_root / summary_dirname
    summary_path = _unique_path(summary_root / f"{output_prefix}_{run_id}_summary.json")
    _write_json(
        summary_path,
        {
            "base_scene_id": base_scene_id,
            "cache_root": str(cache_root),
            "input_summary": base_summary,
            "run_id": run_id,
            "device": device,
            "seed": seed,
            "overwrite": overwrite,
            "reuse_base": reuse_base,
            "disable_snapshots": disable_snapshots,
            "clone_mode": clone_mode,
            "profiles": list(profiles),
            "overrides": overrides,
            "total_seconds": total_seconds,
            "results": [
                {**asdict(item), "config": asdict(_build_config(item.profile, device, seed, disable_snapshots=disable_snapshots, overrides=overrides))}
                for item in results
            ],
        },
    )
    print(f"Wrote summary: {summary_path}")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cache_root = Path(args.cache_root).expanduser()
    run_id = args.run_id.strip() or time.strftime("%Y%m%d_%H%M%S")
    profiles = _split_profiles(args.profiles)
    overrides = _load_overrides(args.override_json)

    if args.print_presets:
        resolved: dict[str, dict[str, object]] = {}
        for profile in profiles:
            config = _build_config(profile, args.device, args.seed, disable_snapshots=args.disable_snapshots, overrides=overrides)
            resolved[profile] = asdict(config)
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return 0

    return _run_profiles(
        profiles,
        base_scene_id=args.scene_id,
        cache_root=cache_root,
        output_prefix=args.output_prefix,
        run_id=run_id,
        device=args.device,
        seed=args.seed,
        overwrite=args.overwrite,
        reuse_base=args.reuse_base,
        disable_snapshots=args.disable_snapshots,
        clone_mode=args.clone_mode,
        summary_dirname=args.summary_dirname,
        fail_fast=args.fail_fast,
        overrides=overrides,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
