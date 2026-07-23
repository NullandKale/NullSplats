"""Optuna hyperparameter search over Pixel Ray Cache tile-model architecture/lr/loss weights.

Every trial trains on a small debug tile-crop (never the full 1920x1080 grid -- see the
parameter/memory budget table in the design plan: dozens-to-hundreds of full-grid trials are not
tractable). Because per-tile networks train largely independently of each other (no shared
weights, no cross-tile receptive field), a smaller tile-crop subset is a reasonable proxy for
architecture/lr/loss-weight search -- unlike typical CNN hyperparameter search where a crop can
distort receptive-field-dependent behavior. After a sweep, build one PixelRayCacheConfig from
`study.best_params` and run a single full-grid confirmation via cli.py.

Kept as a separate entry point from cli.py (one fixed-hyperparameter run) rather than a --sweep
flag on it, mirroring test.py's separate --run-case (single run) vs driver (--headless, loops
over cases) modes.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys

import optuna
import torch

from nullsplats.backend.io_cache import ScenePaths
from nullsplats.research.pixel_ray_cache import data, tiles, train
from nullsplats.research.pixel_ray_cache import optim as optim_registry
from nullsplats.research.pixel_ray_cache.cli import _parse_targets, _parse_tile_crop
from nullsplats.research.pixel_ray_cache.config import PixelRayCacheConfig
from nullsplats.research.pixel_ray_cache.data import RayCacheFrame
from nullsplats.research.pixel_ray_cache.losses import list_losses
from nullsplats.research.pixel_ray_cache.tile_models.base import HyperparamSpec
from nullsplats.research.pixel_ray_cache.tile_models.registry import get_hyperparams, get_tile_model_cls, list_tile_models
from nullsplats.util.logging import get_logger, setup_logging

logger = get_logger("research.pixel_ray_cache.sweep")


def _suggest_for_spec(trial: optuna.Trial, spec: HyperparamSpec):
    """Generic per-architecture-hyperparameter suggestion, driven entirely by the model class's
    own HyperparamSpec schema -- no hardcoded per-architecture knowledge here, so a new
    registered tile model is swept correctly with zero changes to this function."""
    if spec.kind == "choice":
        return trial.suggest_categorical(spec.name, list(spec.choices))
    if spec.kind == "bool":
        return trial.suggest_categorical(spec.name, [True, False])
    if spec.kind == "int":
        default = int(spec.default)
        low = max(1, default // 2)
        high = max(low + 1, default * 2)
        return trial.suggest_int(spec.name, low, high)
    if spec.kind == "float":
        default = float(spec.default)
        low = max(1e-6, default / 10.0)
        high = max(low * 10.0, default * 10.0)
        return trial.suggest_float(spec.name, low, high, log=True)
    raise ValueError(f"Unknown hyperparam kind: {spec.kind!r}")


def objective(trial: optuna.Trial, *, base_config: PixelRayCacheConfig, frames: list[RayCacheFrame]) -> float:
    tile_model = trial.suggest_categorical("tile_model", list_tile_models())
    kwargs = {spec.name: _suggest_for_spec(trial, spec) for spec in get_hyperparams(tile_model)}
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", optim_registry.list_optimizers())
    color_loss = trial.suggest_categorical("color_loss", list_losses())

    depth_w = base_config.depth_loss_weight
    normal_w = base_config.normal_loss_weight
    depth_loss = base_config.depth_loss
    if "depth" in base_config.targets:
        depth_w = trial.suggest_float("depth_loss_weight", 0.01, 1.0, log=True)
        depth_loss = trial.suggest_categorical("depth_loss", list_losses())
    if "normal" in base_config.targets:
        normal_w = trial.suggest_float("normal_loss_weight", 0.01, 1.0, log=True)

    config = dataclasses.replace(
        base_config,
        tile_model=tile_model,
        tile_model_kwargs=kwargs,
        lr=lr,
        optimizer=optimizer_name,
        color_loss=color_loss,
        depth_loss=depth_loss,
        depth_loss_weight=depth_w,
        normal_loss_weight=normal_w,
        out_dir=base_config.resolve_out_dir() / "sweeps" / f"trial_{trial.number:04d}",
    )

    n_tiles = tiles.n_tiles_for(frames[0].height, frames[0].width, crop=config.tile_crop)
    model = get_tile_model_cls(config.tile_model)(n_tiles, config.target_spec, **kwargs)

    def report(epoch: int, loss: float) -> None:
        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    result = train.run_training(config, frames, model, on_epoch_loss=report)
    return result.final_loss


def run_sweep(
    *, base_config: PixelRayCacheConfig, frames: list[RayCacheFrame], n_trials: int, storage: str | None = None
) -> optuna.Study:
    if base_config.tile_crop is None:
        raise ValueError(
            "sweep runs require base_config.tile_crop (a small debug crop) -- full-grid trials "
            "are not tractable at dozens-to-hundreds of trials, see the memory budget table."
        )
    if storage is None:
        db_path = base_config.cache_root / "outputs" / base_config.scene / "pixel_ray_cache" / "sweeps"
        db_path.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{(db_path / f'{base_config.scene}.db').as_posix()}"
    study = optuna.create_study(
        direction="minimize",
        storage=storage,
        study_name=f"pixel_ray_cache_{base_config.scene}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(lambda t: objective(t, base_config=base_config, frames=frames), n_trials=n_trials)
    return study


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for Pixel Ray Cache.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--cache-root", default="cache")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per trial (kept low for search speed).")
    parser.add_argument("--tile-crop", default="64x64+928+508", help="WxH+X+Y in tile units; required for sweeps.")
    parser.add_argument("--targets", default="color")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--normalize-poses",
        action="store_true",
        help="Recenter+isotropically rescale camera positions into [-1,1] before sweeping (see "
        "data.compute_pose_normalization). Affects the range encoding/lr hyperparameters get "
        "tuned against -- a study run without this isn't directly comparable to one with it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = _build_arg_parser().parse_args(argv)
    crop = _parse_tile_crop(args.tile_crop)
    if crop is None:
        raise ValueError("--tile-crop is required for sweep.py")

    device = torch.device(args.device)
    paths = ScenePaths(scene_id=args.scene, cache_root=Path(args.cache_root))
    frames = data.load_scene_frames(paths, device=device, normalize_poses=args.normalize_poses)

    base_config = PixelRayCacheConfig(
        scene=args.scene,
        cache_root=Path(args.cache_root),
        targets=_parse_targets(args.targets),
        epochs=args.epochs,
        tile_crop=crop,
        device=args.device,
        normalize_poses=args.normalize_poses,
    )
    study = run_sweep(base_config=base_config, frames=frames, n_trials=args.n_trials)
    print(
        f"OK scene={args.scene} n_trials={args.n_trials} best_value={study.best_value:.6f} "
        f"best_params={study.best_params}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
