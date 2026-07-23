"""One-command GPU test suite for Pixel Ray Cache.

Mirrors test.py's --run-case flow (same extract_frames / SceneManager.save_selection / run_sfm
calls, same "log to a dedicated file under log/ AND stdout" logging pattern via
_setup_test_logging), then runs Pixel Ray Cache training in three stages against the SAME
prepared scene: color-only, then color+depth, then color+depth+normal.

Trains the FULL tile grid (no --tile-crop) at native input resolution -- i.e. every pixel of
every extracted frame is used, aggregated 2x2-per-tile (so the tile grid is "half resolution" of
the input frame and every tile trains on all 4 of its sub-pixels, not a spatial subset). Extracts
every available frame from the input video by default (candidate/target counts default high
enough to just take everything) rather than a small subset, for more training signal.

COLMAP only runs once: if a scene with the same target frame count has already been extracted
and COLMAP-processed (checked via cache/inputs/<scene>/metadata.json + the COLMAP text model
under cache/outputs/<scene>/sfm), extraction+SfM are skipped entirely and the cached scene is
reused. This also means the three stages below share one scene and, since Stage 2's DA3 depth
cache is keyed on (cache_root, scene) rather than per-stage output dir (see train.py), DA3
inference for depth also only runs once even though two stages need it.

Run (from repo root, with the project venv active or via its python.exe directly):

    .venv\\Scripts\\python.exe -m nullsplats.research.pixel_ray_cache.gpu_smoke_test

Output streams to the terminal live and is also written to log/pixel_ray_cache_test.log (this
script's own log, matching test.py's log/test.log) as well as logs/app.log (via
nullsplats.util.logging, same as every other run of the app/CLI tools). Each stage's outputs
(checkpoint, checkpoint.json, per-epoch preview PNGs) land under
cache/outputs/<scene_id>/pixel_ray_cache/<stage_name>/; the shared DA3 depth cache lives at
cache/outputs/<scene_id>/pixel_ray_cache/depth_cache/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]

STAGES: list[tuple[str, str]] = [
    ("color_only", "color"),
    ("color_depth", "color,depth"),
    ("color_depth_normal", "color,depth,normal"),
]


def _setup_test_logging() -> logging.Logger:
    """Same pattern as test.py's _setup_test_logging: a dedicated file handler under log/."""
    log_dir = REPO_ROOT / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pixel_ray_cache_test.log"
    logger = logging.getLogger("nullsplats.research.pixel_ray_cache_test")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Test log initialized at %s", log_path)
    return logger


def _log(logger: logging.Logger, message: str) -> None:
    logger.info(message)
    print(message)


def _ensure_scene_ready(
    *,
    scene_id: str,
    input_path: Path,
    candidate_count: int,
    target_count: int,
    cache_root: Path,
    colmap_matcher: str,
    colmap_camera_model: str,
    logger: logging.Logger,
) -> int:
    """Extract frames + run COLMAP SfM for scene_id, unless a cached scene with the same
    target_count already has both selected frames and COLMAP output -- then reuse it and skip
    straight to training. Returns the number of selected frames."""
    from nullsplats.backend.colmap_io import find_text_model
    from nullsplats.backend.io_cache import ScenePaths
    from nullsplats.backend.scene_manager import SceneManager
    from nullsplats.backend.sfm_pipeline import SfmConfig, run_sfm
    from nullsplats.backend.video_frames import extract_frames, load_metadata

    paths = ScenePaths(scene_id, cache_root=cache_root)
    cached_target_count: int | None = None
    try:
        metadata = load_metadata(scene_id, cache_root=cache_root)
        cached_target_count = int(metadata.get("target_count", 0)) or None
    except FileNotFoundError:
        pass

    frames_selected_ready = paths.frames_selected_dir.exists() and any(paths.frames_selected_dir.iterdir())
    colmap_ready = False
    if frames_selected_ready:
        try:
            find_text_model(paths)
            colmap_ready = True
        except FileNotFoundError:
            colmap_ready = False

    if frames_selected_ready and colmap_ready and cached_target_count == target_count:
        selected_count = sum(1 for _ in paths.frames_selected_dir.iterdir())
        _log(
            logger,
            f"Reusing cached scene={scene_id} selected={selected_count} "
            "(target_count unchanged, frames + COLMAP already present -- skipping extraction/SfM)",
        )
        return selected_count

    _log(
        logger,
        f"[extract] scene={scene_id} input={input_path} candidate={candidate_count} target={target_count}",
    )
    result = extract_frames(
        scene_id,
        input_path,
        source_type="video",
        candidate_count=candidate_count,
        target_count=target_count,
        cache_root=cache_root,
    )
    _log(logger, f"Extracted available={len(result.available_frames)} selected={len(result.selected_frames)}")

    manager = SceneManager(cache_root=cache_root)
    manager.save_selection(scene_id, list(result.selected_frames), target_px=0, resample="lanczos")

    _log(logger, f"[colmap] scene={scene_id} matcher={colmap_matcher}")
    sfm_config = SfmConfig(colmap_path="", matcher=colmap_matcher, camera_model=colmap_camera_model)
    run_sfm(scene_id, config=sfm_config, cache_root=cache_root)
    _log(logger, "COLMAP SfM complete")
    return len(result.selected_frames)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="assets/input.mp4", help="Video to extract frames from.")
    parser.add_argument("--cache-root", default="cache")
    parser.add_argument(
        "--scene",
        default=None,
        help="Scene id; default is deterministic (prc_full_<target-count>) so repeat runs with "
        "the same --target-count reuse the cached scene instead of re-extracting/re-COLMAPing.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=10000,
        help="Clamped to the video's actual frame count -- default is a large sentinel meaning "
        "'every available frame'.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=10000,
        help="Clamped to candidate-count/available frames -- default is a large sentinel "
        "meaning 'select every extracted frame' (no auto-select subsampling).",
    )
    parser.add_argument("--colmap-matcher", default="exhaustive")
    parser.add_argument("--colmap-camera-model", default="PINHOLE")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per level when --cascade is set.")
    parser.add_argument("--hidden-width", type=int, default=16)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="Coarse-to-fine training instead of flat: 1 network on the whole image, warm-start "
        "copied into quadrants, repeat down to full per-tile resolution.",
    )
    parser.add_argument("--cascade-levels", type=int, default=32)
    parser.add_argument("--da3-pretrained", default="depth-anything/DA3NESTED-GIANT-LARGE")
    parser.add_argument(
        "--da3-chunk-size",
        type=int,
        default=4,
        help="Frames per DA3 inference call when computing depth; lower it if the color_depth "
        "stage OOMs (e.g. --da3-chunk-size 1).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--stages",
        default=None,
        help="Comma-separated subset of stage names to run (default: all three -- "
        "color_only,color_depth,color_depth_normal).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logger = _setup_test_logging()

    from nullsplats.research.pixel_ray_cache import cli as prc_cli

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    cache_root = Path(args.cache_root)
    scene_id = args.scene or f"prc_full_{args.target_count}"

    selected_count = _ensure_scene_ready(
        scene_id=scene_id,
        input_path=input_path,
        candidate_count=args.candidate_count,
        target_count=args.target_count,
        cache_root=cache_root,
        colmap_matcher=args.colmap_matcher,
        colmap_camera_model=args.colmap_camera_model,
        logger=logger,
    )
    _log(logger, f"Scene ready scene={scene_id} frames={selected_count}")

    stage_names = set(args.stages.split(",")) if args.stages else None
    stages = [s for s in STAGES if stage_names is None or s[0] in stage_names]
    if not stages:
        raise ValueError(f"No matching stages for --stages={args.stages!r}; available: {[s[0] for s in STAGES]}")

    for stage_name, targets in stages:
        out_dir = cache_root / "outputs" / scene_id / "pixel_ray_cache" / stage_name
        _log(
            logger,
            f"=== Stage: {stage_name} (targets={targets}) frames={selected_count} "
            f"cascade={args.cascade} out_dir={out_dir} ===",
        )
        cli_args = [
            "--scene", scene_id,
            "--cache-root", str(cache_root),
            "--epochs", str(args.epochs),
            "--hidden-width", str(args.hidden_width),
            "--hidden-layers", str(args.hidden_layers),
            "--targets", targets,
            "--device", args.device,
            "--preview-every", "1",
            "--out-dir", str(out_dir),
            "--da3-pretrained", args.da3_pretrained,
            "--da3-chunk-size", str(args.da3_chunk_size),
        ]
        if args.cascade:
            cli_args += ["--cascade", "--cascade-levels", str(args.cascade_levels)]
        exit_code = prc_cli.main(cli_args)
        _log(logger, f"Stage {stage_name} finished exit_code={exit_code}")
        if exit_code != 0:
            _log(logger, f"Aborting remaining stages after failure in {stage_name}")
            return exit_code

    _log(logger, f"All stages complete scene={scene_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
