from __future__ import annotations

from pathlib import Path
import sys

from nullsplats.backend.io_cache import ScenePaths, ensure_scene_dirs
from nullsplats.backend.sfm_pipeline import SfmConfig, run_sfm
from nullsplats.backend.splat_train import SplatTrainingConfig, train_scene
from nullsplats.util.logging import setup_logging


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _binary_path(tool: str) -> str:
    repo_root = _repo_root()
    if tool == "colmap":
        candidates = [
            repo_root / "tools" / "colmap" / "COLMAP.bat",
            repo_root / "tools" / "colmap" / "colmap.exe",
        ]
    else:
        candidates = []
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return tool


def _ensure_real_frames(scene_id: str) -> ScenePaths:
    paths = ensure_scene_dirs(scene_id)
    selected = sorted(paths.frames_selected_dir.glob("*"))
    if not selected:
        raise RuntimeError(
            f"Real frames missing at {paths.frames_selected_dir}. "
            "Populate frames_selected with real data before running this test."
        )
    return paths


def main() -> None:
    setup_logging()
    scene_id = "20251121_120658_720"
    paths = _ensure_real_frames(scene_id)
    print(f"Using real scene at {paths.frames_selected_dir}")

    sfm_config = SfmConfig(
        colmap_path=_binary_path("colmap"),
    )
    sfm_result = run_sfm(scene_id, config=sfm_config)
    print(f"SfM complete; log: {sfm_result.log_path}")
    if not sfm_result.sparse_model_path.exists():
        raise RuntimeError(f"Sparse model missing at {sfm_result.sparse_model_path}")
    if not any(sfm_result.sparse_model_path.iterdir()):
        raise RuntimeError(f"Sparse model folder is empty at {sfm_result.sparse_model_path}")
    if not sfm_result.converted_model_path.exists():
        raise RuntimeError(f"Converted model missing at {sfm_result.converted_model_path}")

    training_config = SplatTrainingConfig(iterations=8, snapshot_interval=4, max_points=0)
    training_result = train_scene(scene_id, training_config)
    print(f"Training complete; last checkpoint: {training_result.last_checkpoint}")
    if not training_result.last_checkpoint.exists():
        raise RuntimeError(f"Training checkpoint missing at {training_result.last_checkpoint}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        import traceback

        sys.stderr.write(f"Integration test failed: {exc}\n")
        traceback.print_exc()
        sys.exit(1)
