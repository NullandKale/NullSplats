"""Structure-from-motion pipeline integration for NullSplats.

This module orchestrates COLMAP command-line tools to generate camera poses for
a scene. Images are read from the cached frames_selected directory for the
scene. All logging is streamed into sfm/logs and echoed to the configured
logger; no environment variables are consulted.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import atexit
import shutil
import subprocess
import threading
import time
from typing import Iterable, List

from nullsplats.backend.io_cache import ScenePaths, ensure_scene_dirs
from nullsplats.util.logging import get_logger
from nullsplats.util.scene_id import SceneId
from nullsplats.util.tooling_paths import default_colmap_path


logger = get_logger("sfm_pipeline")

_ACTIVE_SFM_PROCESSES: set[subprocess.Popen] = set()
_PROCESS_LOCK = threading.Lock()


def _cleanup_active_processes() -> None:
    with _PROCESS_LOCK:
        procs = list(_ACTIVE_SFM_PROCESSES)
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                continue
            proc.wait()


atexit.register(_cleanup_active_processes)


@dataclass(frozen=True)
class SfmConfig:
    """Configuration for COLMAP execution."""

    colmap_path: str = ""
    matcher: str = "exhaustive"
    camera_model: str = "PINHOLE"


@dataclass(frozen=True)
class SfmResult:
    """Artifacts produced by an SfM run.

    sparse_model_path points to the COLMAP model directory (e.g., sfm/sparse/0).
    converted_model_path points to the exported PLY file (e.g., sfm/sparse/model.ply).
    """

    scene_id: SceneId
    paths: ScenePaths
    database_path: Path
    sparse_model_path: Path
    converted_model_path: Path
    log_path: Path


def run_sfm(
    scene_id: str | SceneId,
    *,
    config: SfmConfig,
    cache_root: str | Path = "cache",
) -> SfmResult:
    """Execute COLMAP + GLOMAP for the provided scene.

    Args:
        scene_id: Scene identifier.
        config: Executable paths and algorithm choices.
        cache_root: Base cache directory containing inputs and outputs.

    Returns:
        SfmResult describing the generated artifacts.
    """
    normalized_scene = SceneId(str(scene_id))
    paths = ensure_scene_dirs(normalized_scene, cache_root=cache_root)
    _require_frames(paths.frames_selected_dir)
    colmap_exe = config.colmap_path or default_colmap_path()
    _assert_executable(colmap_exe, "COLMAP")

    sfm_dir = paths.sfm_dir
    log_dir = sfm_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / f"colmap_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log").resolve()
    database_path = (sfm_dir / "database.db").resolve()
    sparse_path = (sfm_dir / "sparse").resolve()
    sparse_model = sparse_path / "0"
    converted_model = sparse_path / "model.ply"
    _reset_previous_outputs(database_path, sparse_path, converted_model)
    images_path = paths.frames_selected_dir.resolve()
    sparse_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "SfM start scene=%s images=%s colmap=%s matcher=%s camera_model=%s",
        normalized_scene,
        images_path,
        colmap_exe,
        config.matcher,
        config.camera_model,
    )
    start_time = time.perf_counter()
    with ExitStack() as stack:
        _ensure_dll_search_paths(colmap_exe, stack)
        with log_path.open("a", encoding="utf-8") as log_file:
            _log_binary_details(colmap_exe, log_file)
            _stream_command(
                [
                    colmap_exe,
                    "feature_extractor",
                    "--database_path",
                    str(database_path),
                    "--image_path",
                    str(images_path),
                    "--ImageReader.single_camera",
                    "1",
                    "--ImageReader.camera_model",
                    config.camera_model,
                    "--log_to_stderr",
                    "1",
                    "--log_level",
                    "2",
                ],
                log_file,
                log_path,
                "COLMAP feature extraction",
            )
            _stream_command(
                [
                    colmap_exe,
                    f"{config.matcher}_matcher",
                    "--database_path",
                    str(database_path),
                ],
                log_file,
                log_path,
                "COLMAP matching",
            )
            _stream_command(
                [
                    colmap_exe,
                    "mapper",
                    "--database_path",
                    str(database_path),
                    "--image_path",
                    str(images_path),
                    "--output_path",
                    str(sparse_path),
                    "--log_to_stderr",
                    "1",
                    "--log_level",
                    "2",
                ],
                log_file,
                log_path,
                "COLMAP mapping",
            )
            _stream_command(
                [
                    colmap_exe,
                    "model_converter",
                    "--input_path",
                    str(sparse_model),
                    "--output_path",
                    str(converted_model),
                    "--output_type",
                    "PLY",
                ],
                log_file,
                log_path,
                "COLMAP model conversion",
            )
            text_model_dir = sparse_path / "text"
            text_model_dir.mkdir(parents=True, exist_ok=True)
            _stream_command(
                [
                    colmap_exe,
                    "model_converter",
                    "--input_path",
                    str(sparse_model),
                    "--output_path",
                    str(text_model_dir),
                    "--output_type",
                    "TXT",
                ],
                log_file,
                log_path,
                "COLMAP text model export",
            )
    elapsed = time.perf_counter() - start_time
    logger.info(
        "SfM complete scene=%s elapsed=%.2fs log=%s database=%s sparse=%s ply=%s",
        normalized_scene,
        elapsed,
        log_path,
        database_path,
        sparse_model,
        converted_model,
    )
    return SfmResult(
        scene_id=normalized_scene,
        paths=paths,
        database_path=database_path,
        sparse_model_path=sparse_model,
        converted_model_path=converted_model,
        log_path=log_path,
    )


def _reset_previous_outputs(database_path: Path, sparse_path: Path, converted_model: Path) -> None:
    """Ensure the SfM output folder starts empty for a fresh COLMAP run."""
    if database_path.exists():
        database_path.unlink()
        logger.info("Removed previous database at %s", database_path)
    if sparse_path.exists():
        shutil.rmtree(sparse_path)
        logger.info("Removed previous sparse outputs at %s", sparse_path)
    if converted_model.exists():
        converted_model.unlink()


def _require_frames(frames_dir: Path) -> None:
    images = list(_iter_images(frames_dir))
    if not images:
        raise FileNotFoundError(f"No selected frames found in {frames_dir}; extract inputs first.")


def _iter_images(frames_dir: Path) -> Iterable[Path]:
    return (
        path
        for path in sorted(frames_dir.glob("*"))
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )


def _assert_executable(path: str, label: str) -> None:
    resolved = shutil.which(path) if not Path(path).exists() else path
    if resolved is None:
        raise FileNotFoundError(f"{label} executable not found: {path}")


def _ensure_dll_search_paths(colmap_path: str, stack: ExitStack) -> None:
    """Add DLL search paths for COLMAP directories on Windows."""
    try:
        import os
    except ImportError:
        return
    add_dll = getattr(os, "add_dll_directory", None)
    if add_dll is None:
        return
    seen: set[Path] = set()
    for binary in (colmap_path,):
        candidate = Path(binary)
        parents = []
        if candidate.suffix:
            parents.append(candidate.parent)
            parents.append(candidate.parent.parent / "lib")
        else:
            parents.append(candidate)
            parents.append(candidate / "lib")
        for parent in parents:
            if parent and parent.exists() and parent not in seen:
                handle = add_dll(str(parent))
                stack.callback(handle.close)
                seen.add(parent)
                logger.info("Added DLL search path: %s", parent)


def _log_binary_details(colmap_path: str, log_file) -> None:
    """Write basic binary diagnostics for troubleshooting."""
    for label, path_str in (("COLMAP", colmap_path or default_colmap_path()),):
        path = Path(path_str)
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        log_file.write(f"[binary] {label} path={path} exists={exists} size={size}\n")
        logger.info("%s binary path=%s exists=%s size=%d", label, path, exists, size)


def _stream_command(cmd: List[str], log_file, log_path: Path, label: str) -> None:
    joined = " ".join(cmd)
    run_dir = Path(cmd[0]).parent if Path(cmd[0]).exists() else None
    logger.info("%s start cmd=%s cwd=%s", label, joined, run_dir or ".")
    log_file.write(f"$ {joined}\n")
    if run_dir:
        log_file.write(f"[cwd] {run_dir}\n")
    log_file.flush()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=run_dir)
    with _PROCESS_LOCK:
        _ACTIVE_SFM_PROCESSES.add(process)
    try:
        if process.stdout is None:
            raise RuntimeError(f"Failed to start process for: {joined}")
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            logger.info("%s: %s", label, line.rstrip())
        process.wait()
        if process.returncode != 0:
            hint = _describe_failure(process.returncode)
            logger.error("%s failed code=%d%s; see %s", label, process.returncode, hint, log_path)
            raise RuntimeError(f"{label} failed with exit code {process.returncode}{hint}. See log: {log_path}")
        logger.info("%s completed code=%d", label, process.returncode)
        log_file.write(f"[completed] {label}\n")
        log_file.flush()
    finally:
        with _PROCESS_LOCK:
            _ACTIVE_SFM_PROCESSES.discard(process)
    return


def _describe_failure(code: int) -> str:
    if code in {3221225781, -1073741515}:  # STATUS_DLL_NOT_FOUND
        return " (missing DLL dependency for COLMAP; ensure its DLLs sit alongside the exe)"
    if code in {3221225477, -1073741819}:  # STATUS_ACCESS_VIOLATION
        return " (access violation; check GPU/driver and binary compatibility)"
    return f" (exit 0x{code:08X})"


__all__ = ["SfmConfig", "SfmResult", "run_sfm"]
