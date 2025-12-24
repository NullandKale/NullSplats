"""Scene-level helpers for discovery, lifecycle, and selection persistence."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import io
import threading
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, List

from PIL import Image

from nullsplats.backend.io_cache import ScenePaths, delete_scene, ensure_scene_dirs
from nullsplats.backend.video_frames import (
    ExtractionResult,
    load_cached_frames,
    load_metadata,
    save_metadata,
)
from nullsplats.util.logging import get_logger
from nullsplats.util.scene_id import SceneId


_LOGGER = get_logger("backend.scene_manager")


@dataclass(frozen=True)
class SceneStatus:
    """Snapshot of what data exists for a scene."""

    scene_id: SceneId
    has_inputs: bool
    has_sfm: bool
    has_splats: bool
    has_renders: bool


@dataclass
class Scene:
    """In-memory representation of a scene and its metadata."""

    scene_id: SceneId
    paths: ScenePaths
    metadata: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return str(self.scene_id)


class SceneRegistry:
    """Discover scenes on disk and report which assets are present."""

    def __init__(self, cache_root: str | Path = "cache") -> None:
        self.cache_root = Path(cache_root)

    def list_scenes(self) -> List[SceneStatus]:
        """Return all scenes found under cache/inputs or cache/outputs."""
        scene_ids = self._discover_scene_ids()
        return [self._build_status(scene_id) for scene_id in sorted(scene_ids, key=str)]

    def list_names(self) -> List[str]:
        return [str(status.scene_id) for status in self.list_scenes()]

    def _discover_scene_ids(self) -> set[SceneId]:
        scenes: set[SceneId] = set()
        for candidate in self._dirnames(self.cache_root / "inputs"):
            scenes.add(SceneId(candidate))
        for candidate in self._dirnames(self.cache_root / "outputs"):
            scenes.add(SceneId(candidate))
        return scenes

    def _build_status(self, scene_id: SceneId) -> SceneStatus:
        paths = ScenePaths(scene_id, cache_root=self.cache_root)
        has_inputs = (
            self._has_files(paths.frames_all_dir)
            or self._has_files(paths.frames_selected_dir)
            or paths.metadata_path.exists()
        )
        has_sfm = self._has_files(paths.sfm_dir)
        has_splats = self._has_files(paths.splats_dir)
        has_renders = self._has_files(paths.renders_dir)
        return SceneStatus(
            scene_id=scene_id,
            has_inputs=has_inputs,
            has_sfm=has_sfm,
            has_splats=has_splats,
            has_renders=has_renders,
        )

    @staticmethod
    def _dirnames(path: Path) -> set[str]:
        if not path.exists():
            return set()
        return {item.name for item in path.iterdir() if item.is_dir()}

    @staticmethod
    def _has_files(path: Path) -> bool:
        return path.exists() and any(path.iterdir())


@dataclass(frozen=True)
class SceneSaveSummary:
    """Report on a selection save operation."""

    total: int
    processed: int
    skipped: int
    deleted: int


class SceneSelectionManager:
    """Manage selection persistence and resizing work for a scene."""

    def __init__(self, cache_root: str | Path = "cache", max_workers: int | None = None) -> None:
        self.cache_root = Path(cache_root)
        self.max_workers = max(1, max_workers or (os.cpu_count() or 4))

    def save_selection(
        self,
        scene_id: str | SceneId,
        selected_frames: Sequence[str],
        *,
        target_px: int,
        resample: str = "lanczos",
    ) -> Tuple[ExtractionResult, SceneSaveSummary]:
        """Persist selected frames, resizing on the small side to ``target_px`` in parallel."""
        normalized = SceneId(str(scene_id))
        paths = ensure_scene_dirs(normalized, cache_root=self.cache_root)
        metadata = load_metadata(normalized, cache_root=self.cache_root)

        available_frames: Iterable[str] = metadata.get("available_frames", [])
        available_set = set(available_frames)
        missing = [name for name in selected_frames if name not in available_set]
        if missing:
            raise ValueError(f"Selected frames not present in frames_all: {missing}")

        desired = list(dict.fromkeys(selected_frames))
        desired_set = set(desired)
        dest_dir = paths.frames_selected_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        existing_files = {item.name for item in dest_dir.iterdir() if item.is_file()}
        to_delete = existing_files - desired_set

        # Decide whether we can reuse existing resized frames.
        prev_target = int(metadata.get("selected_resolution_px", 0)) if metadata.get("selected_resolution_px") else 0
        prev_resample = str(metadata.get("selected_resample", "")).lower()
        resample_mode = resample.lower().strip() or "lanczos"
        rebuild_all = prev_target != int(target_px) or prev_resample != resample_mode

        resample_filter = self._resample_filter(resample_mode)
        target_px_int = max(0, int(target_px))

        tasks: list[tuple[Path, Path]] = []
        skipped = 0
        for name in desired:
            src = paths.frames_all_dir / name
            dst = dest_dir / name
            if not src.exists():
                raise FileNotFoundError(f"Selected frame missing: {src}")
            if not rebuild_all and dst.exists():
                skipped += 1
                continue
            tasks.append((src, dst))

        _LOGGER.info(
            "SceneSelectionManager start scene=%s total=%d rebuild_all=%s tasks=%d skipped_existing=%d workers=%d target_px=%d mode=%s",
            normalized,
            len(desired),
            rebuild_all,
            len(tasks),
            skipped,
            self.max_workers,
            target_px_int,
            resample_mode,
        )
        processed = self._process_tasks_parallel(tasks, target_px_int, resample_filter)

        for name in to_delete:
            try:
                (dest_dir / name).unlink()
            except FileNotFoundError:
                continue

        metadata["selected_frames"] = desired
        metadata["selected_resolution_px"] = target_px_int
        metadata["selected_resample"] = resample_mode
        save_metadata(normalized, metadata, cache_root=self.cache_root)

        result = load_cached_frames(normalized, cache_root=self.cache_root)
        summary = SceneSaveSummary(
            total=len(desired),
            processed=processed,
            skipped=skipped,
            deleted=len(to_delete),
        )
        _LOGGER.info(
            "Save selection scene=%s total=%d processed=%d skipped=%d deleted=%d target_px=%d resample=%s",
            normalized,
            summary.total,
            summary.processed,
            summary.skipped,
            summary.deleted,
            target_px_int,
            resample_mode,
        )
        return result, summary

    def _process_tasks_parallel(self, tasks: list[tuple[Path, Path]], target_px: int, resample_filter: int) -> int:
        if not tasks:
            return 0
        processed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._resize_or_copy, src, dst, target_px, resample_filter) for src, dst in tasks]
            for future in as_completed(futures):
                # Propagate exceptions to caller.
                future.result()
                processed += 1
        _LOGGER.info("SceneSelectionManager finished tasks=%d processed=%d", len(tasks), processed)
        return processed

    def _resize_or_copy(self, src: Path, dst: Path, target_px: int, resample_filter: int) -> None:
        with Image.open(src) as img:
            img = img.convert("RGB")
            w, h = img.size
            small_side = min(w, h)
            if target_px <= 0 or small_side <= target_px:
                shutil.copy2(src, dst)
                return
            scale = target_px / float(small_side)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            if new_w != w or new_h != h:
                img = img.resize((new_w, new_h), resample_filter)
            img.save(dst, format="PNG")

    @staticmethod
    def _resample_filter(mode: str) -> int:
        if mode == "bicubic":
            return Image.BICUBIC
        if mode == "bilinear":
            return Image.BILINEAR
        if mode == "nearest":
            return Image.NEAREST
        return Image.LANCZOS


class ThumbnailCache:
    """Thumbnail store built asynchronously and persisted to a single on-disk database."""

    def __init__(self, cache_root: str | Path, size_px: int = 256, max_workers: Optional[int] = None) -> None:
        self.cache_root = Path(cache_root)
        self.size_px = size_px
        self.max_workers = max(1, max_workers or (os.cpu_count() or 4))
        self._thumbs: dict[str, dict[str, bytes]] = {}
        self._lock = threading.Lock()
        self._db_lock = threading.Lock()
        self._warmup_thread: Optional[threading.Thread] = None
        self._db_path = self.cache_root / "thumbnails.db"
        self._load_db()

    def start_warmup(self, scene_ids: Iterable[str]) -> None:
        if self._warmup_thread and self._warmup_thread.is_alive():
            return
        scenes = [sid for sid in scene_ids if not self._has_thumbs(str(sid))]
        if not scenes:
            return
        self._warmup_thread = threading.Thread(target=self._warmup, args=(scenes,), daemon=True)
        self._warmup_thread.start()

    def _warmup(self, scene_ids: list[str]) -> None:
        for scene_id in scene_ids:
            try:
                self.build_scene(scene_id)
            except Exception:  # noqa: BLE001
                _LOGGER.debug("Thumbnail warmup failed for scene=%s", scene_id, exc_info=True)
        self._save_db()

    def build_scene(self, scene_id: str) -> None:
        paths = ScenePaths(scene_id, cache_root=self.cache_root)
        frames_dir = paths.frames_all_dir
        if not frames_dir.exists():
            return
        files = [p for p in sorted(frames_dir.iterdir()) if p.is_file()]
        thumbs: dict[str, bytes] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._make_thumb_bytes, path): path.name for path in files}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    data = future.result()
                except Exception:  # noqa: BLE001
                    continue
                if data:
                    thumbs[name] = data
        with self._lock:
            self._thumbs[str(SceneId(scene_id))] = thumbs
        _LOGGER.info(
            "Thumbnail cache built scene=%s count=%d size_px=%d workers=%d",
            scene_id,
            len(thumbs),
            self.size_px,
            self.max_workers,
        )
        self._save_db()

    def get_or_build(self, scene_id: str, filename: str) -> Optional[bytes]:
        scene_key = str(SceneId(scene_id))
        with self._lock:
            data = self._thumbs.get(scene_key, {}).get(filename)
        if data:
            return data
        # Build on demand for a single file.
        path = ScenePaths(scene_key, cache_root=self.cache_root).frames_all_dir / filename
        data = self._make_thumb_bytes(path)
        if data:
            with self._lock:
                self._thumbs.setdefault(scene_key, {})[filename] = data
            self._save_db()
        return data

    def _make_thumb_bytes(self, path: Path) -> Optional[bytes]:
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((self.size_px, self.size_px), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
        except Exception:
            return None

    def _load_db(self) -> None:
        if not self._db_path.exists():
            return
        try:
            import sqlite3

            with self._db_lock:
                conn = sqlite3.connect(self._db_path)
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS thumbs (scene TEXT, name TEXT, data BLOB, PRIMARY KEY(scene, name))"
                )
                cursor = conn.execute("SELECT scene, name, data FROM thumbs")
                with self._lock:
                    for scene, name, data in cursor:
                        self._thumbs.setdefault(scene, {})[name] = data
                conn.close()
                _LOGGER.info(
                    "Loaded thumbnail DB %s entries=%d", self._db_path, sum(len(v) for v in self._thumbs.values())
                )
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Failed to load thumbnail DB", exc_info=True)

    def _save_db(self) -> None:
        try:
            import sqlite3

            with self._db_lock:
                conn = sqlite3.connect(self._db_path)
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS thumbs (scene TEXT, name TEXT, data BLOB, PRIMARY KEY(scene, name))"
                )
                with self._lock:
                    rows = [
                        (scene, name, data) for scene, thumbs in self._thumbs.items() for name, data in thumbs.items()
                    ]
                conn.executemany(
                    "INSERT OR REPLACE INTO thumbs(scene, name, data) VALUES (?, ?, ?)",
                    rows,
                )
                conn.commit()
                conn.close()
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Failed to save thumbnail DB", exc_info=True)

    def _has_thumbs(self, scene_id: str) -> bool:
        with self._lock:
            thumbs = self._thumbs.get(scene_id)
            return bool(thumbs)


class SceneManager:
    """Own scene discovery, lifecycle, and selection operations."""

    def __init__(self, cache_root: str | Path = "cache", *, max_workers: int | None = None) -> None:
        self.cache_root = Path(cache_root)
        self.registry = SceneRegistry(cache_root=self.cache_root)
        self.selection = SceneSelectionManager(cache_root=self.cache_root, max_workers=max_workers)
        self.thumbnails = ThumbnailCache(cache_root=self.cache_root, max_workers=max_workers)
        self.current_scene: Optional[SceneId] = None
        # Begin warming thumbnails for existing scenes.
        try:
            self.thumbnails.start_warmup(self.registry.list_names())
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Thumbnail warmup start failed", exc_info=True)

    def init(self) -> None:
        """Ensure cache root exists."""
        self.cache_root.mkdir(parents=True, exist_ok=True)

    # --- Scene lifecycle ---
    def list_scenes(self) -> List[SceneStatus]:
        return self.registry.list_scenes()

    def list_names(self) -> List[str]:
        return self.registry.list_names()

    def listScenes(self) -> List[str]:  # noqa: N802
        return self.list_names()

    def create_from_video(self, video_path: str, name: Optional[str] = None) -> Scene:
        return self._create_scene("video", video_path, name)

    def create_from_folder(self, folder_path: str, name: Optional[str] = None) -> Scene:
        return self._create_scene("images", folder_path, name)

    # Friendly aliases matching requested interface
    def createFromVideo(self, video_path: str, name: Optional[str] = None) -> Scene:  # noqa: N802
        return self.create_from_video(video_path, name)

    def createFromFolder(self, folder_path: str, name: Optional[str] = None) -> Scene:  # noqa: N802
        return self.create_from_folder(folder_path, name)

    def ensure_scene_for_source(
        self, source_path: str | None, source_type: str, name: Optional[str] = None
    ) -> Scene:
        """Create or ensure a scene for a given source path and type."""
        scene_name = name or (self.derive_scene_id_from_path(source_path) if source_path else None)
        if not scene_name:
            raise ValueError("Scene name is required.")
        path = Path(source_path) if source_path else None
        if path and path.exists():
            if source_type == "video":
                return self.create_from_video(str(path), scene_name)
            return self.create_from_folder(str(path), scene_name)
        return self.ensure_scene(scene_name)

    def set_current_scene(self, scene_id: Optional[str | SceneId]) -> Optional[SceneId]:
        if scene_id is None:
            self.current_scene = None
            return None
        normalized = SceneId(str(scene_id))
        self.current_scene = normalized
        return normalized

    def ensure_scene(self, scene_id: str | SceneId) -> Scene:
        normalized = SceneId(str(scene_id))
        paths = ensure_scene_dirs(normalized, cache_root=self.cache_root)
        metadata = load_metadata(normalized, cache_root=self.cache_root)
        self.current_scene = normalized
        try:
            self.thumbnails.start_warmup([str(normalized)])
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Thumbnail warmup failed for scene=%s", normalized, exc_info=True)
        return Scene(scene_id=normalized, paths=paths, metadata=metadata)

    def delete(self, scene_id: Optional[str | SceneId] = None) -> bool:
        target = SceneId(str(scene_id)) if scene_id is not None else self.current_scene
        if target is None:
            return False
        delete_scene(str(target), cache_root=self.cache_root)
        if self.current_scene == target:
            self.current_scene = None
        return True

    def deleteScene(self, scene_id: Optional[str | SceneId] = None) -> bool:  # noqa: N802
        return self.delete(scene_id)

    def get(self, scene_id: Optional[str | SceneId] = None) -> Scene:
        target = SceneId(str(scene_id)) if scene_id is not None else self.current_scene
        if target is None:
            raise ValueError("No scene selected.")
        paths = ensure_scene_dirs(target, cache_root=self.cache_root)
        metadata = load_metadata(target, cache_root=self.cache_root)
        return Scene(scene_id=target, paths=paths, metadata=metadata)

    def update(self, scene_id: str | SceneId, scene: Scene) -> Scene:
        normalized = SceneId(str(scene_id))
        save_metadata(normalized, scene.metadata, cache_root=self.cache_root)
        self.current_scene = normalized
        return self.get(normalized)
    def updateScene(self, scene_id: str | SceneId, scene: Scene) -> Scene:  # noqa: N802
        return self.update(scene_id, scene)

    # --- IO helpers ---
    def load_cached_frames(self, scene_id: Optional[str | SceneId] = None) -> ExtractionResult:
        target = SceneId(str(scene_id)) if scene_id is not None else self.current_scene
        if target is None:
            raise ValueError("No scene selected.")
        return load_cached_frames(target, cache_root=self.cache_root)

    def get_thumbnail_bytes(self, scene_id: str | SceneId, filename: str) -> Optional[bytes]:
        return self.thumbnails.get_or_build(str(scene_id), filename)

    def save_selection(
        self,
        scene_id: str | SceneId,
        selected_frames: Sequence[str],
        *,
        target_px: int,
        resample: str = "lanczos",
    ) -> Tuple[ExtractionResult, SceneSaveSummary]:
        _LOGGER.info(
            "Save selection request scene=%s target_px=%s resample=%s selected=%d",
            scene_id,
            target_px,
            resample,
            len(selected_frames),
        )
        result = self.selection.save_selection(
            scene_id,
            selected_frames,
            target_px=target_px,
            resample=resample,
        )
        # Refresh thumbnails for this scene asynchronously after a save so UI thumbnails stay in sync.
        try:
            self.thumbnails.start_warmup([str(scene_id)])
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Thumbnail refresh after save failed scene=%s", scene_id, exc_info=True)
        return result

    # --- Internals ---
    def _create_scene(self, source_type: str, source_path: str, name: Optional[str]) -> Scene:
        scene_name = name or self._derive_scene_id_from_path(source_path)
        normalized = SceneId(scene_name)
        paths = ensure_scene_dirs(normalized, cache_root=self.cache_root)
        try:
            metadata = load_metadata(normalized, cache_root=self.cache_root)
        except FileNotFoundError:
            metadata = {}
        metadata["source_type"] = source_type
        metadata["source_path"] = str(Path(source_path))
        save_metadata(normalized, metadata, cache_root=self.cache_root)
        self.current_scene = normalized
        try:
            self.thumbnails.start_warmup([str(normalized)])
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Thumbnail warmup failed for scene=%s", normalized, exc_info=True)
        return Scene(scene_id=normalized, paths=paths, metadata=metadata)

    @staticmethod
    def _derive_scene_id_from_path(path_str: str) -> str:
        name = Path(path_str).stem or Path(path_str).name
        sanitized = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in name)
        sanitized = sanitized.strip("_") or "scene"
        return sanitized

    def derive_scene_id_from_path(self, path_str: str) -> str:
        """Public helper to derive a safe scene id from a path string."""
        return self._derive_scene_id_from_path(path_str)


__all__ = [
    "SceneManager",
    "SceneRegistry",
    "SceneSelectionManager",
    "SceneSaveSummary",
    "Scene",
    "SceneStatus",
]
