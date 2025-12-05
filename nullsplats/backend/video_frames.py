"""Frame extraction and scoring utilities for NullSplats inputs.

The functions here operate on real video files or image folders. Frames are
written to the scene cache, scored with a simple sharpness metric, and the top
frames are auto-selected for downstream training. No environment variables are
consulted; callers must pass explicit parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import subprocess
import shutil
import math
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

from nullsplats.backend.io_cache import ScenePaths, ensure_scene_dirs, load_metadata, save_metadata
from nullsplats.util.logging import get_logger
from nullsplats.util.scene_id import SceneId


logger = get_logger("video_frames")


@dataclass(frozen=True)
class FrameScore:
    """Score assigned to a single extracted frame."""

    filename: str
    score: float


@dataclass(frozen=True)
class ExtractionResult:
    """Result of frame extraction or selection reload."""

    scene_id: SceneId
    paths: ScenePaths
    candidate_count: int
    target_count: int
    available_frames: List[str]
    frame_scores: List[FrameScore]
    selected_frames: List[str]
    source_type: str
    source_path: Path


def extract_frames(
    scene_id: str | SceneId,
    source_path: str | Path,
    *,
    source_type: str,
    candidate_count: int = 200,
    target_count: int = 40,
    cache_root: str | Path = "cache",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> ExtractionResult:
    """Extract frames from a video or image folder into the cache.

    Args:
        scene_id: Scene identifier for cache paths.
        source_path: Path to the input video file or image folder.
        source_type: One of ``"video"`` or ``"images"``.
        candidate_count: Number of frames to extract and score.
        target_count: Number of frames to auto-select for frames_selected.
        cache_root: Root directory for cache storage.

    Returns:
        ExtractionResult describing available and selected frames.
    """
    if candidate_count <= 0 or target_count <= 0:
        raise ValueError("candidate_count and target_count must be positive integers.")
    normalized_scene = SceneId(str(scene_id))
    paths = ensure_scene_dirs(normalized_scene, cache_root=cache_root)
    normalized_type = source_type.lower().strip()
    if normalized_type not in {"video", "images"}:
        raise ValueError('source_type must be either "video" or "images".')

    _clear_directory(paths.source_dir)
    _clear_directory(paths.frames_all_dir)
    _clear_directory(paths.frames_selected_dir)

    source_path_obj = Path(source_path).expanduser()
    if not source_path_obj.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path_obj}")

    logger.info(
        "Extraction start scene=%s source_type=%s source=%s candidates=%d target=%d",
        normalized_scene,
        normalized_type,
        source_path_obj,
        candidate_count,
        target_count,
    )

    saved_source = _copy_source_to_cache(source_path_obj, paths.source_dir, normalized_type)
    if normalized_type == "video":
        frame_scores = _extract_from_video(
            saved_source, paths.frames_all_dir, candidate_count, progress_callback=progress_callback
        )
    else:
        frame_scores = _extract_from_image_folder(
            saved_source, paths.frames_all_dir, candidate_count, progress_callback=progress_callback
        )

    selected_frames = auto_select_best(frame_scores, target_count)
    _write_selected_frames(paths.frames_all_dir, paths.frames_selected_dir, selected_frames)

    metadata = {
        "scene_id": str(normalized_scene),
        "source_type": normalized_type,
        "source_path": str(saved_source),
        "candidate_count": candidate_count,
        "target_count": target_count,
        "available_frames": [item.filename for item in frame_scores],
        "selected_frames": selected_frames,
        "frame_scores": [{"file": item.filename, "score": item.score} for item in frame_scores],
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    save_metadata(normalized_scene, metadata, cache_root=cache_root)
    logger.info(
        "Extraction complete scene=%s total_frames=%d selected=%d metadata=%s",
        normalized_scene,
        len(frame_scores),
        len(selected_frames),
        paths.metadata_path,
    )
    return ExtractionResult(
        scene_id=normalized_scene,
        paths=paths,
        candidate_count=candidate_count,
        target_count=target_count,
        available_frames=[item.filename for item in frame_scores],
        frame_scores=frame_scores,
        selected_frames=selected_frames,
        source_type=normalized_type,
        source_path=saved_source,
    )


def load_cached_frames(scene_id: str | SceneId, cache_root: str | Path = "cache") -> ExtractionResult:
    """Reload metadata and frame information for a scene from disk."""
    normalized_scene = SceneId(str(scene_id))
    paths = ensure_scene_dirs(normalized_scene, cache_root=cache_root)
    metadata = load_metadata(normalized_scene, cache_root=cache_root)

    available_frames = metadata.get("available_frames", [])
    scores = metadata.get("frame_scores", [])
    frame_scores = [
        FrameScore(filename=item["file"], score=float(item["score"])) for item in scores if "file" in item
    ]
    selected_frames = metadata.get("selected_frames", [])
    candidate_count = int(metadata.get("candidate_count", len(available_frames)))
    target_count = int(metadata.get("target_count", len(selected_frames)))

    missing = [name for name in available_frames if not (paths.frames_all_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing frames in cache: {missing}")

    logger.info(
        "Loaded cached frames for scene=%s available=%d selected=%d",
        normalized_scene,
        len(available_frames),
        len(selected_frames),
    )
    return ExtractionResult(
        scene_id=normalized_scene,
        paths=paths,
        candidate_count=candidate_count,
        target_count=target_count,
        available_frames=available_frames,
        frame_scores=frame_scores,
        selected_frames=selected_frames,
        source_type=str(metadata.get("source_type", "")),
        source_path=Path(str(metadata.get("source_path", paths.source_dir))),
    )


def persist_selection(
    scene_id: str | SceneId,
    selected_frames: Sequence[str],
    *,
    cache_root: str | Path = "cache",
) -> ExtractionResult:
    """Write the selected frame subset to frames_selected and metadata."""
    normalized_scene = SceneId(str(scene_id))
    paths = ensure_scene_dirs(normalized_scene, cache_root=cache_root)
    metadata = load_metadata(normalized_scene, cache_root=cache_root)
    available_frames: Iterable[str] = metadata.get("available_frames", [])
    available_set = {name for name in available_frames}
    missing = [name for name in selected_frames if name not in available_set]
    if missing:
        raise ValueError(f"Selected frames not present in frames_all: {missing}")

    _write_selected_frames(paths.frames_all_dir, paths.frames_selected_dir, list(selected_frames))

    metadata["selected_frames"] = list(selected_frames)
    save_metadata(normalized_scene, metadata, cache_root=cache_root)

    scores = metadata.get("frame_scores", [])
    frame_scores = [
        FrameScore(filename=item["file"], score=float(item["score"])) for item in scores if "file" in item
    ]
    candidate_count = int(metadata.get("candidate_count", len(metadata.get("available_frames", []))))
    target_count = int(metadata.get("target_count", len(selected_frames)))
    logger.info(
        "Persisted selection for scene=%s selected=%d metadata=%s",
        normalized_scene,
        len(selected_frames),
        paths.metadata_path,
    )
    return ExtractionResult(
        scene_id=normalized_scene,
        paths=paths,
        candidate_count=candidate_count,
        target_count=target_count,
        available_frames=list(available_frames),
        frame_scores=frame_scores,
        selected_frames=list(selected_frames),
        source_type=str(metadata.get("source_type", "")),
        source_path=Path(str(metadata.get("source_path", paths.source_dir))),
    )


def auto_select_best(frame_scores: Sequence[FrameScore], target_count: int) -> List[str]:
    """Return the filenames of the best frames by sharpness score."""
    sorted_scores = sorted(frame_scores, key=lambda item: item.score, reverse=True)
    limited = sorted_scores[: target_count if target_count > 0 else 0]
    return [item.filename for item in limited]


def _copy_source_to_cache(source: Path, dest_dir: Path, source_type: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if source_type == "video":
        destination = dest_dir / source.name
        shutil.copy2(source, destination)
        logger.info("Copied video source to %s", destination)
        return destination
    if not source.is_dir():
        raise ValueError(f"Expected an image directory for source_type=images, got {source}")
    destination = dest_dir
    _clear_directory(destination)
    copied = 0
    for entry in sorted(source.iterdir()):
        if entry.is_file() and entry.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            shutil.copy2(entry, destination / entry.name)
            copied += 1
    if copied == 0:
        raise FileNotFoundError(f"No image files found under {source}")
    logger.info("Copied %d images into %s", copied, destination)
    return destination


def _extract_from_video(
    source_file: Path,
    output_dir: Path,
    candidate_count: int,
    *,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[FrameScore]:
    reader = FFMPEGVideoReader(str(source_file))
    logger.info(
        "Video reader init path=%s stream=%dx%d rotation=%d output=%dx%d fps=%.3f frames=%s",
        source_file,
        reader.width,
        reader.height,
        reader.rotation,
        reader.output_width,
        reader.output_height,
        reader.fps,
        reader.frame_count if reader.frame_count is not None else "unknown",
    )
    total_frames = reader.frame_count if reader.frame_count is not None else candidate_count
    if total_frames is None or total_frames <= 0:
        raise ValueError(f"Video contains no frames: {source_file}")
    sample_count = min(candidate_count, total_frames)
    logger.info("Video extraction start: total_frames=%d sample_count=%d", total_frames, sample_count)
    scores: List[FrameScore] = []
    try:
        for idx, frame in enumerate(reader):
            if idx >= sample_count:
                break
            filename = f"frame_{idx:04d}.png"
            destination = output_dir / filename
            _save_frame_image(frame, destination)
            score = _sharpness_score(frame)
            scores.append(FrameScore(filename=filename, score=score))
            if progress_callback:
                progress_callback(idx + 1, sample_count)
    finally:
        reader.close()
    if scores:
        mean_score = float(np.mean([item.score for item in scores]))
        min_score = min(item.score for item in scores)
        max_score = max(item.score for item in scores)
        logger.info(
            "Extraction summary wrote=%d mean=%.4f min=%.4f max=%.4f first=%s last=%s",
            len(scores),
            mean_score,
            min_score,
            max_score,
            scores[0].filename,
            scores[-1].filename,
        )
    logger.info("Video extraction stop, wrote %d frames", len(scores))
    return scores


def _extract_from_image_folder(
    source_dir: Path,
    output_dir: Path,
    candidate_count: int,
    *,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[FrameScore]:
    image_files = [
        path for path in sorted(source_dir.iterdir()) if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    ]
    if not image_files:
        raise FileNotFoundError(f"No images to process in {source_dir}")
    use_count = min(candidate_count, len(image_files))
    logger.info("Image extraction start: discovered=%d using=%d", len(image_files), use_count)
    scores: List[FrameScore] = []
    for idx, image_path in enumerate(image_files[:use_count]):
        frame = _load_image_to_array(image_path)
        filename = f"frame_{idx:04d}.png"
        destination = output_dir / filename
        _save_frame_image(frame, destination)
        score = _sharpness_score(frame)
        scores.append(FrameScore(filename=filename, score=score))
        if progress_callback:
            progress_callback(idx + 1, use_count)
    if scores:
        mean_score = float(np.mean([item.score for item in scores]))
        min_score = min(item.score for item in scores)
        max_score = max(item.score for item in scores)
        logger.info(
            "Extraction summary wrote=%d mean=%.4f min=%.4f max=%.4f first=%s last=%s",
            len(scores),
            mean_score,
            min_score,
            max_score,
            scores[0].filename,
            scores[-1].filename,
        )
    logger.info("Image extraction stop, wrote %d frames", len(scores))
    return scores


def _write_selected_frames(frames_dir: Path, selected_dir: Path, selected_filenames: Sequence[str]) -> None:
    selected_dir.mkdir(parents=True, exist_ok=True)
    _clear_directory(selected_dir)
    for name in selected_filenames:
        source_file = frames_dir / name
        destination = selected_dir / name
        if not source_file.exists():
            raise FileNotFoundError(f"Selected frame missing: {source_file}")
        shutil.copy2(source_file, destination)
    logger.info("Wrote %d selected frames into %s", len(selected_filenames), selected_dir)


def _clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def _sharpness_score(frame: np.ndarray) -> float:
    grayscale = _to_grayscale(frame)
    laplacian = (
        -4 * grayscale
        + np.roll(grayscale, 1, axis=0)
        + np.roll(grayscale, -1, axis=0)
        + np.roll(grayscale, 1, axis=1)
        + np.roll(grayscale, -1, axis=1)
    )
    score = float(np.mean(np.abs(laplacian)))
    return score


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32)
    channels = frame[..., :3].astype(np.float32)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.tensordot(channels, weights, axes=([2], [0]))


def _save_frame_image(frame: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame.astype(np.uint8)).save(destination, format="PNG")


def _rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate frame to honor display rotation metadata."""
    normalized = rotation % 360
    if normalized == 90:
        return np.rot90(frame, k=-1)  # clockwise
    if normalized == 180:
        return np.rot90(frame, k=2)
    if normalized == 270:
        return np.rot90(frame, k=1)  # counter-clockwise
    return frame


def _load_image_to_array(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def _ffprobe_stream_props(video_path: str) -> tuple[int, int, float, int | None, int]:
    """Return width/height/fps/frame_count/rotation for the primary video stream."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        video_path,
    ]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True).stdout
    logger.info("ffprobe full output for %s: %s", video_path, out)
    info = json.loads(out)
    streams = info.get("streams", [])
    if not streams:
        raise ValueError(f"No video streams found in {video_path}")
    stream = streams[0]

    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions reported by ffprobe for {video_path}: {width}x{height}")
    rate = stream.get("r_frame_rate", "0/1")
    num, den = map(int, rate.split("/")) if "/" in rate else (0, 1)
    fps = num / den if den else 0.0
    nb = str(stream.get("nb_frames", "") or "")
    frame_count = int(nb) if nb.isdigit() else None
    rotation = _extract_rotation(stream)
    logger.info(
        "ffprobe parsed path=%s width=%d height=%d fps=%.3f frame_count=%s rotation=%d side_data=%s tags=%s",
        video_path,
        width,
        height,
        fps,
        frame_count if frame_count is not None else "unknown",
        rotation,
        stream.get("side_data_list", []),
        stream.get("tags", {}),
    )
    return width, height, fps, frame_count, rotation


def _extract_rotation(stream_info: dict) -> int:
    """Parse rotation from ffprobe stream tags/side data, normalized to [0, 359]."""
    rotate_tag = stream_info.get("tags", {}).get("rotate")
    if rotate_tag is not None:
        try:
            return int(rotate_tag) % 360
        except (TypeError, ValueError):
            pass
    for side in stream_info.get("side_data_list", []):
        rotation_val = side.get("rotation")
        if rotation_val is not None:
            try:
                return int(rotation_val) % 360
            except (TypeError, ValueError):
                continue
        if "displaymatrix" in side:
            parsed = _rotation_from_displaymatrix(side["displaymatrix"])
            if parsed is not None:
                return parsed
    return 0


def _rotation_from_displaymatrix(matrix_str: str) -> Optional[int]:
    """Derive rotation from a display matrix text block produced by ffprobe."""
    try:
        lines = [line.strip() for line in matrix_str.strip().splitlines() if line.strip()]
        values = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                values.extend(parts[1:4])
        nums = [int(val) for val in values[:4]]
        if len(nums) < 4:
            return None
        a, b, c, d = nums
        angle_rad = math.atan2(c, a)
        angle_deg = math.degrees(angle_rad)
        snapped = int(round(angle_deg / 90.0)) * 90
        return snapped % 360
    except Exception:
        return None


class FFMPEGVideoReader:
    """Stream raw RGB frames from a video using ffmpeg."""

    def __init__(self, video_path: str, *, pix_fmt: str = "rgb24") -> None:
        self.video_path = video_path
        self.width, self.height, self.fps, self.frame_count, self.rotation = _ffprobe_stream_props(video_path)
        # Trust ffmpeg autorotate; swap expected dimensions for 90/270.
        if self.rotation in {90, 270}:
            self.output_width, self.output_height = self.height, self.width
        else:
            self.output_width, self.output_height = self.width, self.height
        self.frame_size = self.output_width * self.output_height * 3
        logger.debug(
            "FFMPEGVideoReader setup path=%s width=%d height=%d rotation=%d output_w=%d output_h=%d fps=%.3f frame_count=%s frame_size=%d",
            video_path,
            self.width,
            self.height,
            self.rotation,
            self.output_width,
            self.output_height,
            self.fps,
            self.frame_count if self.frame_count is not None else "unknown",
            self.frame_size,
        )
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-",
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.idx = 0

    def __iter__(self) -> "FFMPEGVideoReader":
        return self

    def __next__(self) -> np.ndarray:
        if self.proc.stdout is None:
            raise StopIteration
        data = self.proc.stdout.read(self.frame_size)
        if len(data) != self.frame_size:
            self.close()
            raise StopIteration
        frame = np.frombuffer(data, np.uint8).reshape(self.output_height, self.output_width, 3)
        self.idx += 1
        return frame

    def close(self) -> None:
        try:
            if self.proc.stdout and not self.proc.stdout.closed:
                self.proc.stdout.close()
        except Exception:
            pass
        if self.proc.poll() is None:
            self.proc.wait()


__all__ = [
    "FrameScore",
    "ExtractionResult",
    "FFMPEGVideoReader",
    "auto_select_best",
    "extract_frames",
    "load_cached_frames",
    "persist_selection",
]
