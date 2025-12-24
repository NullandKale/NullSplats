"""COLMAP parsing helpers and data structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from nullsplats.backend.io_cache import ScenePaths


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    camera_id: int
    name: str
    qvec: list[float]
    tvec: list[float]
    xys: list[list[float]]
    point3D_ids: list[int]


@dataclass(frozen=True)
class ColmapPoint3D:
    point3D_id: int
    xyz: list[float]
    rgb: list[int]
    error: float


@dataclass(frozen=True)
class ColmapData:
    cameras: dict[int, ColmapCamera]
    images: dict[int, ColmapImage]
    points3D: dict[int, ColmapPoint3D]
    model_format: str
    source_dir: Path


def load_colmap_data(paths: ScenePaths) -> ColmapData:
    cameras_txt, images_txt = find_text_model(paths)
    cameras = parse_cameras(cameras_txt)
    images = parse_images(images_txt)
    points_path = find_points3d(paths, cameras_txt.parent)
    points = parse_points3d(points_path) if points_path is not None else {}
    return ColmapData(
        cameras={cid: to_colmap_camera(cid, data) for cid, data in cameras.items()},
        images={img.image_id: img for img in images},
        points3D=points,
        model_format="text",
        source_dir=cameras_txt.parent,
    )


def find_text_model(paths: ScenePaths) -> tuple[Path, Path]:
    candidates = [
        (paths.sfm_dir / "sparse" / "text" / "cameras.txt", paths.sfm_dir / "sparse" / "text" / "images.txt"),
        (paths.sfm_dir / "sparse" / "0" / "cameras.txt", paths.sfm_dir / "sparse" / "0" / "images.txt"),
        (paths.sfm_dir / "sparse" / "cameras.txt", paths.sfm_dir / "sparse" / "images.txt"),
    ]
    for cams, imgs in candidates:
        if cams.exists() and imgs.exists():
            return cams, imgs
    raise FileNotFoundError(
        f"cameras.txt/images.txt not found under {paths.sfm_dir}. Re-run COLMAP so text models are exported."
    )


def parse_cameras(path: Path) -> dict[int, dict]:
    cameras: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            if model not in {"PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
                raise ValueError(f"Unsupported COLMAP camera model: {model}")
            if model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            else:
                fx = fy = params[0]
                cx = params[1]
                cy = params[2] if len(params) > 2 else params[1]
            cameras[cam_id] = {"model": model, "width": width, "height": height, "params": (fx, fy, cx, cy)}
    return cameras


def parse_images(path: Path) -> list[ColmapImage]:
    entries: list[ColmapImage] = []
    with path.open("r", encoding="utf-8") as handle:
        lines = iter(handle.readlines())
        for line in lines:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]
            xys: list[list[float]] = []
            point3D_ids: list[int] = []
            points_line = next(lines, "")
            if points_line:
                points = points_line.strip().split()
                for idx in range(0, len(points) - 2, 3):
                    x = float(points[idx])
                    y = float(points[idx + 1])
                    point_id = int(float(points[idx + 2]))
                    xys.append([x, y])
                    point3D_ids.append(point_id)
            entries.append(
                ColmapImage(
                    image_id=image_id,
                    camera_id=camera_id,
                    name=name,
                    qvec=[qw, qx, qy, qz],
                    tvec=[tx, ty, tz],
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
            )
    return entries


def to_colmap_camera(camera_id: int, data: dict) -> ColmapCamera:
    fx, fy, cx, cy = data["params"]
    return ColmapCamera(
        camera_id=camera_id,
        model=data["model"],
        width=data["width"],
        height=data["height"],
        params=[fx, fy, cx, cy],
    )


def find_points3d(paths: ScenePaths, model_dir: Path) -> Optional[Path]:
    candidates = [
        model_dir / "points3D.txt",
        paths.sfm_dir / "sparse" / "0" / "points3D.txt",
        paths.sfm_dir / "sparse" / "points3D.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_points3d(path: Path) -> dict[int, ColmapPoint3D]:
    points: dict[int, ColmapPoint3D] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            point_id = int(parts[0])
            xyz = list(map(float, parts[1:4]))
            rgb = list(map(int, parts[4:7]))
            error = float(parts[7])
            points[point_id] = ColmapPoint3D(
                point3D_id=point_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
            )
    return points
