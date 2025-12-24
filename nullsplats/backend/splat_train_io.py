"""COLMAP parsing and frame loading for splat training."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

from nullsplats.backend.colmap_io import find_text_model, parse_cameras, parse_images
from nullsplats.backend.io_cache import ScenePaths
from nullsplats.backend.splat_train_config import FrameRecord


def load_colmap_frames(paths: ScenePaths, device: torch.device, *, image_downscale: int) -> List[FrameRecord]:
    cameras_txt, images_txt = find_text_model(paths)
    cameras = parse_cameras(cameras_txt)
    images = parse_images(images_txt)
    records: List[FrameRecord] = []
    frames_dir = paths.frames_selected_dir
    for idx, image_entry in enumerate(images):
        camera = cameras.get(image_entry.camera_id)
        if camera is None:
            raise RuntimeError(f"Camera id {image_entry.camera_id} missing in cameras.txt")
        image_path = frames_dir / image_entry.name
        if not image_path.exists():
            image_path = frames_dir / Path(image_entry.name).name
        if not image_path.exists():
            raise FileNotFoundError(f"Image {image_entry.name} not found under {frames_dir}")
        width = camera["width"]
        height = camera["height"]
        fx, fy, cx, cy = camera["params"]
        if image_downscale > 1:
            width = max(1, width // image_downscale)
            height = max(1, height // image_downscale)
            fx = fx / image_downscale
            fy = fy / image_downscale
            cx = cx / image_downscale
            cy = cy / image_downscale
        K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        camtoworld = _cam_to_world_matrix(tuple(image_entry.qvec), tuple(image_entry.tvec), device=device)
        image_tensor = _load_image_tensor(image_path, height, width)
        records.append(
            FrameRecord(
                index=idx,
                name=image_entry.name,
                image_path=image_path,
                camtoworld=camtoworld,
                K=K,
                width=width,
                height=height,
                image=image_tensor,
            )
        )
    first_size = (records[0].height, records[0].width) if records else None
    for rec in records:
        if (rec.height, rec.width) != first_size:
            raise RuntimeError("All frames must share the same resolution after downscale.")
    return records


def load_sparse_points(paths: ScenePaths) -> Tuple[torch.Tensor, torch.Tensor]:
    ply_path = paths.sfm_dir / "sparse" / "model.ply"
    if not ply_path.exists():
        ply_path = paths.sfm_dir / "sparse" / "0" / "points3D.ply"
    if not ply_path.exists():
        ply_path = paths.sfm_dir / "sparse" / "0" / "points3D.txt"
    if not ply_path.exists():
        raise FileNotFoundError(f"Sparse model not found under {paths.sfm_dir}")
    if ply_path.suffix.lower() == ".txt":
        means, colors, _tracks = _load_colmap_txt_points(ply_path)
    else:
        means, colors, _tracks = _load_ply_points(ply_path)
    return means, colors




def _cam_to_world_matrix(qvec: Tuple[float, float, float, float], tvec: Tuple[float, float, float], device: torch.device) -> torch.Tensor:
    qw, qx, qy, qz = qvec
    q = torch.tensor([qw, qx, qy, qz], dtype=torch.float64, device=device)
    R = _qvec_to_rotmat(q)
    t = torch.tensor(tvec, dtype=torch.float64, device=device)
    c2w = torch.eye(4, dtype=torch.float64, device=device)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w.float()


def _qvec_to_rotmat(qvec: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = qvec
    return torch.tensor(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=torch.float64,
        device=qvec.device,
    )


def _load_ply_points(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    type_map = {
        "float": np.float32,
        "float32": np.float32,
        "double": np.float64,
        "uchar": np.uint8,
        "uint8": np.uint8,
        "uint": np.uint32,
        "int": np.int32,
    }
    with path.open("rb") as handle:
        header: list[str] = []
        while True:
            line_bytes = handle.readline()
            if not line_bytes:
                raise ValueError("Invalid PLY: no end_header")
            line = line_bytes.decode("ascii", errors="ignore").strip()
            header.append(line)
            if line == "end_header":
                break
        data = handle.read()

    format_line = next((line for line in header if line.startswith("format ")), "format ascii 1.0")
    ascii_format = "ascii" in format_line
    vertex_count = 0
    properties: list[tuple[str, type]] = []
    collecting_vertex = False
    for line in header:
        if line.startswith("element vertex"):
            parts = line.split()
            vertex_count = int(parts[-1])
            collecting_vertex = True
            continue
        if line.startswith("element ") and not line.startswith("element vertex"):
            collecting_vertex = False
        if collecting_vertex and line.startswith("property"):
            _, typ, name = line.split()[:3]
            if typ not in type_map:
                continue
            properties.append((name, type_map[typ]))

    def _extract_structured_array() -> np.ndarray:
        dtype = np.dtype([(name, typ) for name, typ in properties])
        return np.frombuffer(data, dtype=dtype, count=vertex_count)

    def _extract_from_ascii() -> np.ndarray:
        rows = []
        text = data.decode("ascii", errors="ignore").strip().splitlines()
        for line in text[: vertex_count or None]:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < len(properties):
                continue
            parsed = []
            for value, (_, typ) in zip(parts, properties):
                parsed.append(typ(type(typ)(float(value)) if np.issubdtype(typ, np.floating) else int(float(value))))
            rows.append(tuple(parsed))
        dtype = np.dtype([(name, typ) for name, typ in properties])
        return np.array(rows, dtype=dtype)

    arr = _extract_from_ascii() if ascii_format else _extract_structured_array()
    if arr.size == 0 or vertex_count == 0:
        return torch.empty((0, 3)), torch.empty((0, 3)), torch.empty((0,))

    def _get_field(candidates: list[str]) -> Optional[np.ndarray]:
        for name in candidates:
            if name in arr.dtype.names:
                return arr[name]
        return None

    xs = _get_field(["x"])
    ys = _get_field(["y"])
    zs = _get_field(["z"])
    rs = _get_field(["red", "r"])
    gs_ = _get_field(["green", "g"])
    bs = _get_field(["blue", "b"])
    if xs is None or ys is None or zs is None or rs is None or gs_ is None or bs is None:
        return torch.empty((0, 3)), torch.empty((0, 3)), torch.empty((0,))

    means = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    colors = np.stack([rs, gs_, bs], axis=1).astype(np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0
    tracks = _get_field(["track_length", "tracks", "track", "num_obs", "n_obs"])
    track_tensor = (
        torch.from_numpy(tracks.astype(np.float32))
        if tracks is not None
        else torch.zeros((means.shape[0],), dtype=torch.float32)
    )
    return torch.from_numpy(means), torch.from_numpy(colors), track_tensor


def _load_colmap_txt_points(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means = []
    colors = []
    tracks = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            x, y, z = map(float, parts[1:4])
            r, g, b = [float(int(val)) / 255.0 for val in parts[4:7]]
            track_len = float(len(parts) - 8) / 2 if len(parts) > 8 else 0.0
            means.append((x, y, z))
            colors.append((r, g, b))
            tracks.append(track_len)
    if not means:
        return torch.empty((0, 3)), torch.empty((0, 3)), torch.empty((0,))
    return (
        torch.tensor(means, dtype=torch.float32),
        torch.tensor(colors, dtype=torch.float32),
        torch.tensor(tracks, dtype=torch.float32),
    )


def _load_image_tensor(path: Path, height: int, width: int) -> torch.Tensor:
    with Image.open(path) as handle:
        img = handle.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), resample=Image.BILINEAR)
        array = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(array)
