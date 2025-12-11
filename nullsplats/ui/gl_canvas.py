"""Gaussian splat viewer that embeds the Depth-Anything-3 OpenGL renderer.

This module now wraps the OpenGL GaussianSplatViewer (pyopengltk + PyOpenGL)
and routes checkpoint loads through `SplatRenderer` for turntables / offline
rasterization while the GL viewer keeps running in its own widget.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable, Iterable, List, Optional, Tuple
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from nullsplats.backend.io_cache import ScenePaths
from nullsplats.util.logging import get_logger


logger = get_logger("ui.gl_canvas")

try:
    from nullsplats.ui.gaussian_splat_viewer import GaussianSplatViewer
except Exception as exc:  # noqa: BLE001
    GaussianSplatViewer = None
    logger.warning("OpenGL viewer import failed: %s", exc)


@dataclass(frozen=True)
class CameraView:
    """Simple orbit camera definition."""

    yaw: float
    pitch: float
    distance: float
    target: torch.Tensor  # (3,)


@dataclass(frozen=True)
class SplatData:
    """Parsed splat checkpoint suitable for rendering."""

    means: torch.Tensor
    scales_log: torch.Tensor
    quats: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor  # (N, C, 3) SH coefficients (band-0 then rest)
    sh_degree: int
    center: torch.Tensor
    radius: float
    path: Path


class SplatRenderer:
    """Load and render Gaussian splat checkpoints."""

    def __init__(self, device: str | torch.device = "cuda:0") -> None:
        self.device = torch.device(device) if isinstance(device, str) else device
        self.data: Optional[SplatData] = None

    def load(self, path: Path) -> SplatData:
        started = time.perf_counter()
        logger.info("Renderer load entry path=%s thread_ident=%s (checking CUDA availability)", path, threading.get_ident())
        cuda_start = time.perf_counter()
        cuda_ok = torch.cuda.is_available()
        logger.info(
            "Renderer CUDA available=%s path=%s cuda_check_ms=%.2f",
            cuda_ok,
            path,
            (time.perf_counter() - cuda_start) * 1000.0,
        )
        if not cuda_ok:
            raise RuntimeError("CUDA required for splat preview; install a CUDA build of PyTorch.")
        try:
            size = path.stat().st_size
        except Exception:  # noqa: BLE001
            size = -1
        logger.info("Renderer loading splats from %s size_bytes=%s", path, size)
        parse_start = time.perf_counter()
        raw = _load_ply_properties(path)
        logger.info("Renderer PLY parsed path=%s parse_ms=%.2f", path, (time.perf_counter() - parse_start) * 1000.0)
        means = _stack_props(raw, ("x", "y", "z"))
        dc = _stack_props(raw, ("f_dc_0", "f_dc_1", "f_dc_2")).reshape(-1, 1, 3)
        rest_props = _collect_rest_props(raw)
        if len(rest_props) % 3 != 0:
            raise ValueError("f_rest properties must be divisible by 3 for SH coefficients.")
        sh_rest = _stack_props(raw, rest_props).reshape(-1, max(1, len(rest_props) // 3), 3)
        sh_channels = dc.shape[1] + sh_rest.shape[1]
        sh_degree = max(0, int(round(math.sqrt(sh_channels) - 1)))
        logger.info(
            "PLY stats path=%s verts=%d sh_channels=%d sh_degree=%d rest_props=%d",
            path,
            means.shape[0],
            sh_channels,
            sh_degree,
            len(rest_props),
        )

        scales_log = _stack_props(raw, ("scale_0", "scale_1", "scale_2"))
        opacities = torch.tensor(raw["opacity"], dtype=torch.float32)
        quats = _stack_props(raw, ("rot_0", "rot_1", "rot_2", "rot_3"))
        quats = F.normalize(quats, dim=1)
        logger.info("Renderer tensors prepared path=%s", path)

        center = means.mean(dim=0)
        radius = float(torch.linalg.norm(means - center, dim=1).max().item())
        radius = radius if radius > 1e-5 else 1.0

        self.data = SplatData(
            means=means.to(self.device),
            scales_log=scales_log.to(self.device),
            quats=quats.to(self.device),
            opacities=opacities.to(self.device),
            colors=torch.cat([dc, sh_rest], dim=1).to(self.device),
            sh_degree=sh_degree,
            center=center.to(self.device),
            radius=radius,
            path=path,
        )
        elapsed = time.perf_counter() - started
        logger.info(
            "Loaded splat checkpoint %s points=%d sh_degree=%d elapsed=%.3fs",
            path,
            means.shape[0],
            sh_degree,
            elapsed,
        )
        return self.data

    def render(self, width: int, height: int, view: CameraView, *, data: Optional[SplatData] = None) -> Image.Image:
        data = data or self.data
        if data is None:
            raise RuntimeError("No splat data loaded.")
        if width <= 0 or height <= 0:
            raise ValueError("Canvas dimensions must be positive.")

        logger.debug("Render step: building camtoworld")
        camtoworld = _camera_to_world(view)
        logger.debug("Render step: camtoworld built")
        logger.debug(
            "Render start path=%s size=%sx%s yaw=%.4f pitch=%.4f dist=%.4f target=%s",
            data.path,
            width,
            height,
            view.yaw,
            view.pitch,
            view.distance,
            tuple(float(x) for x in view.target),
        )
        logger.debug("Render step: inverting camtoworld")
        viewmat = torch.linalg.inv(camtoworld).unsqueeze(0)
        logger.debug("Render step: camtoworld inverted")
        K = _intrinsics(width, height, fov_deg=60.0).to(data.means.device).unsqueeze(0)
        logger.debug("Render step: intrinsics ready")

        scales = torch.exp(data.scales_log)
        renders, alphas, info = _rasterize(
            means=data.means,
            scales=scales,
            quats=data.quats,
            opacities=data.opacities,
            colors=data.colors,
            sh_degree=data.sh_degree,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
        )

        render_time = info.get("render_time_ms", None)
        logger.debug(
            "Rendered splats path=%s size=%sx%s sh_degree=%d render_time_ms=%s",
            data.path,
            width,
            height,
            data.sh_degree,
            render_time,
        )

        image = renders[0].clamp(0.0, 1.0).detach().cpu().numpy()
        image = (image * 255.0).astype(np.uint8)
        return Image.fromarray(image)


class GLCanvas(ttk.Frame):
    """Tkinter widget that uses an OpenGL Gaussian viewer."""

    def __init__(self, master: tk.Misc, *, device: str | torch.device = "cuda:0", width: int = 640, height: int = 480) -> None:
        super().__init__(master)
        self.renderer = SplatRenderer(device=device)
        self._viewer: Optional["GaussianSplatViewer"] = None
        self.canvas: Optional[tk.Widget] = None
        self._last_path: Optional[Path] = None
        self._latest_load_request = 0
        self._load_lock = threading.Lock()
        self._load_thread: Optional[threading.Thread] = None
        self._current_view: Optional[CameraView] = None
        self._init_viewer(width, height)
        self._camera_update_callbacks: list[Callable[[CameraView], None]] = []

    def _init_viewer(self, width: int, height: int) -> None:
        if GaussianSplatViewer is None:
            label = ttk.Label(
                self,
                text="OpenGL viewer unavailable. Install PyOpenGL + pyopengltk.",
                foreground="#f00",
                justify="center",
            )
            label.pack(fill="both", expand=True)
            self.canvas = label
            return
        try:
            self._viewer = GaussianSplatViewer(self, width=width, height=height)
            self._viewer.pack(fill="both", expand=True)
            self.canvas = self._viewer
        except Exception:  # noqa: BLE001
            logger.exception("Failed to initialize GaussianSplatViewer")
            label = ttk.Label(
                self,
                text="OpenGL viewer failed to initialize.",
                foreground="#f00",
                justify="center",
            )
            label.pack(fill="both", expand=True)
            self.canvas = label
            self._viewer = None

    def start_rendering(self) -> None:
        if self._viewer is not None and hasattr(self._viewer, "start_rendering"):
            try:
                self._viewer.start_rendering()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to start OpenGL viewer")

    def stop_rendering(self) -> None:
        if self._viewer is not None and hasattr(self._viewer, "stop_rendering"):
            try:
                self._viewer.stop_rendering()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to stop OpenGL viewer")

    def clear(self) -> None:
        if self._viewer is not None and hasattr(self._viewer, "clear"):
            try:
                self._viewer.clear()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to clear OpenGL viewer")

    def set_camera_pose(self, *args, **kwargs) -> None:
        """Forward camera pose changes to the underlying viewer if available."""
        if self._viewer is None or not hasattr(self._viewer, "set_camera_pose"):
            return
        try:
            self._viewer.set_camera_pose(*args, **kwargs)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to set camera pose on OpenGL viewer")

    def get_scene_center(self) -> Optional[np.ndarray]:
        if self._viewer is None or not hasattr(self._viewer, "scene_center"):
            return None
        return np.array(self._viewer.scene_center, dtype=np.float32)

    def set_point_scale(self, scale: float) -> None:
        if self._viewer is not None and hasattr(self._viewer, "set_point_scale"):
            try:
                self._viewer.set_point_scale(scale)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to set point scale")

    def set_background_color(self, color: str) -> None:
        if self._viewer is not None and hasattr(self._viewer, "set_background_color"):
            try:
                self._viewer.set_background_color(color)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to set background color")

    def set_sort_back_to_front(self, value: bool) -> None:
        if self._viewer is not None and hasattr(self._viewer, "set_sort_back_to_front"):
            try:
                self._viewer.set_sort_back_to_front(value)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to set sort order")

    def set_debug_mode(self, enabled: bool) -> None:
        if self._viewer is not None and hasattr(self._viewer, "set_debug_mode"):
            try:
                self._viewer.set_debug_mode(enabled)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to set debug mode")

    def set_flat_color_mode(self, enabled: bool) -> None:
        if self._viewer is not None and hasattr(self._viewer, "set_debug_flat_color"):
            try:
                self._viewer.set_debug_flat_color(enabled)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to set flat color mode")

    def set_scale_bias(self, bias: Tuple[float, float, float]) -> None:
        if self._viewer is not None and hasattr(self._viewer, "set_scale_bias"):
            try:
                self._viewer.set_scale_bias(bias)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to set scale bias")

    def set_opacity_bias(self, bias: float) -> None:
        if self._viewer is not None and hasattr(self._viewer, "set_opacity_bias"):
            try:
                self._viewer.set_opacity_bias(bias)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to set opacity bias")

    def request_depth_sort(self) -> None:
        if self._viewer is not None and hasattr(self._viewer, "request_depth_sort"):
            try:
                self._viewer.request_depth_sort()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to request depth sort")

    def get_current_view(self) -> Optional[CameraView]:
        return self._current_view

    def adjust_camera_angles(self, *, yaw: Optional[float] = None, pitch: Optional[float] = None, distance: Optional[float] = None) -> None:
        if self._current_view is None:
            return
        new_yaw = self._current_view.yaw if yaw is None else yaw
        new_pitch = self._current_view.pitch if pitch is None else pitch
        new_distance = self._current_view.distance if distance is None else distance
        target = self._current_view.target
        view = CameraView(yaw=new_yaw, pitch=new_pitch, distance=max(new_distance, 0.001), target=target)
        self._current_view = view
        self._update_viewer_camera(view)
        self._notify_camera_listeners(view)

    def recenter_camera(self) -> None:
        if self.renderer.data is None:
            return
        view = self._default_view_from_data(self.renderer.data, self.renderer.data.path)
        self._current_view = view
        self._update_viewer_camera(view)
        self._notify_camera_listeners(view)

    def add_camera_listener(self, callback: Callable[[CameraView], None]) -> None:
        self._camera_update_callbacks.append(callback)
        if self._current_view is not None:
            callback(self._current_view)

    def _notify_camera_listeners(self, view: CameraView) -> None:
        for callback in self._camera_update_callbacks:
            try:
                callback(view)
            except Exception:  # noqa: BLE001
                logger.exception("Camera listener callback failed")

    def load_splat(self, path: Path) -> None:
        target = Path(path)
        self._latest_load_request += 1
        request_id = self._latest_load_request
        thread = threading.Thread(
            target=self._load_worker,
            args=(request_id, target),
            name=f"glcanvas_load_{request_id}",
            daemon=True,
        )
        self._load_thread = thread
        thread.start()
        logger.info("GLCanvas load dispatched request=%d path=%s", request_id, target)

    @property
    def last_path(self) -> Optional[Path]:
        return self._last_path

    def _load_worker(self, request_id: int, path: Path) -> None:
        try:
            data = self.renderer.load(path)
        except Exception:  # noqa: BLE001
            logger.exception("GLCanvas failed to load splat %s", path)
            return
        with self._load_lock:
            if request_id != self._latest_load_request:
                logger.info("GLCanvas load request stale request_id=%d latest=%d", request_id, self._latest_load_request)
                return
            self._last_path = path
        self.after(0, lambda: self._apply_data(data, path))

    def _apply_data(self, data: SplatData, path: Path) -> None:
        if self._viewer is None:
            return
        try:
            means = data.means.detach().cpu().numpy()
            scales = data.scales_log.detach().cpu().numpy()
            quats = data.quats.detach().cpu().numpy()
            opacities = data.opacities.detach().cpu().numpy()
            colors = data.colors.detach().cpu().numpy()
            sh_dc = colors[:, 0, :]
            self._viewer.set_gaussians(means, scales, quats, opacities, sh_dc)
            view = self._default_view_from_data(data, path)
            self._current_view = view
            self._update_viewer_camera(view)
            self._notify_camera_listeners(view)
            self.start_rendering()
            logger.info("GLCanvas applied data %s gaussians=%d", path, means.shape[0])
        except Exception:  # noqa: BLE001
            logger.exception("GLCanvas failed to upload splat %s", path)

    def destroy(self) -> None:  # type: ignore[override]
        self.stop_rendering()
        super().destroy()

    def _default_view_from_data(self, data: SplatData, path: Path) -> CameraView:
        scene_id = path.parent.parent.name if path.parent.name == "splats" else None
        if scene_id:
            view = _colmap_default_view(scene_id, data)
            if view is not None:
                return view
        return _fallback_view(data, self._current_view)

    def _update_viewer_camera(self, view: CameraView) -> None:
        if self._viewer is None:
            return
        camtoworld = _camera_to_world(view)
        cam_pos = camtoworld[:3, 3]
        try:
            self._viewer.set_camera_pose(
                cam_pos.detach().cpu().numpy(),
                rotation=torch.linalg.inv(camtoworld)[:3, :3].detach().cpu().numpy(),
                target=view.target.detach().cpu().numpy(),
            )
            logger.info(
                "Viewer camera update: pos=%s target=%s",
                tuple(float(x) for x in cam_pos.detach().cpu().numpy()),
                tuple(float(x) for x in view.target.detach().cpu().numpy()),
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to update OpenGL viewer camera")


def _load_ply_properties(path: Path) -> np.ndarray:
    start_read = time.perf_counter()
    logger.info("PLY parse start path=%s", path)
    type_map = {
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "uchar": "u1",
        "uint8": "u1",
        "uint": "<u4",
        "int": "<i4",
    }
    with path.open("rb") as handle:
        header_lines: List[str] = []
        while True:
            line_bytes = handle.readline()
            if not line_bytes:
                raise ValueError("Invalid PLY: missing end_header.")
            line = line_bytes.decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        data = handle.read()

    fmt = next((l for l in header_lines if l.startswith("format ")), "")
    if "binary_little_endian" not in fmt:
        raise ValueError("Only binary_little_endian PLY is supported for preview.")

    vertex_count = 0
    props: List[Tuple[str, str]] = []
    for line in header_lines:
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        if line.startswith("property"):
            parts = line.split()
            if len(parts) < 3:
                continue
            prop_type, name = parts[1], parts[2]
            mapped = type_map.get(prop_type)
            if mapped:
                props.append((name, mapped))

    if vertex_count == 0:
        raise ValueError("No vertices found in PLY header.")
    dtype = np.dtype([(name, typ) for name, typ in props])
    arr = np.frombuffer(data, dtype=dtype, count=vertex_count)
    if arr.size != vertex_count:
        raise ValueError(f"Expected {vertex_count} vertices, parsed {arr.size}.")
    elapsed = time.perf_counter() - start_read
    logger.info(
        "PLY parse done path=%s verts=%d props=%d elapsed=%.3fs dtype=%s",
        path,
        vertex_count,
        len(props),
        elapsed,
        arr.dtype,
    )
    return arr


def _stack_props(arr: np.ndarray, names: Iterable[str]) -> torch.Tensor:
    columns: List[np.ndarray] = []
    for name in names:
        if name not in arr.dtype.names:
            raise KeyError(f"Property {name} missing from splat file.")
        columns.append(arr[name].astype(np.float32))
    stacked = np.stack(columns, axis=1)
    return torch.from_numpy(stacked)


def _collect_rest_props(arr: np.ndarray) -> List[str]:
    props = []
    for name in arr.dtype.names:
        if name.startswith("f_rest_"):
            props.append(name)
    props.sort(key=lambda n: int(n.split("_")[-1]))
    return props


def _look_at_torch(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    forward = target - eye
    forward = forward / (torch.linalg.norm(forward) + 1e-8)
    right = torch.cross(forward, up)
    right = right / (torch.linalg.norm(right) + 1e-8)
    true_up = torch.cross(right, forward)

    view = torch.eye(4, device=eye.device, dtype=torch.float32)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[0, 3] = -torch.dot(right, eye)
    view[1, 3] = -torch.dot(true_up, eye)
    view[2, 3] = torch.dot(forward, eye)
    return view


def _camera_to_world(view: CameraView) -> torch.Tensor:
    yaw = view.yaw
    pitch = view.pitch
    distance = max(view.distance, 1e-4)
    target = view.target

    cx = target[0] + distance * math.cos(pitch) * math.sin(yaw)
    cy = target[1] + distance * math.sin(pitch)
    cz = target[2] + distance * math.cos(pitch) * math.cos(yaw)
    cam_pos = torch.tensor([cx, cy, cz], device=target.device, dtype=torch.float32)

    world_up = torch.tensor([0.0, 1.0, 0.0], device=target.device, dtype=torch.float32)
    view_mat = _look_at_torch(cam_pos, target, world_up)
    return torch.linalg.inv(view_mat)


def _intrinsics(width: int, height: int, *, fov_deg: float) -> torch.Tensor:
    fov_rad = math.radians(fov_deg)
    focal = 0.5 * width / math.tan(0.5 * fov_rad)
    fx = float(focal)
    fy = float(focal)
    cx = width / 2.0
    cy = height / 2.0
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)


def _pan_delta(view: CameraView, dx_pixels: float, dy_pixels: float) -> torch.Tensor:
    right = torch.tensor(
        [math.cos(view.yaw), 0.0, -math.sin(view.yaw)],
        device=view.target.device,
        dtype=torch.float32,
    )
    up = torch.tensor([0.0, 1.0, 0.0], device=view.target.device, dtype=torch.float32)
    scale = view.distance * 0.002
    return (-dx_pixels * scale) * right + (dy_pixels * scale) * up


def _rasterize(
    *,
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    sh_degree: int,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
):
    logger.debug("Rasterize import start")
    import gsplat  # noqa: WPS433
    from gsplat.rendering import rasterization  # noqa: WPS433
    logger.debug("Rasterize import complete")

    colors_full = colors[:, : (sh_degree + 1) ** 2, :]
    logger.debug(
        "Rasterize call start verts=%d width=%d height=%d sh_degree=%d device=%s",
        means.shape[0],
        width,
        height,
        sh_degree,
        means.device,
    )
    renders, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors_full,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode="RGB",
        absgrad=False,
    )
    logger.debug("Rasterize call complete info_keys=%s", list(info.keys()))
    return renders, alphas, info


def _vector_to_angles(vec: torch.Tensor) -> tuple[float, float]:
    """Convert a direction vector into yaw/pitch matching CameraView conventions."""
    v = vec.detach().cpu().numpy().astype(float)
    vx, vy, vz = v.tolist()
    horiz = math.hypot(vx, vz)
    yaw = math.atan2(vx, vz if abs(vz) > 1e-9 else 1e-9)
    pitch = math.atan2(vy, horiz if horiz > 1e-9 else 1e-9)
    return yaw, pitch


def _colmap_default_view(scene_id: str, data: SplatData) -> Optional[CameraView]:
    """Load a base camera from COLMAP (first image) to seed the viewer."""
    try:
        paths = ScenePaths(scene_id)
        candidates = [
            (paths.sfm_dir / "sparse" / "text" / "cameras.txt", paths.sfm_dir / "sparse" / "text" / "images.txt"),
            (paths.sfm_dir / "sparse" / "0" / "cameras.txt", paths.sfm_dir / "sparse" / "0" / "images.txt"),
            (paths.sfm_dir / "sparse" / "cameras.txt", paths.sfm_dir / "sparse" / "images.txt"),
        ]
        cameras_txt = images_txt = None
        for cams, imgs in candidates:
            if cams.exists() and imgs.exists():
                cameras_txt, images_txt = cams, imgs
                break
        if cameras_txt is None or images_txt is None:
            return None
        cameras = _parse_cameras_min(cameras_txt)
        image_entry = _parse_first_image(images_txt)
        if image_entry is None:
            return None
        cam = cameras.get(image_entry["camera_id"])
        if cam is None:
            return None
        c2w = _cam_to_world_from_qt(image_entry["qvec"], image_entry["tvec"], device=data.center.device)
        cam_pos = c2w[:3, 3]
        direction = (data.center - cam_pos)
        yaw, pitch = _vector_to_angles(direction)
        dist = float(torch.linalg.norm(direction).item())
        min_dist = max(data.radius * 1.5, 0.5)
        distance = max(dist, min_dist)
        return CameraView(yaw=yaw, pitch=pitch, distance=distance, target=data.center)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to derive default view from COLMAP scene_id=%s", scene_id)
        return None


def _parse_cameras_min(path: Path) -> dict[int, dict]:
    cams: dict[int, dict] = {}
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
            if model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            else:
                fx = fy = params[0]
                cx = params[1]
                cy = params[2] if len(params) > 2 else params[1]
            cams[cam_id] = {"width": width, "height": height, "params": (fx, fy, cx, cy)}
    return cams


def _parse_first_image(path: Path) -> Optional[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
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
            return {
                "image_id": image_id,
                "qvec": (qw, qx, qy, qz),
                "tvec": (tx, ty, tz),
                "camera_id": camera_id,
                "name": name,
            }
    return None


def _cam_to_world_from_qt(qvec: tuple[float, float, float, float], tvec: tuple[float, float, float], device: torch.device) -> torch.Tensor:
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


def _fallback_view(data: SplatData, current: Optional[CameraView]) -> CameraView:
    """Pick a stable fallback view if no camera is set."""
    if current is not None:
        return current
    return CameraView(
        yaw=0.0,
        pitch=0.25,
        distance=max(data.radius * 3.0, 1.0),
        target=data.center,
    )


__all__ = ["GLCanvas", "SplatRenderer", "CameraView", "SplatData"]
