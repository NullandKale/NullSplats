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
from nullsplats.backend.splat_train import PreviewPayload
from nullsplats.util.logging import get_logger
from nullsplats.optional_plugins import load_preview_sinks
from nullsplats.ui.preview_outputs import PreviewFrameInfo, PreviewOutputSink


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
    quats: torch.Tensor  # (N, 4) in wxyz order for gsplat
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
        if path.suffix.lower() == ".splat":
            parsed = _load_splat_binary(path)
            means = parsed["means"]
            scales_log = parsed["scales_log"]
            quats_wxyz = parsed["quats_wxyz"]
            opacities = parsed["opacities"]
            dc = parsed["sh_dc"].reshape(-1, 1, 3)
            sh_rest = torch.zeros((means.shape[0], 0, 3), dtype=torch.float32)
            sh_degree = 0
            logger.info(
                "SPLAT stats path=%s verts=%d sh_degree=%d",
                path,
                means.shape[0],
                sh_degree,
            )
        else:
            raw = _load_ply_properties(path)
            logger.info("Renderer PLY parsed path=%s parse_ms=%.2f", path, (time.perf_counter() - parse_start) * 1000.0)
            means = _stack_props(raw, ("x", "y", "z"))
            dc = _stack_props(raw, ("f_dc_0", "f_dc_1", "f_dc_2")).reshape(-1, 1, 3)
            rest_props = _collect_rest_props(raw)
            if rest_props:
                if len(rest_props) % 3 != 0:
                    raise ValueError("f_rest properties must be divisible by 3 for SH coefficients.")
                sh_rest = _stack_props(raw, rest_props).reshape(-1, len(rest_props) // 3, 3)
            else:
                sh_rest = torch.zeros((means.shape[0], 0, 3), dtype=torch.float32)
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
            opacities = _normalize_opacities(torch.tensor(raw["opacity"], dtype=torch.float32), path)
            quats_wxyz = _stack_props(raw, ("rot_0", "rot_1", "rot_2", "rot_3"))
            quats_wxyz = F.normalize(quats_wxyz, dim=1)
        logger.info("Renderer tensors prepared path=%s", path)

        center = means.mean(dim=0)
        radius = float(torch.linalg.norm(means - center, dim=1).max().item())
        radius = radius if radius > 1e-5 else 1.0

        self.data = SplatData(
            means=means.to(self.device),
            scales_log=scales_log.to(self.device),
            quats=quats_wxyz.to(self.device),
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
        self._active_request_id = 0
        self._load_lock = threading.Lock()
        self._load_in_progress = False
        self._pending_load: Optional[Path] = None
        self._load_thread: Optional[threading.Thread] = None
        self._current_view: Optional[CameraView] = None
        self._resize_job: Optional[str] = None
        self._rendering: bool = False
        self._camera_update_callbacks: list[Callable[[CameraView], None]] = []
        self._preview_sinks: list[PreviewOutputSink] = []
        self._preview_frame_id: int = 0
        self._last_frame_ts: float = 0.0
        self._frame_watchdog_job: Optional[str] = None
        self._setup_preview_sinks()
        self._init_viewer(width, height)
        # Ensure the frame itself expands with the paned container.
        self.pack_propagate(False)

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
            # Throttle render restarts during Tk resize events to avoid GL context churn.
            self._viewer.bind("<Configure>", self._on_resize)
            self.canvas = self._viewer
            self._notify_preview_sinks_viewer_ready(self._viewer)
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
        # The viewer might have been recreated while _rendering stayed True; ensure it is actually running.
        if self._rendering:
            animate_flag = getattr(self._viewer, "animate", None)
            if animate_flag in (0, None):
                logger.info("GLCanvas start_rendering: restarting viewer despite _rendering=True (animate=%s)", animate_flag)
            else:
                return
        if self._viewer is not None and hasattr(self._viewer, "start_rendering"):
            logger.info(
                "GLCanvas start_rendering viewer=%s mapped=%s animate=%s rendering=%s",
                type(self._viewer).__name__,
                self.winfo_ismapped(),
                getattr(self._viewer, "animate", None),
                self._rendering,
            )
            try:
                self._viewer.start_rendering()
                self._rendering = True
                self._schedule_frame_watchdog()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to start OpenGL viewer")
        else:
            logger.info("GLCanvas start_rendering falling back to render_once (no viewer)")
            self.render_once()

    def stop_rendering(self) -> None:
        if self._viewer is not None and hasattr(self._viewer, "stop_rendering"):
            logger.info(
                "GLCanvas stop_rendering viewer=%s animate=%s rendering=%s",
                type(self._viewer).__name__,
                getattr(self._viewer, "animate", None),
                self._rendering,
            )
            try:
                self._viewer.stop_rendering()
                self._rendering = False
                self._cancel_frame_watchdog()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to stop OpenGL viewer")

    def clear(self) -> None:
        """Stop and reset viewer state for a scene change."""
        try:
            self.stop_rendering()
        except Exception:  # noqa: BLE001
            logger.debug("Clear stop_rendering failed", exc_info=True)
        self._cancel_frame_watchdog()
        with self._load_lock:
            self._latest_load_request += 1  # stale any in-flight loads
            self._last_path = None
            self._rendering = False
            self._pending_load = None
        self._current_view = None
        try:
            self.renderer.data = None
        except Exception:
            pass
        if self._viewer is not None and hasattr(self._viewer, "clear"):
            try:
                self._viewer.clear()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to clear OpenGL viewer")

    def render_once(self) -> None:
        """Trigger a single render pass if available."""
        if self._viewer is not None and hasattr(self._viewer, "render_once"):
            try:
                self._viewer.render_once()
                self._notify_preview_frame_rendered()
            except Exception:  # noqa: BLE001
                logger.debug("Render-once failed", exc_info=True)

    def _on_resize(self, event: tk.Event) -> None:
        """Pause rendering during resize to prevent context errors, then restart."""
        if self._viewer is None:
            return
        try:
            # Keep the viewer's idea of width/height in sync with the Tk widget.
            w = max(1, int(getattr(event, "width", self.winfo_width())))
            h = max(1, int(getattr(event, "height", self.winfo_height())))
            self._viewer.width = w
            self._viewer.height = h
            try:
                self._viewer.configure(width=w, height=h)
            except Exception:
                pass
        except Exception:  # noqa: BLE001
            logger.debug("Failed to sync viewer size on resize", exc_info=True)
        try:
            self._viewer.stop_rendering()
        except Exception:  # noqa: BLE001
            logger.debug("Resize stop_rendering failed", exc_info=True)
        if self._resize_job is not None:
            try:
                self.after_cancel(self._resize_job)
            except Exception:
                pass
        # Restart after resize settles.
        self._resize_job = self.after(200, self._resume_after_resize)

    def _resume_after_resize(self) -> None:
        self._resize_job = None
        if self._viewer is None:
            return
        try:
            w = max(1, int(self.winfo_width()))
            h = max(1, int(self.winfo_height()))
            self._viewer.width = w
            self._viewer.height = h
            try:
                self._viewer.configure(width=w, height=h)
            except Exception:
                pass
        except Exception:  # noqa: BLE001
            logger.debug("Failed to sync viewer size after resize", exc_info=True)
        try:
            self._viewer.start_rendering()
        except Exception:  # noqa: BLE001
            logger.debug("Resize start_rendering failed", exc_info=True)

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

    def _setup_preview_sinks(self) -> None:
        try:
            sinks = load_preview_sinks()
        except Exception:  # noqa: BLE001
            logger.exception("Preview sink load failed; continuing without sinks.")
            sinks = []
        self._preview_sinks = sinks
        for sink in self._preview_sinks:
            if hasattr(sink, "on_camera_updated"):
                self.add_camera_listener(lambda view, sink=sink: self._safe_sink_call(sink.on_camera_updated, view))

    def _safe_sink_call(self, fn: Callable[..., object], *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception:  # noqa: BLE001
            logger.debug("Preview sink callback failed", exc_info=True)

    def _notify_preview_sinks_viewer_ready(self, viewer: tk.Widget) -> None:
        if not self._preview_sinks:
            return
        # Prefer viewer-level frame hooks so sinks can mirror live renders.
        add_frame_listener = getattr(viewer, "add_frame_listener", None)
        if callable(add_frame_listener):
            try:
                add_frame_listener(self._notify_preview_frame_rendered)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to attach frame listener to viewer", exc_info=True)
        for sink in self._preview_sinks:
            if hasattr(sink, "on_viewer_ready"):
                self._safe_sink_call(sink.on_viewer_ready, viewer)

    def _notify_preview_sinks_viewer_destroyed(self) -> None:
        if not self._preview_sinks:
            return
        for sink in self._preview_sinks:
            if hasattr(sink, "on_viewer_destroyed"):
                self._safe_sink_call(sink.on_viewer_destroyed)
            if hasattr(sink, "stop"):
                self._safe_sink_call(sink.stop)
        self._preview_sinks = []

    def _notify_preview_sinks_camera(self, view: Optional[CameraView]) -> None:
        if view is None or not self._preview_sinks:
            return
        for sink in self._preview_sinks:
            if hasattr(sink, "on_camera_updated"):
                self._safe_sink_call(sink.on_camera_updated, view)

    def _notify_preview_frame_rendered(self) -> None:
        if not self._preview_sinks:
            return
        self._last_frame_ts = time.time()
        self._schedule_frame_watchdog()
        # Sync camera from the live viewer so preview sinks track actual mouse movement.
        current = self._capture_viewer_camera()
        if current is not None and _view_changed(self._current_view, current):
            self._current_view = current
            self._notify_camera_listeners(current)
        self._preview_frame_id += 1
        frame = PreviewFrameInfo(frame_id=self._preview_frame_id, timestamp=time.time())
        for sink in self._preview_sinks:
            if hasattr(sink, "on_frame_rendered"):
                self._safe_sink_call(sink.on_frame_rendered, frame)

    def get_preview_sinks(self) -> list[PreviewOutputSink]:
        return list(self._preview_sinks)

    def reset_preview_pipelines(self) -> None:
        """Stop rendering and rebuild preview sinks for a clean restart."""
        try:
            self.stop_rendering()
        except Exception:
            logger.debug("reset_preview_pipelines: stop_rendering failed", exc_info=True)
        try:
            self._notify_preview_sinks_viewer_destroyed()
        except Exception:
            logger.debug("reset_preview_pipelines: sink teardown failed", exc_info=True)
        try:
            self._setup_preview_sinks()
            if self._viewer is not None:
                self._notify_preview_sinks_viewer_ready(self._viewer)
                if self._current_view is not None:
                    self._notify_preview_sinks_camera(self._current_view)
        except Exception:
            logger.debug("reset_preview_pipelines: sink reattach failed", exc_info=True)

    def _capture_viewer_camera(self) -> Optional[CameraView]:
        """Read the current viewer camera into a CameraView if available."""
        viewer = self._viewer
        if viewer is None:
            return None
        try:
            cam = getattr(viewer, "camera", None)
            if cam is None:
                return None
            pos = np.array(cam.position, dtype=np.float32)
            target = np.array(cam.target, dtype=np.float32)
            direction = pos - target
            distance = float(np.linalg.norm(direction))
            if distance < 1e-6:
                return None
            yaw, pitch = _vector_to_angles(torch.tensor(direction))
            device = self.renderer.data.center.device if self.renderer.data is not None else torch.device("cpu")
            return CameraView(
                yaw=float(yaw),
                pitch=float(pitch),
                distance=distance,
                target=torch.tensor(target, device=device, dtype=torch.float32),
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to capture viewer camera", exc_info=True)
            return None

    def _capture_viewer_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        viewer = self._viewer
        if viewer is None:
            return None
        try:
            cam = getattr(viewer, "camera", None)
            if cam is None:
                return None
            pos = np.array(cam.position, dtype=np.float32)
            target = np.array(cam.target, dtype=np.float32)
            return pos, target
        except Exception:  # noqa: BLE001
            logger.debug("Failed to capture viewer pose", exc_info=True)
            return None

    def _apply_viewer_pose(self, pos: np.ndarray, target: np.ndarray) -> None:
        if self._viewer is None:
            return
        try:
            eye = torch.tensor(pos, dtype=torch.float32)
            tgt = torch.tensor(target, dtype=torch.float32)
            up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
            self._viewer.set_camera_pose(
                eye.detach().cpu().numpy(),
                rotation=None,
                target=tgt.detach().cpu().numpy(),
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to apply viewer pose", exc_info=True)

    def _view_from_pose(self, pos: np.ndarray, target: np.ndarray) -> CameraView:
        direction = target - pos
        distance = float(np.linalg.norm(direction))
        yaw, pitch = _vector_to_angles(torch.tensor(direction))
        return CameraView(
            yaw=float(yaw),
            pitch=float(pitch),
            distance=max(distance, 0.001),
            target=torch.tensor(target, dtype=torch.float32),
        )

    def load_splat(self, path: Path) -> None:
        target = Path(path)
        with self._load_lock:
            self._latest_load_request += 1
            request_id = self._latest_load_request
            if self._load_in_progress:
                self._pending_load = target
                logger.info(
                    "GLCanvas load_splat queued request=%d path=%s active_request=%s pending=%s",
                    request_id,
                    target,
                    self._active_request_id,
                    self._pending_load,
                )
                return
            self._load_in_progress = True
            self._active_request_id = request_id
        logger.info(
            "GLCanvas load_splat request=%d path=%s viewer=%s mapped=%s rendering=%s last_path=%s",
            request_id,
            target,
            type(self._viewer).__name__ if self._viewer is not None else None,
            self.winfo_ismapped(),
            self._rendering,
            self._last_path,
        )
        thread = threading.Thread(
            target=self._load_worker,
            args=(request_id, target),
            name=f"glcanvas_load_{request_id}",
            daemon=True,
        )
        self._load_thread = thread
        thread.start()
        logger.info("GLCanvas load dispatched request=%d path=%s", request_id, target)

    def load_preview_data(self, payload: PreviewPayload) -> None:
        """Apply an in-memory splat payload without touching disk."""
        if self._viewer is None:
            logger.info("GLCanvas preview skipped (no viewer)")
            return
        if not self.winfo_exists():
            logger.info("GLCanvas preview skipped (widget destroyed)")
            return
        prior_pose = self._capture_viewer_pose()
        prior_view = self._capture_viewer_camera() or self._current_view
        try:
            means = _to_numpy(payload.means)
            scales = _to_numpy(payload.scales_log)
            quats_wxyz = _to_numpy(payload.quats_wxyz)
            opacities = _to_numpy(payload.opacities)
            sh_dc = _to_numpy(payload.sh_dc)

            if means.size == 0:
                logger.info("GLCanvas preview skipped (empty payload)")
                return

            quats = np.ascontiguousarray(quats_wxyz[:, [1, 2, 3, 0]])
            self._viewer.set_gaussians(means, scales, quats, opacities, sh_dc, preserve_camera=True)

            center_np = means.mean(axis=0)
            center = torch.tensor(center_np, dtype=torch.float32)
            radius = float(np.linalg.norm(means - center_np, axis=1).max())
            radius = radius if radius > 1e-5 else 1.0
            data = SplatData(
                means=torch.from_numpy(means),
                scales_log=torch.from_numpy(scales),
                quats=torch.from_numpy(quats_wxyz),
                opacities=torch.from_numpy(opacities),
                colors=torch.from_numpy(sh_dc).unsqueeze(1),
                sh_degree=0,
                center=center,
                radius=radius,
                path=Path("__preview__"),
            )
            self.renderer.data = data

            if prior_pose is not None:
                self._apply_viewer_pose(*prior_pose)
                self._current_view = self._view_from_pose(*prior_pose)
            elif prior_view is not None:
                self._current_view = prior_view
                self._update_viewer_camera(prior_view)
            elif self._current_view is None:
                view = _fallback_view(data, self._current_view)
                self._current_view = view
                self._update_viewer_camera(view)

            self._notify_camera_listeners(self._current_view)
            self.start_rendering()
            if hasattr(self._viewer, "request_depth_sort"):
                try:
                    self._viewer.request_depth_sort()
                except Exception:  # noqa: BLE001
                    logger.debug("Depth sort request failed", exc_info=True)
            if hasattr(self._viewer, "render_once"):
                try:
                    self._viewer.render_once()
                    self._notify_preview_frame_rendered()
                except Exception:  # noqa: BLE001
                    logger.debug("Render-once failed", exc_info=True)
            logger.info(
                "GLCanvas preview applied iteration=%d gaussians=%d",
                payload.iteration,
                means.shape[0],
            )
        except Exception:  # noqa: BLE001
            logger.exception("GLCanvas failed to upload preview payload")

    @property
    def last_path(self) -> Optional[Path]:
        return self._last_path

    def _load_worker(self, request_id: int, path: Path) -> None:
        data: Optional[SplatData] = None
        try:
            data = self.renderer.load(path)
        except Exception:  # noqa: BLE001
            logger.exception("GLCanvas failed to load splat %s (request=%d)", path, request_id)
            self._finish_load(request_id)
            return
        with self._load_lock:
            previous_path = self._last_path
            is_stale = request_id != self._latest_load_request
            if not is_stale:
                self._last_path = path
            latest_request = self._latest_load_request
        def _apply_and_finish() -> None:
            try:
                if is_stale:
                    logger.info(
                        "GLCanvas load request stale request_id=%d latest=%d pending=%s",
                        request_id,
                        latest_request,
                        self._pending_load,
                    )
                    return
                self._apply_data(data, path, previous_path)
            finally:
                self._finish_load(request_id)
        self.after(0, _apply_and_finish)

    def _finish_load(self, request_id: int) -> None:
        next_path: Optional[Path] = None
        with self._load_lock:
            self._load_thread = None
            self._active_request_id = 0
            self._load_in_progress = False
            if self._pending_load is not None:
                next_path = self._pending_load
                self._pending_load = None
        if next_path is not None:
            logger.info("GLCanvas dispatching queued load path=%s after request=%d", next_path, request_id)
            self.load_splat(next_path)

    def _apply_data(self, data: SplatData, path: Path, previous_path: Optional[Path]) -> None:
        if self._viewer is None:
            logger.info("GLCanvas apply_data skipped (no viewer) path=%s request_path=%s last_path=%s", path, path, self._last_path)
            return
        if not self.winfo_exists():
            logger.info("GLCanvas apply_data skipped (widget destroyed) path=%s", path)
            return
        logger.debug(
            "GLCanvas apply_data enter path=%s viewer=%s mapped=%s animate=%s rendering=%s",
            path,
            type(self._viewer).__name__,
            self.winfo_ismapped(),
            getattr(self._viewer, "animate", None),
            self._rendering,
        )
        # Reset camera when switching to a new checkpoint/scene so we don't stare at empty space.
        try:
            if previous_path is None or previous_path.parent != path.parent:
                logger.debug(
                    "GLCanvas resetting camera for new scene load path=%s previous_path=%s current_view=%s",
                    path,
                    previous_path,
                    self._current_view,
                )
                self._current_view = None
        except Exception:
            self._current_view = None
        view: Optional[CameraView] = self._current_view
        try:
            means = data.means.detach().cpu().numpy()
            scales = data.scales_log.detach().cpu().numpy()
            quats_wxyz = data.quats.detach().cpu().numpy()
            quats = np.ascontiguousarray(quats_wxyz[:, [1, 2, 3, 0]])
            opacities = data.opacities.detach().cpu().numpy()
            colors = data.colors.detach().cpu().numpy()
            sh_dc = colors[:, 0, :]
            self._viewer.set_gaussians(means, scales, quats, opacities, sh_dc)
            # Preserve user camera if they orbited since the last load.
            captured_view = self._capture_viewer_camera()
            if captured_view is not None and self._current_view is not None:
                self._current_view = captured_view
            if self._current_view is None:
                view = self._default_view_from_data(data, path)
                self._current_view = view
                self._update_viewer_camera(view)
            else:
                # Keep current yaw/pitch/distance but retarget to the new data center.
                try:
                    self._current_view = CameraView(
                        yaw=self._current_view.yaw,
                        pitch=self._current_view.pitch,
                        distance=self._current_view.distance,
                        target=data.center,
                    )
                    self._update_viewer_camera(self._current_view)
                except Exception:
                    self._current_view = self._default_view_from_data(data, path)
                    self._update_viewer_camera(self._current_view)
            logger.info(
                "GLCanvas apply_data success path=%s gaussians=%d current_view=%s viewer_running=%s",
                path,
                means.shape[0],
                self._current_view,
                self._rendering,
            )
            if view is not None:
                self._notify_camera_listeners(view)
            self.start_rendering()
            # Force an immediate render and depth sort so the user sees the latest checkpoint.
            if hasattr(self._viewer, "request_depth_sort"):
                try:
                    self._viewer.request_depth_sort()
                except Exception:  # noqa: BLE001
                    logger.debug("Depth sort request failed", exc_info=True)
            if hasattr(self._viewer, "render_once"):
                try:
                    self._viewer.render_once()
                    self._notify_preview_frame_rendered()
                except Exception:  # noqa: BLE001
                    logger.debug("Render-once failed", exc_info=True)
            logger.info("GLCanvas applied data %s gaussians=%d", path, means.shape[0])
        except Exception:  # noqa: BLE001
            logger.exception("GLCanvas failed to upload splat %s", path)

    def destroy(self) -> None:  # type: ignore[override]
        self.stop_rendering()
        self._cancel_frame_watchdog()
        self._notify_preview_sinks_viewer_destroyed()
        super().destroy()

    def _schedule_frame_watchdog(self) -> None:
        if self._frame_watchdog_job is not None:
            return
        self._frame_watchdog_job = self.after(250, self._frame_watchdog_tick)

    def _cancel_frame_watchdog(self) -> None:
        if self._frame_watchdog_job is None:
            return
        try:
            self.after_cancel(self._frame_watchdog_job)
        except Exception:
            pass
        self._frame_watchdog_job = None

    def _frame_watchdog_tick(self) -> None:
        self._frame_watchdog_job = None
        if not self._rendering or self._viewer is None:
            return
        now = time.time()
        if self._last_frame_ts and (now - self._last_frame_ts) > 0.5:
            try:
                if hasattr(self._viewer, "render_once"):
                    self._viewer.render_once()
            except Exception:
                logger.debug("Frame watchdog render_once failed", exc_info=True)
        self._schedule_frame_watchdog()

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
            # Avoid passing rotation so the viewer uses target + stable world-up.
            self._viewer.set_camera_pose(
                cam_pos.detach().cpu().numpy(),
                rotation=None,
                target=view.target.detach().cpu().numpy(),
            )
            logger.info(
                "Viewer camera update: pos=%s target=%s",
                tuple(float(x) for x in cam_pos.detach().cpu().numpy()),
                tuple(float(x) for x in view.target.detach().cpu().numpy()),
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to update OpenGL viewer camera")
        self._notify_preview_sinks_camera(view)


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
    ascii_format = "ascii" in fmt
    binary_format = "binary_little_endian" in fmt
    if not (ascii_format or binary_format):
        raise ValueError(f"Unsupported PLY format: {fmt or '<missing>'}")

    vertex_count = 0
    props: List[Tuple[str, str]] = []
    current_element = None
    for line in header_lines:
        if line.startswith("element"):
            parts = line.split()
            current_element = parts[1] if len(parts) > 1 else None
            if current_element == "vertex" and len(parts) > 2:
                vertex_count = int(parts[2])
            continue
        if line.startswith("property") and current_element == "vertex":
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
    if ascii_format:
        text = data.decode("ascii", errors="ignore").strip().splitlines()
        rows: List[tuple] = []
        for line in text:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < len(props):
                continue
            parsed = []
            for value, (_, typ) in zip(parts, props):
                if np.issubdtype(np.dtype(typ), np.floating):
                    parsed.append(float(value))
                else:
                    parsed.append(int(float(value)))
            rows.append(tuple(parsed))
            if len(rows) >= vertex_count:
                break
        arr = np.array(rows, dtype=dtype)
        if arr.size != vertex_count:
            raise ValueError(f"Expected {vertex_count} vertices, parsed {arr.size}.")
    else:
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


def _load_splat_binary(path: Path) -> dict[str, torch.Tensor]:
    data = path.read_bytes()
    if not data:
        raise ValueError("Empty .splat file.")
    record_size = 32
    if len(data) % record_size != 0:
        raise ValueError(f"Invalid .splat file size ({len(data)} bytes).")
    count = len(data) // record_size
    dtype = np.dtype(
        [
            ("mean", "<f4", (3,)),
            ("scale", "<f4", (3,)),
            ("color", "u1", (4,)),
            ("rot", "u1", (4,)),
        ]
    )
    records = np.frombuffer(data, dtype=dtype, count=count)
    means = torch.from_numpy(np.array(records["mean"], dtype=np.float32))
    scales = torch.from_numpy(np.array(records["scale"], dtype=np.float32))
    scales = torch.clamp(scales, min=1e-8)
    scales_log = torch.log(scales)
    colors = torch.from_numpy(records["color"].astype(np.float32) / 255.0)
    rgb = colors[:, :3]
    opacities = torch.clamp(colors[:, 3], 0.0, 1.0)
    sh_dc = (rgb - 0.5) / 0.28209479177387814
    quats = torch.from_numpy(records["rot"].astype(np.float32))
    quats = (quats - 128.0) / 128.0
    quats = F.normalize(quats, dim=1)
    return {
        "means": means,
        "scales_log": scales_log,
        "opacities": opacities,
        "sh_dc": sh_dc,
        "quats_wxyz": quats,
    }


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


def _normalize_opacities(opacities: torch.Tensor, path: Path) -> torch.Tensor:
    """Ensure opacities are in linear [0, 1] space for rendering."""
    if opacities.numel() == 0:
        return opacities
    sanitized = torch.nan_to_num(opacities, nan=0.0, posinf=20.0, neginf=-20.0)
    finite = torch.isfinite(sanitized)
    if not finite.any():
        logger.warning("PLY opacities are non-finite for %s; defaulting to zeros.", path)
        return torch.zeros_like(opacities)
    finite_vals = sanitized[finite]
    min_val = float(finite_vals.min().item())
    max_val = float(finite_vals.max().item())
    if min_val < 0.0 or max_val > 1.0:
        logger.info(
            "PLY opacities look like logits for %s (min=%.3f max=%.3f); applying sigmoid.",
            path,
            min_val,
            max_val,
        )
        return torch.sigmoid(sanitized)
    return sanitized


def _to_numpy(data: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(data, np.ndarray):
        arr = data
    else:
        arr = data.detach().cpu().numpy()
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return np.ascontiguousarray(arr)


def _look_at_torch(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    forward = target - eye
    forward = forward / (torch.linalg.norm(forward) + 1e-8)
    right = torch.cross(forward, up, dim=0)
    right = right / (torch.linalg.norm(right) + 1e-8)
    true_up = torch.cross(right, forward, dim=0)

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


def _view_changed(prev: Optional[CameraView], curr: CameraView, *, eps: float = 1e-4) -> bool:
    if prev is None:
        return True
    if abs(prev.yaw - curr.yaw) > eps:
        return True
    if abs(prev.pitch - curr.pitch) > eps:
        return True
    if abs(prev.distance - curr.distance) > eps:
        return True
    try:
        delta = (prev.target - curr.target).abs().max().item()
        return float(delta) > eps
    except Exception:
        return True


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
    """Convert a target-to-camera direction vector into yaw/pitch."""
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
        direction = (cam_pos - data.center)
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
