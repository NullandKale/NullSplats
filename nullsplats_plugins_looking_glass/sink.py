"""Preview output sink that renders a quilt and submits it to Bridge."""

from __future__ import annotations

import time
from typing import Optional

import os
import numpy as np
from nullsplats.ui.gl_canvas import CameraView
from nullsplats.ui.preview_outputs import PreviewFrameInfo, PreviewOutputSink
from nullsplats.util.logging import get_logger

from .camera_adapter import base_pose_from_view, generate_view_offsets
from .config import LKGConfig
from .quilt_renderer import QuiltRenderer, QuiltSettings
from .bridge_session import BridgeSession

logger = get_logger("plugins.looking_glass.sink")


class LookingGlassSink(PreviewOutputSink):
    def __init__(self, config: LKGConfig) -> None:
        self.config = config
        self.viewer = None
        self.bridge = BridgeSession(display_index=config.display_index)
        self.quilt_renderer: Optional[QuiltRenderer] = None
        self.last_camera: Optional[CameraView] = None
        self._last_submit_ts: float = 0.0
        self._render_scheduled: bool = False
        self._render_job: Optional[str] = None
        self._started: bool = False
        self._start_failed: bool = False
        self._last_error: Optional[str] = None
        self._submitted_frames: int = 0
        self._dumped_quilt: bool = False

    def on_viewer_ready(self, viewer) -> None:  # noqa: ANN001
        self.viewer = viewer
        logger.info("Looking Glass sink attached to viewer; awaiting context.")

    def on_viewer_destroyed(self) -> None:
        self.stop()

    def on_camera_updated(self, camera_view: CameraView) -> None:
        self.last_camera = camera_view

    def on_frame_rendered(self, frame_info: PreviewFrameInfo) -> None:
        if not self._started and not self._start_failed:
            if not self._maybe_start():
                return
        self._ensure_render_loop()

    def _render_quilt_frame(self) -> None:
        self._render_scheduled = False
        if self.viewer is None or self.bridge.api is None or self.quilt_renderer is None:
            return
        if self.last_camera is None:
            return
        view_count = self.quilt_renderer.settings.vx * self.quilt_renderer.settings.vy
        offsets = generate_view_offsets(
            self.last_camera,
            view_count,
            self.config.depthiness,
            aspect=self.quilt_renderer.settings.aspect,
            fov_deg=self.config.fov,
            viewcone_deg=self.config.viewcone,
            focus=self.config.focus,
        )
        if not offsets:
            return
        base_pose = base_pose_from_view(self.last_camera)
        debug_colors = None
        if self.config.debug_quilt:
            # Simple repeating palette to visualize tile layout.
            palette = [
                (1.0, 0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0, 1.0),
                (0.0, 0.0, 1.0, 1.0),
                (1.0, 1.0, 0.0, 1.0),
                (1.0, 0.0, 1.0, 1.0),
                (0.0, 1.0, 1.0, 1.0),
            ]
            debug_colors = [palette[i % len(palette)] for i in range(len(offsets))]
        try:
            tex = self.quilt_renderer.render_quilt(
                self.viewer,
                offsets,
                base_pose=base_pose,
                debug_colors=debug_colors,
            )
            if self.quilt_renderer.last_successes() == 0:
                logger.debug("Skipping Bridge submit (no quilt tiles rendered)")
                return
            ok = self.bridge.submit_quilt_texture(
                tex,
                width=self.quilt_renderer.settings.quilt_width,
                height=self.quilt_renderer.settings.quilt_height,
                vx=self.quilt_renderer.settings.vx,
                vy=self.quilt_renderer.settings.vy,
                aspect=self.quilt_renderer.settings.aspect,
                zoom=self.quilt_renderer.settings.zoom,
            )
            try:
                if hasattr(self.viewer, "render_once"):
                    self.viewer.render_once()
            except Exception:
                pass
            self._submitted_frames += 1
            if self._submitted_frames <= 3:
                logger.info(
                    "Looking Glass submitted quilt frame #%s (quilt=%sx%s tiles=%sx%s)",
                    self._submitted_frames,
                    self.quilt_renderer.settings.quilt_width,
                    self.quilt_renderer.settings.quilt_height,
                    self.quilt_renderer.settings.vx,
                    self.quilt_renderer.settings.vy,
                )
            if not ok:
                logger.warning("Looking Glass submit returned False")
            # Log a few quilt samples to confirm channels are non-zero.
            if self._submitted_frames <= 2:
                try:
                    from OpenGL.GL import (
                        glBindFramebuffer,
                        glReadBuffer,
                        glReadPixels,
                        GL_FRAMEBUFFER,
                        GL_COLOR_ATTACHMENT0,
                        GL_RGBA,
                        GL_UNSIGNED_BYTE,
                    )
                    import numpy as np

                    glBindFramebuffer(GL_FRAMEBUFFER, self.quilt_renderer.active_fbo())
                    glReadBuffer(GL_COLOR_ATTACHMENT0)
                    sample = glReadPixels(0, 0, 2, 2, GL_RGBA, GL_UNSIGNED_BYTE)
                    arr = np.frombuffer(sample, dtype=np.uint8)
                    if arr.size:
                        logger.info("Quilt sample RGBA bytes=%s", arr.tolist())
                except Exception:
                    logger.exception("Quilt sample read failed")
        except Exception:  # noqa: BLE001
            logger.exception("Looking Glass frame submission failed")

    def _ensure_render_loop(self) -> None:
        if self.viewer is None:
            return
        if self._render_job is not None:
            return
        try:
            self._render_job = self.viewer.after(10, self._render_tick)
        except Exception:
            self._render_job = None

    def _render_tick(self) -> None:
        self._render_job = None
        if self.viewer is None:
            return
        try:
            if not getattr(self.viewer, "context_created", False):
                self._render_job = self.viewer.after(200, self._render_tick)
                return
            if hasattr(self.viewer, "winfo_ismapped") and not self.viewer.winfo_ismapped():
                self._render_job = self.viewer.after(200, self._render_tick)
                return
            if getattr(self.viewer, "animate", 0) == 0:
                self._render_job = self.viewer.after(200, self._render_tick)
                return
        except Exception:
            self._render_job = self.viewer.after(200, self._render_tick)
            return
        if not self._started and not self._start_failed:
            if not self._maybe_start():
                self._render_job = self.viewer.after(200, self._render_tick)
                return
        if self.bridge.api is None or self.quilt_renderer is None or self.last_camera is None:
            self._render_job = self.viewer.after(200, self._render_tick)
            return
        min_interval = 1.0 / max(self.config.max_fps, 1.0)
        now = time.time()
        elapsed = now - self._last_submit_ts
        if elapsed < min_interval:
            delay_ms = max(1, int((min_interval - elapsed) * 1000.0))
            self._render_job = self.viewer.after(delay_ms, self._render_tick)
            return
        self._last_submit_ts = now
        self._render_quilt_frame()
        self._render_job = self.viewer.after(max(1, int(min_interval * 1000.0)), self._render_tick)

    def _dump_quilt_texture(self) -> None:
        """Always attempt to dump the quilt to disk and exit once."""
        if self.quilt_renderer is None:
            logger.info("Quilt dump skipped: no quilt_renderer")
            return
        logger.info("Attempting quilt dump for debugging")
        try:
            from OpenGL.GL import (
                glBindFramebuffer,
                glReadPixels,
                glReadBuffer,
                glPixelStorei,
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                GL_PACK_ALIGNMENT,
            )
        except Exception:
            logger.exception("GL imports for quilt dump failed")
            return
        try:
            glBindFramebuffer(GL_FRAMEBUFFER, self.quilt_renderer.active_fbo())
            try:
                glReadBuffer(GL_COLOR_ATTACHMENT0)
            except Exception:
                pass
            glPixelStorei(GL_PACK_ALIGNMENT, 1)
            w = int(self.quilt_renderer.settings.quilt_width)
            h = int(self.quilt_renderer.settings.quilt_height)
            data = glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            np_arr = np.frombuffer(data, dtype=np.uint8)
            if np_arr.size == 0:
                logger.info("Quilt dump had 0 bytes")
                return
            logger.info("Quilt tex first 16 bytes=%s", np_arr[:16].tolist())
            np_arr = np_arr.reshape((h, w, 4))
            np_arr = np.flip(np_arr, axis=0)  # flip vertically for PNG
            out_path = os.path.join("tmp", "quilt_debug.png")
            try:
                from PIL import Image  # type: ignore

                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                Image.fromarray(np_arr, mode="RGBA").save(out_path)
                logger.info("Saved quilt debug PNG to %s", out_path)
            except Exception:
                logger.debug("Failed to save quilt debug PNG", exc_info=True)
                try:
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    raw_path = out_path + ".npy"
                    np.save(raw_path, np_arr)
                    logger.info("Saved raw quilt data to %s", raw_path)
                except Exception:
                    logger.debug("Failed to save quilt raw data", exc_info=True)
            self._dumped_quilt = True
            logger.info("Quilt dump complete (continuing to run)")
        except Exception:
            logger.exception("Quilt texture dump failed")

    def stop(self) -> None:
        try:
            if self.quilt_renderer is not None:
                self.quilt_renderer.stop()
        except Exception:
            pass
        self.quilt_renderer = None
        self.viewer = None
        self.bridge.stop()
        self._started = False
        self._start_failed = False
        self._last_error = None

    def retry_start(self) -> None:
        self.stop()

    def update_settings(
        self,
        *,
        depthiness: Optional[float] = None,
        focus: Optional[float] = None,
        fov: Optional[float] = None,
        viewcone: Optional[float] = None,
        zoom: Optional[float] = None,
        baseline: Optional[float] = None,
    ) -> None:
        if depthiness is None and baseline is not None:
            depthiness = baseline
        if depthiness is not None:
            self.config.depthiness = max(0.0, float(depthiness))
        if focus is not None:
            self.config.focus = float(focus)
        if fov is not None:
            self.config.fov = max(5.0, float(fov))
        if viewcone is not None:
            self.config.viewcone = max(0.0, min(89.0, float(viewcone)))
        if zoom is not None:
            self.config.zoom = max(0.1, float(zoom))
        try:
            if self.viewer is not None and hasattr(self.viewer, "set_fov_deg"):
                self.viewer.set_fov_deg(self.config.fov)
        except Exception:
            pass
        if self.quilt_renderer is not None:
            settings = self.quilt_renderer.settings
            self.quilt_renderer.update_settings(
                QuiltSettings(
                    quilt_width=settings.quilt_width,
                    quilt_height=settings.quilt_height,
                    vx=settings.vx,
                    vy=settings.vy,
                    aspect=settings.aspect,
                    zoom=self.config.zoom,
                )
            )

    def _maybe_start(self) -> bool:
        """Initialize Bridge and quilt resources; run only when GL context is active."""
        if self.viewer is None:
            return False
        try:
            # Check viewer flag indicating context creation (pyopengltk).
            if not getattr(self.viewer, "context_created", False):
                logger.debug("Viewer context not ready; deferring Looking Glass start.")
                return False
        except Exception:
            return False

        if not self.bridge.start():
            logger.info("Looking Glass sink disabled (Bridge start failed).")
            self._start_failed = True
            self._last_error = "Bridge start failed"
            return False
        defaults = self.bridge.default_quilt()
        if defaults is None:
            logger.info("Looking Glass sink disabled (no quilt defaults).")
            self._start_failed = True
            self._last_error = "No quilt defaults"
            return False
        aspect, qx, qy, vx, vy = defaults
        if self.config.views_override:
            vx = max(1, int(self.config.views_override))
            vy = 1
        quilt_w = int(qx * self.config.quilt_scale)
        quilt_h = int(qy * self.config.quilt_scale)
        settings = QuiltSettings(quilt_width=quilt_w, quilt_height=quilt_h, vx=vx, vy=vy, aspect=aspect, zoom=self.config.zoom)
        try:
            self.quilt_renderer = QuiltRenderer(settings)
            self._started = True
            self._last_error = None
            logger.info(
                "Looking Glass quilt renderer initialized quilt=%sx%s tiles=%sx%s aspect=%.3f zoom=%.2f",
                quilt_w,
                quilt_h,
                vx,
                vy,
                aspect,
                self.config.zoom,
            )
            logger.info("Looking Glass sink streaming to display index %s", self.bridge.display_index)
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Failed to initialize quilt renderer.")
            self.quilt_renderer = None
            self._start_failed = True
            self._last_error = "Quilt renderer init failed"
            return False

    def current_status(self) -> tuple[str, str]:
        if self._started and self.quilt_renderer is not None:
            qs = self.quilt_renderer.settings
            return (
                "Streaming",
                (
                    f"Quilt {qs.quilt_width}x{qs.quilt_height} ({qs.vx}x{qs.vy} views) "
                    f"zoom={qs.zoom:.2f} depthiness={self.config.depthiness:.2f} focus={self.config.focus:.2f} "
                    f"fov={self.config.fov:.1f} viewcone={self.config.viewcone:.1f}"
                ),
            )
        if self._start_failed:
            return ("Error", self._last_error or "Start failed")
        if self.viewer is None:
            return ("Waiting", "No viewer attached")
        try:
            if not getattr(self.viewer, "context_created", False):
                return ("Waiting", "GL context not ready yet")
        except Exception:
            pass
        return ("Pending", "Waiting for first frame to start Bridge")
