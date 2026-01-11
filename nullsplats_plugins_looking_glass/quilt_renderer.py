"""OpenGL quilt renderer for Looking Glass output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from OpenGL import GL

from nullsplats.util.logging import get_logger

logger = get_logger("plugins.looking_glass.quilt")


@dataclass
class QuiltSettings:
    quilt_width: int
    quilt_height: int
    vx: int
    vy: int
    aspect: float
    zoom: float


class QuiltRenderer:
    def __init__(self, settings: QuiltSettings) -> None:
        self.settings = settings
        self._fbo: list[int] = []
        self._color_tex: list[int] = []
        self._depth_rb: list[int] = []
        self._active_idx: int = 0
        self._debug_sampled: bool = False
        self._last_successes: int = 0
        self._allocate()

    def update_settings(self, settings: QuiltSettings) -> None:
        if settings == self.settings:
            return
        self.settings = settings
        self._release()
        self._allocate()

    def _allocate(self) -> None:
        """Allocate framebuffer, color texture, and depth buffer."""
        prev_fb = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
        self._fbo = []
        self._color_tex = []
        self._depth_rb = []
        self._active_idx = 0
        for _ in range(2):
            color_tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, color_tex)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA8,
                int(self.settings.quilt_width),
                int(self.settings.quilt_height),
                0,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                None,
            )
            try:
                fmt = GL.glGetTexLevelParameteriv(GL.GL_TEXTURE_2D, 0, GL.GL_TEXTURE_INTERNAL_FORMAT)
                logger.info("Quilt texture internal format=0x%x", fmt)
            except Exception:
                pass

            depth_rb = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_rb)
            GL.glRenderbufferStorage(
                GL.GL_RENDERBUFFER,
                GL.GL_DEPTH_COMPONENT24,
                int(self.settings.quilt_width),
                int(self.settings.quilt_height),
            )

            fbo = GL.glGenFramebuffers(1)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, color_tex, 0)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, depth_rb)
            # Ensure the color attachment is the active draw target.
            GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)

            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"Quilt framebuffer incomplete: 0x{status:x}")

            self._fbo.append(fbo)
            self._color_tex.append(color_tex)
            self._depth_rb.append(depth_rb)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, prev_fb)

    def render_quilt(
        self,
        viewer,
        offsets: List[Tuple[float, float]],
        *,
        base_pose: Tuple[np.ndarray, np.ndarray],
        clear: bool = True,
        debug_colors: List[Tuple[float, float, float, float]] | None = None,
    ) -> int:
        """Render the quilt using the provided poses into the FBO.

        Returns the color texture handle.
        """
        if not self._fbo or not self._color_tex:
            raise RuntimeError("QuiltRenderer not initialized.")
        tile_w = max(1, int(self.settings.quilt_width // max(1, self.settings.vx)))
        tile_h = max(1, int(self.settings.quilt_height // max(1, self.settings.vy)))
        idx = self._active_idx % len(self._fbo)
        fbo = self._fbo[idx]
        color_tex = self._color_tex[idx]
        prev_fb = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
        prev_viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
        prev_scissor_enabled = GL.glIsEnabled(GL.GL_SCISSOR_TEST)
        prev_scissor_box = GL.glGetIntegerv(GL.GL_SCISSOR_BOX)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        if clear:
            GL.glViewport(0, 0, int(self.settings.quilt_width), int(self.settings.quilt_height))
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Constrain clears from the viewer (which always calls glClear) to the current tile.
        GL.glEnable(GL.GL_SCISSOR_TEST)
        successes = 0
        for idx, (view_offset_x, proj_shift_x) in enumerate(offsets):
            x_idx = idx % self.settings.vx
            y_idx = idx // self.settings.vx
            # Bridge samples expect rows flipped (origin bottom-left), so flip Y like MinimalCube example.
            viewport_x = x_idx * tile_w
            viewport_y = (self.settings.vy - 1 - y_idx) * tile_h
            GL.glScissor(int(viewport_x), int(viewport_y), tile_w, tile_h)
            GL.glViewport(int(viewport_x), int(viewport_y), tile_w, tile_h)
            try:
                if debug_colors is not None:
                    r, g, b, a = debug_colors[idx % len(debug_colors)]
                    GL.glClearColor(r, g, b, a)
                    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
                    successes += 1
                else:
                    if viewer.render_offscreen(
                        tile_w,
                        tile_h,
                        base_pose,
                        flip_y=True,
                        view_offset_x=view_offset_x,
                        projection_shift_x=proj_shift_x,
                        invert_view_y=True,
                    ):
                        successes += 1
                    else:
                        logger.debug("render_offscreen returned False for tile %s", idx)
            except Exception:
                # Fail soft; continue rendering remaining tiles.
                logger.exception("Quilt tile %s render failed", idx)

        self._last_successes = successes

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, prev_fb)
        if prev_viewport is not None and len(prev_viewport) >= 4:
            GL.glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3])
        if prev_scissor_enabled:
            GL.glEnable(GL.GL_SCISSOR_TEST)
        else:
            GL.glDisable(GL.GL_SCISSOR_TEST)
        if prev_scissor_box is not None and len(prev_scissor_box) >= 4:
            GL.glScissor(prev_scissor_box[0], prev_scissor_box[1], prev_scissor_box[2], prev_scissor_box[3])
        if not self._debug_sampled:
            try:
                # Sample a small region to confirm nonzero output.
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
                GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
                import numpy as np

                sample = GL.glReadPixels(
                    max(0, tile_w // 2),
                    max(0, tile_h // 2),
                    4,
                    4,
                    GL.GL_RGBA,
                    GL.GL_UNSIGNED_BYTE,
                )
                arr = np.frombuffer(sample, dtype=np.uint8)
                if arr.size > 0:
                    mean_val = float(arr.mean())
                    logger.info("Quilt sample mean=%.2f (first frame)", mean_val)
                self._debug_sampled = True
            except Exception:
                pass
            finally:
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, prev_fb)
                if prev_viewport is not None and len(prev_viewport) >= 4:
                    GL.glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3])
                if prev_scissor_enabled:
                    GL.glEnable(GL.GL_SCISSOR_TEST)
                else:
                    GL.glDisable(GL.GL_SCISSOR_TEST)
                if prev_scissor_box is not None and len(prev_scissor_box) >= 4:
                    GL.glScissor(prev_scissor_box[0], prev_scissor_box[1], prev_scissor_box[2], prev_scissor_box[3])

        if successes != len(offsets):
            logger.warning(
                "Quilt render successes=%s of %s (vx=%s vy=%s tile=%sx%s)",
                successes,
                len(offsets),
                self.settings.vx,
                self.settings.vy,
                tile_w,
                tile_h,
            )
        self._active_idx = (self._active_idx + 1) % max(1, len(self._fbo))
        return color_tex

    def last_successes(self) -> int:
        return self._last_successes

    def texture_handle(self) -> int:
        if not self._color_tex:
            raise RuntimeError("Quilt texture not allocated.")
        idx = (self._active_idx - 1) % len(self._color_tex)
        return self._color_tex[idx]

    def active_fbo(self) -> int:
        if not self._fbo:
            raise RuntimeError("Quilt framebuffer not allocated.")
        idx = (self._active_idx - 1) % len(self._fbo)
        return self._fbo[idx]

    def _release(self) -> None:
        try:
            if self._color_tex:
                GL.glDeleteTextures(len(self._color_tex), self._color_tex)
        except Exception:
            pass
        try:
            if self._depth_rb:
                GL.glDeleteRenderbuffers(len(self._depth_rb), self._depth_rb)
        except Exception:
            pass
        try:
            if self._fbo:
                GL.glDeleteFramebuffers(len(self._fbo), self._fbo)
        except Exception:
            pass
        self._fbo = []
        self._color_tex = []
        self._depth_rb = []
        self._active_idx = 0

    def stop(self) -> None:
        self._release()
