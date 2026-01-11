"""Preview helpers for TrainingTab."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

from nullsplats.backend.splat_train import PreviewPayload


class TrainingTabPreviewMixin:
    """Preview polling and in-memory preview helpers."""

    def _handle_preview_payload(self, payload: PreviewPayload) -> None:
        try:
            self._preview_queue.put_nowait(payload)
        except Exception:
            self.logger.debug("Preview payload dropped", exc_info=True)

    def _schedule_preview_drain(self) -> None:
        if not self.frame.winfo_exists():
            return
        self._preview_drain_job = self.frame.after(150, self._drain_preview_queue)

    def _drain_preview_queue(self) -> None:
        if not self.frame.winfo_exists():
            return
        latest: Optional[PreviewPayload] = None
        while True:
            try:
                latest = self._preview_queue.get_nowait()
            except Exception:
                break
        if latest is not None and not self._preview_paused_for_sfm:
            if self.preview_canvas is not None and self._preview_toggle.get() and self._tab_active:
                try:
                    self.preview_canvas.start_rendering()
                    self.preview_canvas.load_preview_data(latest)
                    self.preview_status_var.set(
                        f"In-memory preview (iter {latest.iteration}, {latest.means.shape[0]} pts)"
                    )
                except Exception:
                    self.logger.debug("Preview apply failed", exc_info=True)
        self._schedule_preview_drain()

    def _pause_preview_for_sfm(self) -> None:
        if self._preview_paused_for_sfm:
            return
        self._preview_toggle_before_sfm = bool(self._preview_toggle.get())
        self._preview_paused_for_sfm = True
        self._preview_polling = False
        if self.preview_canvas is not None:
            try:
                self.preview_canvas.stop_rendering()
            except Exception:
                self.logger.debug("Failed to stop preview rendering for SFM", exc_info=True)

    def _resume_preview_after_sfm(self) -> None:
        if not self._preview_paused_for_sfm:
            return
        self._preview_paused_for_sfm = False
        if not self._preview_toggle_before_sfm:
            return
        if self.preview_canvas is not None:
            self.preview_canvas.start_rendering()
        if self._tab_active:
            self._toggle_preview_poll(force_on=True)

    def _toggle_preview_poll(self, force_on: bool = False) -> None:
        if self._preview_paused_for_sfm:
            return
        if self._in_memory_preview_active:
            return
        desired = bool(self._preview_toggle.get()) or force_on
        if desired and not self._preview_polling and self._tab_active:
            self._preview_polling = True
            self.logger.info("Preview poll loop start (enabled)")
            if self.preview_canvas is not None:
                self.preview_canvas.start_rendering()
            self._schedule_preview_poll()
        elif (not desired or not self._tab_active) and self._preview_polling:
            self._preview_polling = False
            self.logger.info("Preview poll loop stopped")
            if self.preview_canvas is not None:
                self.preview_canvas.stop_rendering()

    def _refresh_preview_now(self) -> None:
        """Manually force the viewer to reload the latest checkpoint, resetting the renderer."""
        scene_id = self.app_state.current_scene_id
        if scene_id is None:
            self._set_status("Select a scene before refreshing preview.", is_error=True)
            return
        if self.preview_canvas is not None:
            try:
                self.preview_canvas.reset_preview_pipelines()
            except Exception:
                self.logger.debug("Preview reset failed", exc_info=True)
        self._last_preview_path = None
        # Ensure polling is active for this manual refresh and force a fresh load.
        self._toggle_preview_poll(force_on=True)
        self._poll_latest_checkpoint(force=True)

    def _schedule_preview_poll(self) -> None:
        if not self._preview_polling:
            return
        self.frame.after(3000, self._poll_latest_checkpoint)

    def _poll_latest_checkpoint(self, force: bool = False) -> None:
        if self._in_memory_preview_active:
            return
        if (not self._preview_polling and not force) or not self._tab_active:
            self.logger.debug(
                "Preview poll skipped force=%s polling=%s tab_active=%s", force, self._preview_polling, self._tab_active
            )
            return
        self._preview_cycle += 1
        cycle_id = self._preview_cycle
        scene_id = self.app_state.current_scene_id
        self.logger.debug(
            "Preview poll cycle=%d scene=%s start force=%s polling=%s",
            cycle_id,
            scene_id,
            force,
            self._preview_polling,
        )
        if scene_id is None:
            self.logger.debug("Preview poll cycle=%d: no active scene set", cycle_id)
            self._schedule_preview_poll()
            return
        try:
            paths = self.app_state.scene_manager.get(scene_id).paths
            if not paths.splats_dir.exists():
                self.logger.debug(
                    "Preview poll cycle=%d scene=%s: splats dir missing at %s", cycle_id, scene_id, paths.splats_dir
                )
                self.preview_status_var.set("No splats directory yet.")
                self._last_preview_path = None
                self._schedule_preview_poll()
                return
            latest = self._latest_checkpoint(paths.splats_dir)
            if latest is None:
                self.logger.debug(
                    "Preview poll cycle=%d scene=%s: no checkpoints found (splats_dir=%s)",
                    cycle_id,
                    scene_id,
                    paths.splats_dir,
                )
                self.preview_status_var.set("No checkpoints found yet.")
            elif force or latest != self._last_preview_path:
                self.logger.debug(
                    "Preview poll cycle=%d scene=%s: loading %s last_preview=%s",
                    cycle_id,
                    scene_id,
                    latest.name,
                    self._last_preview_path,
                )
                self._load_preview(latest, allow_when_disabled=force)
            else:
                self.logger.debug("Preview poll cycle=%d scene=%s: no new checkpoints", cycle_id, scene_id)
                self.preview_status_var.set(f"Previewing {latest.name}")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Preview poll cycle=%d scene=%s failed: %s", cycle_id, scene_id, exc)
        finally:
            self.logger.debug("Preview poll cycle=%d scene=%s complete", cycle_id, scene_id)
            self._schedule_preview_poll()

    def _latest_checkpoint(self, splat_dir: Path) -> Optional[Path]:
        if not splat_dir.exists():
            return None
        candidates = [p for p in splat_dir.iterdir() if p.suffix.lower() in {".ply", ".splat"}]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _load_preview(self, checkpoint_path: Path, *, allow_when_disabled: bool = False) -> None:
        if self.preview_canvas is None:
            self.logger.info("Preview load skipped: no preview canvas for %s", checkpoint_path)
            return
        if checkpoint_path.suffix.lower() != ".ply":
            warn = f"Preview supports .ply checkpoints; found {checkpoint_path.name}"
            self.logger.warning(warn)
            self._set_status(warn, is_error=True)
            self.preview_status_var.set(warn)
            self._last_preview_path = checkpoint_path
            return
        try:
            call_started = time.perf_counter()
            self.logger.debug(
                "Preview load enter path=%s allow_when_disabled=%s polling=%s",
                checkpoint_path,
                allow_when_disabled,
                self._preview_polling,
            )
            if not self._preview_polling and not allow_when_disabled:
                self.logger.info("Preview load skipped (polling disabled) for %s", checkpoint_path)
                return
            if self.preview_canvas is not None:
                viewer = getattr(self.preview_canvas, "_viewer", None)
                self.logger.debug(
                    "Preview load start_rendering path=%s viewer=%s mapped=%s last_path=%s allow_when_disabled=%s",
                    checkpoint_path,
                    type(viewer).__name__ if viewer is not None else None,
                    self.preview_canvas.winfo_ismapped(),
                    getattr(self.preview_canvas, "last_path", None),
                    allow_when_disabled,
                )
                self.preview_canvas.start_rendering()
                self.logger.debug(
                    "Preview load after start_rendering path=%s elapsed_ms=%.2f",
                    checkpoint_path,
                    (time.perf_counter() - call_started) * 1000.0,
                )
            load_start = time.perf_counter()
            self.logger.info("Preview load calling load_splat for %s", checkpoint_path)
            self.preview_canvas.load_splat(checkpoint_path)
            self.logger.debug(
                "Preview load dispatched to canvas path=%s load_elapsed_ms=%.2f total_elapsed_ms=%.2f",
                checkpoint_path,
                (time.perf_counter() - load_start) * 1000.0,
                (time.perf_counter() - call_started) * 1000.0,
            )
            self._last_preview_path = checkpoint_path
            self.logger.info("Preview load completed queue for %s", checkpoint_path)
            self._set_status(f"Previewing {checkpoint_path.name}", is_error=False)
            self.preview_status_var.set(f"Previewing {checkpoint_path.name}")
            try:
                self.preview_canvas.render_once()
            except Exception:
                self.logger.debug("Preview render_once failed for %s", checkpoint_path, exc_info=True)
                pass
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to load preview for %s", checkpoint_path)
            self._set_status(f"Preview load failed: {exc}", is_error=True)
            self.preview_status_var.set(f"Preview failed: {exc}")
            self._preview_toggle.set(False)
            self._preview_polling = False

    def _ensure_preview_running(self) -> None:
        if not self.preview_canvas:
            return
        if self._preview_paused_for_sfm:
            return
        self._clear_stale_preview_for_scene()
        if self._preview_toggle.get():
            self.preview_canvas.start_rendering()
            self._toggle_preview_poll(force_on=True)
            self._poll_latest_checkpoint(force=True)

    def _clear_stale_preview_for_scene(self) -> None:
        """If the current viewer content belongs to another scene, clear it."""
        scene = self.app_state.current_scene_id
        if scene is None or self.preview_canvas is None:
            return
        try:
            last_path = self.preview_canvas.last_path
        except Exception:
            last_path = None
        if last_path is None:
            return
        try:
            last_scene = last_path.parent.parent.name if last_path.parent.name == "splats" else None
        except Exception:
            last_scene = None
        if last_scene is not None and str(last_scene) != str(scene):
            self.logger.info(
                "Clearing stale preview: viewer scene=%s current_scene=%s path=%s", last_scene, scene, last_path
            )
            try:
                self.preview_canvas.clear()
            except Exception:
                self.logger.debug("Failed to clear stale preview", exc_info=True)
            self._last_preview_path = None

