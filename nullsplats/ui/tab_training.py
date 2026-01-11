"""Training tab UI for NullSplats."""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
import os
import queue
import subprocess
import sys
from tkinter import scrolledtext, ttk
from typing import Any, Optional
from nullsplats.app_state import AppState
from nullsplats.backend.splat_backends.dispatch import train_with_trainer
from nullsplats.backend.splat_backends.registry import get_trainer, list_trainers
from nullsplats.backend.splat_backends.types import TrainingOutput
from nullsplats.backend.splat_train_config import PreviewPayload, SplatTrainingConfig
from nullsplats.util.logging import get_logger
from nullsplats.util.tooling_paths import app_root, default_cuda_path
from nullsplats.util.threading import run_in_background
from nullsplats.ui.gl_canvas import GLCanvas
from nullsplats.ui.tab_training_layout import TrainingTabLayoutMixin
from nullsplats.ui.tab_training_preview import TrainingTabPreviewMixin


class TrainingTab(TrainingTabLayoutMixin, TrainingTabPreviewMixin):
    """Training tab with training controls."""

    def __init__(self, master: tk.Misc, app_state: AppState) -> None:
        self.app_state = app_state
        self.logger = get_logger("ui.training")
        self.logger.setLevel(logging.DEBUG)
        self.frame = ttk.Frame(master)

        default_cfg = SplatTrainingConfig()
        trainers = list_trainers()
        self._trainers = {trainer.name: trainer for trainer in trainers}
        self.training_method_var = tk.StringVar(value="gsplat")
        self.method_hint_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Configure training, then run.")
        self.preview_status_var = tk.StringVar(value="Viewer idle.")
        self.sfm_hint_var = tk.StringVar(value="")
        self.cuda_path_var = tk.StringVar(value=default_cuda_path())
        self.iterations_var = tk.IntVar(value=default_cfg.iterations)
        self.snapshot_var = tk.IntVar(value=default_cfg.snapshot_interval)
        self.max_points_var = tk.IntVar(value=default_cfg.max_points)
        self.export_format_var = tk.StringVar(value=default_cfg.export_format)
        self.device_var = tk.StringVar(value=default_cfg.device)
        self.image_downscale_var = tk.IntVar(value=1)
        self.batch_size_var = tk.IntVar(value=default_cfg.batch_size)
        self.sh_degree_var = tk.IntVar(value=default_cfg.sh_degree)
        self.sh_interval_var = tk.IntVar(value=default_cfg.sh_degree_interval)
        self.init_scale_var = tk.DoubleVar(value=default_cfg.init_scale)
        self.min_scale_var = tk.DoubleVar(value=default_cfg.min_scale)
        self.max_scale_var = tk.DoubleVar(value=default_cfg.max_scale)
        self.opacity_bias_var = tk.DoubleVar(value=default_cfg.opacity_bias)
        self.random_background_var = tk.BooleanVar(value=default_cfg.random_background)
        self.means_lr_var = tk.DoubleVar(value=default_cfg.means_lr)
        self.scales_lr_var = tk.DoubleVar(value=default_cfg.scales_lr)
        self.opacities_lr_var = tk.DoubleVar(value=default_cfg.opacities_lr)
        self.sh_lr_var = tk.DoubleVar(value=default_cfg.sh_lr)
        self.lr_final_scale_var = tk.DoubleVar(value=default_cfg.lr_final_scale)
        self.densify_start_var = tk.IntVar(value=default_cfg.densify_start)
        self.densify_interval_var = tk.IntVar(value=default_cfg.densify_interval)
        self.densify_opacity_var = tk.DoubleVar(value=default_cfg.densify_opacity_threshold)
        self.densify_scale_var = tk.DoubleVar(value=default_cfg.densify_scale_threshold)
        self.prune_opacity_var = tk.DoubleVar(value=default_cfg.prune_opacity_threshold)
        self.prune_scale_var = tk.DoubleVar(value=default_cfg.prune_scale_threshold)
        self.densify_max_points_var = tk.IntVar(value=default_cfg.densify_max_points)
        self.preview_interval_var = tk.DoubleVar(value=default_cfg.preview_interval_seconds)
        self.preview_min_iters_var = tk.IntVar(value=default_cfg.preview_min_iters)
        self.preview_max_points_var = tk.IntVar(value=default_cfg.max_preview_points)
        self.da3_pretrained_id_var = tk.StringVar(value="depth-anything/DA3NESTED-GIANT-LARGE")
        self.da3_process_res_var = tk.IntVar(value=504)
        self.da3_process_res_method_var = tk.StringVar(value="upper_bound_resize")
        self.da3_ref_view_var = tk.StringVar(value="saddle_balanced")
        self.da3_use_ray_pose_var = tk.BooleanVar(value=False)
        self.da3_gs_views_interval_var = tk.IntVar(value=1)
        self.da3_use_input_res_var = tk.BooleanVar(value=False)
        self.da3_view_stride_var = tk.IntVar(value=1)
        self.da3_max_views_var = tk.IntVar(value=20)
        self.da3_align_scale_var = tk.BooleanVar(value=True)
        self.da3_infer_gs_var = tk.BooleanVar(value=True)
        self.da3_allow_unposed_var = tk.BooleanVar(value=False)
        self.lkg_status_var = tk.StringVar(value="Looking Glass: not started")
        self.lkg_detail_var = tk.StringVar(value="")
        self.lkg_depthiness_var = tk.DoubleVar(value=1.0)
        self.lkg_focus_var = tk.DoubleVar(value=2.0)
        self.lkg_fov_var = tk.DoubleVar(value=14.0)
        self.lkg_viewcone_var = tk.DoubleVar(value=40.0)
        self.lkg_zoom_var = tk.DoubleVar(value=1.0)
        self.sharp_checkpoint_path_var = tk.StringVar(value="")
        self.sharp_intrinsics_source_var = tk.StringVar(value="colmap")
        self.sharp_image_index_var = tk.IntVar(value=0)
        self.sharp_focal_px_var = tk.DoubleVar(value=0.0)
        self.sharp_fov_deg_var = tk.DoubleVar(value=0.0)

        self.scene_label: Optional[ttk.Label] = None
        self.scene_status_label: Optional[ttk.Label] = None
        self.status_label: Optional[ttk.Label] = None
        self.log_view: Optional[scrolledtext.ScrolledText] = None
        self._log_handler: Optional[logging.Handler] = None
        self._working = False

        self.preview_canvas: Optional[GLCanvas] = None
        self._last_preview_path: Optional[Path] = None
        self._preview_cycle = 0
        self._preview_polling = False
        self._preview_toggle = tk.BooleanVar(value=True)
        self._preview_queue: "queue.SimpleQueue[PreviewPayload]" = queue.SimpleQueue()
        self._preview_drain_job: Optional[str] = None
        self._preview_paused_for_sfm = False
        self._preview_toggle_before_sfm = True
        self._in_memory_preview_active = False
        self._tab_active = False
        self.training_preset_var = tk.StringVar(value="medium")
        self._warmup_started = False
        self._interactive_controls: list[tk.Widget] = []
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar: Optional[ttk.Progressbar] = None
        self._gsplat_section_visible = True
        self._da3_section_visible = True
        self._sharp_section_visible = True
        self._lkg_last_sink_id: Optional[int] = None
        self._lkg_status_job: Optional[str] = None
        self._lkg_apply_job: Optional[str] = None

        self._build_contents()
        self._apply_training_preset()  # default to medium preset settings
        if getattr(self, "_lkg_enabled", False):
            self._refresh_lkg_status()
        self._update_scene_label()
        self._schedule_preview_drain()
        self._apply_trainer_capabilities()

    def _reset_progress(self, *, indeterminate: bool = False) -> None:
        if self.progress_bar is None:
            return
        self.progress_bar.stop()
        self.progress_bar.configure(mode="indeterminate" if indeterminate else "determinate", maximum=1.0)
        self.progress_var.set(0.0)
        if indeterminate:
            self.progress_bar.start(90)

    def _set_progress(self, fraction: float) -> None:
        if self.progress_bar is None:
            return
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate", maximum=1.0)
        self.progress_var.set(max(0.0, min(1.0, fraction)))

    def _scene_status_text(self) -> str:
        scene = self.app_state.current_scene_id
        if scene is None:
            return "No active scene set."
        try:
            statuses = self.app_state.refresh_scene_status()
            status = next((s for s in statuses if str(s.scene_id) == str(scene)), None)
            if status is None:
                return "Scene status unknown."
            parts = [
                f"Inputs {'OK' if status.has_inputs else '--'}",
                f"SfM {'OK' if status.has_sfm else '--'}",
                f"Splats {'OK' if status.has_splats else '--'}",
            ]
            return " > ".join(parts)
        except Exception:  # noqa: BLE001
            return "Scene status unavailable."

    def _lkg_sink(self):
        if self.preview_canvas is None:
            return None
        try:
            sinks = self.preview_canvas.get_preview_sinks()
            for sink in sinks:
                if sink.__class__.__name__ == "LookingGlassSink":
                    return sink
        except Exception:
            return None
        return None

    def _refresh_lkg_status(self) -> None:
        if not getattr(self, "_lkg_enabled", False):
            return
        sink = self._lkg_sink()
        if sink is None:
            self.lkg_status_var.set("Looking Glass: not available")
            self.lkg_detail_var.set("Connect a Looking Glass and ensure Bridge is running.")
        else:
            try:
                status, detail = sink.current_status() if hasattr(sink, "current_status") else ("Unknown", "")
            except Exception:
                status, detail = "Unknown", ""
            self.lkg_status_var.set(f"Looking Glass: {status}")
            self.lkg_detail_var.set(detail)
            sid = id(sink)
            if self._lkg_last_sink_id != sid:
                self._lkg_last_sink_id = sid
                try:
                    self.lkg_depthiness_var.set(float(getattr(sink.config, "depthiness", self.lkg_depthiness_var.get())))
                    self.lkg_focus_var.set(float(getattr(sink.config, "focus", self.lkg_focus_var.get())))
                    self.lkg_fov_var.set(float(getattr(sink.config, "fov", self.lkg_fov_var.get())))
                    self.lkg_viewcone_var.set(float(getattr(sink.config, "viewcone", self.lkg_viewcone_var.get())))
                    self.lkg_zoom_var.set(float(getattr(sink.config, "zoom", self.lkg_zoom_var.get())))
                except Exception:
                    pass
        try:
            if self._lkg_status_job is not None:
                self.frame.after_cancel(self._lkg_status_job)
        except Exception:
            pass
        self._lkg_status_job = self.frame.after(1000, self._refresh_lkg_status)

    def _lkg_retry_clicked(self) -> None:
        if not getattr(self, "_lkg_enabled", False):
            return
        sink = self._lkg_sink()
        if sink is None:
            self.lkg_status_var.set("Looking Glass: not available")
            return
        try:
            if self.preview_canvas is not None:
                try:
                    self.preview_canvas.reset_preview_pipelines()
                except Exception:
                    self.logger.debug("Preview pipeline reset during LKG retry failed", exc_info=True)
            sink.retry_start()
            self.lkg_status_var.set("Looking Glass: retrying…")
            self.lkg_detail_var.set("Will start on next frame once GL context is ready.")
        except Exception:
            self.logger.debug("Looking Glass retry failed", exc_info=True)

    def _lkg_apply_clicked(self) -> None:
        if not getattr(self, "_lkg_enabled", False):
            return
        sink = self._lkg_sink()
        if sink is None:
            return
        try:
            sink.update_settings(
                depthiness=self.lkg_depthiness_var.get(),
                focus=self.lkg_focus_var.get(),
                fov=self.lkg_fov_var.get(),
                viewcone=self.lkg_viewcone_var.get(),
                zoom=self.lkg_zoom_var.get(),
            )
            self.lkg_detail_var.set("Settings applied; next frame will use updated values.")
        except Exception:
            self.logger.debug("Looking Glass apply settings failed", exc_info=True)

    def _lkg_schedule_apply(self) -> None:
        if not getattr(self, "_lkg_enabled", False):
            return
        try:
            if self._lkg_apply_job is not None:
                self.frame.after_cancel(self._lkg_apply_job)
        except Exception:
            pass
        self._lkg_apply_job = self.frame.after(150, self._lkg_apply_clicked)

    def _sfm_hint_text(self) -> str:
        scene = self.app_state.current_scene_id
        if scene is None:
            return "SfM: no active scene selected."
        return "SfM: ready." if self._has_sfm_outputs(str(scene)) else "SfM: missing. Run COLMAP in the COLMAP tab first."

    def on_tab_selected(self, selected: bool) -> None:
        self._tab_active = selected
        if not self.preview_canvas:
            return
        if selected:
            self._update_scene_label()
            if self._preview_paused_for_sfm:
                return
            if not self._warmup_started:
                # Defer renderer warmup until the tab is visible to avoid slowing startup.
                self._warmup_renderer(trigger="tab_select")
            # Resume polling if enabled and ensure preview starts rendering.
            self._clear_stale_preview_for_scene()
            self.preview_canvas.start_rendering()
            self._ensure_preview_running()
            self._poll_latest_checkpoint(force=True)
        else:
            # Stop polling when hidden, but allow manual refresh when reactivated.
            if self._preview_polling:
                self._preview_polling = False
            self.preview_canvas.stop_rendering()

    def deactivate_viewer(self) -> None:
        """Fully stop preview rendering and release GPU state while hidden."""
        self._preview_polling = False
        if self.preview_canvas is not None:
            try:
                self.preview_canvas.stop_rendering()
            except Exception:  # noqa: BLE001
                self.logger.exception("Failed to deactivate training preview viewer")
        if getattr(self, "_lkg_enabled", False):
            self.lkg_status_var.set("Looking Glass: idle")
            self.lkg_detail_var.set("")

    def _scene_text(self) -> str:
        scene = self.app_state.current_scene_id
        if scene is None:
            return "No active scene selected."
        return f"Active scene: {scene}"

    def on_scene_changed(self, scene_id: Optional[str]) -> None:
        if scene_id is not None:
            self.app_state.set_current_scene(scene_id)
        self._last_preview_path = None
        if self.preview_canvas is not None:
            try:
                self.preview_canvas.stop_rendering()
            except Exception:
                self.logger.debug("Failed to stop preview canvas on scene change", exc_info=True)
            try:
                self.preview_canvas.clear()
            except Exception:
                pass
        self.preview_status_var.set("No checkpoints found yet.")
        self._update_scene_label()
        if scene_id is not None:
            if self._tab_active and self._preview_toggle.get():
                self._toggle_preview_poll(force_on=True)
                self._poll_latest_checkpoint(force=True)

    def _require_scene(self) -> Optional[str]:
        scene = self.app_state.current_scene_id
        if scene is None:
            self._set_status("Select or create a scene in the Inputs tab first.", is_error=True)
            return None
        return str(scene)

    def _update_scene_label(self) -> None:
        if self.scene_label is not None:
            self.scene_label.config(text=self._scene_text())
        if self.scene_status_label is not None:
            self.scene_status_label.config(text=self._scene_status_text())
        self.sfm_hint_var.set(self._sfm_hint_text())

    def _run_training_only(self) -> None:
        if self._working:
            self._set_status("Another operation is running; wait for it to finish.", is_error=True)
            return
        method = self._selected_trainer_name()
        if method == "gsplat":
            self._apply_training_preset()
        scene_id = self._require_scene()
        if scene_id is None:
            return
        allow_missing_colmap = method == "depth_anything_3" and self.da3_allow_unposed_var.get()
        if method == "sharp":
            allow_missing_colmap = self.sharp_intrinsics_source_var.get().strip().lower() != "colmap"
        if not self._has_sfm_outputs(scene_id) and not allow_missing_colmap:
            self._set_status("COLMAP outputs not found. Run COLMAP first.", is_error=True)
            return
        train_config = self._build_training_config(method)
        if self.preview_canvas is not None and method == "gsplat":
            self.preview_canvas.start_rendering()
        if method == "gsplat":
            self._preview_toggle.set(True)
            self._preview_polling = False
            self._in_memory_preview_active = True
        else:
            self._preview_toggle.set(False)
            self._preview_polling = False
            self._in_memory_preview_active = False
            self.preview_status_var.set("Live preview unavailable for this method.")
        self._set_status("Training only...")
        self._reset_progress(indeterminate=method != "gsplat")
        self._working = True
        self._set_controls_enabled(False)
        run_in_background(
            self._execute_training,
            scene_id,
            method,
            train_config,
            tk_root=self.frame.winfo_toplevel(),
            on_success=self._handle_training_success,
            on_error=self._handle_error,
            thread_name=f"train_only_{scene_id}",
        )

    def _execute_training(self, scene_id: str, trainer_name: str, train_config: dict[str, Any] | SplatTrainingConfig) -> TrainingOutput:
        preview_callback = None
        try:
            trainer = get_trainer(trainer_name)
            if trainer.capabilities.live_preview:
                preview_callback = self._handle_preview_payload
        except Exception:
            preview_callback = self._handle_preview_payload if trainer_name == "gsplat" else None
        return train_with_trainer(
            scene_id,
            trainer_name,
            train_config,
            cache_root=self.app_state.config.cache_root,
            allow_missing_colmap=(
                (trainer_name == "depth_anything_3" and self.da3_allow_unposed_var.get())
                or (trainer_name == "sharp" and self.sharp_intrinsics_source_var.get().strip().lower() != "colmap")
            ),
            progress_callback=self._report_progress,
            checkpoint_callback=self._handle_checkpoint,
            preview_callback=preview_callback,
        )

    def _report_progress(self, iteration: int, total: int, metric: float) -> None:
        # Throttle UI updates to avoid blocking the Tk loop on every iteration.
        if iteration not in (1, total) and iteration % 100 != 0:
            return

        def _update() -> None:
            self.status_var.set(f"Training {iteration}/{total} loss={metric:.4f}")
            if total > 0:
                self._set_progress(iteration / float(total))
            if self.preview_canvas is not None:
                self.preview_canvas.render_once()

        self.frame.after(0, _update)

    def _handle_checkpoint(self, iteration: int, checkpoint_path: Path) -> None:
        def _update() -> None:
            self.status_var.set(f"Checkpoint {iteration}: {checkpoint_path.name}")
            self.preview_status_var.set(f"Previewing {checkpoint_path.name}")
            if self.preview_canvas is not None:
                # If the checkpoint belongs to a different scene than what's currently shown, clear first.
                try:
                    last_path = self.preview_canvas.last_path
                    if last_path is not None and last_path.parent.parent != checkpoint_path.parent.parent:
                        self.logger.info(
                            "Checkpoint load clearing stale viewer content last=%s new=%s", last_path, checkpoint_path
                        )
                        self.preview_canvas.clear()
                        self._last_preview_path = None
                except Exception:
                    self.logger.debug("Failed to check/clear stale checkpoint before load", exc_info=True)
                self.preview_canvas.start_rendering()
                self.logger.info(
                    "Checkpoint trigger preview load for %s polling=%s toggle=%s tab_active=%s canvas_mapped=%s",
                    checkpoint_path,
                    self._preview_polling,
                    self._preview_toggle.get(),
                    self._tab_active,
                    self.preview_canvas.winfo_ismapped(),
                )
                self._load_preview(checkpoint_path, allow_when_disabled=True)

        self.frame.after(0, _update)

    def _handle_training_success(self, training_result: TrainingOutput) -> None:
        self._working = False
        self._in_memory_preview_active = False
        self._set_controls_enabled(True)
        self.app_state.refresh_scene_status()
        self._update_scene_label()
        self._set_progress(1.0)
        self._set_status(
            f"Training finished. Last checkpoint: {training_result.primary_path}",
            is_error=False,
        )

    def _handle_error(self, exc: Exception) -> None:
        self._working = False
        self._in_memory_preview_active = False
        self._set_controls_enabled(True)
        self._reset_progress()
        self.logger.exception("Training tab operation failed")
        self._set_status(f"Operation failed: {exc}", is_error=True)

    def apply_training_preset(self, preset: Optional[str] = None) -> None:
        if preset:
            self.training_preset_var.set(preset)
        self._apply_training_preset()

    def run_training(self) -> None:
        self._run_training_only()

    def is_working(self) -> bool:
        return self._working

    def _apply_training_preset(self) -> None:
        preset = self.training_preset_var.get()
        self._apply_gsplat_preset(preset)
        self._apply_da3_preset(preset)

    def _apply_gsplat_preset(self, preset: str) -> None:
        if preset == "high":
            iterations = 30_000
            max_points = 2_500_000
            prune_opacity = 0.01
            prune_scale = 0.12
        elif preset == "medium":
            iterations = 12_000
            max_points = 1_000_000
            prune_opacity = 0.015
            prune_scale = 0.15
        else:
            iterations = self.iterations_var.get() or 3000
            max_points = 500_000
            prune_opacity = 0.02
            prune_scale = 0.2

        if preset == "medium":
            self.iterations_var.set(iterations)
        elif preset == "high":
            self.iterations_var.set(iterations)
        else:
            self.iterations_var.set(iterations)

        self.snapshot_var.set(max(1, self.iterations_var.get() // 5))
        self.max_points_var.set(max_points)
        self.densify_max_points_var.set(max_points)
        self.prune_opacity_var.set(prune_opacity)
        self.prune_scale_var.set(prune_scale)

    def _apply_da3_preset(self, preset: str) -> None:
        if preset == "high":
            self.da3_process_res_var.set(720)
            self.da3_process_res_method_var.set("lower_bound_resize")
            self.da3_max_views_var.set(50)
        elif preset == "medium":
            self.da3_process_res_var.set(504)
            self.da3_process_res_method_var.set("lower_bound_resize")
            self.da3_max_views_var.set(20)
        else:
            self.da3_process_res_var.set(504)
            self.da3_process_res_method_var.set("upper_bound_resize")
            self.da3_max_views_var.set(10)
        self.da3_view_stride_var.set(1)

    def _warmup_renderer(self, trigger: str = "manual") -> None:
        """Pre-compile gsplat rasterization in a subprocess to avoid UI blocking."""

        if self._warmup_started:
            return
        self._warmup_started = True
        device = self.device_var.get().strip() or "cuda:0"
        self._set_status(f"Warmup started on {device} (subprocess, trigger={trigger}).")

        def _do_warmup() -> str:
            script = f"""
import torch
import json
from pathlib import Path
import gsplat
from gsplat.rendering import rasterization

device = torch.device("{device}")
means = torch.zeros((1,3), device=device)
scales = torch.zeros((1,3), device=device)
quats = torch.tensor([[1.0,0.0,0.0,0.0]], device=device)
opacities = torch.ones((1,), device=device) * 0.5
colors = torch.zeros((1,1,3), device=device)
viewmats = torch.eye(4, device=device).unsqueeze(0)
Ks = torch.eye(3, device=device).unsqueeze(0)
print("Warmup init: device", device, "cuda_available", torch.cuda.is_available())
_renders, _alphas, info = rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=colors,
    viewmats=viewmats,
    Ks=Ks,
    width=16,
    height=16,
    sh_degree=0,
    render_mode="RGB",
    absgrad=False,
)
print(json.dumps({{"render_time_ms": float(info.get("render_time_ms", 0.0))}}))
"""
            env = os.environ.copy()
            env.setdefault("TORCH_EXTENSIONS_DIR", str(app_root() / "torch_extensions"))
            env.setdefault("PYTHONPATH", env.get("PYTHONPATH", ""))
            env.setdefault("PYTHONNOUSERSITE", "1")
            env.setdefault("PATH", env.get("PATH", ""))
            torch_lib = app_root() / "venv" / "Lib" / "site-packages" / "torch" / "lib"
            if torch_lib.exists():
                env["PATH"] = f"{torch_lib};{env['PATH']}"
            cuda_bin = app_root() / "cuda" / "bin"
            if cuda_bin.exists():
                env["PATH"] = f"{cuda_bin};{env['PATH']}"
            cuda_lib = app_root() / "cuda" / "lib" / "x64"
            if cuda_lib.exists():
                env["PATH"] = f"{cuda_lib};{env['PATH']}"
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
            )
            if result.returncode != 0:
                stdout = result.stdout.strip()
                stderr = result.stderr.strip()
                raise RuntimeError(
                    f"Warmup subprocess failed (code {result.returncode}). "
                    f"stdout: {stdout or '<empty>'} | stderr: {stderr or '<empty>'}"
                )
            return result.stdout.strip()

        def _on_success(output: str) -> None:
            self._set_status(f"Warmup done: {output or 'ok'}")

        def _on_error(exc: Exception) -> None:
            self.logger.exception("Renderer warmup failed")
            self._set_status(f"Warmup failed: {exc}", is_error=True)

        run_in_background(_do_warmup, tk_root=self.frame, on_success=_on_success, on_error=_on_error, thread_name="warmup_renderer")

    def _set_status(self, message: str, *, is_error: bool = False) -> None:
        self.status_var.set(message)
        if self.status_label is not None:
            self.status_label.config(foreground="#a00" if is_error else "#444")

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        # Disable all tracked controls except preview-related ones.
        for widget in self._interactive_controls:
            try:
                widget.configure(state=state)
            except Exception:
                continue
        if enabled and self._preview_toggle.get() and self._tab_active:
            self._toggle_preview_poll(force_on=True)
            self._poll_latest_checkpoint(force=True)

    def _selected_trainer_name(self) -> str:
        name = self.training_method_var.get().strip().lower()
        return name or "gsplat"

    def _apply_trainer_capabilities(self) -> None:
        name = self._selected_trainer_name()
        trainer = self._trainers.get(name)
        if trainer is None:
            self.preview_status_var.set("Unknown training method.")
            self.method_hint_var.set("Select a valid training method.")
            return
        hint = "Live preview available." if trainer.capabilities.live_preview else "Live preview unavailable."
        self.method_hint_var.set(hint)
        self._show_method_sections(name)
        if trainer.capabilities.live_preview:
            if self.preview_canvas is not None and self._tab_active:
                self.preview_canvas.start_rendering()
            self.preview_status_var.set("Viewer idle.")
            return
        self._preview_toggle.set(False)
        self._preview_polling = False
        self._in_memory_preview_active = False
        self.preview_status_var.set("Live preview unavailable for this method.")

    def _show_method_sections(self, method: str) -> None:
        if not hasattr(self, "gsplat_settings_frame") or not hasattr(self, "da3_settings_frame"):
            return
        if not hasattr(self, "sharp_settings_frame"):
            return
        log_label = getattr(self, "log_label", None)
        if method == "gsplat":
            if not self._gsplat_section_visible:
                if log_label is not None:
                    self.gsplat_settings_frame.pack(fill="x", padx=10, pady=(0, 6), before=log_label)
                else:
                    self.gsplat_settings_frame.pack(fill="x", padx=10, pady=(0, 6))
                self._gsplat_section_visible = True
            if self._da3_section_visible:
                self.da3_settings_frame.pack_forget()
                self._da3_section_visible = False
            if self._sharp_section_visible:
                self.sharp_settings_frame.pack_forget()
                self._sharp_section_visible = False
        elif method == "depth_anything_3":
            if self._gsplat_section_visible:
                self.gsplat_settings_frame.pack_forget()
                self._gsplat_section_visible = False
            if not self._da3_section_visible:
                if log_label is not None:
                    self.da3_settings_frame.pack(fill="x", padx=10, pady=(0, 6), before=log_label)
                else:
                    self.da3_settings_frame.pack(fill="x", padx=10, pady=(0, 6))
                self._da3_section_visible = True
            if self._sharp_section_visible:
                self.sharp_settings_frame.pack_forget()
                self._sharp_section_visible = False
        elif method == "sharp":
            if self._gsplat_section_visible:
                self.gsplat_settings_frame.pack_forget()
                self._gsplat_section_visible = False
            if self._da3_section_visible:
                self.da3_settings_frame.pack_forget()
                self._da3_section_visible = False
            if not self._sharp_section_visible:
                if log_label is not None:
                    self.sharp_settings_frame.pack(fill="x", padx=10, pady=(0, 6), before=log_label)
                else:
                    self.sharp_settings_frame.pack(fill="x", padx=10, pady=(0, 6))
                self._sharp_section_visible = True
        else:
            if self._gsplat_section_visible:
                self.gsplat_settings_frame.pack_forget()
                self._gsplat_section_visible = False
            if self._da3_section_visible:
                self.da3_settings_frame.pack_forget()
                self._da3_section_visible = False
            if self._sharp_section_visible:
                self.sharp_settings_frame.pack_forget()
                self._sharp_section_visible = False

    def _build_training_config(self, method: str) -> dict[str, Any] | SplatTrainingConfig:
        if method == "gsplat":
            return SplatTrainingConfig(
                cuda_toolkit_path=self.cuda_path_var.get().strip() or SplatTrainingConfig().cuda_toolkit_path,
                iterations=int(self.iterations_var.get()),
                snapshot_interval=int(self.snapshot_var.get()),
                max_points=int(self.max_points_var.get()),
                export_format=self.export_format_var.get().strip() or "ply",
                device=self.device_var.get().strip() or "cuda:0",
                image_downscale=int(self.image_downscale_var.get()),
                batch_size=int(self.batch_size_var.get()),
                sh_degree=int(self.sh_degree_var.get()),
                sh_degree_interval=int(self.sh_interval_var.get()),
                init_scale=float(self.init_scale_var.get()),
                min_scale=float(self.min_scale_var.get()),
                max_scale=float(self.max_scale_var.get()),
                opacity_bias=float(self.opacity_bias_var.get()),
                random_background=bool(self.random_background_var.get()),
                means_lr=float(self.means_lr_var.get()),
                scales_lr=float(self.scales_lr_var.get()),
                opacities_lr=float(self.opacities_lr_var.get()),
                sh_lr=float(self.sh_lr_var.get()),
                lr_final_scale=float(self.lr_final_scale_var.get()),
                densify_start=int(self.densify_start_var.get()),
                densify_interval=int(self.densify_interval_var.get()),
                densify_opacity_threshold=float(self.densify_opacity_var.get()),
                densify_scale_threshold=float(self.densify_scale_var.get()),
                prune_opacity_threshold=float(self.prune_opacity_var.get()),
                prune_scale_threshold=float(self.prune_scale_var.get()),
                densify_max_points=int(self.densify_max_points_var.get()),
                preview_interval_seconds=float(self.preview_interval_var.get()),
                preview_min_iters=int(self.preview_min_iters_var.get()),
                max_preview_points=int(self.preview_max_points_var.get()),
            )
        if method == "depth_anything_3":
            pretrained_id = self.da3_pretrained_id_var.get().strip()
            if not pretrained_id:
                raise ValueError("Depth Anything 3 requires a pretrained model id or local directory.")
            return {
                "pretrained_id": pretrained_id,
                "device": self.device_var.get().strip() or "cuda",
                "use_input_resolution": bool(self.da3_use_input_res_var.get()),
                "process_res": int(self.da3_process_res_var.get()),
                "process_res_method": self.da3_process_res_method_var.get().strip() or "upper_bound_resize",
                "align_to_input_ext_scale": bool(self.da3_align_scale_var.get()),
                "infer_gs": bool(self.da3_infer_gs_var.get()),
                "use_ray_pose": bool(self.da3_use_ray_pose_var.get()),
                "ref_view_strategy": self.da3_ref_view_var.get().strip() or "saddle_balanced",
                "gs_views_interval": int(self.da3_gs_views_interval_var.get()),
                "view_stride": int(self.da3_view_stride_var.get()),
                "max_views": int(self.da3_max_views_var.get()),
            }
        if method == "sharp":
            checkpoint_path = self.sharp_checkpoint_path_var.get().strip()
            focal_px_override = float(self.sharp_focal_px_var.get() or 0.0)
            fov_override = float(self.sharp_fov_deg_var.get() or 0.0)
            return {
                "checkpoint_path": checkpoint_path or None,
                "device": self.device_var.get().strip() or "default",
                "intrinsics_source": self.sharp_intrinsics_source_var.get().strip() or "colmap",
                "image_index": int(self.sharp_image_index_var.get() or 0),
                "focal_px_override": focal_px_override if focal_px_override > 0.0 else None,
                "fov_override_deg": fov_override if fov_override > 0.0 else None,
            }
        raise ValueError(f"Unsupported training method: {method}")

    def _has_sfm_outputs(self, scene_id: str) -> bool:
        paths = self.app_state.scene_manager.get(scene_id).paths
        candidates = [
            paths.sfm_dir / "sparse" / "text" / "images.txt",
            paths.sfm_dir / "sparse" / "0" / "images.txt",
            paths.sfm_dir / "sparse" / "images.txt",
        ]
        return any(path.exists() for path in candidates)

    def _attach_log_handler(self) -> None:
        if self.log_view is None:
            return
        root_logger = logging.getLogger("nullsplats")
        if self._log_handler is not None:
            return

        class TkLogHandler(logging.Handler):
            def __init__(self, widget: scrolledtext.ScrolledText) -> None:
                super().__init__()
                self.widget = widget
                self._queue: "queue.SimpleQueue[str]" = queue.SimpleQueue()
                # Start a flush loop on the Tk thread; emit only enqueues.
                try:
                    self.widget.after(50, self._flush)
                except Exception:  # noqa: BLE001
                    pass

            def emit(self, record: logging.LogRecord) -> None:
                try:
                    msg = self.format(record)
                    self._queue.put_nowait(msg)
                except Exception:  # noqa: BLE001
                    return

            def _flush(self) -> None:
                if not self.widget.winfo_exists():
                    return
                try:
                    self.widget.configure(state="normal")
                    while not self._queue.empty():
                        try:
                            msg = self._queue.get_nowait()
                        except Exception:  # noqa: BLE001
                            break
                        self.widget.insert("end", msg + "\n")
                    self.widget.see("end")
                    self.widget.configure(state="disabled")
                except Exception:  # noqa: BLE001
                    pass
                try:
                    self.widget.after(50, self._flush)
                except Exception:  # noqa: BLE001
                    pass

        handler = TkLogHandler(self.log_view)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        root_logger.addHandler(handler)
        self._log_handler = handler
