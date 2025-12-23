"""Training tab UI for NullSplats."""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
import os
import queue
import shutil
import subprocess
import sys
from tkinter import scrolledtext, ttk
from typing import Optional, Tuple
from nullsplats.app_state import AppState
from nullsplats.backend.sfm_pipeline import SfmConfig, SfmResult, run_sfm
from nullsplats.backend.splat_train import PreviewPayload, SplatTrainingConfig, TrainingResult, train_scene
from nullsplats.util.logging import get_logger
from nullsplats.util.tooling_paths import app_root, default_colmap_path, default_cuda_path
from nullsplats.util.threading import run_in_background
from nullsplats.ui.gl_canvas import GLCanvas
from nullsplats.ui.tab_training_layout import TrainingTabLayoutMixin
from nullsplats.ui.tab_training_preview import TrainingTabPreviewMixin


class TrainingTab(TrainingTabLayoutMixin, TrainingTabPreviewMixin):
    """Training tab with SfM and training controls."""

    def __init__(self, master: tk.Misc, app_state: AppState) -> None:
        self.app_state = app_state
        self.logger = get_logger("ui.training")
        self.logger.setLevel(logging.DEBUG)
        self.frame = ttk.Frame(master)

        default_cfg = SplatTrainingConfig()
        self.status_var = tk.StringVar(value="Configure SfM and training, then run.")
        self.preview_status_var = tk.StringVar(value="Viewer idle.")
        self.colmap_path_var = tk.StringVar(value=default_colmap_path())
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
        self.training_preset_var = tk.StringVar(value="low")
        self._warmup_started = False
        self._interactive_controls: list[tk.Widget] = []
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar: Optional[ttk.Progressbar] = None
        self.force_sfm_var = tk.BooleanVar(value=False)

        self._build_contents()
        self._apply_training_preset()  # default to medium preset settings
        self._update_scene_label()
        self._schedule_preview_drain()

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
                f"Inputs {'✓' if status.has_inputs else '–'}",
                f"SfM {'✓' if status.has_sfm else '–'}",
                f"Splats {'✓' if status.has_splats else '–'}",
            ]
            return " • ".join(parts)
        except Exception:  # noqa: BLE001
            return "Scene status unavailable."

    def on_tab_selected(self, selected: bool) -> None:
        self._tab_active = selected
        if not self.preview_canvas:
            return
        if selected:
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

    def _run_pipeline(self) -> None:
        if self._working:
            self._set_status("Another operation is running; wait for it to finish.", is_error=True)
            return
        self._apply_training_preset()
        scene_id = self._require_scene()
        if scene_id is None:
            return
        if self.force_sfm_var.get():
            if not self._clear_outputs(scene_id):
                return
        else:
            self._last_preview_path = None
            if self.preview_canvas is not None:
                try:
                    self.preview_canvas.clear()
                except Exception:
                    pass
        sfm_config = SfmConfig(
            colmap_path=self.colmap_path_var.get().strip() or "colmap",
        )
        train_config = SplatTrainingConfig(
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
        self._pause_preview_for_sfm()
        self._in_memory_preview_active = True
        self._set_status("Running COLMAP then training...")
        self._reset_progress(indeterminate=True)
        self._working = True
        self._set_controls_enabled(False)

        run_in_background(
            self._execute_pipeline,
            scene_id,
            sfm_config,
            train_config,
            tk_root=self.frame.winfo_toplevel(),
            on_success=self._handle_pipeline_success,
            on_error=self._handle_error,
            thread_name=f"sfm_train_{scene_id}",
        )

    def _run_sfm_only(self) -> None:
        if self._working:
            self._set_status("Another operation is running; wait for it to finish.", is_error=True)
            return
        scene_id = self._require_scene()
        if scene_id is None:
            return
        if self.force_sfm_var.get():
            if not self._clear_outputs(scene_id):
                return
        sfm_config = SfmConfig(
            colmap_path=self.colmap_path_var.get().strip() or "colmap",
        )
        self._pause_preview_for_sfm()
        self._working = True
        self._set_status("Running COLMAP...")
        self._reset_progress(indeterminate=True)
        self._set_controls_enabled(False)
        run_in_background(
            self._execute_sfm,
            scene_id,
            sfm_config,
            tk_root=self.frame.winfo_toplevel(),
            on_success=self._handle_sfm_success,
            on_error=self._handle_error,
            thread_name=f"sfm_only_{scene_id}",
        )

    def _run_training_only(self) -> None:
        if self._working:
            self._set_status("Another operation is running; wait for it to finish.", is_error=True)
            return
        self._apply_training_preset()
        scene_id = self._require_scene()
        if scene_id is None:
            return
        if not self._has_sfm_outputs(scene_id):
            self._set_status("COLMAP outputs not found. Run COLMAP first.", is_error=True)
            return
        train_config = SplatTrainingConfig(
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
        if self.preview_canvas is not None:
            self.preview_canvas.start_rendering()
        self._preview_toggle.set(True)
        self._preview_polling = False
        self._in_memory_preview_active = True
        self._set_status("Training only...")
        self._reset_progress()
        self._working = True
        self._set_controls_enabled(False)
        run_in_background(
            self._execute_training,
            scene_id,
            train_config,
            tk_root=self.frame.winfo_toplevel(),
            on_success=lambda result: self._handle_pipeline_success((None, result)),
            on_error=self._handle_error,
            thread_name=f"train_only_{scene_id}",
        )

    def _execute_pipeline(
        self, scene_id: str, sfm_config: SfmConfig, train_config: SplatTrainingConfig
    ) -> Tuple[SfmResult, TrainingResult]:
        sfm_result = run_sfm(
            scene_id,
            config=sfm_config,
            cache_root=self.app_state.config.cache_root,
        )
        self.frame.after(0, self._resume_preview_after_sfm)
        training_result = train_scene(
            scene_id,
            train_config,
            cache_root=self.app_state.config.cache_root,
            progress_callback=self._report_progress,
            checkpoint_callback=self._handle_checkpoint,
            preview_callback=self._handle_preview_payload,
        )
        return sfm_result, training_result

    def _execute_sfm(self, scene_id: str, sfm_config: SfmConfig) -> SfmResult:
        result = run_sfm(scene_id, config=sfm_config, cache_root=self.app_state.config.cache_root)
        self.frame.after(0, self._resume_preview_after_sfm)
        return result

    def _execute_training(self, scene_id: str, train_config: SplatTrainingConfig) -> TrainingResult:
        return train_scene(
            scene_id,
            train_config,
            cache_root=self.app_state.config.cache_root,
            progress_callback=self._report_progress,
            checkpoint_callback=self._handle_checkpoint,
            preview_callback=self._handle_preview_payload,
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

    def _handle_pipeline_success(self, result: Tuple[SfmResult | None, TrainingResult]) -> None:
        _sfm_result, training_result = result
        self._working = False
        self._in_memory_preview_active = False
        self._set_controls_enabled(True)
        self.app_state.refresh_scene_status()
        self._update_scene_label()
        self._set_progress(1.0)
        self._set_status(
            f"SfM + training finished. Last checkpoint: {training_result.last_checkpoint}",
            is_error=False,
        )

    def _handle_sfm_success(self, sfm_result: SfmResult) -> None:
        self._working = False
        self._in_memory_preview_active = False
        self._set_controls_enabled(True)
        self.app_state.refresh_scene_status()
        self._update_scene_label()
        self._set_progress(1.0)
        self._set_status(f"SfM finished. Output: {sfm_result.converted_model_path}", is_error=False)

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

    def run_pipeline(self) -> None:
        self._run_pipeline()

    def is_working(self) -> bool:
        return self._working

    def _apply_training_preset(self) -> None:
        preset = self.training_preset_var.get()
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

    def _has_sfm_outputs(self, scene_id: str) -> bool:
        paths = self.app_state.scene_manager.get(scene_id).paths
        candidates = [
            paths.sfm_dir / "sparse" / "text" / "images.txt",
            paths.sfm_dir / "sparse" / "0" / "images.txt",
            paths.sfm_dir / "sparse" / "images.txt",
        ]
        return any(path.exists() for path in candidates)

    def _clear_outputs(self, scene_id: str) -> bool:
        try:
            paths = self.app_state.scene_manager.get(scene_id).paths
            if paths.outputs_root.exists():
                shutil.rmtree(paths.outputs_root)
            self.logger.info("Cleared outputs for scene=%s at %s", scene_id, paths.outputs_root)
            self._last_preview_path = None
            if self.preview_canvas is not None:
                self.preview_canvas.clear()
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to clear outputs for scene %s", scene_id)
            self._set_status(f"Failed to clear outputs: {exc}", is_error=True)
            return False

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
