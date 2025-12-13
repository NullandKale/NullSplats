"""Training tab UI for NullSplats."""

from __future__ import annotations

import logging
import math
import tkinter as tk
from pathlib import Path
import os
import queue
import shutil
import subprocess
import sys
import time
from tkinter import scrolledtext, ttk
from typing import Optional, Tuple, List

import numpy as np
from nullsplats.app_state import AppState
from nullsplats.backend.sfm_pipeline import SfmConfig, SfmResult, run_sfm
from nullsplats.backend.splat_train import SplatTrainingConfig, TrainingResult, train_scene
from nullsplats.util.logging import get_logger
from nullsplats.util.threading import run_in_background
from nullsplats.ui.gl_canvas import GLCanvas
from nullsplats.ui.advanced_render_controls import AdvancedRenderSettingsPanel
from nullsplats.ui.render_controls import RenderSettingsPanel
from nullsplats.ui.colmap_camera_panel import ColmapCameraPanel


def _default_binary_path(tool: str) -> str:
    """Return a repo-local binary path if present, otherwise the bare tool name."""
    repo_root = Path(__file__).resolve().parents[2]
    if tool == "colmap":
        return str(repo_root / "tools" / "colmap" / "COLMAP.bat")
    return tool


def _default_cuda_path() -> str:
    """Return preferred CUDA toolkit path if present."""
    preferred = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
    if preferred.exists():
        return str(preferred)
    env_home = Path(os.environ.get("CUDA_HOME", "")) if "CUDA_HOME" in os.environ else None
    return str(env_home) if env_home and env_home.exists() else ""


class TrainingTab:
    """Training tab with SfM and training controls."""

    def __init__(self, master: tk.Misc, app_state: AppState) -> None:
        self.app_state = app_state
        self.logger = get_logger("ui.training")
        self.logger.setLevel(logging.DEBUG)
        self.frame = ttk.Frame(master)

        default_cfg = SplatTrainingConfig()
        self.status_var = tk.StringVar(value="Configure SfM and training, then run.")
        self.preview_status_var = tk.StringVar(value="Viewer idle.")
        self.colmap_path_var = tk.StringVar(value=_default_binary_path("colmap"))
        self.cuda_path_var = tk.StringVar(value=_default_cuda_path())
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
        self._tab_active = False
        self.training_preset_var = tk.StringVar(value="low")
        self._warmup_started = False
        self._interactive_controls: list[tk.Widget] = []
        self._autoplay_job: Optional[str] = None
        self._last_user_interaction: float = 0.0
        self._setting_autoplay_pose = False
        self._pose_list: List[tuple[np.ndarray, np.ndarray]] = []
        self._pose_idx: int = 0
        self._autoplay_enabled = tk.BooleanVar(value=True)
        self._autoplay_interval_var = tk.DoubleVar(value=2.0)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar: Optional[ttk.Progressbar] = None
        self.force_sfm_var = tk.BooleanVar(value=False)

        self._build_contents()
        self._apply_training_preset()  # default to medium preset settings
        self._update_scene_label()

    def _register_control(self, widget: tk.Widget) -> None:
        """Track interactive widgets so we can disable/enable during training."""
        self._interactive_controls.append(widget)
        widget.bind(
            "<Destroy>",
            lambda e: self._interactive_controls.remove(widget)
            if widget in self._interactive_controls
            else None,
        )

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

    def _build_contents(self) -> None:
        paned = ttk.Panedwindow(self.frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        right_col = ttk.Frame(paned)
        left_col = ttk.Frame(paned, width=420)
        paned.add(right_col, weight=3)
        paned.add(left_col, weight=2)

        # --- Right: viewer-first layout ---
        preview_shell = ttk.Frame(right_col)
        preview_shell.pack(fill="both", expand=True, padx=10, pady=(10, 8))

        header = ttk.Frame(preview_shell)
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(header, text="Live preview (latest checkpoint)", font=("Segoe UI", 11, "bold")).pack(side="left")
        ttk.Label(header, textvariable=self.preview_status_var, foreground="#444").pack(side="left", padx=(8, 0))
        ttk.Button(
            header, text="Refresh now", command=self._refresh_preview_now
        ).pack(side="right")
        ttk.Checkbutton(
            header,
            text="Enable live preview polling",
            variable=self._preview_toggle,
            command=self._toggle_preview_poll,
        ).pack(side="right", padx=(0, 8))

        autoplay_row = ttk.Frame(preview_shell)
        autoplay_row.pack(fill="x", pady=(0, 6))
        ttk.Checkbutton(
            autoplay_row,
            text="Auto camera rotation (uses COLMAP poses when available)",
            variable=self._autoplay_enabled,
            command=self._update_autoplay_setting,
        ).pack(side="left")
        ttk.Label(autoplay_row, text="Every (s):").pack(side="left", padx=(8, 2))
        interval_spin = ttk.Spinbox(
            autoplay_row,
            from_=0.5,
            to=30.0,
            increment=0.5,
            textvariable=self._autoplay_interval_var,
            width=6,
            command=self._update_autoplay_setting,
        )
        interval_spin.pack(side="left")
        self._register_control(interval_spin)
        ttk.Button(autoplay_row, text="Rotate now", command=lambda: self._autoplay_tick(manual=True)).pack(
            side="right"
        )

        preview_frame = ttk.LabelFrame(preview_shell, text="Viewer")
        preview_frame.pack(fill="both", expand=True)
        preview_inner = ttk.Frame(preview_frame)
        preview_inner.pack(fill="both", expand=True, padx=6, pady=(6, 4))
        self.preview_canvas = GLCanvas(preview_inner, device=self.device_var.get(), width=1200, height=720)
        self.preview_canvas.pack(fill="both", expand=True)
        try:
            self.preview_canvas.add_camera_listener(self._on_camera_move)
        except Exception:
            pass

        notebook = ttk.Notebook(preview_frame)
        notebook.pack(fill="x", padx=6, pady=(0, 6))
        # Cameras first for faster pose tweaks
        camera_tab = ttk.Frame(notebook)
        notebook.add(camera_tab, text="Cameras")
        self.colmap_panel = ColmapCameraPanel(
            camera_tab,
            viewer_getter=lambda: self.preview_canvas,
            scene_getter=lambda: self.app_state.current_scene_id,
            paths_getter=lambda scene: self.app_state.scene_manager.get(scene).paths,
        )
        self.colmap_panel.pack(fill="both", expand=True, padx=4, pady=4)
        render_tab = ttk.Frame(notebook)
        notebook.add(render_tab, text="Render settings")
        render_wrap = ttk.Frame(render_tab)
        render_wrap.pack(fill="both", expand=True, padx=4, pady=4)
        ttk.Label(render_wrap, text="Viewport", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.preview_controls = RenderSettingsPanel(render_wrap, lambda: self.preview_canvas)
        self.preview_controls.pack(fill="x", padx=(0, 0), pady=(2, 6))
        ttk.Label(render_wrap, text="Advanced", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(4, 0))
        self.preview_advanced_controls = AdvancedRenderSettingsPanel(render_wrap, lambda: self.preview_canvas)
        self.preview_advanced_controls.pack(fill="x", padx=(0, 0), pady=(2, 0))

        # --- Left: workflow-focused, minimal primary controls ---
        ttk.Label(left_col, text="Training workflow", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=10, pady=(10, 6)
        )

        scene_card = ttk.LabelFrame(left_col, text="Scene context")
        scene_card.pack(fill="x", padx=10, pady=(0, 6))
        self.scene_label = ttk.Label(scene_card, text=self._scene_text(), anchor="w", justify="left", font=("Segoe UI", 10, "bold"))
        self.scene_label.pack(fill="x", padx=6, pady=(6, 2))
        self.scene_status_label = ttk.Label(
            scene_card,
            text=self._scene_status_text(),
            anchor="w",
            justify="left",
            foreground="#444",
        )
        self.scene_status_label.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(
            scene_card,
            text="Set scenes in Inputs tab; selecting a scene there auto-loads cached checkpoints for preview.",
            foreground="#666",
            wraplength=380,
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=6, pady=(0, 6))

        status_card = ttk.LabelFrame(left_col, text="Status & run control")
        status_card.pack(fill="x", padx=10, pady=(0, 6))
        self.status_label = ttk.Label(
            status_card, textvariable=self.status_var, foreground="#333", font=("Segoe UI", 11, "bold"), wraplength=380
        )
        self.status_label.pack(fill="x", padx=6, pady=(6, 4))
        self.progress_bar = ttk.Progressbar(status_card, variable=self.progress_var, mode="determinate", maximum=1.0)
        self.progress_bar.pack(fill="x", padx=6, pady=(0, 6))
        primary_row = ttk.Frame(status_card)
        primary_row.pack(fill="x", padx=6, pady=(0, 6))
        btn_run = ttk.Button(primary_row, text="Run COLMAP + Train", command=self._run_pipeline)
        btn_run.pack(side="left")
        self._register_control(btn_run)
        btn_train_only = ttk.Button(primary_row, text="Train only", command=self._run_training_only)
        btn_train_only.pack(side="left", padx=(6, 0))
        self._register_control(btn_train_only)
        btn_sfm_only = ttk.Button(primary_row, text="Run COLMAP only", command=self._run_sfm_only)
        btn_sfm_only.pack(side="left", padx=(6, 0))
        self._register_control(btn_sfm_only)
        btn_warm = ttk.Button(primary_row, text="Warm up renderer", command=self._warmup_renderer)
        btn_warm.pack(side="right")
        self._register_control(btn_warm)

        primary_cfg = ttk.LabelFrame(left_col, text="Primary settings")
        primary_cfg.pack(fill="x", padx=10, pady=(0, 6))
        preset_row = ttk.Frame(primary_cfg)
        preset_row.pack(fill="x", padx=6, pady=(6, 4))
        ttk.Label(preset_row, text="Preset:").pack(side="left")
        preset_combo = ttk.Combobox(
            preset_row,
            textvariable=self.training_preset_var,
            values=["low", "medium", "high"],
            state="readonly",
            width=10,
        )
        preset_combo.pack(side="left", padx=(4, 0))
        preset_combo.bind("<<ComboboxSelected>>", lambda _: self._apply_training_preset())
        self._register_control(preset_combo)
        btn_apply = ttk.Button(preset_row, text="Apply", command=self._apply_training_preset)
        btn_apply.pack(side="left", padx=(6, 0))
        self._register_control(btn_apply)

        core_row1 = ttk.Frame(primary_cfg)
        core_row1.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(core_row1, text="Device:").pack(side="left")
        ttk.Entry(core_row1, textvariable=self.device_var, width=14).pack(side="left", padx=(4, 12))
        ttk.Label(core_row1, text="Iterations:").pack(side="left")
        ttk.Spinbox(core_row1, from_=1, to=1_000_000, textvariable=self.iterations_var, width=10).pack(side="left", padx=(4, 12))
        ttk.Label(core_row1, text="Snapshot interval:").pack(side="left")
        ttk.Spinbox(core_row1, from_=1, to=1_000_000, textvariable=self.snapshot_var, width=9).pack(side="left", padx=(4, 0))

        core_row2 = ttk.Frame(primary_cfg)
        core_row2.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(core_row2, text="Batch size:").pack(side="left")
        ttk.Spinbox(core_row2, from_=1, to=16, textvariable=self.batch_size_var, width=6).pack(side="left", padx=(4, 12))
        ttk.Label(core_row2, text="Export format:").pack(side="left")
        ttk.Combobox(core_row2, textvariable=self.export_format_var, values=("ply", "splat"), width=8).pack(side="left", padx=(4, 12))
        ttk.Checkbutton(core_row2, text="Re-run COLMAP from scratch", variable=self.force_sfm_var).pack(side="left")

        # Advanced controls tucked away
        sfm_body = self._collapsible_section(left_col, "Structure-from-Motion (COLMAP)", start_hidden=True)
        path_row = ttk.Frame(sfm_body)
        path_row.pack(fill="x", padx=6, pady=(6, 4))
        ttk.Label(path_row, text="COLMAP path:").grid(row=0, column=0, sticky="w")
        ttk.Entry(path_row, textvariable=self.colmap_path_var, width=32).grid(row=0, column=1, sticky="ew", padx=(4, 8))
        ttk.Label(path_row, text="CUDA toolkit path:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(path_row, textvariable=self.cuda_path_var, width=32).grid(
            row=1, column=1, sticky="ew", padx=(4, 8), pady=(4, 0)
        )
        path_row.columnconfigure(1, weight=1)

        training_body = self._collapsible_section(left_col, "Training config (advanced)", start_hidden=True)
        row1 = ttk.Frame(training_body)
        row1.pack(fill="x", padx=6, pady=(6, 4))
        ttk.Label(row1, text="Max points (0=all):").pack(side="left")
        ttk.Spinbox(row1, from_=0, to=2_000_000, textvariable=self.max_points_var, width=12).pack(side="left", padx=(4, 12))
        ttk.Label(row1, text="Image downscale:").pack(side="left")
        ttk.Spinbox(row1, from_=1, to=16, textvariable=self.image_downscale_var, width=6).pack(side="left", padx=(4, 0))

        row3 = ttk.Frame(training_body)
        row3.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(row3, text="SH degree / interval:").pack(side="left")
        sh_frame = ttk.Frame(row3)
        sh_frame.pack(side="left")
        ttk.Spinbox(sh_frame, from_=0, to=5, textvariable=self.sh_degree_var, width=4).pack(side="left")
        ttk.Label(sh_frame, text="/").pack(side="left", padx=2)
        ttk.Spinbox(sh_frame, from_=1, to=100000, textvariable=self.sh_interval_var, width=8).pack(side="left")

        opt_body = self._collapsible_section(left_col, "Optimization", start_hidden=True)
        opt_row1 = ttk.Frame(opt_body)
        opt_row1.pack(fill="x", padx=6, pady=(6, 4))
        ttk.Label(opt_row1, text="Init scale:").pack(side="left")
        ttk.Spinbox(opt_row1, from_=0.01, to=5.0, increment=0.01, textvariable=self.init_scale_var, width=8).pack(
            side="left", padx=(4, 12)
        )
        ttk.Label(opt_row1, text="Scale min/max:").pack(side="left")
        scale_frame = ttk.Frame(opt_row1)
        scale_frame.pack(side="left")
        ttk.Spinbox(scale_frame, from_=1e-5, to=1.0, increment=1e-5, textvariable=self.min_scale_var, width=8).pack(
            side="left"
        )
        ttk.Label(scale_frame, text="to").pack(side="left", padx=2)
        ttk.Spinbox(scale_frame, from_=1e-4, to=1.0, increment=1e-4, textvariable=self.max_scale_var, width=8).pack(
            side="left"
        )

        opt_row2 = ttk.Frame(opt_body)
        opt_row2.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(opt_row2, text="Opacity bias:").pack(side="left")
        ttk.Spinbox(opt_row2, from_=0.01, to=0.99, increment=0.01, textvariable=self.opacity_bias_var, width=8).pack(
            side="left", padx=(4, 12)
        )
        ttk.Checkbutton(opt_row2, text="Random background", variable=self.random_background_var).pack(side="left")

        opt_row3 = ttk.Frame(opt_body)
        opt_row3.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(opt_row3, text="LR (means / scales / opacities / SH):").pack(side="left")
        lr_frame = ttk.Frame(opt_row3)
        lr_frame.pack(side="left", padx=(4, 0))
        ttk.Entry(lr_frame, textvariable=self.means_lr_var, width=8).pack(side="left", padx=(0, 4))
        ttk.Entry(lr_frame, textvariable=self.scales_lr_var, width=8).pack(side="left", padx=(0, 4))
        ttk.Entry(lr_frame, textvariable=self.opacities_lr_var, width=8).pack(side="left", padx=(0, 4))
        ttk.Entry(lr_frame, textvariable=self.sh_lr_var, width=8).pack(side="left")

        opt_row4 = ttk.Frame(opt_body)
        opt_row4.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(opt_row4, text="Means LR final scale:").pack(side="left")
        ttk.Spinbox(
            opt_row4,
            from_=0.0001,
            to=1.0,
            increment=0.0001,
            textvariable=self.lr_final_scale_var,
            width=8,
        ).pack(side="left", padx=(4, 0))

        densify_body = self._collapsible_section(left_col, "Densify / prune", start_hidden=True)
        densify_row1 = ttk.Frame(densify_body)
        densify_row1.pack(fill="x", padx=6, pady=(6, 4))
        ttk.Label(densify_row1, text="Start / interval (iters):").pack(side="left")
        dt_frame = ttk.Frame(densify_row1)
        dt_frame.pack(side="left", padx=(4, 12))
        ttk.Spinbox(dt_frame, from_=0, to=100000, textvariable=self.densify_start_var, width=8).pack(side="left")
        ttk.Label(dt_frame, text="/").pack(side="left", padx=2)
        ttk.Spinbox(dt_frame, from_=1, to=100000, textvariable=self.densify_interval_var, width=6).pack(side="left")
        ttk.Label(densify_row1, text="Densify thresholds (opacity / scale):").pack(side="left")
        thresh_frame = ttk.Frame(densify_row1)
        thresh_frame.pack(side="left", padx=(4, 0))
        ttk.Entry(thresh_frame, textvariable=self.densify_opacity_var, width=8).pack(side="left", padx=(0, 4))
        ttk.Entry(thresh_frame, textvariable=self.densify_scale_var, width=8).pack(side="left")

        densify_row2 = ttk.Frame(densify_body)
        densify_row2.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(densify_row2, text="Prune thresholds (opacity / scale):").pack(side="left")
        prune_frame = ttk.Frame(densify_row2)
        prune_frame.pack(side="left", padx=(4, 12))
        ttk.Entry(prune_frame, textvariable=self.prune_opacity_var, width=8).pack(side="left", padx=(0, 4))
        ttk.Entry(prune_frame, textvariable=self.prune_scale_var, width=8).pack(side="left")
        ttk.Label(densify_row2, text="Max points during densify (0=cap by output):").pack(side="left")
        ttk.Spinbox(densify_row2, from_=0, to=5_000_000, textvariable=self.densify_max_points_var, width=12).pack(
            side="left", padx=(4, 0)
        )

        ttk.Label(
            left_col, text="Live logs stream below; full logs live in logs/app.log.", anchor="w", justify="left"
        ).pack(anchor="w", padx=10, pady=(4, 4))

        log_frame = ttk.LabelFrame(left_col, text="Live logs")
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_view = scrolledtext.ScrolledText(log_frame, wrap="none", height=12, width=80)
        self.log_view.pack(fill="both", expand=True, padx=6, pady=6)
        self.log_view.configure(state="disabled")
        self._attach_log_handler()

    def _collapsible_section(self, parent: tk.Misc, title: str, *, start_hidden: bool = False) -> ttk.Frame:
        """Create a section with a show/hide toggle to reduce clutter."""
        shell = ttk.Frame(parent)
        shell.pack(fill="x", padx=10, pady=(0, 8))
        header = ttk.Frame(shell)
        header.pack(fill="x")
        ttk.Label(header, text=title, font=("Segoe UI", 10, "bold")).pack(side="left")
        body = ttk.Frame(shell)
        if start_hidden:
            body.pack_forget()
        else:
            body.pack(fill="x", pady=(4, 0))

        def _toggle() -> None:
            if body.winfo_manager():
                body.pack_forget()
                toggle_btn.config(text="Show")
            else:
                body.pack(fill="x", pady=(4, 0))
                toggle_btn.config(text="Hide")

        toggle_btn = ttk.Button(header, text="Show" if start_hidden else "Hide", width=6, command=_toggle)
        toggle_btn.pack(side="right")
        return body

    def on_tab_selected(self, selected: bool) -> None:
        self._tab_active = selected
        if not self.preview_canvas:
            return
        if selected:
            if not self._warmup_started:
                # Defer renderer warmup until the tab is visible to avoid slowing startup.
                self._warmup_renderer(trigger="tab_select")
            # Resume polling if enabled and ensure preview starts rendering.
            self._clear_stale_preview_for_scene()
            self.preview_canvas.start_rendering()
            self._ensure_preview_running()
            self._schedule_autoplay()
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
            self._refresh_autoplay_poses(str(scene_id))
            self._schedule_autoplay()
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
        self._refresh_autoplay_poses(scene_id)
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
        )
        # Keep preview rendering active during training if enabled.
        if self.preview_canvas is not None:
            self.preview_canvas.start_rendering()
        self._preview_toggle.set(True)
        self._toggle_preview_poll(force_on=True)
        self._schedule_autoplay()
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
        self._refresh_autoplay_poses(scene_id)
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
        )
        if self.preview_canvas is not None:
            self.preview_canvas.start_rendering()
        self._preview_toggle.set(True)
        self._toggle_preview_poll(force_on=True)
        self._refresh_autoplay_poses(scene_id)
        self._schedule_autoplay()
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
        training_result = train_scene(
            scene_id,
            train_config,
            cache_root=self.app_state.config.cache_root,
            progress_callback=self._report_progress,
            checkpoint_callback=self._handle_checkpoint,
        )
        return sfm_result, training_result

    def _execute_sfm(self, scene_id: str, sfm_config: SfmConfig) -> SfmResult:
        return run_sfm(scene_id, config=sfm_config, cache_root=self.app_state.config.cache_root)

    def _execute_training(self, scene_id: str, train_config: SplatTrainingConfig) -> TrainingResult:
        return train_scene(
            scene_id,
            train_config,
            cache_root=self.app_state.config.cache_root,
            progress_callback=self._report_progress,
            checkpoint_callback=self._handle_checkpoint,
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
            self._schedule_autoplay()

        self.frame.after(0, _update)

    def _handle_pipeline_success(self, result: Tuple[SfmResult | None, TrainingResult]) -> None:
        _sfm_result, training_result = result
        self._working = False
        self._set_controls_enabled(True)
        self.app_state.refresh_scene_status()
        self._update_scene_label()
        self._set_progress(1.0)
        self._set_status(
            f"SfM + training finished. Last checkpoint: {training_result.last_checkpoint}",
            is_error=False,
        )
        self._schedule_autoplay()

    def _handle_sfm_success(self, sfm_result: SfmResult) -> None:
        self._working = False
        self._set_controls_enabled(True)
        self.app_state.refresh_scene_status()
        self._update_scene_label()
        self._set_progress(1.0)
        self._set_status(f"SfM finished. Output: {sfm_result.converted_model_path}", is_error=False)
        self._schedule_autoplay()

    def _handle_error(self, exc: Exception) -> None:
        self._working = False
        self._set_controls_enabled(True)
        self._reset_progress()
        self.logger.exception("Training tab operation failed")
        self._set_status(f"Operation failed: {exc}", is_error=True)

    def _apply_training_preset(self) -> None:
        preset = self.training_preset_var.get()
        if preset == "medium":
            self.iterations_var.set(30_000)
            self.snapshot_var.set(max(1, self.iterations_var.get() // 5))
        elif preset == "high":
            self.iterations_var.set(30_000)
            self.snapshot_var.set(max(1, self.iterations_var.get() // 5))
        else:
            # Low: keep existing iterations/downscale, ensure snapshots ~5 per run
            if self.iterations_var.get() > 0:
                self.snapshot_var.set(max(1, self.iterations_var.get() // 5))

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
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()

        def _on_success(output: str) -> None:
            self._set_status(f"Warmup done: {output or 'ok'}")

        def _on_error(exc: Exception) -> None:
            self.logger.exception("Renderer warmup failed")
            self._set_status(f"Warmup failed: {exc}", is_error=True)

        run_in_background(_do_warmup, tk_root=self.frame, on_success=_on_success, on_error=_on_error, thread_name="warmup_renderer")

    def _toggle_preview_poll(self, force_on: bool = False) -> None:
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
        self._last_preview_path = None
        # Ensure polling is active for this manual refresh and force a fresh load.
        self._toggle_preview_poll(force_on=True)
        self._poll_latest_checkpoint(force=True)

    def _schedule_preview_poll(self) -> None:
        if not self._preview_polling:
            return
        self.frame.after(3000, self._poll_latest_checkpoint)

    def _poll_latest_checkpoint(self, force: bool = False) -> None:
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

    def _ensure_preview_running(self) -> None:
        if not self.preview_canvas:
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
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root_logger.addHandler(handler)
        self._log_handler = handler

    def _refresh_autoplay_poses(self, scene_id: str) -> None:
        try:
            paths = self.app_state.scene_manager.get(scene_id).paths
            images_file = self._find_images_file(paths.sfm_dir)
            if images_file is None:
                self._pose_list = []
                return
            self._pose_list = self._parse_images_file(images_file)
            self._pose_idx = 0
        except Exception:  # noqa: BLE001
            self.logger.debug("Failed to refresh autoplay poses", exc_info=True)
            self._pose_list = []

    def _find_images_file(self, sfm_dir: Path) -> Optional[Path]:
        candidates = [
            sfm_dir / "sparse" / "text" / "images.txt",
            sfm_dir / "sparse" / "0" / "images.txt",
            sfm_dir / "sparse" / "images.txt",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _parse_images_file(self, images_file: Path) -> List[tuple[np.ndarray, np.ndarray]]:
        poses: List[tuple[np.ndarray, np.ndarray]] = []
        with images_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                try:
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                except ValueError:
                    continue
                rot_w2c = _quat_to_rotation_matrix_np(np.array([qw, qx, qy, qz], dtype=np.float64))
                translation = np.array([tx, ty, tz], dtype=np.float64)
                position = (-rot_w2c.T @ translation).astype(np.float32)
                poses.append((position, rot_w2c.astype(np.float32)))
        return poses

    def _autoplay_interval_ms(self) -> int:
        try:
            return max(500, int(float(self._autoplay_interval_var.get()) * 1000))
        except Exception:
            return 2000

    def _update_autoplay_setting(self) -> None:
        if self._autoplay_enabled.get():
            self._schedule_autoplay()
        else:
            if self._autoplay_job is not None:
                try:
                    self.frame.after_cancel(self._autoplay_job)
                except Exception:
                    pass
                self._autoplay_job = None

    def _schedule_autoplay(self, delay_ms: Optional[int] = None) -> None:
        if self._autoplay_job is not None:
            try:
                self.frame.after_cancel(self._autoplay_job)
            except Exception:
                pass
            self._autoplay_job = None
        if not self._tab_active or self.preview_canvas is None or not self._autoplay_enabled.get():
            return
        wait_ms = delay_ms if delay_ms is not None else self._autoplay_interval_ms()
        self._autoplay_job = self.frame.after(wait_ms, self._autoplay_tick)

    def _autoplay_tick(self, *, manual: bool = False) -> None:
        self._autoplay_job = None
        if not self._tab_active or self.preview_canvas is None:
            return
        if not manual and not self._autoplay_enabled.get():
            return
        if not manual and (time.monotonic() - self._last_user_interaction) < 5.0:
            self._schedule_autoplay(1000)
            return
        if not self._pose_list:
            scene_id = self.app_state.current_scene_id
            if scene_id is not None:
                self._refresh_autoplay_poses(scene_id)
            if not self._pose_list:
                self._spin_camera(self.preview_canvas)
                self._schedule_autoplay()
                return
        pos, rot = self._pose_list[self._pose_idx % len(self._pose_list)]
        self._pose_idx += 1
        viewer = self.preview_canvas
        try:
            self._setting_autoplay_pose = True
            viewer.set_camera_pose(pos, rotation=rot)
        finally:
            self._setting_autoplay_pose = False
        viewer.start_rendering()
        viewer.render_once()
        if self._autoplay_enabled.get():
            self._schedule_autoplay()

    def _on_camera_move(self, _view) -> None:
        if self._setting_autoplay_pose:
            return
        self._last_user_interaction = time.monotonic()
        if self._autoplay_job is not None:
            try:
                self.frame.after_cancel(self._autoplay_job)
            except Exception:
                pass
            self._autoplay_job = None

    def _spin_camera(self, viewer: GLCanvas) -> None:
        """Fallback auto-rotation when COLMAP poses are not available."""
        view = viewer.get_current_view()
        if view is None:
            return
        viewer.adjust_camera_angles(yaw=view.yaw + math.radians(8.0))


def _quat_to_rotation_matrix_np(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    qw, qx, qy, qz = quat
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )
