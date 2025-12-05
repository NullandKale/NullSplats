"""Training tab UI for NullSplats."""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
import os
import shutil
from tkinter import scrolledtext, ttk
from typing import Optional, Tuple

from nullsplats.app_state import AppState
from nullsplats.backend.sfm_pipeline import SfmConfig, SfmResult, run_sfm
from nullsplats.backend.io_cache import ScenePaths
from nullsplats.backend.splat_train import SplatTrainingConfig, TrainingResult, train_scene
from nullsplats.util.logging import get_logger
from nullsplats.util.threading import run_in_background


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
        self.frame = ttk.Frame(master)

        default_cfg = SplatTrainingConfig()
        self.status_var = tk.StringVar(value="Configure SfM and training, then run.")
        self.colmap_path_var = tk.StringVar(value=_default_binary_path("colmap"))
        self.cuda_path_var = tk.StringVar(value=_default_cuda_path())
        self.iterations_var = tk.IntVar(value=default_cfg.iterations)
        self.snapshot_var = tk.IntVar(value=default_cfg.snapshot_interval)
        self.max_points_var = tk.IntVar(value=default_cfg.max_points)
        self.export_format_var = tk.StringVar(value=default_cfg.export_format)
        self.device_var = tk.StringVar(value=default_cfg.device)
        self.image_downscale_var = tk.IntVar(value=default_cfg.image_downscale)
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
        self.status_label: Optional[ttk.Label] = None
        self.log_view: Optional[scrolledtext.ScrolledText] = None
        self._log_handler: Optional[logging.Handler] = None
        self._working = False

        self._build_contents()
        self._update_scene_label()

    def _build_contents(self) -> None:
        ttk.Label(self.frame, text="Training and SfM", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=10, pady=(10, 4)
        )

        scene_row = ttk.Frame(self.frame)
        scene_row.pack(fill="x", padx=10, pady=(0, 6))
        self.scene_label = ttk.Label(scene_row, text=self._scene_text(), anchor="w", justify="left")
        self.scene_label.pack(side="left")

        self.status_label = ttk.Label(self.frame, textvariable=self.status_var, foreground="#444")
        self.status_label.pack(anchor="w", padx=10, pady=(0, 8))

        sfm_frame = ttk.LabelFrame(self.frame, text="Structure-from-Motion")
        sfm_frame.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(
            sfm_frame,
            text=(
                "Run COLMAP on selected frames, then immediately start training. "
                "GPU is required; CPU training is rejected."
            ),
            wraplength=780,
            justify="left",
        ).pack(anchor="w", padx=6, pady=(6, 4))

        path_row = ttk.Frame(sfm_frame)
        path_row.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(path_row, text="COLMAP path:").grid(row=0, column=0, sticky="w")
        ttk.Entry(path_row, textvariable=self.colmap_path_var, width=40).grid(row=0, column=1, sticky="ew", padx=(4, 8))
        ttk.Label(path_row, text="CUDA toolkit path:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(path_row, textvariable=self.cuda_path_var, width=40).grid(
            row=1, column=1, sticky="ew", padx=(4, 8), pady=(4, 0)
        )
        path_row.columnconfigure(1, weight=1)

        sfm_buttons = ttk.Frame(sfm_frame)
        sfm_buttons.pack(fill="x", padx=6, pady=(0, 8))
        ttk.Button(sfm_buttons, text="Run COLMAP + Train", command=self._run_pipeline).pack(side="left")

        training_frame = ttk.LabelFrame(self.frame, text="Training loop")
        training_frame.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(
            training_frame,
            text="Train splats from selected frames with checkpoints in cache/outputs/<scene>/splats.",
            wraplength=780,
            justify="left",
        ).pack(anchor="w", padx=6, pady=(6, 4))

        form = ttk.Frame(training_frame)
        form.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(form, text="Iterations:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(form, from_=1, to=1_000_000, textvariable=self.iterations_var, width=10).grid(
            row=0, column=1, sticky="w", padx=(4, 16)
        )
        ttk.Label(form, text="Snapshot interval:").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(form, from_=1, to=1_000_000, textvariable=self.snapshot_var, width=10).grid(
            row=0, column=3, sticky="w", padx=(4, 16)
        )
        ttk.Label(form, text="Max points (0 = all):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(form, from_=0, to=2_000_000, textvariable=self.max_points_var, width=12).grid(
            row=1, column=1, sticky="w", padx=(4, 12), pady=(6, 0)
        )
        ttk.Label(form, text="Export format:").grid(row=1, column=2, sticky="w", pady=(6, 0))
        ttk.Combobox(form, textvariable=self.export_format_var, values=("ply", "splat"), width=14).grid(
            row=1, column=3, sticky="w", padx=(4, 0), pady=(6, 0)
        )
        ttk.Label(form, text="Device (CUDA only):").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(form, textvariable=self.device_var, width=16).grid(
            row=2, column=1, sticky="w", padx=(4, 12), pady=(6, 0)
        )
        ttk.Label(form, text="Image downscale:").grid(row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Spinbox(form, from_=1, to=8, textvariable=self.image_downscale_var, width=10).grid(
            row=2, column=3, sticky="w", padx=(4, 0), pady=(6, 0)
        )
        ttk.Label(form, text="Batch size (images):").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(form, from_=1, to=16, textvariable=self.batch_size_var, width=10).grid(
            row=3, column=1, sticky="w", padx=(4, 0), pady=(6, 0)
        )
        ttk.Label(form, text="SH degree / interval:").grid(row=3, column=2, sticky="w", pady=(6, 0))
        sh_frame = ttk.Frame(form)
        sh_frame.grid(row=3, column=3, sticky="w", padx=(4, 0), pady=(6, 0))
        ttk.Spinbox(sh_frame, from_=0, to=5, textvariable=self.sh_degree_var, width=6).pack(side="left")
        ttk.Label(sh_frame, text="/").pack(side="left", padx=2)
        ttk.Spinbox(sh_frame, from_=1, to=100000, textvariable=self.sh_interval_var, width=8).pack(side="left")

        opt_frame = ttk.LabelFrame(training_frame, text="Optimization")
        opt_frame.pack(fill="x", padx=6, pady=(6, 6))
        ttk.Label(opt_frame, text="Init scale:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(opt_frame, from_=0.01, to=5.0, increment=0.01, textvariable=self.init_scale_var, width=10).grid(
            row=0, column=1, sticky="w", padx=(4, 12)
        )
        ttk.Label(opt_frame, text="Scale min/max:").grid(row=0, column=2, sticky="w")
        scale_frame = ttk.Frame(opt_frame)
        scale_frame.grid(row=0, column=3, sticky="w")
        ttk.Spinbox(scale_frame, from_=1e-5, to=1.0, increment=1e-5, textvariable=self.min_scale_var, width=10).pack(
            side="left"
        )
        ttk.Label(scale_frame, text="to").pack(side="left", padx=2)
        ttk.Spinbox(scale_frame, from_=1e-4, to=1.0, increment=1e-4, textvariable=self.max_scale_var, width=10).pack(
            side="left"
        )
        ttk.Label(opt_frame, text="Opacity bias:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(opt_frame, from_=0.01, to=0.99, increment=0.01, textvariable=self.opacity_bias_var, width=10).grid(
            row=1, column=1, sticky="w", padx=(4, 0), pady=(6, 0)
        )
        ttk.Checkbutton(opt_frame, text="Random background", variable=self.random_background_var).grid(
            row=1, column=2, columnspan=2, sticky="w", pady=(6, 0)
        )
        ttk.Label(opt_frame, text="LR (means / scales / opacities / SH):").grid(row=2, column=0, sticky="w", pady=(6, 0))
        lr_frame = ttk.Frame(opt_frame)
        lr_frame.grid(row=2, column=1, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Entry(lr_frame, textvariable=self.means_lr_var, width=10).pack(side="left", padx=(0, 4))
        ttk.Entry(lr_frame, textvariable=self.scales_lr_var, width=10).pack(side="left", padx=(0, 4))
        ttk.Entry(lr_frame, textvariable=self.opacities_lr_var, width=10).pack(side="left", padx=(0, 4))
        ttk.Entry(lr_frame, textvariable=self.sh_lr_var, width=10).pack(side="left")
        ttk.Label(
            opt_frame,
            text="Higher-order SH uses SH LR / 20 automatically; quaternion LR is fixed at 1e-3 to match gsplat defaults.",
            wraplength=760,
            justify="left",
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Label(opt_frame, text="Means LR final scale:").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(
            opt_frame,
            from_=0.0001,
            to=1.0,
            increment=0.0001,
            textvariable=self.lr_final_scale_var,
            width=10,
        ).grid(row=4, column=1, sticky="w", padx=(4, 0), pady=(6, 0))

        densify_frame = ttk.LabelFrame(training_frame, text="Densify / prune (gsplat DefaultStrategy)")
        densify_frame.pack(fill="x", padx=6, pady=(6, 6))
        ttk.Label(densify_frame, text="Start / interval (iters):").grid(row=0, column=0, sticky="w")
        densify_timing = ttk.Frame(densify_frame)
        densify_timing.grid(row=0, column=1, sticky="w", padx=(4, 12))
        ttk.Spinbox(
            densify_timing, from_=0, to=100000, textvariable=self.densify_start_var, width=10
        ).pack(side="left")
        ttk.Label(densify_timing, text="/").pack(side="left", padx=2)
        ttk.Spinbox(
            densify_timing, from_=1, to=100000, textvariable=self.densify_interval_var, width=8
        ).pack(side="left")
        ttk.Label(densify_frame, text="Densify thresholds (opacity / scale):").grid(row=0, column=2, sticky="w")
        densify_thresh = ttk.Frame(densify_frame)
        densify_thresh.grid(row=0, column=3, sticky="w", padx=(4, 0))
        ttk.Entry(densify_thresh, textvariable=self.densify_opacity_var, width=10).pack(side="left", padx=(0, 4))
        ttk.Entry(densify_thresh, textvariable=self.densify_scale_var, width=10).pack(side="left")
        ttk.Label(densify_frame, text="Prune thresholds (opacity / scale):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        prune_thresh = ttk.Frame(densify_frame)
        prune_thresh.grid(row=1, column=1, sticky="w", padx=(4, 12), pady=(6, 0))
        ttk.Entry(prune_thresh, textvariable=self.prune_opacity_var, width=10).pack(side="left", padx=(0, 4))
        ttk.Entry(prune_thresh, textvariable=self.prune_scale_var, width=10).pack(side="left")
        ttk.Label(densify_frame, text="Max points during densify (0=limit by output cap):").grid(
            row=1, column=2, sticky="w", pady=(6, 0)
        )
        ttk.Spinbox(densify_frame, from_=0, to=5_000_000, textvariable=self.densify_max_points_var, width=12).grid(
            row=1, column=3, sticky="w", padx=(4, 0), pady=(6, 0)
        )
        ttk.Label(
            densify_frame,
            text="Strategy matches tools/gsplat_examples/simple_trainer: gradients trigger splits/duplication; values tune sensitivity.",
            wraplength=760,
            justify="left",
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))

        ttk.Label(training_frame, text="Live logs stream below; full logs live in logs/app.log.", anchor="w").pack(
            anchor="w", padx=6, pady=(4, 4)
        )

        log_frame = ttk.LabelFrame(self.frame, text="Live logs (streamed from logger)")
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_view = scrolledtext.ScrolledText(log_frame, wrap="none", height=16, width=110)
        self.log_view.pack(fill="both", expand=True, padx=6, pady=6)
        self.log_view.configure(state="disabled")
        self._attach_log_handler()

    def _scene_text(self) -> str:
        scene = self.app_state.current_scene_id
        if scene is None:
            return "No active scene selected."
        return f"Active scene: {scene}"

    def on_scene_changed(self, scene_id: Optional[str]) -> None:
        if scene_id is not None:
            self.app_state.set_current_scene(scene_id)
        self._update_scene_label()

    def _require_scene(self) -> Optional[str]:
        scene = self.app_state.current_scene_id
        if scene is None:
            self._set_status("Select or create a scene in the Inputs tab first.", is_error=True)
            return None
        return str(scene)

    def _update_scene_label(self) -> None:
        if self.scene_label is not None:
            self.scene_label.config(text=self._scene_text())

    def _run_pipeline(self) -> None:
        if self._working:
            self._set_status("Another operation is running; wait for it to finish.", is_error=True)
            return
        scene_id = self._require_scene()
        if scene_id is None:
            return
        if not self._clear_outputs(scene_id):
            return
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
        self._set_status("Running COLMAP then training...")
        self._working = True

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

    def _report_progress(self, iteration: int, total: int, metric: float) -> None:
        self.frame.after(
            0, lambda: self.status_var.set(f"Training {iteration}/{total} loss={metric:.4f}")
        )

    def _handle_checkpoint(self, iteration: int, checkpoint_path: Path) -> None:
        def _update() -> None:
            self.status_var.set(f"Checkpoint {iteration}: {checkpoint_path.name}")

        self.frame.after(0, _update)

    def _handle_pipeline_success(self, result: Tuple[SfmResult, TrainingResult]) -> None:
        _sfm_result, training_result = result
        self._working = False
        self.app_state.refresh_scene_status()
        self._update_scene_label()
        self._set_status(
            f"SfM + training finished. Last checkpoint: {training_result.last_checkpoint}",
            is_error=False,
        )

    def _handle_error(self, exc: Exception) -> None:
        self._working = False
        self.logger.exception("Training tab operation failed")
        self._set_status(f"Operation failed: {exc}", is_error=True)

    def _set_status(self, message: str, *, is_error: bool = False) -> None:
        self.status_var.set(message)
        if self.status_label is not None:
            self.status_label.config(foreground="#a00" if is_error else "#444")

    def _clear_outputs(self, scene_id: str) -> bool:
        try:
            paths = ScenePaths(scene_id, cache_root=self.app_state.config.cache_root)
            if paths.outputs_root.exists():
                shutil.rmtree(paths.outputs_root)
            self.logger.info("Cleared outputs for scene=%s at %s", scene_id, paths.outputs_root)
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

            def emit(self, record: logging.LogRecord) -> None:
                msg = self.format(record)

                def _append() -> None:
                    if not self.widget.winfo_exists():
                        return
                    self.widget.configure(state="normal")
                    self.widget.insert("end", msg + "\n")
                    self.widget.see("end")
                    self.widget.configure(state="disabled")

                try:
                    self.widget.after(0, _append)
                except Exception:  # noqa: BLE001
                    pass

        handler = TkLogHandler(self.log_view)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root_logger.addHandler(handler)
        self._log_handler = handler
