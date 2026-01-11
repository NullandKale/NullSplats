"""Layout helpers for the Training tab UI."""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext, ttk

from nullsplats.ui.advanced_render_controls import AdvancedRenderSettingsPanel
from nullsplats.ui.colmap_camera_panel import ColmapCameraPanel
from nullsplats.ui.gl_canvas import GLCanvas
from nullsplats.ui.render_controls import RenderSettingsPanel
from nullsplats.optional_plugins import looking_glass_available


class TrainingTabLayoutMixin:
    """UI construction helpers for TrainingTab."""

    def _register_control(self, widget: tk.Widget) -> None:
        """Track interactive widgets so we can disable/enable during training."""
        self._interactive_controls.append(widget)
        widget.bind(
            "<Destroy>",
            lambda e: self._interactive_controls.remove(widget)
            if widget in self._interactive_controls
            else None,
        )

    def _build_contents(self) -> None:
        paned = ttk.Panedwindow(self.frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        right_col = ttk.Frame(paned)
        left_outer = ttk.Frame(paned, width=420)
        paned.add(right_col, weight=3)
        paned.add(left_outer, weight=2)

        # Scrollable left column to keep long training controls accessible.
        left_canvas = tk.Canvas(left_outer, highlightthickness=0)
        left_scroll = ttk.Scrollbar(left_outer, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_scroll.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="both", expand=True)

        left_col = ttk.Frame(left_canvas)
        left_window_id = left_canvas.create_window((0, 0), window=left_col, anchor="nw")

        def _update_left_scroll(_event=None) -> None:
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        left_col.bind("<Configure>", _update_left_scroll)

        def _on_left_resize(event) -> None:
            left_canvas.itemconfigure(left_window_id, width=event.width)

        left_canvas.bind("<Configure>", _on_left_resize)

        def _on_mousewheel(event) -> None:
            if left_canvas.winfo_height() < left_col.winfo_height():
                left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        left_canvas.bind("<MouseWheel>", _on_mousewheel)

        # --- Right: viewer-first layout ---
        preview_shell = ttk.Frame(right_col)
        preview_shell.pack(fill="both", expand=True, padx=10, pady=(10, 8))

        preview_frame = ttk.LabelFrame(preview_shell, text="Preview")
        preview_frame.pack(fill="both", expand=True)
        header = ttk.Frame(preview_frame)
        header.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(header, textvariable=self.preview_status_var, foreground="#444").pack(side="left")
        ttk.Button(header, text="Refresh preview", command=self._refresh_preview_now).pack(side="right")
        ttk.Checkbutton(
            header,
            text="Enable live preview polling",
            variable=self._preview_toggle,
            command=self._toggle_preview_poll,
        ).pack(side="right", padx=(0, 8))
        preview_inner = ttk.Frame(preview_frame)
        preview_inner.pack(fill="both", expand=True, padx=6, pady=(6, 4))
        self.preview_canvas = GLCanvas(preview_inner, device=self.device_var.get(), width=1200, height=720)
        self.preview_canvas.pack(fill="both", expand=True)

        notebook = ttk.Notebook(preview_frame)
        notebook.pack(fill="x", padx=6, pady=(0, 6))
        render_tab = ttk.Frame(notebook)
        notebook.add(render_tab, text="Render controls")
        self.preview_controls = RenderSettingsPanel(render_tab, lambda: self.preview_canvas)
        self.preview_controls.pack(fill="x", padx=4, pady=4)
        advanced_tab = ttk.Frame(notebook)
        notebook.add(advanced_tab, text="Advanced")
        self.preview_advanced_controls = AdvancedRenderSettingsPanel(advanced_tab, lambda: self.preview_canvas)
        self.preview_advanced_controls.pack(fill="x", padx=4, pady=4)
        camera_tab = ttk.Frame(notebook)
        notebook.add(camera_tab, text="Cameras")
        self.colmap_panel = ColmapCameraPanel(
            camera_tab,
            viewer_getter=lambda: self.preview_canvas,
            scene_getter=lambda: self.app_state.current_scene_id,
            paths_getter=lambda scene: self.app_state.scene_manager.get(scene).paths,
        )
        self.colmap_panel.pack(fill="both", expand=True, padx=4, pady=(4, 4))
        self._lkg_enabled = looking_glass_available()
        if self._lkg_enabled:
            bridge_tab = ttk.Frame(notebook)
            notebook.add(bridge_tab, text="Looking Glass")
            self._build_lkg_panel(bridge_tab)
        notebook.select(camera_tab)

        # --- Left: workflow-focused, minimal primary controls ---
        ttk.Label(left_col, text="Training workflow", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=10, pady=(10, 6)
        )

        method_card = ttk.LabelFrame(left_col, text="Training method")
        method_card.pack(fill="x", padx=10, pady=(0, 6))
        method_row = ttk.Frame(method_card)
        method_row.pack(fill="x", padx=6, pady=(6, 4))
        ttk.Label(method_row, text="Method:").pack(side="left")
        method_combo = ttk.Combobox(
            method_row,
            textvariable=self.training_method_var,
            values=sorted(self._trainers.keys()),
            state="readonly",
            width=18,
        )
        method_combo.pack(side="left", padx=(4, 12))
        method_combo.bind("<<ComboboxSelected>>", lambda _: self._apply_trainer_capabilities())
        self._register_control(method_combo)
        ttk.Label(method_row, text="Device:").pack(side="left")
        device_entry = ttk.Entry(method_row, textvariable=self.device_var, width=12)
        device_entry.pack(side="left", padx=(4, 0))
        self._register_control(device_entry)
        ttk.Label(method_card, textvariable=self.method_hint_var, foreground="#666").pack(
            anchor="w", padx=6, pady=(0, 6)
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
            textvariable=self.sfm_hint_var,
            anchor="w",
            justify="left",
            foreground="#666",
        ).pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(
            scene_card,
            text="Set scenes in Inputs tab; selecting a scene there auto-loads cached checkpoints for preview.",
            foreground="#666",
            wraplength=380,
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=6, pady=(0, 6))

        status_card = ttk.LabelFrame(left_col, text="Run status")
        status_card.pack(fill="x", padx=10, pady=(0, 6))
        self.status_label = ttk.Label(
            status_card, textvariable=self.status_var, foreground="#333", font=("Segoe UI", 11, "bold"), wraplength=380
        )
        self.status_label.pack(fill="x", padx=6, pady=(6, 4))
        self.progress_bar = ttk.Progressbar(status_card, variable=self.progress_var, mode="determinate", maximum=1.0)
        self.progress_bar.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(
            status_card,
            text="Run training after COLMAP completes. Preview updates while training.",
            foreground="#666",
            wraplength=380,
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=6, pady=(0, 4))
        primary_row = ttk.Frame(status_card)
        primary_row.pack(fill="x", padx=6, pady=(0, 6))
        btn_train_only = ttk.Button(primary_row, text="Run training", command=self._run_training_only)
        btn_train_only.pack(side="left")
        self._register_control(btn_train_only)
        btn_warm = ttk.Button(primary_row, text="Warm up renderer", command=self._warmup_renderer)
        btn_warm.pack(side="right")
        self._register_control(btn_warm)

        self.gsplat_settings_frame = ttk.Frame(left_col)
        self.gsplat_settings_frame.pack(fill="x", padx=10, pady=(0, 6))
        self.da3_settings_frame = ttk.Frame(left_col)
        self.da3_settings_frame.pack(fill="x", padx=10, pady=(0, 6))
        self.sharp_settings_frame = ttk.Frame(left_col)
        self.sharp_settings_frame.pack(fill="x", padx=10, pady=(0, 6))

        primary_cfg = ttk.LabelFrame(self.gsplat_settings_frame, text="Gsplat settings")
        primary_cfg.pack(fill="x", padx=0, pady=(0, 6))
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

        da3_body = ttk.LabelFrame(self.da3_settings_frame, text="Depth Anything 3 settings")
        da3_body.pack(fill="x", padx=0, pady=(0, 6))
        da3_row1 = ttk.Frame(da3_body)
        da3_row1.pack(fill="x", padx=6, pady=(6, 4))
        ttk.Label(da3_row1, text="Model id/path:").pack(side="left")
        da3_model_entry = ttk.Entry(da3_row1, textvariable=self.da3_pretrained_id_var, width=32)
        da3_model_entry.pack(
            side="left", padx=(4, 0), fill="x", expand=True
        )
        self._register_control(da3_model_entry)

        da3_row2 = ttk.Frame(da3_body)
        da3_row2.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(da3_row2, text="Preset:").pack(side="left")
        da3_preset_combo = ttk.Combobox(
            da3_row2,
            textvariable=self.training_preset_var,
            values=["low", "medium", "high"],
            state="readonly",
            width=10,
        )
        da3_preset_combo.pack(side="left", padx=(4, 12))
        da3_preset_combo.bind("<<ComboboxSelected>>", lambda _: self._apply_training_preset())
        self._register_control(da3_preset_combo)
        ttk.Label(da3_row2, text="Ref view:").pack(side="left")
        da3_ref_combo = ttk.Combobox(
            da3_row2,
            textvariable=self.da3_ref_view_var,
            values=("saddle_balanced", "saddle_sim_range", "first", "middle"),
            width=16,
            state="readonly",
        )
        da3_ref_combo.pack(side="left", padx=(4, 12))
        self._register_control(da3_ref_combo)
        da3_ray_toggle = ttk.Checkbutton(da3_row2, text="Use ray pose", variable=self.da3_use_ray_pose_var)
        da3_ray_toggle.pack(side="left")
        self._register_control(da3_ray_toggle)

        da3_row3 = ttk.Frame(da3_body)
        da3_row3.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(da3_row3, text="Process res:").pack(side="left")
        da3_res_spin = ttk.Spinbox(da3_row3, from_=128, to=2048, textvariable=self.da3_process_res_var, width=8)
        da3_res_spin.pack(side="left", padx=(4, 12))
        self._register_control(da3_res_spin)
        ttk.Label(da3_row3, text="Resize method:").pack(side="left")
        da3_resize_combo = ttk.Combobox(
            da3_row3,
            textvariable=self.da3_process_res_method_var,
            values=("upper_bound_resize", "lower_bound_resize"),
            width=18,
            state="readonly",
        )
        da3_resize_combo.pack(side="left", padx=(4, 12))
        self._register_control(da3_resize_combo)

        da3_row4 = ttk.Frame(da3_body)
        da3_row4.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(da3_row4, text="View stride:").pack(side="left")
        da3_stride_spin = ttk.Spinbox(
            da3_row4, from_=1, to=100, textvariable=self.da3_view_stride_var, width=6
        )
        da3_stride_spin.pack(side="left", padx=(4, 12))
        self._register_control(da3_stride_spin)
        ttk.Label(da3_row4, text="Max views:").pack(side="left")
        da3_max_views_spin = ttk.Spinbox(
            da3_row4, from_=0, to=10_000, textvariable=self.da3_max_views_var, width=8
        )
        da3_max_views_spin.pack(side="left", padx=(4, 12))
        self._register_control(da3_max_views_spin)
        ttk.Label(da3_row4, text="GS views interval:").pack(side="left")
        da3_interval_spin = ttk.Spinbox(
            da3_row4, from_=1, to=100, textvariable=self.da3_gs_views_interval_var, width=6
        )
        da3_interval_spin.pack(side="left", padx=(4, 12))
        self._register_control(da3_interval_spin)

        da3_row5 = ttk.Frame(da3_body)
        da3_row5.pack(fill="x", padx=6, pady=(0, 4))
        da3_align_toggle = ttk.Checkbutton(
            da3_row5, text="Align to input scale", variable=self.da3_align_scale_var
        )
        da3_align_toggle.pack(side="left", padx=(0, 12))
        self._register_control(da3_align_toggle)
        da3_infer_toggle = ttk.Checkbutton(da3_row5, text="Infer GS", variable=self.da3_infer_gs_var)
        da3_infer_toggle.pack(side="left", padx=(0, 12))
        self._register_control(da3_infer_toggle)
        da3_input_toggle = ttk.Checkbutton(
            da3_row5, text="Use input resolution", variable=self.da3_use_input_res_var
        )
        da3_input_toggle.pack(side="left")
        self._register_control(da3_input_toggle)

        da3_row6 = ttk.Frame(da3_body)
        da3_row6.pack(fill="x", padx=6, pady=(0, 4))
        da3_unposed_toggle = ttk.Checkbutton(
            da3_row6,
            text="Allow DA3 without poses (skip COLMAP)",
            variable=self.da3_allow_unposed_var,
        )
        da3_unposed_toggle.pack(side="left")
        self._register_control(da3_unposed_toggle)

        sharp_body = ttk.LabelFrame(self.sharp_settings_frame, text="SHARP settings")
        sharp_body.pack(fill="x", padx=0, pady=(0, 6))
        sharp_row1 = ttk.Frame(sharp_body)
        sharp_row1.pack(fill="x", padx=6, pady=(6, 4))
        ttk.Label(sharp_row1, text="Checkpoint path:").pack(side="left")
        sharp_ckpt_entry = ttk.Entry(sharp_row1, textvariable=self.sharp_checkpoint_path_var, width=28)
        sharp_ckpt_entry.pack(side="left", padx=(4, 0), fill="x", expand=True)
        self._register_control(sharp_ckpt_entry)

        sharp_row2 = ttk.Frame(sharp_body)
        sharp_row2.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(sharp_row2, text="Image index:").pack(side="left")
        sharp_index_spin = ttk.Spinbox(sharp_row2, from_=0, to=100000, textvariable=self.sharp_image_index_var, width=8)
        sharp_index_spin.pack(side="left", padx=(4, 12))
        self._register_control(sharp_index_spin)
        ttk.Label(sharp_row2, text="Intrinsics:").pack(side="left")
        sharp_intrinsics_combo = ttk.Combobox(
            sharp_row2,
            textvariable=self.sharp_intrinsics_source_var,
            values=("colmap", "exif", "manual"),
            width=10,
            state="readonly",
        )
        sharp_intrinsics_combo.pack(side="left", padx=(4, 12))
        self._register_control(sharp_intrinsics_combo)

        sharp_row3 = ttk.Frame(sharp_body)
        sharp_row3.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(sharp_row3, text="Focal px override:").pack(side="left")
        sharp_focal_entry = ttk.Entry(sharp_row3, textvariable=self.sharp_focal_px_var, width=10)
        sharp_focal_entry.pack(side="left", padx=(4, 12))
        self._register_control(sharp_focal_entry)
        ttk.Label(sharp_row3, text="FOV override (deg):").pack(side="left")
        sharp_fov_entry = ttk.Entry(sharp_row3, textvariable=self.sharp_fov_deg_var, width=10)
        sharp_fov_entry.pack(side="left", padx=(4, 0))
        self._register_control(sharp_fov_entry)

        # Advanced controls tucked away
        training_body = self._collapsible_section(self.gsplat_settings_frame, "Training config (advanced)", start_hidden=True)
        path_row = ttk.Frame(training_body)
        path_row.pack(fill="x", padx=6, pady=(6, 4))
        ttk.Label(path_row, text="CUDA toolkit path:").pack(side="left")
        ttk.Entry(path_row, textvariable=self.cuda_path_var, width=32).pack(side="left", padx=(4, 0), fill="x", expand=True)
        row1 = ttk.Frame(training_body)
        row1.pack(fill="x", padx=6, pady=(0, 4))
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

        row4 = ttk.Frame(training_body)
        row4.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(row4, text="Preview interval (sec / min iters):").pack(side="left")
        preview_frame = ttk.Frame(row4)
        preview_frame.pack(side="left")
        ttk.Spinbox(preview_frame, from_=0.1, to=60.0, increment=0.1, textvariable=self.preview_interval_var, width=6).pack(
            side="left"
        )
        ttk.Label(preview_frame, text="/").pack(side="left", padx=2)
        ttk.Spinbox(preview_frame, from_=0, to=100000, textvariable=self.preview_min_iters_var, width=8).pack(side="left")

        row5 = ttk.Frame(training_body)
        row5.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(row5, text="Preview max points (0=all):").pack(side="left")
        ttk.Spinbox(row5, from_=0, to=2_000_000, textvariable=self.preview_max_points_var, width=12).pack(
            side="left", padx=(4, 0)
        )

        opt_body = self._collapsible_section(self.gsplat_settings_frame, "Optimization", start_hidden=True)
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

        densify_body = self._collapsible_section(self.gsplat_settings_frame, "Densify / prune", start_hidden=True)
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

        self.log_label = ttk.Label(
            left_col,
            text="Live logs stream below; full logs live in logs/app.log.",
            anchor="w",
            justify="left",
        )
        self.log_label.pack(anchor="w", padx=10, pady=(4, 4))

        self.log_frame = ttk.LabelFrame(left_col, text="Live logs")
        self.log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_view = scrolledtext.ScrolledText(self.log_frame, wrap="word", height=12, width=80)
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

    def _build_lkg_panel(self, parent: ttk.Frame) -> None:
        status_card = ttk.LabelFrame(parent, text="Looking Glass Bridge")
        status_card.pack(fill="x", padx=6, pady=6)
        ttk.Label(status_card, textvariable=self.lkg_status_var, font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=6, pady=(6, 2)
        )
        ttk.Label(status_card, textvariable=self.lkg_detail_var, foreground="#444", wraplength=520).pack(
            anchor="w", padx=6, pady=(0, 6)
        )
        btn_row = ttk.Frame(status_card)
        btn_row.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(btn_row, text="Retry Bridge", command=self._lkg_retry_clicked).pack(side="left")
        ttk.Button(btn_row, text="Refresh status", command=self._refresh_lkg_status).pack(side="left", padx=(6, 0))

        ctrl_card = ttk.LabelFrame(parent, text="Depth / focus")
        ctrl_card.pack(fill="x", padx=6, pady=(0, 6))
        form = ttk.Frame(ctrl_card)
        form.pack(fill="x", padx=6, pady=6)
        self._lkg_depthiness_label = tk.StringVar(value=f"{self.lkg_depthiness_var.get():.2f}")
        ttk.Label(form, text="Depthiness:").grid(row=0, column=0, sticky="w")
        ttk.Scale(
            form,
            from_=0.0,
            to=5.0,
            variable=self.lkg_depthiness_var,
            command=lambda _v: (self._lkg_depthiness_label.set(f"{self.lkg_depthiness_var.get():.2f}"), self._lkg_schedule_apply()),
        ).grid(row=0, column=1, sticky="ew", padx=(4, 8))
        ttk.Label(form, textvariable=self._lkg_depthiness_label, width=6).grid(row=0, column=2, sticky="w")
        ttk.Label(form, text="Focus:").grid(row=0, column=3, sticky="w", padx=(8, 0))
        self._lkg_focus_label = tk.StringVar(value=f"{self.lkg_focus_var.get():.2f}")
        ttk.Scale(
            form,
            from_=-10.0,
            to=10.0,
            variable=self.lkg_focus_var,
            command=lambda _v: (self._lkg_focus_label.set(f"{self.lkg_focus_var.get():.2f}"), self._lkg_schedule_apply()),
        ).grid(row=0, column=4, sticky="ew", padx=(4, 8))
        ttk.Label(form, textvariable=self._lkg_focus_label, width=6).grid(row=0, column=5, sticky="w")

        row_zoom = ttk.Frame(ctrl_card)
        row_zoom.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(row_zoom, text="Zoom:").grid(row=0, column=0, sticky="w")
        self._lkg_zoom_label = tk.StringVar(value=f"{self.lkg_zoom_var.get():.2f}")
        ttk.Scale(
            row_zoom,
            from_=0.1,
            to=4.0,
            variable=self.lkg_zoom_var,
            command=lambda _v: (self._lkg_zoom_label.set(f"{self.lkg_zoom_var.get():.2f}"), self._lkg_schedule_apply()),
        ).grid(row=0, column=1, sticky="ew", padx=(4, 8))
        ttk.Label(row_zoom, textvariable=self._lkg_zoom_label, width=6).grid(row=0, column=2, sticky="w")
        ttk.Button(row_zoom, text="Apply", command=self._lkg_apply_clicked).grid(row=0, column=3, sticky="w", padx=(8, 0))

        row2 = ttk.Frame(ctrl_card)
        row2.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(row2, text="FOV (deg):").grid(row=0, column=0, sticky="w")
        self._lkg_fov_label = tk.StringVar(value=f"{self.lkg_fov_var.get():.1f}")
        ttk.Scale(
            row2,
            from_=5.0,
            to=120.0,
            variable=self.lkg_fov_var,
            command=lambda _v: (self._lkg_fov_label.set(f"{self.lkg_fov_var.get():.1f}"), self._lkg_schedule_apply()),
        ).grid(row=0, column=1, sticky="ew", padx=(4, 8))
        ttk.Label(row2, textvariable=self._lkg_fov_label, width=6).grid(row=0, column=2, sticky="w")
        ttk.Label(row2, text="Viewcone (deg):").grid(row=0, column=3, sticky="w", padx=(8, 0))
        self._lkg_viewcone_label = tk.StringVar(value=f"{self.lkg_viewcone_var.get():.1f}")
        ttk.Scale(
            row2,
            from_=0.0,
            to=89.0,
            variable=self.lkg_viewcone_var,
            command=lambda _v: (self._lkg_viewcone_label.set(f"{self.lkg_viewcone_var.get():.1f}"), self._lkg_schedule_apply()),
        ).grid(row=0, column=4, sticky="ew", padx=(4, 8))
        ttk.Label(row2, textvariable=self._lkg_viewcone_label, width=6).grid(row=0, column=5, sticky="w")

        form.columnconfigure(1, weight=1)
        form.columnconfigure(4, weight=1)
        row_zoom.columnconfigure(1, weight=1)
        row2.columnconfigure(1, weight=1)
        row2.columnconfigure(4, weight=1)

