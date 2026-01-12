"""Video-to-SHARP workflow UI (app and tab variants)."""

from __future__ import annotations

import json
import re
import subprocess
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk
from typing import List, Optional

import torch

from nullsplats.app_state import AppState
from nullsplats.backend.io_cache import ScenePaths, ensure_scene_dirs
from nullsplats.backend.splat_backends.sharp_trainer import run_sharp_video_stream
from nullsplats.backend.splat_train_config import PreviewPayload
from nullsplats.ui.gl_canvas import GLCanvas
from nullsplats.ui.advanced_render_controls import AdvancedRenderSettingsPanel
from nullsplats.ui.render_controls import RenderSettingsPanel
from nullsplats.ui.colmap_camera_panel import ColmapCameraPanel
from nullsplats.optional_plugins import looking_glass_available
from nullsplats.util.logging import get_logger
from nullsplats.util.threading import run_in_background

import importlib.util
import sys


class _VideoSharpUI:
    """Shared UI logic for video-to-SHARP processing."""

    def __init__(self, parent: tk.Misc, tk_root: tk.Misc, cache_root: Path) -> None:
        self.logger = get_logger("video_sharp.ui")
        self.parent = parent
        self.root = tk_root
        self.cache_root = Path(cache_root)

        self.video_path: Optional[Path] = None
        self.scene_id: Optional[str] = None
        self.scene_paths: Optional[ScenePaths] = None
        self.splat_paths: List[Path] = []
        self._play_job: Optional[str] = None
        self._play_index = 0
        self._is_playing = False
        self._tab_active = True
        self._lkg_enabled = looking_glass_available()
        self._lkg_status_job: Optional[str] = None
        self._lkg_apply_job: Optional[str] = None

        self.video_path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Pick a video to get started.")
        self.progress_var = tk.StringVar(value="")
        self.max_frames_var = tk.IntVar(value=60)
        self.focal_px_var = tk.DoubleVar(value=0.0)
        self.fov_deg_var = tk.DoubleVar(value=55.0)
        self.play_fps_var = tk.DoubleVar(value=0.0)
        self.video_fps: Optional[float] = None
        self.skip_save_var = tk.BooleanVar(value=False)
        self.use_compile_var = tk.BooleanVar(value=False)
        self.use_amp_var = tk.BooleanVar(value=False)
        self.skip_preview_var = tk.BooleanVar(value=False)
        self.output_format_var = tk.StringVar(value="ply")
        self.lkg_status_var = tk.StringVar(value="Looking Glass: not available")
        self.lkg_detail_var = tk.StringVar(value="")
        self.lkg_depthiness_var = tk.DoubleVar(value=1.0)
        self.lkg_focus_var = tk.DoubleVar(value=2.0)
        self.lkg_fov_var = tk.DoubleVar(value=14.0)
        self.lkg_viewcone_var = tk.DoubleVar(value=40.0)
        self.lkg_zoom_var = tk.DoubleVar(value=1.0)

        self._build_ui()

    def _build_ui(self) -> None:
        paned = ttk.Panedwindow(self.parent, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        ttk.Label(left, text="Video SHARP", font=("Segoe UI", 12, "bold")).pack(
            anchor="w", padx=10, pady=(10, 6)
        )

        src_frame = ttk.LabelFrame(left, text="Source video")
        src_frame.pack(fill="x", padx=10, pady=(0, 8))
        src_row = ttk.Frame(src_frame)
        src_row.pack(fill="x", padx=6, pady=6)
        src_entry = ttk.Entry(src_row, textvariable=self.video_path_var, state="readonly")
        src_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(src_row, text="Browse", command=self._choose_video).pack(side="left", padx=(6, 0))

        params = ttk.LabelFrame(left, text="Frame & camera settings")
        params.pack(fill="x", padx=10, pady=(0, 8))
        row1 = ttk.Frame(params)
        row1.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(row1, text="Max frames:").pack(side="left")
        ttk.Spinbox(row1, from_=1, to=500, textvariable=self.max_frames_var, width=6).pack(
            side="left", padx=(6, 12)
        )
        ttk.Label(row1, text="Video FPS:").pack(side="left")
        ttk.Entry(row1, textvariable=self.play_fps_var, width=6, state="readonly").pack(
            side="left", padx=(6, 0)
        )

        row2 = ttk.Frame(params)
        row2.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(row2, text="Focal px override:").pack(side="left")
        ttk.Entry(row2, textvariable=self.focal_px_var, width=8).pack(side="left", padx=(6, 12))
        ttk.Label(row2, text="FOV override (deg):").pack(side="left")
        ttk.Entry(row2, textvariable=self.fov_deg_var, width=8).pack(side="left", padx=(6, 0))

        row3 = ttk.Frame(params)
        row3.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Checkbutton(
            row3,
            text="Preview only (skip saving)",
            variable=self.skip_save_var,
        ).pack(side="left")
        ttk.Checkbutton(
            row3,
            text="torch.compile",
            variable=self.use_compile_var,
        ).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(
            row3,
            text="AMP (fp16)",
            variable=self.use_amp_var,
        ).pack(side="left", padx=(12, 0))

        row4 = ttk.Frame(params)
        row4.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Checkbutton(
            row4,
            text="Skip preview rendering",
            variable=self.skip_preview_var,
        ).pack(side="left")
        ttk.Label(row4, text="Output:").pack(side="left", padx=(12, 4))
        ttk.Combobox(
            row4,
            textvariable=self.output_format_var,
            values=["ply", "splat"],
            state="readonly",
            width=8,
        ).pack(side="left")

        actions = ttk.LabelFrame(left, text="Actions")
        actions.pack(fill="x", padx=10, pady=(0, 8))
        actions_row = ttk.Frame(actions)
        actions_row.pack(fill="x", padx=6, pady=6)
        ttk.Button(actions_row, text="Run SHARP (stream)", command=self._run_sharp).pack(side="left")
        ttk.Button(actions_row, text="Play", command=self._play).pack(side="left", padx=(12, 0))
        ttk.Button(actions_row, text="Pause", command=self._pause).pack(side="left", padx=(6, 0))
        ttk.Button(actions_row, text="Stop", command=self._stop).pack(side="left", padx=(6, 0))

        list_frame = ttk.LabelFrame(left, text="Generated splats")
        list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        list_inner = ttk.Frame(list_frame)
        list_inner.pack(fill="both", expand=True, padx=6, pady=6)
        self.splat_list = tk.Listbox(list_inner, height=12, exportselection=False)
        self.splat_list.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(list_inner, orient="vertical", command=self.splat_list.yview)
        scroll.pack(side="right", fill="y")
        self.splat_list.config(yscrollcommand=scroll.set)
        self.splat_list.bind("<<ListboxSelect>>", lambda _event: self._preview_selected())

        ttk.Label(left, textvariable=self.progress_var, foreground="#666").pack(
            anchor="w", padx=12, pady=(0, 2)
        )
        ttk.Label(left, textvariable=self.status_var, foreground="#444", wraplength=320).pack(
            anchor="w", padx=12, pady=(0, 10)
        )

        preview_frame = ttk.LabelFrame(right, text="Preview")
        preview_frame.pack(fill="both", expand=True, padx=10, pady=10)
        preview_inner = ttk.Frame(preview_frame)
        preview_inner.pack(fill="both", expand=True, padx=6, pady=6)
        self.viewer = GLCanvas(preview_inner, device="cuda:0", width=960, height=540)
        self.viewer.pack(fill="both", expand=True)

        notebook = ttk.Notebook(preview_frame)
        notebook.pack(fill="x", padx=6, pady=(0, 6))
        render_tab = ttk.Frame(notebook)
        notebook.add(render_tab, text="Render controls")
        self.render_controls = RenderSettingsPanel(render_tab, lambda: self.viewer)
        self.render_controls.pack(fill="x", padx=4, pady=4)
        advanced_tab = ttk.Frame(notebook)
        notebook.add(advanced_tab, text="Advanced")
        self.render_advanced_controls = AdvancedRenderSettingsPanel(advanced_tab, lambda: self.viewer)
        self.render_advanced_controls.pack(fill="x", padx=4, pady=4)
        camera_tab = ttk.Frame(notebook)
        notebook.add(camera_tab, text="Cameras")
        self.colmap_panel = ColmapCameraPanel(
            camera_tab,
            viewer_getter=lambda: self.viewer,
            scene_getter=lambda: self.scene_id,
            paths_getter=lambda scene: self._resolve_scene_paths(scene),
        )
        self.colmap_panel.pack(fill="both", expand=True, padx=4, pady=(4, 4))
        if self._lkg_enabled:
            lkg_tab = ttk.Frame(notebook)
            notebook.add(lkg_tab, text="Looking Glass")
            self._build_lkg_panel(lkg_tab)
        notebook.select(camera_tab)

    def set_tab_active(self, active: bool) -> None:
        self._tab_active = active
        if not active:
            self._pause()
            try:
                self.viewer.stop_rendering()
            except Exception:
                self.logger.debug("Failed to stop viewer on tab deactivate", exc_info=True)
            try:
                if self._lkg_status_job is not None:
                    self.root.after_cancel(self._lkg_status_job)
            except Exception:
                pass
        elif self._lkg_enabled:
            self._refresh_lkg_status()

    def deactivate_viewer(self) -> None:
        self._pause()
        if self.viewer is not None:
            try:
                self.viewer.stop_rendering()
            except Exception:
                self.logger.debug("Failed to stop viewer", exc_info=True)
        try:
            if self._lkg_status_job is not None:
                self.root.after_cancel(self._lkg_status_job)
        except Exception:
            pass

    def _choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video files", "*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v"), ("All files", "*.*")],
        )
        if not path:
            return
        self.set_video_path(Path(path))

    def set_video_path(self, path: Path) -> None:
        self.video_path = Path(path)
        self.scene_id = None
        self.video_fps = _probe_video_fps(self.video_path)
        self.play_fps_var.set(float(self.video_fps or 0.0))
        self.video_path_var.set(str(self.video_path))
        self.scene_paths = None
        self.splat_paths = []
        self._refresh_splat_list()
        self.status_var.set("Video selected. Ready to run SHARP (streamed).")

    def run_sharp(self) -> None:
        self._run_sharp()

    def _run_sharp(self) -> None:
        if self.video_path is None:
            self.status_var.set("Pick a video file first.")
            return
        focal_px = float(self.focal_px_var.get() or 0.0)
        fov_deg = float(self.fov_deg_var.get() or 0.0)
        if focal_px <= 0.0 and fov_deg <= 0.0:
            self.status_var.set("Set a focal px override or FOV (deg) for SHARP.")
            return
        max_frames = int(self.max_frames_var.get() or 0)
        if max_frames <= 0:
            self.status_var.set("Max frames must be at least 1.")
            return

        self.scene_id = self._new_scene_id(self.video_path)
        self.scene_paths = ensure_scene_dirs(self.scene_id, cache_root=self.cache_root)
        try:
            self.colmap_panel.refresh()
        except Exception:
            pass

        output_dir = self.scene_paths.splats_dir / "sharp_video"
        profiler_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        profiler = _load_profiler_class()(f"run_{profiler_label}")
        self.status_var.set("Running SHARP on frames...")
        self.progress_var.set("Starting SHARP...")

        def _on_frame(path: Optional[Path], done: int, total: int) -> None:
            def _update() -> None:
                if self._tab_active:
                    if path is not None and not skip_preview:
                        self._preview_path(path)

            self.root.after(0, _update)

        def _on_preview(payload: dict) -> None:
            def _update() -> None:
                if skip_preview:
                    return
                preview = PreviewPayload(
                    iteration=int(payload["iteration"]),
                    means=torch.from_numpy(payload["means"]),
                    scales_log=torch.from_numpy(payload["scales_log"]),
                    quats_wxyz=torch.from_numpy(payload["quats_wxyz"]),
                    opacities=torch.from_numpy(payload["opacities"]),
                    sh_dc=torch.from_numpy(payload["sh_dc"]),
                )
                self.viewer.load_preview_data(preview)

            self.root.after(0, _update)

        def _on_timing(payload: dict) -> None:
            profiler.record(payload)

        skip_save = bool(self.skip_save_var.get())
        skip_preview = bool(self.skip_preview_var.get())
        use_compile = bool(self.use_compile_var.get())
        use_amp = bool(self.use_amp_var.get())
        output_format = self.output_format_var.get().strip().lower() or "ply"
        if output_format not in {"ply", "splat"}:
            output_format = "ply"

        def _do_sharp():
            return run_sharp_video_stream(
                self.video_path,
                output_dir,
                {
                    "device": "default",
                    "focal_px_override": focal_px if focal_px > 0.0 else None,
                    "fov_override_deg": fov_deg if fov_deg > 0.0 else None,
                },
                max_frames=max_frames,
                skip_save=skip_save,
                use_compile=use_compile,
                use_amp=use_amp,
                output_format=output_format,
                on_frame=_on_frame,
                on_preview=None if skip_preview else _on_preview,
                on_timing=_on_timing,
            )

        def _on_success(outputs: List[Path]) -> None:
            self.splat_paths = outputs
            self._refresh_splat_list()
            self.progress_var.set("")
            summary, json_path, _md_path = profiler.finalize()
            stats = summary.get("stats", {})
            decode_mean = stats.get("decode_s", {}).get("mean", 0.0)
            load_mean = stats.get("load_s", {}).get("mean", 0.0)
            infer_mean = stats.get("infer_s", {}).get("mean", 0.0)
            save_mean = stats.get("save_s", {}).get("mean", 0.0)
            total_mean = stats.get("total_s", {}).get("mean", 0.0)
            summary_msg = (
                "SHARP complete. avg decode={:.3f}s load={:.3f}s infer={:.3f}s "
                "save={:.3f}s total={:.3f}s profile={} saved={} compile={} amp={} format={} preview={}"
            ).format(
                decode_mean,
                load_mean,
                infer_mean,
                save_mean,
                total_mean,
                json_path.name,
                "no" if skip_save else "yes",
                "yes" if use_compile else "no",
                "yes" if use_amp else "no",
                output_format,
                "no" if skip_preview else "yes",
            )
            self.status_var.set(summary_msg)
            self.logger.info(summary_msg)
            self.logger.info("Video SHARP profile written: %s", json_path)
            if outputs and self._tab_active:
                self._preview_path(outputs[0])

        def _on_error(exc: Exception) -> None:
            self.progress_var.set("")
            self.status_var.set(f"SHARP failed: {exc}")

        run_in_background(
            _do_sharp,
            tk_root=self.root,
            on_success=_on_success,
            on_error=_on_error,
            thread_name="video_sharp_run",
        )

    def _refresh_splat_list(self) -> None:
        self.splat_list.delete(0, tk.END)
        for path in self.splat_paths:
            self.splat_list.insert(tk.END, path.name)

    def _selected_splat(self) -> Optional[Path]:
        if not self.splat_paths:
            return None
        selection = self.splat_list.curselection()
        if not selection:
            return None
        return self.splat_paths[selection[0]]

    def _preview_selected(self) -> None:
        path = self._selected_splat()
        if path is None:
            return
        self._preview_path(path)

    def _preview_path(self, path: Path) -> None:
        if not self._tab_active:
            return
        if not self.viewer.winfo_ismapped():
            self.root.after(100, lambda: self._preview_path(path))
            return
        try:
            self.viewer.start_rendering()
            self.viewer.load_splat(path)
            self.status_var.set(f"Previewing {path.name}")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to preview %s", path)
            self.status_var.set(f"Preview failed: {exc}")

    def _play(self) -> None:
        if not self.splat_paths:
            self.status_var.set("Generate splats before playback.")
            return
        if self._is_playing:
            return
        self._is_playing = True
        self._play_index = 0
        self._play_next()

    def _play_next(self) -> None:
        if not self._is_playing or not self.splat_paths or not self._tab_active:
            return
        if not self.viewer.winfo_ismapped():
            self._play_job = self.root.after(100, self._play_next)
            return
        path = self.splat_paths[self._play_index]
        try:
            self.viewer.start_rendering()
            self.viewer.load_splat(path)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Playback failed at %s", path)
            self.status_var.set(f"Playback failed: {exc}")
            self._is_playing = False
            return
        self._play_index = (self._play_index + 1) % len(self.splat_paths)
        fps = float(self.play_fps_var.get() or 0.0) or float(self.video_fps or 0.0) or 1.0
        delay = max(1, int(1000 / max(1.0, fps)))
        self._play_job = self.root.after(delay, self._play_next)

    def _pause(self) -> None:
        self._is_playing = False
        if self._play_job is not None:
            try:
                self.root.after_cancel(self._play_job)
            except Exception:
                pass
            self._play_job = None
        self.status_var.set("Playback paused.")

    def _stop(self) -> None:
        self._pause()
        self._play_index = 0
        self.status_var.set("Playback stopped.")

    def _lkg_sink(self):
        if self.viewer is None:
            return None
        try:
            sinks = self.viewer.get_preview_sinks()
            for sink in sinks:
                if sink.__class__.__name__ == "LookingGlassSink":
                    return sink
        except Exception:
            return None
        return None

    def _refresh_lkg_status(self) -> None:
        if not self._lkg_enabled:
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
                self.root.after_cancel(self._lkg_status_job)
        except Exception:
            pass
        self._lkg_status_job = self.root.after(1000, self._refresh_lkg_status)

    def _lkg_retry_clicked(self) -> None:
        sink = self._lkg_sink()
        if sink is None:
            self.lkg_status_var.set("Looking Glass: not available")
            return
        try:
            if self.viewer is not None:
                try:
                    self.viewer.reset_preview_pipelines()
                except Exception:
                    self.logger.debug("Preview pipeline reset during LKG retry failed", exc_info=True)
            sink.retry_start()
            self.lkg_status_var.set("Looking Glass: retrying...")
            self.lkg_detail_var.set("Will start on next frame once GL context is ready.")
        except Exception:
            self.logger.debug("Looking Glass retry failed", exc_info=True)

    def _lkg_apply_clicked(self) -> None:
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
        if not self._lkg_enabled:
            return
        try:
            if self._lkg_apply_job is not None:
                self.root.after_cancel(self._lkg_apply_job)
        except Exception:
            pass
        self._lkg_apply_job = self.root.after(150, self._lkg_apply_clicked)

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

        self._refresh_lkg_status()

    @staticmethod
    def _new_scene_id(video_path: Path) -> str:
        base = re.sub(r"[^a-zA-Z0-9_-]+", "_", video_path.stem).strip("_") or "video"
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"video_sharp_{base}_{stamp}"

    def _resolve_scene_paths(self, scene: str) -> ScenePaths:
        if self.scene_paths is not None:
            return self.scene_paths
        return ensure_scene_dirs(scene, cache_root=self.cache_root)


class VideoSharpApp:
    """Standalone window for the video-to-SHARP workflow."""

    def __init__(self, cache_root: Path, *, input_path: Optional[Path] = None, auto_run: bool = False) -> None:
        self.root = tk.Tk()
        self.root.title("NullSplats Video SHARP")
        self.root.minsize(1100, 700)
        self.ui = _VideoSharpUI(self.root, self.root, cache_root)
        if input_path is not None:
            self.ui.set_video_path(input_path)
            if auto_run:
                self.root.after(200, self.ui.run_sharp)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        self.ui.deactivate_viewer()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


class VideoSharpTab:
    """Notebook tab for video-to-SHARP processing."""

    def __init__(self, master: tk.Misc, app_state: AppState) -> None:
        self.app_state = app_state
        self.frame = ttk.Frame(master)
        root = self.frame.winfo_toplevel()
        self.ui = _VideoSharpUI(self.frame, root, Path(app_state.config.cache_root))

    def on_tab_selected(self, selected: bool) -> None:
        self.ui.set_tab_active(selected)

    def deactivate_viewer(self) -> None:
        self.ui.deactivate_viewer()


def run_video_sharp_app(cache_root: Optional[Path] = None, *, input_path: Optional[Path] = None) -> None:
    auto_run = input_path is not None
    app = VideoSharpApp(cache_root or Path("cache"), input_path=input_path, auto_run=auto_run)
    app.run()


__all__ = ["run_video_sharp_app", "VideoSharpApp", "VideoSharpTab"]


def _load_profiler_class():
    module_name = "nullsplats_video_sharp_profiler"
    if module_name in sys.modules:
        return sys.modules[module_name].VideoSharpProfiler
    module_path = Path(__file__).with_name("video_sharp_profiler.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Video SHARP profiler from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.VideoSharpProfiler


def _probe_video_fps(path: Path) -> Optional[float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-print_format",
        "json",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True).stdout
    except Exception:
        return None
    try:
        info = json.loads(out)
        streams = info.get("streams", [])
        if not streams:
            return None
        stream = streams[0]
        rate = stream.get("r_frame_rate", "0/1")
        num, den = map(int, rate.split("/")) if "/" in rate else (0, 1)
        if den == 0:
            return None
        fps = num / den
        return float(fps) if fps > 0 else None
    except Exception:
        return None
