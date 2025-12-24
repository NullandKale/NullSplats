"""Exports tab with checkpoint listing, preview, and turntable rendering."""

from __future__ import annotations

import math
import os
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional
import logging

import imageio
import numpy as np

from nullsplats.app_state import AppState
from nullsplats.ui.advanced_render_controls import AdvancedRenderSettingsPanel
from nullsplats.ui.colmap_camera_panel import ColmapCameraPanel
from nullsplats.ui.gl_canvas import CameraView, GLCanvas
from nullsplats.ui.render_controls import RenderSettingsPanel
from nullsplats.util.logging import get_logger
from nullsplats.util.threading import run_in_background


class ExportsTab:
    """Manage exported checkpoints and previews for the active scene."""

    def __init__(self, master: tk.Misc, app_state: AppState) -> None:
        self.logger = get_logger("ui.exports")
        self.logger.setLevel(logging.DEBUG)
        self.app_state = app_state
        self.frame = ttk.Frame(master)
        self.scene_var = tk.StringVar(value=str(app_state.current_scene_id or ""))
        self.export_dir_var = tk.StringVar(value=str(Path("exports")))
        self.status_var = tk.StringVar(value="Select a scene to list checkpoints.")
        self.preview_note_var = tk.StringVar(value="Preview idle.")
        self.checkpoint_paths: List[Path] = []
        self.viewer: Optional[GLCanvas] = None
        self._tab_active = False
        self._pending_preview_path: Optional[Path] = None

        self._build_contents()
        self._refresh_scenes()
        self._load_checkpoints()

    def _build_contents(self) -> None:
        paned = ttk.Panedwindow(self.frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        right_col = ttk.Frame(paned)
        left_col = ttk.Frame(paned)
        paned.add(right_col, weight=3)
        paned.add(left_col, weight=1)

        ttk.Label(left_col, text="Exports and viewer", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=10, pady=(10, 4)
        )

        scene_row = ttk.Frame(left_col)
        scene_row.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Label(scene_row, text="Scene:").pack(side="left")
        self.scene_combo = ttk.Combobox(scene_row, textvariable=self.scene_var, state="readonly", width=30)
        self.scene_combo.pack(side="left", padx=(4, 8))
        ttk.Button(scene_row, text="Refresh scenes", command=self._refresh_scenes).pack(side="left")

        list_frame = ttk.LabelFrame(left_col, text="Checkpoints (.ply)")
        list_frame.pack(fill="both", expand=False, padx=10, pady=(0, 8))
        list_inner = ttk.Frame(list_frame)
        list_inner.pack(fill="both", expand=True, padx=6, pady=6)
        self.checkpoint_list = tk.Listbox(list_inner, height=8, exportselection=False)
        self.checkpoint_list.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(list_inner, orient="vertical", command=self.checkpoint_list.yview)
        scroll.pack(side="right", fill="y")
        self.checkpoint_list.config(yscrollcommand=scroll.set)
        self.checkpoint_list.bind("<<ListboxSelect>>", lambda _event: self._preview_selected())

        buttons = ttk.Frame(list_frame)
        buttons.pack(fill="x", padx=6, pady=(4, 2))
        ttk.Button(buttons, text="Refresh list", command=self._load_checkpoints).pack(side="left", padx=(0, 4))
        ttk.Button(buttons, text="Preview selected", command=self._preview_selected).pack(side="left", padx=(0, 4))
        ttk.Button(buttons, text="Open splats folder", command=self._open_splats_folder).pack(side="right")

        export_frame = ttk.LabelFrame(left_col, text="Export actions")
        export_frame.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(export_frame, text="Export directory:").grid(row=0, column=0, sticky="w", padx=(6, 4), pady=(6, 4))
        ttk.Entry(export_frame, textvariable=self.export_dir_var, width=50).grid(row=0, column=1, sticky="ew", pady=(6, 4))
        ttk.Button(export_frame, text="Browse", command=self._choose_export_dir).grid(row=0, column=2, sticky="w", padx=6, pady=(6, 4))
        ttk.Button(export_frame, text="Copy selected .ply", command=self._export_checkpoint).grid(row=1, column=0, sticky="w", padx=6, pady=(0, 6))
        ttk.Button(export_frame, text="Render turntable.mp4", command=self._render_turntable).grid(row=1, column=1, sticky="w", padx=(0, 6), pady=(0, 6))
        export_frame.columnconfigure(1, weight=1)

        preview_frame = ttk.LabelFrame(right_col, text="Preview")
        preview_frame.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        header = ttk.Frame(preview_frame)
        header.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(header, textvariable=self.preview_note_var, foreground="#444").pack(side="left")
        ttk.Button(header, text="Refresh preview", command=self._preview_selected).pack(side="right")
        preview_inner = ttk.Frame(preview_frame)
        preview_inner.pack(fill="both", expand=True, padx=6, pady=(6, 4))
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
            scene_getter=lambda: self.app_state.current_scene_id,
            paths_getter=lambda scene: self.app_state.scene_manager.get(scene).paths,
        )
        self.colmap_panel.pack(fill="both", expand=True, padx=4, pady=(4, 4))
        notebook.select(camera_tab)

        ttk.Label(left_col, textvariable=self.status_var, foreground="#444").pack(
            anchor="w", padx=10, pady=(0, 10)
        )

    def deactivate_viewer(self) -> None:
        """Stop rendering and clear the exports viewer when hidden."""
        if self.viewer is not None:
            try:
                self.viewer.stop_rendering()
            except Exception:  # noqa: BLE001
                self.logger.exception("Failed to deactivate exports viewer")

    def on_tab_selected(self, selected: bool) -> None:
        self._tab_active = selected
        if self.viewer is None:
            return
        if selected:
            # Only spin up the renderer when the tab is visible.
            self.logger.debug("Exports tab selected; pending preview=%s", self._pending_preview_path)
            if self._pending_preview_path:
                pending = self._pending_preview_path
                self.frame.after(150, lambda path=pending: self._start_preview(path))
        else:
            self.logger.debug("Exports tab hidden; stopping renderer")
            try:
                self.viewer.stop_rendering()
            except Exception:  # noqa: BLE001
                self.logger.exception("Failed to stop exports viewer")

    def _refresh_scenes(self) -> None:
        scenes = [str(status.scene_id) for status in self.app_state.scene_manager.list_scenes()]
        self.scene_combo["values"] = scenes
        current = self.app_state.current_scene_id
        if current is None and scenes:
            current = scenes[0]
            self.app_state.set_current_scene(current)
        if current is not None:
            self.scene_var.set(str(current))
        self.status_var.set("Scene list refreshed.")

    def _load_checkpoints(self) -> None:
        scene_id = self.scene_var.get().strip()
        if not scene_id:
            self.status_var.set("No scene selected.")
            return
        self.app_state.set_current_scene(scene_id)
        paths = self.app_state.scene_manager.get(scene_id).paths
        splat_dir = paths.splats_dir
        if not splat_dir.exists():
            splat_dir.mkdir(parents=True, exist_ok=True)
        checkpoints = sorted(
            [p for p in splat_dir.iterdir() if p.suffix.lower() == ".ply"],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        self.checkpoint_paths = checkpoints
        self.checkpoint_list.delete(0, tk.END)
        for path in checkpoints:
            self.checkpoint_list.insert(tk.END, path.name)
        if checkpoints:
            self.checkpoint_list.selection_set(0)
            self._pending_preview_path = checkpoints[0]
            if self._tab_active:
                self._preview_selected()
        else:
            self._pending_preview_path = None
        self.preview_note_var.set("Preview idle." if checkpoints else "No checkpoints yet.")
        self.status_var.set(f"Found {len(checkpoints)} checkpoint(s) in {splat_dir}.")
        if hasattr(self, "colmap_panel"):
            self.colmap_panel.refresh()

    def _selected_checkpoint(self) -> Optional[Path]:
        if not self.checkpoint_paths:
            return None
        selection = self.checkpoint_list.curselection()
        if not selection:
            return None
        return self.checkpoint_paths[selection[0]]

    def _preview_selected(self) -> None:
        checkpoint = self._selected_checkpoint()
        if checkpoint is None:
            self._pending_preview_path = None
            self.status_var.set("Select a checkpoint to preview.")
            return
        self._pending_preview_path = checkpoint
        if not self._tab_active:
            # Defer expensive renderer startup until the tab is visible.
            self.logger.debug("Deferring preview for %s until Exports tab is active", checkpoint)
            return
        self._start_preview(checkpoint)

    def _start_preview(self, checkpoint: Path, *, retry: bool = True) -> None:
        """Start rendering/preview after ensuring the viewer is mapped."""
        if not self._tab_active:
            return
        if self.viewer is None:
            return
        if not self.viewer.winfo_ismapped():
            if retry:
                self.logger.debug("Viewer not mapped yet; retrying preview for %s", checkpoint)
                self.frame.after(100, lambda: self._start_preview(checkpoint, retry=False))
            else:
                self.logger.warning("Viewer still not mapped; skipping preview for %s", checkpoint)
            return
        try:
            self.viewer.start_rendering()
            self.viewer.load_splat(checkpoint)
            self.logger.info(
                "Exports preview request path=%s last_path=%s",
                checkpoint,
                self.viewer.last_path,
            )
            self.status_var.set(f"Previewing {checkpoint.name}")
            self.preview_note_var.set(f"Previewing {checkpoint.name}")
            self.logger.info("Preview loaded: %s", checkpoint)
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to preview %s", checkpoint)
            self.status_var.set(f"Preview failed: {exc}")
            self.preview_note_var.set(f"Preview failed: {exc}")

    def _choose_export_dir(self) -> None:
        chosen = filedialog.askdirectory()
        if chosen:
            self.export_dir_var.set(chosen)

    def _export_checkpoint(self) -> None:
        checkpoint = self._selected_checkpoint()
        if checkpoint is None:
            self.status_var.set("Select a checkpoint to export.")
            return
        export_dir = Path(self.export_dir_var.get().strip())
        if not export_dir:
            self.status_var.set("Provide a target export directory.")
            return
        export_dir.mkdir(parents=True, exist_ok=True)
        target = export_dir / checkpoint.name
        shutil.copy2(checkpoint, target)
        self.status_var.set(f"Copied to {target}")
        self.logger.info("Exported checkpoint %s to %s", checkpoint, target)

    def _render_turntable(self) -> None:
        checkpoint = self._selected_checkpoint()
        if checkpoint is None:
            self.status_var.set("Select a checkpoint before rendering a turntable.")
            return
        scene_id = self.scene_var.get().strip()
        if not scene_id:
            self.status_var.set("Select a scene before rendering a turntable.")
            return
        if self.viewer is None:
            self.status_var.set("Viewer not initialized.")
            return
        paths = self.app_state.scene_manager.get(scene_id).paths
        render_dir = paths.renders_dir
        render_dir.mkdir(parents=True, exist_ok=True)
        turntable_path = render_dir / "turntable.mp4"
        export_dir = Path(self.export_dir_var.get().strip()) if self.export_dir_var.get().strip() else None

        def _build_turntable() -> Path:
            renderer = self.viewer.renderer
            data = renderer.data
            if data is None or self.viewer.last_path != checkpoint:
                data = renderer.load(checkpoint)
                self.logger.info("Turntable loaded %s for rendering", checkpoint)
            base_view = CameraView(
                yaw=0.0,
                pitch=0.2,
                distance=max(data.radius * 4.0, 1.0),
                target=data.center,
            )
            width = max(1, int(self.viewer.canvas.winfo_width()))
            height = max(1, int(self.viewer.canvas.winfo_height()))
            frames = 60
            self.logger.info(
                "Turntable loop start scene=%s checkpoint=%s frames=%d size=%sx%s",
                scene_id,
                checkpoint,
                frames,
                width,
                height,
            )
            with imageio.get_writer(turntable_path, fps=24) as writer:
                for idx in range(frames):
                    yaw = base_view.yaw + (2.0 * math.pi * idx / frames)
                    cam_view = CameraView(
                        yaw=yaw,
                        pitch=base_view.pitch,
                        distance=base_view.distance,
                        target=base_view.target,
                    )
                    img = renderer.render(width, height, cam_view)
                    writer.append_data(np.array(img))
                    self.logger.info(
                        "Turntable frame cycle scene=%s idx=%d/%d", scene_id, idx + 1, frames
                    )
            self.logger.info("Turntable loop stop scene=%s output=%s", scene_id, turntable_path)
            if export_dir:
                export_dir.mkdir(parents=True, exist_ok=True)
                export_target = export_dir / turntable_path.name
                shutil.copy2(turntable_path, export_target)
                self.logger.info("Turntable copied to %s", export_target)
            return turntable_path

        def _on_success(path: Path) -> None:
            self.status_var.set(f"Turntable rendered to {path}")

        def _on_error(exc: Exception) -> None:
            self.logger.exception("Turntable rendering failed")
            self.status_var.set(f"Turntable failed: {exc}")
            messagebox.showerror("Turntable failed", str(exc))

        run_in_background(
            _build_turntable,
            tk_root=self.frame,
            on_success=_on_success,
            on_error=_on_error,
            thread_name=f"turntable_{scene_id}",
        )

    def _open_splats_folder(self) -> None:
        scene_id = self.scene_var.get().strip()
        if not scene_id:
            self.status_var.set("Select a scene to open its splats folder.")
            return
        paths = self.app_state.scene_manager.get(scene_id).paths
        folder = paths.splats_dir
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(folder))
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to open splats folder")
            self.status_var.set(f"Open folder failed: {exc}")

    def on_scene_changed(self, scene_id: str | None) -> None:
        if scene_id is not None:
            self.scene_var.set(str(scene_id))
            self.app_state.set_current_scene(scene_id)
        self._load_checkpoints()
