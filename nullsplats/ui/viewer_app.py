"""Standalone viewer app for cached splat checkpoints."""

from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import List, Optional

from nullsplats.app_state import AppState
from nullsplats.ui.advanced_render_controls import AdvancedRenderSettingsPanel
from nullsplats.ui.colmap_camera_panel import ColmapCameraPanel
from nullsplats.ui.gl_canvas import GLCanvas
from nullsplats.ui.render_controls import RenderSettingsPanel
from nullsplats.optional_plugins import looking_glass_available
from nullsplats.util.config import AppConfig
from nullsplats.util.logging import get_logger
from nullsplats.util.threading import run_in_background


class ViewerApp:
    """Minimal viewer window for cached .splat/.ply outputs."""

    def __init__(self, cache_root: Path) -> None:
        self.logger = get_logger("ui.viewer")
        self.cache_root = Path(cache_root)
        self.app_state = AppState(config=AppConfig(cache_root=self.cache_root))
        self._current_scene_id: Optional[str] = None
        self._tab_active = True
        self._pending_preview_path: Optional[Path] = None
        self._entries: List[Path] = []
        self.root = tk.Tk()
        title_root = self.cache_root.resolve()
        self.root.title(f"NullSplats Viewer - {title_root}")
        self.root.minsize(1000, 650)
        self.lkg_status_var = tk.StringVar(value="Looking Glass: not available")
        self.lkg_detail_var = tk.StringVar(value="")
        self.lkg_depthiness_var = tk.DoubleVar(value=1.0)
        self.lkg_focus_var = tk.DoubleVar(value=2.0)
        self.lkg_fov_var = tk.DoubleVar(value=14.0)
        self.lkg_viewcone_var = tk.DoubleVar(value=40.0)
        self.lkg_zoom_var = tk.DoubleVar(value=1.0)
        self._lkg_status_job: Optional[str] = None
        self._lkg_apply_job: Optional[str] = None
        self._lkg_enabled = looking_glass_available()
        self._build_ui()
        self._refresh_list()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        paned = ttk.Panedwindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)

        right_col = ttk.Frame(paned)
        left_col = ttk.Frame(paned)
        paned.add(right_col, weight=3)
        paned.add(left_col, weight=1)

        ttk.Label(left_col, text="Cached outputs", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=10, pady=(10, 4)
        )

        list_frame = ttk.LabelFrame(left_col, text="Checkpoints (.splat / .ply)")
        list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        list_inner = ttk.Frame(list_frame)
        list_inner.pack(fill="both", expand=True, padx=6, pady=6)
        self.checkpoint_list = tk.Listbox(list_inner, height=16, exportselection=False)
        self.checkpoint_list.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(list_inner, orient="vertical", command=self.checkpoint_list.yview)
        scroll.pack(side="right", fill="y")
        self.checkpoint_list.config(yscrollcommand=scroll.set)
        self.checkpoint_list.bind("<<ListboxSelect>>", lambda _event: self._preview_selected())

        buttons = ttk.Frame(list_frame)
        buttons.pack(fill="x", padx=6, pady=(4, 2))
        ttk.Button(buttons, text="Refresh list", command=self._refresh_list).pack(side="left", padx=(0, 4))
        ttk.Button(buttons, text="Preview selected", command=self._preview_selected).pack(side="left", padx=(0, 4))
        ttk.Button(buttons, text="Open folder", command=self._open_selected_folder).pack(side="right")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(left_col, textvariable=self.status_var, foreground="#444").pack(
            anchor="w", padx=10, pady=(0, 10)
        )

        preview_frame = ttk.LabelFrame(right_col, text="Preview")
        preview_frame.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        header = ttk.Frame(preview_frame)
        header.pack(fill="x", padx=6, pady=(6, 2))
        self.preview_note_var = tk.StringVar(value="Preview idle.")
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
            scene_getter=lambda: self._current_scene_id,
            paths_getter=lambda scene: self.app_state.scene_manager.get(scene).paths,
        )
        self.colmap_panel.pack(fill="both", expand=True, padx=4, pady=(4, 4))
        if self._lkg_enabled:
            lkg_tab = ttk.Frame(notebook)
            notebook.add(lkg_tab, text="Looking Glass")
            self._build_lkg_panel(lkg_tab)

    def _open_selected_folder(self) -> None:
        path = self._selected_checkpoint()
        if path is None:
            self.status_var.set("Select a checkpoint to open its folder.")
            return
        try:
            os.startfile(str(path.parent))
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to open folder")
            self.status_var.set(f"Open folder failed: {exc}")

    def _scan_cache_outputs(self) -> List[Path]:
        outputs_root = self.cache_root / "outputs"
        if not outputs_root.exists():
            return []
        candidates: List[Path] = []
        for path in outputs_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".splat", ".ply"}:
                candidates.append(path)
        return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)

    def _refresh_list(self) -> None:
        self.status_var.set("Scanning cache outputs...")

        def _do_scan() -> List[Path]:
            return self._scan_cache_outputs()

        def _on_success(paths: List[Path]) -> None:
            self._entries = paths
            self.checkpoint_list.delete(0, tk.END)
            for path in paths:
                label = self._display_label(path)
                self.checkpoint_list.insert(tk.END, label)
            if paths:
                self.checkpoint_list.selection_set(0)
                self._pending_preview_path = paths[0]
            else:
                self._pending_preview_path = None
            self.status_var.set(f"Found {len(paths)} checkpoint(s) under cache outputs.")
            self.preview_note_var.set("Preview idle." if paths else "No checkpoints yet.")
            if paths:
                self._preview_selected()

        def _on_error(exc: Exception) -> None:
            self.logger.exception("Cache scan failed")
            self.status_var.set(f"Scan failed: {exc}")

        run_in_background(
            _do_scan,
            tk_root=self.root,
            on_success=_on_success,
            on_error=_on_error,
            thread_name="viewer_cache_scan",
        )

    def _display_label(self, path: Path) -> str:
        scene_id = self._scene_from_path(path)
        if scene_id:
            return f"{scene_id} / {path.name}"
        try:
            rel = path.relative_to(self.cache_root)
            return str(rel)
        except ValueError:
            return str(path)

    def _scene_from_path(self, path: Path) -> Optional[str]:
        try:
            rel = path.relative_to(self.cache_root)
        except ValueError:
            return None
        if len(rel.parts) >= 2 and rel.parts[0] == "outputs":
            return rel.parts[1]
        return None

    def _selected_checkpoint(self) -> Optional[Path]:
        if not self._entries:
            return None
        selection = self.checkpoint_list.curselection()
        if not selection:
            return None
        return self._entries[selection[0]]

    def _preview_selected(self) -> None:
        checkpoint = self._selected_checkpoint()
        if checkpoint is None:
            self._pending_preview_path = None
            self.status_var.set("Select a checkpoint to preview.")
            return
        self._pending_preview_path = checkpoint
        self._current_scene_id = self._scene_from_path(checkpoint)
        try:
            if hasattr(self, "colmap_panel"):
                self.colmap_panel.refresh()
        except FileNotFoundError:
            self.logger.debug("COLMAP panel refresh skipped; scene metadata missing.")
        self._start_preview(checkpoint)

    def _start_preview(self, checkpoint: Path, *, retry: bool = True) -> None:
        if self.viewer is None:
            return
        if not self.viewer.winfo_ismapped():
            if retry:
                self.logger.debug("Viewer not mapped yet; retrying preview for %s", checkpoint)
                self.root.after(100, lambda: self._start_preview(checkpoint, retry=False))
            else:
                self.logger.warning("Viewer still not mapped; skipping preview for %s", checkpoint)
            return
        try:
            self.viewer.start_rendering()
            self.viewer.load_splat(checkpoint)
            self.status_var.set(f"Previewing {checkpoint.name}")
            self.preview_note_var.set(f"Previewing {checkpoint.name}")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed to preview %s", checkpoint)
            self.status_var.set(f"Preview failed: {exc}")
            self.preview_note_var.set(f"Preview failed: {exc}")

    def _on_close(self) -> None:
        try:
            if self.viewer is not None:
                self.viewer.stop_rendering()
        except Exception:  # noqa: BLE001
            self.logger.exception("Failed to stop viewer on close")
        try:
            if self._lkg_status_job is not None:
                self.root.after_cancel(self._lkg_status_job)
        except Exception:
            pass
        self.root.destroy()

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

    def run(self) -> None:
        self.root.mainloop()


def run_viewer_app(cache_root: Optional[Path] = None) -> None:
    app = ViewerApp(cache_root or Path("cache"))
    app.run()


__all__ = ["run_viewer_app", "ViewerApp"]
