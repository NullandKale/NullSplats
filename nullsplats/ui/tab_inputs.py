"""Inputs tab UI for NullSplats."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Dict, List, Optional
import time

from nullsplats.app_state import AppState
from nullsplats.backend.scene_manager import SceneManager
from nullsplats.backend.video_frames import ExtractionResult, extract_frames
from nullsplats.ui.tab_inputs_grid import InputsTabGridMixin
from nullsplats.ui.tab_inputs_scenes import InputsTabScenesMixin
from nullsplats.ui.tab_inputs_wizard import InputsTabWizardMixin
from nullsplats.util.logging import get_logger
from nullsplats.util.threading import run_in_background


class InputsTab(InputsTabScenesMixin, InputsTabGridMixin, InputsTabWizardMixin):
    """Inputs tab with scene creation, frame extraction, and selection UI."""

    def __init__(
        self,
        master: tk.Misc,
        app_state: AppState,
        on_scene_selected: Callable[[str], None],
        colmap_tab=None,
        training_tab=None,
        exports_tab=None,
        notebook: ttk.Notebook | None = None,
    ) -> None:
        self.app_state = app_state
        self.on_scene_selected = on_scene_selected
        self.colmap_tab = colmap_tab
        self.training_tab = training_tab
        self.exports_tab = exports_tab
        self.notebook = notebook
        self.logger = get_logger("ui.inputs")
        self.frame = ttk.Frame(master)

        self.status_var = tk.StringVar(value="Select or create a scene to begin.")
        self.status_label: Optional[ttk.Label] = None
        self.source_type_var = tk.StringVar(value="video")
        self.video_path_var = tk.StringVar()
        self.image_dir_var = tk.StringVar()
        self.candidate_var = tk.IntVar(value=100)
        self.target_var = tk.IntVar(value=50)
        self.training_resolution_var = tk.IntVar(value=getattr(app_state, "training_image_target_px", 1080))
        self.training_resample_var = tk.StringVar(value=getattr(app_state, "training_image_resample", "lanczos").lower())
        self.scene_entry: Optional[ttk.Entry] = None
        self._extracting = False
        self._saving = False
        self._interactive_controls: List[tk.Widget] = []

        self.frame_scores: Dict[str, float] = {}
        self.thumbnail_refs: List[tk.PhotoImage] = []
        self._thumbnail_job: Optional[str] = None
        self._thumb_queue: List[str] = []
        self._loaded_thumbs: set[str] = set()
        self._progress_bar: Optional[ttk.Progressbar] = None
        self._wizard_running = False
        self._card_height_px = 240
        self._grid_columns = 3
        self._thumb_cache: Dict[tuple[str, str], tk.PhotoImage] = {}
        self._card_width_px = 200
        self._tile_pool: list[dict] = []
        self._visible_tiles: Dict[int, dict] = {}
        self._virtual_items: list[str] = []
        self.selection_state: Dict[str, bool] = {}
        self.current_result: Optional[ExtractionResult] = None
        self._dirty_selection = False
        self._autosave_job: Optional[str] = None
        self.scene_manager: SceneManager = app_state.scene_manager
        self._pending_nav_to_training = False
        self._scene_images: list[tk.PhotoImage] = []
        self._scene_cards: dict[str, tk.Widget] = {}
        self._advanced_extract_frame: Optional[ttk.Frame] = None
        self._scene_thumb_pending: set[str] = set()
        self._scene_thumb_labels: dict[str, ttk.Label] = {}
        self._thumb_workers: int = 0
        self._max_thumb_workers: int = 4
        self._thumb_inflight: set[tuple[str, str]] = set()
        style = ttk.Style()
        try:
            style.configure("ActiveCard.TFrame", background="#dbeafe")
        except Exception:
            pass

        self._build_layout()
        self.logger.info("InputsTab layout built; refreshing scenes...")
        self.refresh_scenes()
        # Nudge focus to the source input so the user starts with extraction.
        self.frame.after(100, lambda: self.video_path_entry.focus_set() if hasattr(self, "video_path_entry") else None)

    def _register_control(self, widget: tk.Widget) -> None:
        """Track interactive widgets so we can disable/enable during long tasks."""
        self._interactive_controls.append(widget)
        widget.bind(
            "<Destroy>",
            lambda e: self._interactive_controls.remove(widget)
            if widget in self._interactive_controls
            else None,
        )

    def _build_layout(self) -> None:
        paned = ttk.Panedwindow(self.frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left_col = ttk.Frame(paned, width=340)
        grid_area = ttk.Frame(paned)
        paned.add(left_col, weight=2)
        paned.add(grid_area, weight=5)

        # Left: start here â€” source/extract, then scenes list beneath.
        self._build_action_stack(left_col)
        self._build_scene_sidebar(left_col)

        # Center: selection toolbar and grid.
        self._build_selection_controls(grid_area)
        self._build_grid(grid_area)

    def _build_action_stack(self, parent: tk.Misc) -> None:
        parent.pack_propagate(False)
        source_card = ttk.LabelFrame(parent, text="Step 1 — Input → new scene + extract")
        source_card.pack(fill="x", padx=8, pady=(10, 6))

        path_row = ttk.Frame(source_card)
        path_row.pack(fill="x", padx=6, pady=(6, 4))
        ttk.Label(path_row, text="Input (video or folder):").grid(row=0, column=0, sticky="w")
        self.video_path_entry = ttk.Entry(path_row, textvariable=self.video_path_var, width=38)
        self.video_path_entry.grid(row=0, column=1, sticky="ew", padx=(4, 4))
        btn_pick_video = ttk.Button(path_row, text="Choose video", command=self._choose_video)
        btn_pick_video.grid(row=0, column=2, sticky="e")
        self._register_control(btn_pick_video)
        btn_pick_folder = ttk.Button(path_row, text="Choose folder", command=self._choose_image_dir)
        btn_pick_folder.grid(row=1, column=2, sticky="e", pady=(4, 0))
        self._register_control(btn_pick_folder)
        path_row.columnconfigure(1, weight=1)

        wizard_row = ttk.Frame(source_card)
        wizard_row.pack(fill="x", padx=6, pady=(0, 4))
        btn_wizard = ttk.Button(wizard_row, text="Wizard mode", command=self._start_inline_wizard)
        btn_wizard.pack(side="left")
        self._register_control(btn_wizard)

        # Always show extraction settings.
        self._advanced_extract_frame = ttk.Frame(source_card)
        self._advanced_extract_frame.pack(fill="x", padx=6, pady=(4, 4))
        ttk.Label(self._advanced_extract_frame, text="Total").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(self._advanced_extract_frame, from_=1, to=10000, textvariable=self.candidate_var, width=7).grid(
            row=0, column=1, sticky="w", padx=(4, 8)
        )
        ttk.Label(self._advanced_extract_frame, text="Selected").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(self._advanced_extract_frame, from_=1, to=10000, textvariable=self.target_var, width=7).grid(
            row=0, column=3, sticky="w", padx=(4, 0)
        )

        actions = ttk.Frame(source_card)
        actions.pack(fill="x", padx=6, pady=(4, 6))
        btn_extract = ttk.Button(actions, text="Extract to new scene", command=self._start_extraction)
        btn_extract.pack(side="left")
        self._register_control(btn_extract)

        # Status label now lives in the center column for emphasis.


    def _build_selection_controls(self, parent: tk.Misc) -> None:
        controls = ttk.Frame(parent)
        controls.pack(fill="x", padx=10, pady=(10, 6))
        btn_all = ttk.Button(controls, text="Select All", command=self._select_all)
        btn_all.pack(side="left")
        self._register_control(btn_all)
        btn_none = ttk.Button(controls, text="Select None", command=self._select_none)
        btn_none.pack(side="left", padx=(6, 0))
        self._register_control(btn_none)
        btn_auto = ttk.Button(controls, text="Auto-Select Top N", command=self._auto_select)
        btn_auto.pack(side="left", padx=(6, 0))
        self._register_control(btn_auto)
        ttk.Label(controls, text="Resolution (px, small side):", padding=(12, 0)).pack(side="left")
        res_combo = ttk.Combobox(
            controls,
            textvariable=self.training_resolution_var,
            values=[720, 1080, 2160],
            state="readonly",
            width=8,
        )
        res_combo.pack(side="left", padx=(4, 4))
        res_combo.bind("<<ComboboxSelected>>", lambda _: self._on_resolution_change())
        mode_combo = ttk.Combobox(
            controls,
            textvariable=self.training_resample_var,
            values=["lanczos", "bicubic", "bilinear", "nearest"],
            state="readonly",
            width=10,
        )
        mode_combo.pack(side="left")
        self._register_control(res_combo)
        self._register_control(mode_combo)

        # Status + progress + continue CTA in center column.
        status_row = ttk.Frame(parent)
        status_row.pack(fill="x", padx=10, pady=(4, 6))
        self.status_label = ttk.Label(
            status_row, textvariable=self.status_var, foreground="#333", font=("Segoe UI", 11, "bold"), wraplength=620
        )
        self.status_label.pack(side="left", fill="x", expand=True)
        btn_continue = ttk.Button(
            status_row,
            text="Continue to COLMAP",
            command=self._go_to_colmap,
            style="Accent.TButton" if "Accent.TButton" in ttk.Style().element_names() else None,
        )
        btn_continue.pack(side="right", padx=(10, 0))
        self._register_control(btn_continue)

        progress_row = ttk.Frame(parent)
        progress_row.pack(fill="x", padx=10, pady=(0, 8))
        self._progress_bar = ttk.Progressbar(progress_row, mode="determinate", length=240)
        self._progress_bar.pack(fill="x", expand=True)

    def on_tab_selected(self, selected: bool) -> None:
        if not selected and self._dirty_selection:
            try:
                self._persist_selection()
            except Exception:
                pass

    def can_leave_tab(self) -> bool:
        """Block leaving the tab while extraction/saving runs; trigger save if dirty."""
        if self._extracting:
            self._set_status("Finish extraction before leaving this tab.", is_error=True)
            return False
        if self._saving:
            self._set_status("Saving selection... please wait.", is_error=False)
            return False
        if self._dirty_selection:
            self._set_status("Saving selection before switching tabs...")
            self._persist_selection()
            return False
        return True


    def refresh_scenes(self) -> None:
        """Refresh scenes asynchronously to avoid blocking UI."""
        self.logger.info("refresh_scenes: start")

        def _load() -> List[SceneStatus]:
            return self.app_state.refresh_scene_status()

        def _update(scenes: List[SceneStatus]) -> None:
            self.logger.info("refresh_scenes: update start count=%d", len(scenes))
            self._populate_listbox(scenes)
            self._sync_status()
            if self.app_state.current_scene_id is not None:
                self._highlight_active_scene(str(self.app_state.current_scene_id))
            self.logger.info("refresh_scenes: update done")

        run_in_background(
            _load,
            tk_root=self.frame,
            on_success=_update,
            on_error=lambda exc: self.logger.exception("Scene refresh failed"),
        )


    def _start_extraction(self) -> None:
        candidate_count = int(self.candidate_var.get())
        target_count = int(self.target_var.get())
        source_type = self.source_type_var.get()
        source_path = self.video_path_var.get().strip() if source_type == "video" else self.image_dir_var.get().strip()

        if not source_path:
            self._set_status("Select a video file or image folder first.", is_error=True)
            return
        scene_id = self._ensure_scene_for_source(source_path, force_new=True)
        if scene_id is None:
            return
        self._set_status(f"Preparing scene '{scene_id}' from the input and extracting frames...", is_error=False)
        paths = self.app_state.scene_manager.get(scene_id).paths
        if paths.frames_all_dir.exists() and any(paths.frames_all_dir.iterdir()):
            if not messagebox.askyesno(
                "Re-extract frames",
                "Existing frames will be replaced. Continue?",
                parent=self.frame.winfo_toplevel(),
            ):
                return
        self._begin_extraction(scene_id, source_path, source_type, candidate_count, target_count)





    def _handle_extraction_success(self, result: ExtractionResult) -> None:
        self._render_result(result)
        self.status_var.set(
            f"Extracted {len(result.available_frames)} frames; auto-selected {len(result.selected_frames)}."
        )
        try:
            # Warm thumbnails for the new extraction so they load quickly.
            self.app_state.scene_manager.thumbnails.start_warmup([str(result.scene_id)])
        except Exception:
            pass
        if self._progress_bar is not None:
            self._progress_bar["value"] = self._progress_bar["maximum"]
        self._extracting = False
        self._set_busy_ui(False)
        self._dirty_selection = True
        self._schedule_autosave("post_extraction", delay_ms=600)

    def _handle_error(self, exc: Exception) -> None:
        self.logger.exception("Inputs tab operation failed")
        self._set_status(f"Operation failed: {exc}", is_error=True)
        self._extracting = False
        self._saving = False
        self._autosave_job = None
        if self._progress_bar is not None:
            try:
                self._progress_bar.stop()
                self._progress_bar.config(mode="determinate", value=0)
            except Exception:
                self._progress_bar["value"] = 0
        self._set_busy_ui(False)













    # --- Virtual grid utilities ---










    def _set_status(self, message: str, *, is_error: bool = False) -> None:
        self.status_var.set(message)
        if self.status_label is not None:
            self.status_label.config(foreground="#a00" if is_error else "#333")

    def _update_progress(self, current: int, total: int) -> None:
        if self.status_label is not None and self.status_label.cget("foreground") != "#a00":
            self.status_var.set(f"Extracting frames... {current}/{total}")
        if self._progress_bar is not None:
            self._progress_bar["maximum"] = max(1, total)
            self._progress_bar["value"] = current

    def _set_busy_ui(self, busy: bool, message: str | None = None) -> None:
        try:
            cursor = "watch" if busy else ""
            self.frame.winfo_toplevel().config(cursor=cursor)
        except Exception:
            pass
        for widget in self._interactive_controls:
            try:
                widget_state = "disabled" if busy else "normal"
                widget.configure(state=widget_state)
            except Exception:
                continue
        if busy and message:
            self.status_var.set(message)

    def _sync_status(self) -> None:
        scene_text = (
            f"Active scene: {self.app_state.current_scene_id}"
            if self.app_state.current_scene_id is not None
            else "No active scene."
        )
        self.active_scene_label.config(text=scene_text)
        if self.status_label is not None and self.status_label.cget("foreground") != "#a00":
            selection_text = (
                f"Selected frames: {len(self._selected_frames())}" if self.selection_state else "Selected frames: 0"
            )
            self.status_var.set(selection_text)
        if self.current_result is None and hasattr(self, "canvas"):
            # Show an empty state placeholder when nothing is loaded.
            self.canvas.delete("placeholder")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="Pick or create a scene to view frames.\nUse the right-hand panel to choose a video or image folder.",
                anchor="center",
                justify="center",
                tags="placeholder",
            )

    def _schedule_autosave(self, reason: str = "selection", delay_ms: int = 400) -> None:
        self._dirty_selection = True
        if self._autosave_job is not None:
            try:
                self.frame.after_cancel(self._autosave_job)
            except Exception:
                pass
        self._autosave_job = self.frame.after(delay_ms, self._persist_selection)
        self.logger.info(
            "Autosave scheduled reason=%s delay_ms=%d selected=%d dirty=%s saving=%s",
            reason,
            delay_ms,
            len(self._selected_frames()),
            self._dirty_selection,
            self._saving,
        )

    def _on_resolution_change(self) -> None:
        try:
            target_px = int(self.training_resolution_var.get())
            if target_px > 0:
                self.app_state.training_image_target_px = target_px
        except Exception:
            pass
        resample = self.training_resample_var.get().lower().strip()
        if resample in {"lanczos", "bicubic", "bilinear", "nearest"}:
            self.app_state.training_image_resample = resample
        if self.selection_state:
            self._schedule_autosave("resolution_change")


    def _begin_extraction(
        self, scene_id: str, source_path: str, source_type: str, candidate_count: int, target_count: int
    ) -> None:
        if self._extracting:
            self._set_status("Extraction already running.", is_error=True)
            return
        self._set_busy_ui(True, "Extracting frames...")
        self._extracting = True
        if self._progress_bar is not None:
            try:
                self._progress_bar.stop()
            except Exception:
                pass
            self._progress_bar.config(mode="determinate")
            self._progress_bar["maximum"] = max(1, candidate_count)
            self._progress_bar["value"] = 0
        self.status_var.set("Extracting frames in background...")
        run_in_background(
            extract_frames,
            scene_id,
            source_path,
            source_type=source_type,
            candidate_count=candidate_count,
            target_count=target_count,
            cache_root=self.app_state.config.cache_root,
            progress_callback=self._update_progress,
            tk_root=self.frame.winfo_toplevel(),
            on_success=self._handle_extraction_success,
            on_error=self._handle_error,
            thread_name=f"extract_{scene_id}",
        )

    def _maybe_autofill_scene_from_path(self, path_str: str) -> None:
        if self.scene_entry is None:
            return
        current = self.scene_entry.get().strip()
        if current:
            return
        candidate = self.app_state.scene_manager.derive_scene_id_from_path(path_str)
        self.scene_entry.delete(0, tk.END)
        self.scene_entry.insert(0, candidate)
        self._set_status(f"Scene name set to '{candidate}' from input path.")

    def _ensure_scene_for_source(self, source_path: str, *, force_new: bool = False) -> Optional[str]:
        name = self.scene_entry.get().strip() if self.scene_entry else None
        if force_new:
            base = name or self.app_state.scene_manager.derive_scene_id_from_path(source_path)
            name = self._generate_unique_scene_name(base)
        try:
            scene = self.app_state.scene_manager.ensure_scene_for_source(
                source_path, self.source_type_var.get(), name or None
            )
        except ValueError as exc:
            self._set_status(f"Invalid Scene ID: {exc}", is_error=True)
            return None
        normalized = scene.scene_id
        self.on_scene_selected(str(normalized))  # type: ignore[arg-type]
        self.refresh_scenes()
        if self.scene_entry is not None:
            self.scene_entry.delete(0, tk.END)
            self.scene_entry.insert(0, str(normalized))
        self._set_status(f"Active scene set to {normalized}")
        return str(normalized)

    def _generate_unique_scene_name(self, base: str) -> str:
        """Return a scene name that does not collide with existing scenes."""
        existing = set(self.app_state.scene_manager.list_names())
        if base not in existing:
            return base
        suffix = 2
        candidate = f"{base}_{suffix}"
        while candidate in existing:
            suffix += 1
            candidate = f"{base}_{suffix}"
        return candidate

    def _current_source_path(self) -> str:
        if self.source_type_var.get() == "video":
            return self.video_path_var.get().strip()
        return self.image_dir_var.get().strip()

    def _handle_toggle_var(self, filename: str, var: tk.BooleanVar) -> None:
        self.selection_state[filename] = bool(var.get())
        self._dirty_selection = True
        self._sync_status()
        self._schedule_autosave("toggle")

    def _auto_load_cached_for_scene(self, scene_id: str) -> None:
        if self._extracting:
            return
        paths = self.app_state.scene_manager.get(scene_id).paths
        if paths.metadata_path.exists():
            self._load_cached()

    def _persist_selection(self) -> None:
        if self._autosave_job is not None:
            try:
                self.frame.after_cancel(self._autosave_job)
            except Exception:
                pass
            self._autosave_job = None
        if not self._dirty_selection:
            return
        if self._extracting:
            self._set_status("Waiting for extraction to finish before saving selection...")
            self._schedule_autosave("waiting_for_extraction", delay_ms=500)
            return
        scene_id = self._require_scene()
        if scene_id is None:
            return
        if self._saving:
            # Try again shortly after the current save finishes.
            self._schedule_autosave("waiting_for_save", delay_ms=300)
            return
        # Sync state without triggering another autosave.
        try:
            target_px = max(0, int(self.training_resolution_var.get()))
            if target_px > 0:
                self.app_state.training_image_target_px = target_px
        except Exception:
            target_px = self.app_state.training_image_target_px
        resample = self.training_resample_var.get().lower().strip()
        if resample in {"lanczos", "bicubic", "bilinear", "nearest"}:
            self.app_state.training_image_resample = resample
        else:
            resample = self.app_state.training_image_resample
        selected = self._selected_frames()
        if self.current_result and not self._dirty_selection and selected == self.current_result.selected_frames:
            return
        self.logger.info(
            "Persist selection start scene=%s selected=%d target_px=%d resample=%s dirty=%s",
            scene_id,
            len(selected),
            target_px,
            resample,
            self._dirty_selection,
        )
        self._saving = True
        self.status_var.set("Saving selection...")
        if self._progress_bar is not None:
            try:
                self._progress_bar.config(mode="indeterminate")
                self._progress_bar.start(10)
            except Exception:
                pass
        start = time.perf_counter()
        run_in_background(
            self._persist_selection_worker,
            scene_id,
            list(selected),
            target_px,
            resample,
            tk_root=self.frame.winfo_toplevel(),
            on_success=lambda res, st=start: self._handle_persist_success(res, st),
            on_error=self._handle_error,
            thread_name=f"persist_{scene_id}",
        )

    def _persist_selection_worker(self, scene_id: str, selected: List[str], target_px: int, resample: str) -> tuple:
        t0 = time.perf_counter()
        result, summary = self.scene_manager.save_selection(
            scene_id,
            selected,
            target_px=target_px,
            resample=resample,
        )
        t1 = time.perf_counter()
        return result, summary, (t1 - t0)

    def _handle_persist_success(self, payload: tuple, start: float) -> None:
        result, summary, elapsed = payload
        self._autosave_job = None
        self._saving = False
        self._render_result(result)
        total = time.perf_counter() - start
        if self._progress_bar is not None:
            try:
                self._progress_bar.stop()
                self._progress_bar.config(mode="determinate", value=0)
            except Exception:
                pass
        self.logger.info(
            "Persist selection timing: total=%.3fs copy+resize=%.3fs (processed=%d skipped=%d deleted=%d)",
            total,
            elapsed,
            summary.processed,
            summary.skipped,
            summary.deleted,
        )
        self.status_var.set(
            f"Saved {len(result.selected_frames)} frames in {total:.2f}s "
            f"(processed {summary.processed}, skipped {summary.skipped}, deleted {summary.deleted})."
        )
        self._dirty_selection = False
        self._set_busy_ui(False)
        if self._pending_nav_to_training:
            self._pending_nav_to_training = False
            self._navigate_to_colmap_tab()

    def _go_to_colmap(self) -> None:
        if self._extracting:
            self._set_status("Wait for extraction to finish before continuing.", is_error=True)
            return
        if self._saving:
            self._pending_nav_to_training = True
            self._set_status("Waiting for save to finish before continuing...")
            return
        if self.app_state.current_scene_id is None:
            self._set_status("Select or create a scene first.", is_error=True)
            return
        if not self._selected_frames():
            self._set_status("Select at least one frame, then continue.", is_error=True)
            return
        # Persist the current selection so Training uses it, then navigate.
        self._pending_nav_to_training = True
        self._persist_selection()
        if not self._dirty_selection and not self._saving:
            # Nothing to save; go now.
            self._pending_nav_to_training = False
            self._navigate_to_colmap_tab()











    def _navigate_to_colmap_tab(self) -> None:
        target_notebook = self.notebook or self.frame.master
        try:
            target_notebook.select(1)
        except Exception:
            pass

