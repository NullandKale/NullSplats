"""Inputs tab UI for NullSplats."""

from __future__ import annotations

import io
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter.simpledialog as simpledialog
from typing import Callable, Dict, List, Optional, Tuple
import time

from PIL import Image, ImageTk

from nullsplats.app_state import AppState, SceneStatus
from nullsplats.backend.scene_manager import SceneManager
from nullsplats.backend.video_frames import ExtractionResult, FrameScore, auto_select_best, extract_frames
from nullsplats.backend.io_cache import load_metadata
from nullsplats.util.logging import get_logger
from nullsplats.util.threading import run_in_background


class InputsTab:
    """Inputs tab with scene creation, frame extraction, and selection UI."""

    def __init__(
        self,
        master: tk.Misc,
        app_state: AppState,
        on_scene_selected: Callable[[str], None],
        training_tab=None,
        exports_tab=None,
        notebook: ttk.Notebook | None = None,
    ) -> None:
        self.app_state = app_state
        self.on_scene_selected = on_scene_selected
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
        self._progress_total: int = 0
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
        self._timers: Dict[str, float] = {}
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
        btn_pick_input = ttk.Button(path_row, text="Choose input…", command=self._choose_input)
        btn_pick_input.grid(row=0, column=2, sticky="e")
        self._register_control(btn_pick_input)
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

    def _build_scene_sidebar(self, parent: tk.Misc) -> None:
        ttk.Label(parent, text="Scenes", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=8, pady=(8, 4))
        name_row = ttk.Frame(parent)
        name_row.pack(fill="x", padx=8, pady=(0, 6))
        self.scene_entry = ttk.Entry(name_row)
        self.scene_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        btn_create = ttk.Button(name_row, text="Create/Set", command=self._create_scene)
        btn_create.pack(side="right")
        self._register_control(btn_create)

        list_frame = ttk.Frame(parent)
        list_frame.pack(fill="both", expand=True, padx=8, pady=(0, 6))
        self.scene_canvas = tk.Canvas(list_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.scene_canvas.yview)
        self.scene_cards_container = ttk.Frame(self.scene_canvas)
        self.scene_canvas_window = self.scene_canvas.create_window((0, 0), window=self.scene_cards_container, anchor="nw")
        self.scene_cards_container.bind(
            "<Configure>", lambda _: self.scene_canvas.configure(scrollregion=self.scene_canvas.bbox("all"))
        )
        self.scene_canvas.bind(
            "<Configure>", lambda e: self.scene_canvas.itemconfigure(self.scene_canvas_window, width=e.width)
        )
        self.scene_canvas.configure(yscrollcommand=scrollbar.set, height=340)
        self.scene_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        actions = ttk.Frame(parent)
        actions.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Button(actions, text="Refresh", command=self.refresh_scenes).pack(side="left")
        manage_actions = ttk.Frame(actions)
        manage_actions.pack(side="right")
        reextract_btn = ttk.Button(manage_actions, text="Re-extract selected", command=self._reextract_selected_scene)
        reextract_btn.pack(side="left")
        self._register_control(reextract_btn)
        delete_btn = ttk.Button(manage_actions, text="Delete selected", command=self._delete_scene)
        delete_btn.pack(side="left", padx=(6, 0))
        self._register_control(delete_btn)

        self.active_scene_label = ttk.Label(parent, text="No active scene.", foreground="#444")
        self.active_scene_label.pack(anchor="w", padx=8, pady=(0, 4))

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
            text="Continue to Training",
            command=self._go_to_training,
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

    def _build_grid(self, parent: tk.Misc) -> None:
        container = ttk.LabelFrame(parent, text="Frame selection")
        container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.canvas = tk.Canvas(container, borderwidth=0, height=360)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self._scrollbar_set = scrollbar.set
        self.grid_inner = ttk.Frame(self.canvas)
        self._canvas_window = self.canvas.create_window((0, 0), window=self.grid_inner, anchor="nw")
        self.grid_inner.bind("<Configure>", lambda _: self._update_scrollregion())
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.configure(yscrollcommand=self._on_canvas_scroll)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

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

    def _populate_listbox(self, scenes: List[SceneStatus]) -> None:
        # Clear existing cards.
        start = time.perf_counter()
        for child in self.scene_cards_container.winfo_children():
            child.destroy()
        self._scene_images.clear()
        self._scene_cards.clear()
        self._scene_thumb_labels.clear()
        self._scene_thumb_pending.clear()
        active = str(self.app_state.current_scene_id) if self.app_state.current_scene_id is not None else None

        for status in scenes:
            sid = str(status.scene_id)
            card = ttk.Frame(self.scene_cards_container, padding=6, relief="groove", borderwidth=1)
            card.pack(fill="x", padx=2, pady=4)
            card.bind("<Button-1>", lambda _e, scene_id=sid: self._select_scene_card(scene_id))
            thumb_lbl = ttk.Label(card, text="Preview", width=12)
            thumb_lbl.grid(row=0, column=0, rowspan=2, sticky="nw", padx=(0, 8))
            self._scene_thumb_labels[sid] = thumb_lbl

            title = ttk.Label(card, text=sid, font=("Segoe UI", 10, "bold"))
            title.grid(row=0, column=1, sticky="w")
            summary = self._scene_summary(sid, status)
            ttk.Label(card, text=summary, foreground="#555").grid(row=1, column=1, sticky="w", pady=(2, 0))

            if active == sid:
                card.configure(style="ActiveCard.TFrame")
            self._scene_cards[sid] = card
            self._request_scene_thumbnail(sid)

        self.scene_cards_container.update_idletasks()
        elapsed = (time.perf_counter() - start) * 1000.0
        self.logger.info(
            "_populate_listbox done count=%d active=%s queued_thumbs=%d elapsed_ms=%.1f",
            len(scenes),
            active,
            len(self._scene_thumb_pending),
            elapsed,
        )

    def _handle_selection(self, _: object) -> None:
        # Deprecated; card selection handled separately.
        return

    def _select_scene_card(self, scene_id: str) -> None:
        normalized = self.app_state.set_current_scene(scene_id)
        self.on_scene_selected(str(normalized))  # type: ignore[arg-type]
        if self.scene_entry is not None:
            self.scene_entry.delete(0, tk.END)
            self.scene_entry.insert(0, str(normalized))
        self._sync_status()
        self._auto_load_cached_for_scene(str(normalized))
        self._highlight_active_scene(str(normalized))

    def _highlight_active_scene(self, scene_id: str) -> None:
        for sid, card in self._scene_cards.items():
            try:
                card.configure(style="TFrame", relief="groove")
            except Exception:
                pass
        if scene_id in self._scene_cards:
            try:
                self._scene_cards[scene_id].configure(style="ActiveCard.TFrame", relief="solid")
            except Exception:
                pass

    def _create_scene(self) -> None:
        source_path = self._current_source_path()
        name = self.scene_entry.get().strip() if self.scene_entry else ""
        if not name and not source_path:
            self._set_status("Pick a video or image folder first, or enter a scene name.", is_error=True)
            return
        try:
            scene = self.app_state.scene_manager.ensure_scene_for_source(
                source_path or None, self.source_type_var.get(), name or None
            )
            if not name and self.scene_entry is not None:
                self.scene_entry.delete(0, tk.END)
                self.scene_entry.insert(0, str(scene.scene_id))
        except ValueError as exc:
            self._set_status(f"Invalid Scene ID: {exc}", is_error=True)
            return
        normalized = scene.scene_id
        self.on_scene_selected(str(normalized))  # type: ignore[arg-type]
        self.refresh_scenes()
        self._set_status(f"Active scene set to {normalized}")

    def _delete_scene(self, scene_id: Optional[str | SceneId] = None) -> None:
        target = scene_id or self.app_state.current_scene_id
        if target is None:
            self._set_status("Select a scene before deleting.", is_error=True)
            return
        self.app_state.scene_manager.delete(target)
        self.refresh_scenes()
        self._clear_grid()
        self._set_status(f"Deleted scene {target}.")

    def _reextract_selected_scene(self) -> None:
        """Re-run extraction for the active scene using its saved source path."""
        if self._extracting:
            self._set_status("Extraction already running.", is_error=True)
            return
        scene_id = self._require_scene()
        if scene_id is None:
            return
        try:
            metadata = load_metadata(scene_id, cache_root=self.app_state.config.cache_root)
        except FileNotFoundError:
            self._set_status("No metadata for this scene. Pick the input and extract again.", is_error=True)
            return
        source_path = metadata.get("source_path")
        source_type = metadata.get("source_type", "video")
        if not source_path:
            self._set_status("Scene is missing its source path. Pick the input and extract again.", is_error=True)
            return
        if not Path(source_path).exists():
            self._set_status("Saved source path is missing on disk. Pick a new input and extract.", is_error=True)
            return

        # Update UI fields to mirror the source being used.
        if source_type == "video":
            self.video_path_var.set(source_path)
            self.image_dir_var.set("")
        else:
            self.image_dir_var.set(source_path)
            self.video_path_var.set("")
        self.source_type_var.set(source_type)
        candidate_count = int(self.candidate_var.get())
        target_count = int(self.target_var.get())

        paths = self.app_state.scene_manager.get(scene_id).paths
        if paths.frames_all_dir.exists() and any(paths.frames_all_dir.iterdir()):
            if not messagebox.askyesno(
                "Re-extract frames",
                "Existing frames will be replaced. Continue?",
                parent=self.frame.winfo_toplevel(),
            ):
                return
        self._set_status(f"Re-extracting frames for scene {scene_id}...", is_error=False)
        self._begin_extraction(scene_id, source_path, source_type, candidate_count, target_count)

    def _choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if path:
            self.video_path_var.set(path)
            self.source_type_var.set("video")
            self._maybe_autofill_scene_from_path(path)
            self._sync_status()
            self._set_status("Input ready. Click Extract to create a new scene.", is_error=False)

    def _choose_image_dir(self) -> None:
        path = filedialog.askdirectory(title="Select image folder")
        if path:
            self.image_dir_var.set(path)
            self.source_type_var.set("images")
            self._maybe_autofill_scene_from_path(path)
            self._sync_status()
            self._set_status("Input ready. Click Extract to create a new scene.", is_error=False)

    def _choose_input(self) -> None:
        """Single entry point to pick a video file or image folder."""
        file_path = filedialog.askopenfilename(
            title="Select video file (or cancel to pick a folder)",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if file_path:
            self.video_path_var.set(file_path)
            self.image_dir_var.set("")
            self.source_type_var.set("video")
            self._maybe_autofill_scene_from_path(file_path)
            self._sync_status()
            self._set_status("Input ready. Click Extract to create a new scene.", is_error=False)
            return
        folder_path = filedialog.askdirectory(title="Select image folder")
        if folder_path:
            self.image_dir_var.set(folder_path)
            self.video_path_var.set("")
            self.source_type_var.set("images")
            self._maybe_autofill_scene_from_path(folder_path)
            self._sync_status()
            self._set_status("Input ready. Click Extract to create a new scene.", is_error=False)

    def _require_scene(self) -> Optional[str]:
        if self.app_state.current_scene_id is None:
            self._set_status("Create or select a scene before extracting frames.", is_error=True)
            return None
        return str(self.app_state.current_scene_id)

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

    def _load_cached(self) -> None:
        scene_id = self._require_scene()
        if scene_id is None:
            return
        self._set_busy_ui(True, "Loading cached frames...")
        if self._progress_bar is not None:
            try:
                self._progress_bar.config(mode="indeterminate")
                self._progress_bar.start(10)
            except Exception:
                pass
        start = time.perf_counter()
        self.status_var.set("Loading cached frames...")
        run_in_background(
            self._load_cached_worker,
            scene_id,
            tk_root=self.frame.winfo_toplevel(),
            on_success=lambda res, st=start: self._handle_load_cached_success(res, st),
            on_error=self._handle_error,
            thread_name=f"load_cached_{scene_id}",
        )

    def _load_cached_for_scene(self, scene_id: str) -> None:
        self.app_state.set_current_scene(scene_id)
        self._highlight_active_scene(scene_id)
        self._load_cached()

    def _load_cached_worker(self, scene_id: str) -> ExtractionResult:
        # Load metadata and ensure frames exist, but avoid re-copying/downscaling during load to keep it fast.
        return self.app_state.scene_manager.load_cached_frames(scene_id)

    def _handle_load_cached_success(self, result: ExtractionResult, start: float) -> None:
        self._render_result(result)
        elapsed = time.perf_counter() - start
        self.logger.info("Load cached timing: total=%.3fs", elapsed)
        self.status_var.set(f"Loaded cached frames and selection in {elapsed:.2f}s.")
        try:
            # Proactively warm thumbnails so UI fetches stay snappy.
            self.app_state.scene_manager.thumbnails.start_warmup([str(result.scene_id)])
        except Exception:
            pass
        if self._progress_bar is not None:
            try:
                self._progress_bar.stop()
                self._progress_bar.config(mode="determinate", value=0)
            except Exception:
                pass
        self._set_busy_ui(False)

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

    def _render_result(self, result: ExtractionResult) -> None:
        t0 = time.perf_counter()
        self.current_result = result
        self.target_var.set(result.target_count)
        self.candidate_var.set(result.candidate_count)
        self.frame_scores = {item.filename: item.score for item in result.frame_scores}
        # Update selection state from current data.
        selected = set(result.selected_frames)
        self.selection_state = {name: (name in selected) for name in result.available_frames}

        # Reset caches/state.
        if self._thumbnail_job is not None:
            try:
                self.frame.after_cancel(self._thumbnail_job)
            except Exception:
                pass
            self._thumbnail_job = None
        self.thumbnail_refs.clear()
        self._thumb_queue = []
        self._loaded_thumbs.clear()
        self._tile_pool = []
        self._visible_tiles = {}
        self._virtual_items = list(result.available_frames)
        self._thumb_workers = 0
        self._thumb_inflight.clear()
        self.canvas.delete("placeholder")

        # Prepare layout.
        self._grid_columns = self._desired_columns(max(self.canvas.winfo_width(), self.canvas.winfo_reqwidth()))
        # Calculate scroll area and render visible tiles.
        self._update_scrollregion()
        self._update_visible_tiles(force=True)

        # Start thumbnail loading for visible items only.
        self._schedule_thumbnail_job()

        self.logger.info(
            "Grid render start items=%d columns=%d card_w=%d card_h=%d canvas_w=%d canvas_h=%d",
            len(result.available_frames),
            self._grid_columns,
            self._card_width_px,
            self._card_height_px,
            self.canvas.winfo_width(),
            self.canvas.winfo_height(),
        )
        self._sync_status()
        self._dirty_selection = False
        self.logger.info("Render_result done items=%d elapsed_ms=%.1f", len(result.available_frames), (time.perf_counter() - t0) * 1000.0)

    def _clear_grid(self) -> None:
        if self._thumbnail_job is not None:
            try:
                self.frame.after_cancel(self._thumbnail_job)
            except Exception:
                pass
            self._thumbnail_job = None
        self._thumb_queue.clear()
        self._loaded_thumbs.clear()
        self.frame_scores.clear()
        self.thumbnail_refs.clear()
        self._thumb_cache.clear()
        self._tile_pool = []
        self._visible_tiles = {}
        self._virtual_items = []
        self.selection_state = {}
        self._scene_thumb_queue.clear()
        if self._scene_thumb_job is not None:
            try:
                self.frame.after_cancel(self._scene_thumb_job)
            except Exception:
                pass
            self._scene_thumb_job = None
        for child in getattr(self, "grid_inner", []).winfo_children() if hasattr(self, "grid_inner") else []:
            try:
                child.destroy()
            except Exception:
                pass
        self._saving = False
        if self._autosave_job is not None:
            try:
                self.frame.after_cancel(self._autosave_job)
            except Exception:
                pass
            self._autosave_job = None
        self._update_scrollregion()

    def _get_scene_thumbnail(self, scene_id: str) -> Optional[tk.PhotoImage]:
        try:
            scene = self.app_state.scene_manager.get(scene_id)
        except Exception:
            return None
        metadata = scene.metadata or {}
        available = metadata.get("available_frames") or []
        filename: Optional[str] = None
        if available:
            filename = available[0]
        else:
            try:
                files = sorted(p.name for p in scene.paths.frames_all_dir.iterdir() if p.is_file())
                filename = files[0] if files else None
            except Exception:
                filename = None
        if not filename:
            return None
        data = self.app_state.scene_manager.get_thumbnail_bytes(str(scene_id), filename)
        if not data:
            return None
        try:
            img = Image.open(io.BytesIO(data))
            img.thumbnail((96, 96), Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception:
            return None
    def _get_scene_thumbnail_bytes(self, scene_id: str) -> Optional[bytes]:
        try:
            scene = self.app_state.scene_manager.get(scene_id)
        except Exception:
            return None
        metadata = scene.metadata or {}
        available = metadata.get("available_frames") or []
        filename: Optional[str] = None
        if available:
            filename = available[0]
        else:
            try:
                files = sorted(p.name for p in scene.paths.frames_all_dir.iterdir() if p.is_file())
                filename = files[0] if files else None
            except Exception:
                filename = None
        if not filename:
            return None
        data = self.app_state.scene_manager.get_thumbnail_bytes(str(scene_id), filename)
        if data:
            return data
        # fallback to raw file
        try:
            with Image.open(scene.paths.frames_all_dir / filename) as img:
                img = img.convert("RGB")
                img.thumbnail((96, 96), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
        except Exception:
            return None

    def _request_scene_thumbnail(self, scene_id: str) -> None:
        if scene_id in self._scene_thumb_pending:
            return
        label = self._scene_thumb_labels.get(scene_id)
        if label is None or not label.winfo_exists():
            return
        self._scene_thumb_pending.add(scene_id)

        def _load_bytes(sid: str) -> tuple[str, Optional[bytes]]:
            return sid, self._get_scene_thumbnail_bytes(sid)

        def _on_success(payload: tuple[str, Optional[bytes]]) -> None:
            sid, data = payload
            self._scene_thumb_pending.discard(sid)
            label_ref = self._scene_thumb_labels.get(sid)
            if label_ref is None or not label_ref.winfo_exists():
                return
            if data:
                try:
                    img = Image.open(io.BytesIO(data))
                    img.thumbnail((96, 96), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self._scene_images.append(photo)
                    label_ref.configure(image=photo, text="")
                except Exception:
                    label_ref.configure(text="No preview")
            else:
                label_ref.configure(text="No preview")

        def _on_error(exc: Exception) -> None:
            self._scene_thumb_pending.discard(scene_id)
            label_ref = self._scene_thumb_labels.get(scene_id)
            if label_ref and label_ref.winfo_exists():
                try:
                    label_ref.configure(text="No preview")
                except Exception:
                    pass
            self.logger.debug("Scene thumb load failed sid=%s exc=%s", scene_id, exc)

        run_in_background(_load_bytes, scene_id, tk_root=self.frame, on_success=_on_success, on_error=_on_error)

    def _scene_summary(self, scene_id: str, status: SceneStatus) -> str:
        available = "?"
        selected = "?"
        source_type = "unknown"
        try:
            meta = self.app_state.scene_manager.get(scene_id).metadata or {}
            available_frames = meta.get("available_frames") or []
            selected_frames = meta.get("selected_frames") or []
            available = str(len(available_frames))
            selected = str(len(selected_frames))
            source_type = str(meta.get("source_type") or "unknown")
        except Exception:
            pass
        return (
            f"{available} frames ({selected} selected) Â· source:{source_type} Â· "
            f"inputs:{status.has_inputs} sfm:{status.has_sfm} splats:{status.has_splats}"
        )

    def _drain_thumbnail_queue(self, result: ExtractionResult, chunk_size: int = 8) -> None:
        try:
            if self.current_result is None or result.scene_id != self.current_result.scene_id:
                self._thumbnail_job = None
                self._thumb_queue.clear()
                return
            loaded = 0
            while self._thumb_queue and loaded < chunk_size and self._thumb_workers < self._max_thumb_workers:
                name = self._thumb_queue.pop(0)
                self._load_thumbnail_async(str(result.scene_id), name)
                loaded += 1
            if self._thumb_queue:
                self._thumbnail_job = self.frame.after(10, lambda: self._drain_thumbnail_queue(result, chunk_size))
            else:
                self._thumbnail_job = None
        except Exception as exc:
            self._thumbnail_job = None
            self.logger.debug("Thumbnail queue drain failed scene=%s exc=%s", result.scene_id, exc)
            self._schedule_thumbnail_job()

    def _load_thumbnail_async(self, scene_id: str, filename: str) -> None:
        cache_key = (scene_id, filename)
        if cache_key in self._thumb_cache:
            self._apply_thumbnail(filename, self._thumb_cache[cache_key])
            return
        inflight_key = (scene_id, filename)
        if inflight_key in self._thumb_inflight:
            return
        self._thumb_workers += 1
        self._thumb_inflight.add(inflight_key)
        done = {"value": False}

        def _load_bytes() -> tuple[str, Optional[bytes]]:
            data = self.app_state.scene_manager.get_thumbnail_bytes(scene_id, filename)
            return filename, data

        def _on_success(payload: tuple[str, Optional[bytes]]) -> None:
            fname, data = payload
            try:
                if done["value"]:
                    return
                done["value"] = True
                self._thumb_inflight.discard((scene_id, fname))
                if self.current_result is None or scene_id != str(self.current_result.scene_id):
                    return
                if data:
                    try:
                        photo = ImageTk.PhotoImage(Image.open(io.BytesIO(data)))
                    except Exception:
                        photo = tk.PhotoImage(width=1, height=1)
                else:
                    photo = tk.PhotoImage(width=1, height=1)
                self._thumb_cache[cache_key] = photo
                self._apply_thumbnail(fname, photo)
            finally:
                self._thumb_workers = max(0, self._thumb_workers - 1)
                self._schedule_thumbnail_job()

        def _on_error(exc: Exception) -> None:
            if done["value"]:
                return
            done["value"] = True
            self._thumb_inflight.discard((scene_id, filename))
            self.logger.debug("Thumbnail load failed file=%s exc=%s", filename, exc)
            self._thumb_workers = max(0, self._thumb_workers - 1)
            self._schedule_thumbnail_job()

        def _on_timeout() -> None:
            if done["value"]:
                return
            done["value"] = True
            self._thumb_inflight.discard((scene_id, filename))
            self._thumb_workers = max(0, self._thumb_workers - 1)
            self._schedule_thumbnail_job()

        # Guard: if a loader never returns, mark it timed-out so the queue keeps moving.
        try:
            self.frame.after(5000, _on_timeout)
        except Exception:
            pass

        try:
            run_in_background(
                _load_bytes,
                tk_root=self.frame,
                on_success=_on_success,
                on_error=_on_error,
                thread_name=f"thumb_{filename}",
            )
        except Exception as exc:
            self._thumb_inflight.discard(inflight_key)
            self._thumb_workers = max(0, self._thumb_workers - 1)
            self.logger.debug("Thumbnail thread start failed file=%s exc=%s", filename, exc)
            self._schedule_thumbnail_job()

    def _apply_thumbnail(self, filename: str, photo: tk.PhotoImage) -> None:
        self.thumbnail_refs.append(photo)
        self._loaded_thumbs.add(filename)
        for tile in self._tile_pool:
            if tile.get("file") == filename:
                try:
                    tile["thumb"].configure(image=photo, text="")
                    tile["thumb"].image = photo
                except Exception:
                    pass

    def _desired_columns(self, width: int) -> int:
        card_width = self._card_width_px  # approximate card width including padding
        return max(1, min(8, width // card_width if width > 0 else 3))

    def _on_canvas_scroll(self, *args) -> None:
        try:
            self._scrollbar_set(*args)
        except Exception:
            pass
        self._update_visible_tiles()

    def _on_mousewheel(self, event: tk.Event) -> None:
        try:
            delta = int(-1 * (event.delta / 120))
            self.canvas.yview_scroll(delta, "units")
            self._update_visible_tiles()
        except Exception:
            pass

    def _on_canvas_resize(self, event: tk.Event) -> None:
        # Resize inner window to match canvas width and recompute columns.
        try:
            self.canvas.itemconfigure(self._canvas_window, width=event.width)
        except Exception:
            pass
        desired = self._desired_columns(event.width)
        if desired != self._grid_columns:
            self._grid_columns = desired
            self._update_scrollregion()
            self._update_visible_tiles(force=True)
        else:
            self._update_scrollregion()
            self._update_visible_tiles(force=True)

    # --- Virtual grid utilities ---
    def _update_visible_tiles(self, force: bool = False) -> None:
        if self.current_result is None or not self._virtual_items:
            return
        view_top = self.canvas.canvasy(0)
        view_bottom = view_top + self.canvas.winfo_height()
        items_per_row = max(1, self._grid_columns)
        row_height = self._card_height_px
        buffer_rows = 1
        first_row = max(0, int(view_top // row_height) - buffer_rows)
        last_row = int(view_bottom // row_height) + buffer_rows
        first_idx = first_row * items_per_row
        last_idx = min(len(self._virtual_items), (last_row + 1) * items_per_row)
        needed = list(range(first_idx, last_idx))

        # Hide tiles that moved out of view.
        for idx in list(self._visible_tiles.keys()):
            if idx not in needed:
                tile = self._visible_tiles.pop(idx)
                tile["frame"].place_forget()

        visible_count = min(len(needed), len(self._virtual_items))
        self._ensure_tile_pool(count=visible_count)

        for pool_idx, item_idx in enumerate(needed[:visible_count]):
            pool_tile = self._tile_pool[pool_idx]
            filename = self._virtual_items[item_idx]
            row = item_idx // items_per_row
            col = item_idx % items_per_row
            x = col * self._card_width_px
            y = row * row_height
            self._populate_tile(pool_tile, filename, x, y, force=force)
            self._visible_tiles[item_idx] = pool_tile

        # Hide any unused tiles in the pool (e.g., when fewer items are visible).
        for extra in self._tile_pool[visible_count:]:
            extra["frame"].place_forget()

    def _ensure_tile_pool(self, count: Optional[int] = None) -> None:
        if count is None:
            visible_rows = int(max(2, (self.canvas.winfo_height() or 600) // self._card_height_px) + 2)
            count = visible_rows * max(1, self._grid_columns)
        while len(self._tile_pool) < count:
            frame = ttk.Frame(self.grid_inner, padding=6)
            thumb_container = ttk.Frame(frame)
            thumb_container.pack(fill="both", expand=True)
            thumb_lbl = ttk.Label(thumb_container, text="Loading...", anchor="center", width=24)
            thumb_lbl.pack(fill="both", expand=True, padx=2, pady=2)
            chk_var = tk.BooleanVar(value=False)
            chk = ttk.Checkbutton(thumb_container, text="", variable=chk_var)
            chk.place(x=6, y=6)
            caption = ttk.Label(
                frame,
                text="",
                anchor="w",
                justify="left",
                wraplength=self._card_width_px - 12,
                padding=(2, 0),
            )
            caption.pack(fill="x", pady=(4, 2))
            self._tile_pool.append(
                {"frame": frame, "thumb": thumb_lbl, "chk": chk, "var": chk_var, "caption": caption, "file": None}
            )

    def _populate_tile(self, tile: dict, filename: str, x: int, y: int, force: bool = False) -> None:
        if not force and tile.get("file") == filename:
            tile["frame"].place(x=x, y=y, width=self._card_width_px, height=self._card_height_px)
            return
        tile["file"] = filename
        sel = self.selection_state.get(filename, False)
        tile["var"].set(sel)
        tile["chk"].configure(command=lambda name=filename, v=tile["var"]: self._handle_toggle_var(name, v))
        tile["chk"].config(text="Selected")
        label_text = filename
        if filename in self.frame_scores:
            label_text = f"{filename} ({self.frame_scores[filename]:.2f})"
        tile["caption"].configure(text=label_text, wraplength=self._card_width_px - 12)

        photo = None
        if self.current_result:
            cache_key = (str(self.current_result.scene_id), filename)
            photo = self._thumb_cache.get(cache_key)
            if photo is None and filename in self._loaded_thumbs:
                photo = self._thumb_cache.get(cache_key)
            if photo is None:
                self._queue_thumbnail(filename)
        if photo:
            tile["thumb"].config(image=photo, text="")
            tile["thumb"].image = photo
        else:
            tile["thumb"].config(image="", text="Loading...")

        tile["frame"].place(x=x, y=y, width=self._card_width_px, height=self._card_height_px)

    def _update_scrollregion(self) -> None:
        if not hasattr(self, "canvas"):
            return
        total_items = max(1, len(self._virtual_items))
        columns = max(1, self._grid_columns)
        rows = (total_items + columns - 1) // columns
        total_height = rows * self._card_height_px
        total_width = max(self.canvas.winfo_width(), columns * self._card_width_px)
        try:
            self.canvas.configure(scrollregion=(0, 0, total_width, total_height))
            self.canvas.itemconfigure(self._canvas_window, width=self.canvas.winfo_width(), height=total_height)
            self.grid_inner.configure(width=total_width, height=total_height)
        except Exception:
            pass

    def _queue_thumbnail(self, filename: str) -> None:
        if filename in self._loaded_thumbs or filename in self._thumb_queue:
            return
        self._thumb_queue.append(filename)
        self._schedule_thumbnail_job()

    def _schedule_thumbnail_job(self) -> None:
        if self._thumbnail_job is not None:
            return
        if not self._thumb_queue or self.current_result is None:
            return
        self._thumbnail_job = self.frame.after(10, lambda: self._drain_thumbnail_queue(self.current_result))

    def _selected_frames(self) -> List[str]:
        return [name for name, selected in self.selection_state.items() if selected]

    def _select_all(self) -> None:
        for name in list(self.selection_state.keys()):
            self.selection_state[name] = True
        self._dirty_selection = True
        self._sync_status()
        self._update_visible_tiles(force=True)
        self._schedule_autosave("select_all")

    def _select_none(self) -> None:
        for name in list(self.selection_state.keys()):
            self.selection_state[name] = False
        self._dirty_selection = True
        self._sync_status()
        self._update_visible_tiles(force=True)
        self._schedule_autosave("select_none")

    def _auto_select(self) -> None:
        if not self.frame_scores:
            self._set_status("Extract frames before auto-selecting.", is_error=True)
            return
        target_count = int(self.target_var.get())
        frame_score_objects = [FrameScore(filename=name, score=score) for name, score in self.frame_scores.items()]
        chosen = auto_select_best(frame_score_objects, target_count)
        for name in list(self.selection_state.keys()):
            self.selection_state[name] = name in chosen
        self._dirty_selection = True
        self._sync_status()
        self._update_visible_tiles(force=True)
        self._schedule_autosave("auto_select")

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

    def _auto_create_and_extract(self, source_path: str, source_type: str) -> None:
        if self._extracting:
            self._set_status("Extraction already running.", is_error=True)
            return
        if not source_path.strip():
            self._set_status("Pick a path first, then create + extract.", is_error=True)
            return
        self.source_type_var.set(source_type)
        scene_id = self._ensure_scene_for_source(source_path, force_new=True)
        if scene_id is None:
            return
        candidate_count = int(self.candidate_var.get())
        target_count = int(self.target_var.get())
        self._begin_extraction(scene_id, source_path, source_type, candidate_count, target_count)

    def _begin_extraction(
        self, scene_id: str, source_path: str, source_type: str, candidate_count: int, target_count: int
    ) -> None:
        if self._extracting:
            self._set_status("Extraction already running.", is_error=True)
            return
        self._set_busy_ui(True, "Extracting frames...")
        self._extracting = True
        self._progress_total = candidate_count
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
            self._navigate_to_training_tab()
    def _build_continue_to_training(self, parent: tk.Misc) -> None:
        box = ttk.LabelFrame(parent, text="Next step")
        box.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(
            box,
            text="Ready? Continue to Training tab with the current scene and presets.",
            wraplength=420,
            justify="left",
        ).pack(anchor="w", padx=6, pady=(4, 6))
        btn_continue = ttk.Button(
            box,
            text="Continue to Training",
            command=self._go_to_training,
            style="Accent.TButton" if "Accent.TButton" in ttk.Style().element_names() else None,
        )
        btn_continue.pack(anchor="w", padx=6, pady=(0, 8))
        self._register_control(btn_continue)

    def _go_to_training(self) -> None:
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
            self._navigate_to_training_tab()

    def _open_wizard(self) -> None:
        self._start_inline_wizard()

    def _start_inline_wizard(self) -> None:
        if self._wizard_running:
            return
        params = self._wizard_prompt_inputs()
        if params is None:
            return
        self._wizard_running = True
        self.source_type_var.set(params["source_type"])
        if params["source_type"] == "video":
            self.video_path_var.set(params["video_path"])
        else:
            self.image_dir_var.set(params["image_dir"])
        self.candidate_var.set(params["candidate"])
        self.target_var.set(params["target"])
        self.training_resolution_var.set(params["resolution"])
        self.training_resample_var.set(params["mode"])
        self._on_resolution_change()
        self._start_extraction()
        self.frame.after(500, self._wizard_wait_for_extract)

    def _wizard_wait_for_extract(self) -> None:
        if self._extracting:
            self.frame.after(500, self._wizard_wait_for_extract)
            return
        self._persist_selection()
        self.frame.after(500, self._wizard_wait_for_save)

    def _wizard_wait_for_save(self) -> None:
        if self._saving:
            self.frame.after(500, self._wizard_wait_for_save)
            return
        if self.notebook is not None:
            try:
                self.notebook.select(1)
            except Exception:
                pass
        preset = self._wizard_prompt_preset()
        if preset and self.training_tab is not None:
            try:
                self.training_tab.training_preset_var.set(preset)
                self.training_tab._apply_training_preset()
                self.training_tab._run_pipeline()
                self.frame.after(1000, self._wizard_wait_for_training)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Wizard", f"Training failed: {exc}", parent=self.frame.winfo_toplevel())
                self._wizard_running = False
        else:
            self._wizard_running = False

    def _wizard_wait_for_training(self) -> None:
        if self.training_tab is not None and getattr(self.training_tab, "_working", False):
            self.frame.after(1000, self._wizard_wait_for_training)
            return
        if self.exports_tab is not None:
            try:
                self.exports_tab._load_checkpoints()
            except Exception:
                pass
        if self.notebook is not None:
            try:
                self.notebook.select(2)
            except Exception:
                pass
        self._wizard_finish_exports()
        self._wizard_running = False

    def _wizard_prompt_inputs(self) -> dict | None:
        dialog = tk.Toplevel(self.frame)
        dialog.title("Wizard: Inputs")
        dialog.transient(self.frame.winfo_toplevel())
        dialog.grab_set()
        self._center_dialog(dialog)
        result: dict | None = None

        source_type = tk.StringVar(value=self.source_type_var.get())
        video_var = tk.StringVar(value=self.video_path_var.get())
        folder_var = tk.StringVar(value=self.image_dir_var.get())
        cand_var = tk.IntVar(value=self.candidate_var.get())
        target_var = tk.IntVar(value=self.target_var.get())
        res_var = tk.IntVar(value=self.training_resolution_var.get())
        mode_var = tk.StringVar(value=self.training_resample_var.get())

        dialog.columnconfigure(1, weight=1)
        ttk.Radiobutton(dialog, text="Video", variable=source_type, value="video").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Radiobutton(dialog, text="Image folder", variable=source_type, value="images").grid(row=0, column=1, sticky="w", padx=8, pady=4)

        ttk.Label(dialog, text="Video file:").grid(row=1, column=0, sticky="w", padx=8)
        ttk.Entry(dialog, textvariable=video_var, width=50).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(dialog, text="Browse", command=lambda: self._wizard_browse_file(video_var)).grid(row=1, column=2, sticky="e", padx=8)
        ttk.Label(dialog, text="Image folder:").grid(row=2, column=0, sticky="w", padx=8, pady=(4, 0))
        ttk.Entry(dialog, textvariable=folder_var, width=50).grid(row=2, column=1, sticky="ew", padx=4, pady=(4, 0))
        ttk.Button(dialog, text="Browse", command=lambda: self._wizard_browse_folder(folder_var)).grid(row=2, column=2, sticky="e", padx=8, pady=(4, 0))

        row3 = ttk.Frame(dialog)
        row3.grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=(8, 4))
        ttk.Label(row3, text="Candidates").pack(side="left")
        ttk.Spinbox(row3, from_=1, to=10000, textvariable=cand_var, width=7).pack(side="left", padx=(4, 12))
        ttk.Label(row3, text="Targets").pack(side="left")
        ttk.Spinbox(row3, from_=1, to=10000, textvariable=target_var, width=7).pack(side="left", padx=(4, 12))
        ttk.Label(row3, text="Resolution (px, small side)").pack(side="left")
        ttk.Combobox(row3, values=[720, 1080, 2160], textvariable=res_var, state="readonly", width=10).pack(side="left", padx=(4, 6))
        ttk.Combobox(row3, values=["lanczos", "bicubic", "bilinear", "nearest"], textvariable=mode_var, state="readonly", width=10).pack(side="left")

        def _ok() -> None:
            nonlocal result
            path = video_var.get().strip() if source_type.get() == "video" else folder_var.get().strip()
            if not path:
                messagebox.showerror("Wizard", "Provide a video file or image folder.", parent=dialog)
                return
            result = {
                "source_type": source_type.get(),
                "video_path": video_var.get().strip(),
                "image_dir": folder_var.get().strip(),
                "candidate": max(1, int(cand_var.get())),
                "target": max(1, int(target_var.get())),
                "resolution": max(1, int(res_var.get())),
                "mode": mode_var.get(),
            }
            dialog.destroy()

        def _cancel() -> None:
            dialog.destroy()

        btn_row = ttk.Frame(dialog)
        btn_row.grid(row=4, column=0, columnspan=3, sticky="e", padx=8, pady=(8, 8))
        ttk.Button(btn_row, text="Cancel", command=_cancel).pack(side="right", padx=(6, 0))
        ttk.Button(btn_row, text="OK", command=_ok).pack(side="right")

        dialog.wait_window()
        return result

    def _wizard_browse_file(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            parent=self.frame.winfo_toplevel(),
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if path:
            var.set(path)

    def _wizard_browse_folder(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory(parent=self.frame.winfo_toplevel(), title="Select image folder")
        if path:
            var.set(path)

    def _wizard_prompt_preset(self) -> str | None:
        dialog = tk.Toplevel(self.frame)
        dialog.title("Wizard: Training preset")
        dialog.transient(self.frame.winfo_toplevel())
        dialog.grab_set()
        self._center_dialog(dialog)
        preset_var = tk.StringVar(value="low")
        ttk.Label(dialog, text="Choose training preset:").pack(anchor="w", padx=10, pady=(10, 4))
        ttk.Combobox(dialog, values=["low", "medium", "high"], textvariable=preset_var, state="readonly", width=10).pack(
            anchor="w", padx=10, pady=(0, 10)
        )
        result: list[str | None] = [None]

        def _ok() -> None:
            result[0] = preset_var.get()
            dialog.destroy()

        def _cancel() -> None:
            dialog.destroy()

        row = ttk.Frame(dialog)
        row.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(row, text="Cancel", command=_cancel).pack(side="right", padx=(6, 0))
        ttk.Button(row, text="OK", command=_ok).pack(side="right")
        dialog.wait_window()
        return result[0]

    def _center_dialog(self, dialog: tk.Toplevel) -> None:
        """Center a dialog over the main window."""
        try:
            dialog.update_idletasks()
            parent = self.frame.winfo_toplevel()
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            w = dialog.winfo_width()
            h = dialog.winfo_height()
            x = px + max(0, (pw - w) // 2)
            y = py + max(0, (ph - h) // 2)
            dialog.geometry(f"+{x}+{y}")
        except Exception:
            return

    def _navigate_to_training_tab(self) -> None:
        target_notebook = self.notebook or self.frame.master
        try:
            target_notebook.select(1)
        except Exception:
            pass

    def _wizard_finish_exports(self) -> None:
        latest = None
        if self.exports_tab is not None and getattr(self.exports_tab, "checkpoint_paths", None):
            latest = self.exports_tab.checkpoint_paths[0] if self.exports_tab.checkpoint_paths else None
        msg = "Extraction and training complete."
        if latest:
            msg += f"\nLatest checkpoint: {latest.name}"
        if messagebox.askyesno("Wizard complete", msg + "\nOpen output folder?"):
            try:
                if latest:
                    Path(latest).parent.mkdir(parents=True, exist_ok=True)
                    import os
                    os.startfile(str(Path(latest).parent))
            except Exception:
                pass
