"""Inputs tab UI for NullSplats."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable, Dict, List, Optional

from nullsplats.app_state import AppState, SceneStatus
from nullsplats.backend.io_cache import ScenePaths, ensure_scene_dirs, delete_scene
from nullsplats.backend.video_frames import (
    ExtractionResult,
    FrameScore,
    auto_select_best,
    extract_frames,
    load_cached_frames,
    persist_selection,
)
from nullsplats.util.logging import get_logger
from nullsplats.util.threading import run_in_background


class InputsTab:
    """Inputs tab with scene creation, frame extraction, and selection UI."""

    def __init__(self, master: tk.Misc, app_state: AppState, on_scene_selected: Callable[[str], None]) -> None:
        self.app_state = app_state
        self.on_scene_selected = on_scene_selected
        self.logger = get_logger("ui.inputs")
        self.frame = ttk.Frame(master)

        self.status_var = tk.StringVar(value="Select or create a scene to begin.")
        self.status_label: Optional[ttk.Label] = None
        self.source_type_var = tk.StringVar(value="video")
        self.video_path_var = tk.StringVar()
        self.image_dir_var = tk.StringVar()
        self.candidate_var = tk.IntVar(value=200)
        self.target_var = tk.IntVar(value=40)
        self.scene_entry: Optional[ttk.Entry] = None
        self._extracting = False

        self.frame_vars: Dict[str, tk.BooleanVar] = {}
        self.frame_scores: Dict[str, float] = {}
        self.thumbnail_refs: List[tk.PhotoImage] = []
        self.current_result: Optional[ExtractionResult] = None

        self._build_header()
        self._build_source_controls()
        self._build_scene_controls()
        self._build_action_buttons()
        self._build_selection_controls()
        self._build_grid()
        self.refresh_scenes()

    def _build_header(self) -> None:
        container = ttk.Frame(self.frame)
        container.pack(fill="x", padx=10, pady=(10, 6))
        ttk.Label(
            container,
            text="Step 1: Choose input (video or image folder).  Step 2: Scene name auto-fills from input; adjust if needed.  Step 3: Extract, review, and save selection.",
            wraplength=780,
            justify="left",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

    def _build_scene_controls(self) -> None:
        container = ttk.LabelFrame(self.frame, text="Step 2: Scene name and cached scenes")
        container.pack(fill="x", padx=10, pady=(6, 8))

        ttk.Label(container, text="Scene name (auto-filled from input):").pack(anchor="w", padx=6, pady=(4, 2))
        name_row = ttk.Frame(container)
        name_row.pack(fill="x", padx=6, pady=(0, 4))
        self.scene_entry = ttk.Entry(name_row, width=30)
        self.scene_entry.pack(side="left", padx=(0, 6))
        ttk.Button(name_row, text="New Scene", command=self._create_scene).pack(side="left")
        ttk.Button(name_row, text="Delete Scene", command=self._delete_scene).pack(side="left", padx=(6, 0))
        ttk.Button(name_row, text="Refresh", command=self.refresh_scenes).pack(side="left", padx=(6, 0))
        self.active_scene_label = ttk.Label(name_row, text="No active scene.")
        self.active_scene_label.pack(side="left", padx=(12, 0))

        ttk.Label(container, text="Scenes discovered in cache:", padding=(0, 4)).pack(anchor="w", padx=6)
        list_frame = ttk.Frame(container)
        list_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self.scene_list = tk.Listbox(list_frame, height=5)
        self.scene_list.pack(side="left", fill="both", expand=True)
        self.scene_list.bind("<<ListboxSelect>>", self._handle_selection)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.scene_list.yview)
        scrollbar.pack(side="right", fill="y")
        self.scene_list.config(yscrollcommand=scrollbar.set)

    def _build_source_controls(self) -> None:
        wrapper = ttk.LabelFrame(self.frame, text="Step 1: Pick input")
        wrapper.pack(fill="x", padx=10, pady=(0, 8))

        source_row = ttk.Frame(wrapper)
        source_row.pack(fill="x", padx=6, pady=6)

        ttk.Radiobutton(
            source_row, text="Video file", variable=self.source_type_var, value="video", command=self._sync_status
        ).pack(side="left")
        ttk.Radiobutton(
            source_row, text="Image folder", variable=self.source_type_var, value="images", command=self._sync_status
        ).pack(side="left", padx=(8, 0))

        ttk.Label(
            wrapper,
            text="Pick the input path. The chosen file/folder is copied into the scene cache for reuse. Scene name auto-fills from this selection.",
            wraplength=780,
            justify="left",
        ).pack(anchor="w", padx=6, pady=(0, 6))

        paths = ttk.Frame(wrapper)
        paths.pack(fill="x", padx=6, pady=(0, 6))

        ttk.Label(paths, text="Video:").grid(row=0, column=0, sticky="w")
        ttk.Entry(paths, textvariable=self.video_path_var, width=55).grid(row=0, column=1, sticky="ew", padx=(4, 4))
        ttk.Button(paths, text="Browse", command=self._choose_video).grid(row=0, column=2, sticky="e")

        ttk.Label(paths, text="Images:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(paths, textvariable=self.image_dir_var, width=55).grid(
            row=1, column=1, sticky="ew", padx=(4, 4), pady=(4, 0)
        )
        ttk.Button(paths, text="Browse", command=self._choose_image_dir).grid(row=1, column=2, sticky="e", pady=(4, 0))
        paths.columnconfigure(1, weight=1)

    def _build_action_buttons(self) -> None:
        wrapper = ttk.LabelFrame(self.frame, text="Step 3: Extract and review")
        wrapper.pack(fill="x", padx=10, pady=(0, 8))

        params = ttk.Frame(wrapper)
        params.pack(fill="x", padx=6, pady=(6, 6))
        ttk.Label(params, text="Candidate frames:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(params, from_=1, to=10000, textvariable=self.candidate_var, width=8).grid(
            row=0, column=1, sticky="w", padx=(4, 12)
        )
        ttk.Label(params, text="Target frames:").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(params, from_=1, to=10000, textvariable=self.target_var, width=8).grid(
            row=0, column=3, sticky="w", padx=(4, 0)
        )

        actions = ttk.Frame(wrapper)
        actions.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(actions, text="Extract Frames", command=self._start_extraction).pack(side="left")
        ttk.Button(actions, text="Reuse Cached Frames", command=self._load_cached).pack(side="left", padx=(8, 0))
        self.status_label = ttk.Label(actions, textvariable=self.status_var, foreground="#444")
        self.status_label.pack(side="left", padx=(12, 0))

    def _build_selection_controls(self) -> None:
        controls = ttk.Frame(self.frame)
        controls.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Button(controls, text="Select All", command=self._select_all).pack(side="left")
        ttk.Button(controls, text="Select None", command=self._select_none).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="Auto-Select Best N", command=self._auto_select).pack(side="left", padx=(6, 0))

    def _build_grid(self) -> None:
        container = ttk.LabelFrame(self.frame, text="Frame selection")
        container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.canvas = tk.Canvas(container, borderwidth=0, height=360)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.grid_inner = ttk.Frame(self.canvas)

        self.grid_inner.bind(
            "<Configure>", lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.grid_inner, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def refresh_scenes(self) -> None:
        scenes = self.app_state.refresh_scene_status()
        self._populate_listbox(scenes)
        self._sync_status()

    def _populate_listbox(self, scenes: List[SceneStatus]) -> None:
        self.scene_list.delete(0, tk.END)
        for status in scenes:
            label = (
                f"{status.scene_id} | inputs:{status.has_inputs} sfm:{status.has_sfm} "
                f"splats:{status.has_splats}"
            )
            self.scene_list.insert(tk.END, label)

    def _handle_selection(self, _: object) -> None:
        if not self.scene_list.curselection():
            return
        index = self.scene_list.curselection()[0]
        selection = self.scene_list.get(index)
        scene_id = selection.split("|", 1)[0].strip()
        normalized = self.app_state.set_current_scene(scene_id)
        self.on_scene_selected(str(normalized))  # type: ignore[arg-type]
        if self.scene_entry is not None:
            self.scene_entry.delete(0, tk.END)
            self.scene_entry.insert(0, str(normalized))
        self._sync_status()
        self._auto_load_cached_for_scene(str(normalized))

    def _create_scene(self) -> None:
        source_path = self._current_source_path()
        name = self.scene_entry.get().strip() if self.scene_entry else ""
        if not name:
            if source_path:
                name = self._derive_scene_id_from_path(source_path)
                if self.scene_entry is not None:
                    self.scene_entry.delete(0, tk.END)
                    self.scene_entry.insert(0, name)
            else:
                self._set_status("Pick a video or image folder first, or enter a scene name.", is_error=True)
                return
        try:
            normalized = self.app_state.set_current_scene(name)
        except ValueError as exc:
            self._set_status(f"Invalid Scene ID: {exc}", is_error=True)
            return
        ensure_scene_dirs(str(normalized), cache_root=self.app_state.config.cache_root)
        self.on_scene_selected(str(normalized))  # type: ignore[arg-type]
        self.refresh_scenes()
        self._set_status(f"Active scene set to {normalized}")

    def _delete_scene(self) -> None:
        scene_id = self.app_state.current_scene_id
        if scene_id is None:
            self._set_status("Select a scene before deleting.", is_error=True)
            return
        delete_scene(str(scene_id), cache_root=self.app_state.config.cache_root)
        if self.app_state.current_scene_id == scene_id:
            self.app_state.set_current_scene(None)
        self.refresh_scenes()
        self._clear_grid()
        self._set_status(f"Deleted scene {scene_id}.")

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
            self._auto_create_and_extract(path, "video")

    def _choose_image_dir(self) -> None:
        path = filedialog.askdirectory(title="Select image folder")
        if path:
            self.image_dir_var.set(path)
            self.source_type_var.set("images")
            self._maybe_autofill_scene_from_path(path)
            self._sync_status()
            self._auto_create_and_extract(path, "images")

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
        scene_id = self._ensure_scene_for_source(source_path)
        if scene_id is None:
            return
        self._begin_extraction(scene_id, source_path, source_type, candidate_count, target_count)

    def _load_cached(self) -> None:
        scene_id = self._require_scene()
        if scene_id is None:
            return
        try:
            result = load_cached_frames(scene_id, cache_root=self.app_state.config.cache_root)
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            self._handle_error(exc)
            return
        self._render_result(result)
        self.status_var.set("Loaded cached frames and selection.")

    def _handle_extraction_success(self, result: ExtractionResult) -> None:
        self._render_result(result)
        self.status_var.set(
            f"Extracted {len(result.available_frames)} frames; auto-selected {len(result.selected_frames)}."
        )
        self._extracting = False

    def _handle_error(self, exc: Exception) -> None:
        self.logger.exception("Inputs tab operation failed")
        self._set_status(f"Operation failed: {exc}", is_error=True)
        self._extracting = False

    def _render_result(self, result: ExtractionResult) -> None:
        self.current_result = result
        self.target_var.set(result.target_count)
        self.candidate_var.set(result.candidate_count)
        self.frame_scores = {item.filename: item.score for item in result.frame_scores}
        self.frame_vars.clear()
        self.thumbnail_refs.clear()
        for child in self.grid_inner.winfo_children():
            child.destroy()

        columns = 3
        for idx, filename in enumerate(result.available_frames):
            row = idx // columns
            column = idx % columns
            holder = ttk.Frame(self.grid_inner, padding=4)
            holder.grid(row=row, column=column, sticky="nwes")

            image_path = result.paths.frames_all_dir / filename
            photo = self._load_thumbnail(image_path)
            self.thumbnail_refs.append(photo)
            ttk.Label(holder, image=photo).pack(anchor="center")

            score = self.frame_scores.get(filename, 0.0)
            var = tk.BooleanVar(value=filename in result.selected_frames)
            self.frame_vars[filename] = var
            ttk.Checkbutton(
                holder,
                text=f"{filename}\nsharpness={score:.4f}",
                variable=var,
                command=lambda name=filename: self._handle_toggle(name),
            ).pack(anchor="center")

        for col in range(columns):
            self.grid_inner.grid_columnconfigure(col, weight=1)
        self.grid_inner.update_idletasks()
        self._sync_status()

    def _clear_grid(self) -> None:
        self.frame_vars.clear()
        self.frame_scores.clear()
        self.thumbnail_refs.clear()
        for child in self.grid_inner.winfo_children():
            child.destroy()

    def _load_thumbnail(self, path: Path) -> tk.PhotoImage:
        photo = tk.PhotoImage(file=str(path))
        max_width = 240
        scale = max(1, int(photo.width() / max_width)) if photo.width() > max_width else 1
        if scale > 1:
            photo = photo.subsample(scale, scale)
        return photo

    def _selected_frames(self) -> List[str]:
        return [name for name, var in self.frame_vars.items() if var.get()]

    def _select_all(self) -> None:
        for var in self.frame_vars.values():
            var.set(True)
        self._sync_status()

    def _select_none(self) -> None:
        for var in self.frame_vars.values():
            var.set(False)
        self._sync_status()

    def _auto_select(self) -> None:
        if not self.frame_scores:
            self._set_status("Extract frames before auto-selecting.", is_error=True)
            return
        target_count = int(self.target_var.get())
        frame_score_objects = [FrameScore(filename=name, score=score) for name, score in self.frame_scores.items()]
        chosen = auto_select_best(frame_score_objects, target_count)
        for name, var in self.frame_vars.items():
            var.set(name in chosen)
        self._persist_selection()

    def _set_status(self, message: str, *, is_error: bool = False) -> None:
        self.status_var.set(message)
        if self.status_label is not None:
            self.status_label.config(foreground="#a00" if is_error else "#444")

    def _update_progress(self, current: int, total: int) -> None:
        if self.status_label is not None and self.status_label.cget("foreground") != "#a00":
            self.status_var.set(f"Extracting frames... {current}/{total}")

    def _sync_status(self) -> None:
        scene_text = (
            f"Active scene: {self.app_state.current_scene_id}"
            if self.app_state.current_scene_id is not None
            else "No active scene."
        )
        self.active_scene_label.config(text=scene_text)
        if self.status_label is not None and self.status_label.cget("foreground") != "#a00":
            selection_text = (
                f"Selected frames: {len(self._selected_frames())}" if self.frame_vars else "Selected frames: 0"
            )
            self.status_var.set(selection_text)

    def _auto_create_and_extract(self, source_path: str, source_type: str) -> None:
        if self._extracting:
            self._set_status("Extraction already running.", is_error=True)
            return
        scene_id = self._ensure_scene_for_source(source_path)
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
        self._extracting = True
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

    def _derive_scene_id_from_path(self, path_str: str) -> str:
        name = Path(path_str).stem or Path(path_str).name
        sanitized = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in name)
        sanitized = sanitized.strip("_") or "scene"
        return sanitized

    def _maybe_autofill_scene_from_path(self, path_str: str) -> None:
        if self.scene_entry is None:
            return
        current = self.scene_entry.get().strip()
        if current:
            return
        candidate = self._derive_scene_id_from_path(path_str)
        self.scene_entry.delete(0, tk.END)
        self.scene_entry.insert(0, candidate)
        self._set_status(f"Scene name set to '{candidate}' from input path.")

    def _ensure_scene_for_source(self, source_path: str) -> Optional[str]:
        if self.scene_entry is None:
            self._set_status("Scene entry unavailable.", is_error=True)
            return None
        name = self.scene_entry.get().strip() or self._derive_scene_id_from_path(source_path)
        if not name:
            self._set_status("Scene name is required.", is_error=True)
            return None
        try:
            normalized = self.app_state.set_current_scene(name)
        except ValueError as exc:
            self._set_status(f"Invalid Scene ID: {exc}", is_error=True)
            return None
        ensure_scene_dirs(str(normalized), cache_root=self.app_state.config.cache_root)
        self.on_scene_selected(str(normalized))  # type: ignore[arg-type]
        self.refresh_scenes()
        if self.scene_entry is not None:
            self.scene_entry.delete(0, tk.END)
            self.scene_entry.insert(0, str(normalized))
        self._set_status(f"Active scene set to {normalized}")
        return str(normalized)

    def _current_source_path(self) -> str:
        if self.source_type_var.get() == "video":
            return self.video_path_var.get().strip()
        return self.image_dir_var.get().strip()

    def _handle_toggle(self, filename: str) -> None:
        if filename in self.frame_vars:
            # ensure UI state is synced before saving
            self.frame_vars[filename].get()
        self._persist_selection()

    def _auto_load_cached_for_scene(self, scene_id: str) -> None:
        if self._extracting:
            return
        paths = ScenePaths(scene_id, cache_root=self.app_state.config.cache_root)
        if paths.metadata_path.exists():
            self._load_cached()

    def _persist_selection(self) -> None:
        scene_id = self._require_scene()
        if scene_id is None:
            return
        selected = self._selected_frames()
        if not selected:
            self._set_status("Select at least one frame to save.", is_error=True)
            return
        try:
            result = persist_selection(scene_id, selected, cache_root=self.app_state.config.cache_root)
        except Exception as exc:  # noqa: BLE001 - surfaced to user
            self._handle_error(exc)
            return
        self._render_result(result)
        self.status_var.set(f"Saved {len(selected)} selected frames.")
