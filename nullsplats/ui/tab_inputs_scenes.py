"""Scene sidebar and scene list behavior for InputsTab."""

from __future__ import annotations

import io
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

from PIL import Image, ImageTk

from nullsplats.app_state import SceneStatus
from nullsplats.backend.video_frames import ExtractionResult
from nullsplats.backend.io_cache import load_metadata
from nullsplats.util.threading import run_in_background


class InputsTabScenesMixin:
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
            self.image_dir_var.set("")
            self.source_type_var.set("video")
            self._maybe_autofill_scene_from_path(path)
            self._sync_status()
            self._set_status("Input ready. Click Extract to create a new scene.", is_error=False)

    def _choose_image_dir(self) -> None:
        path = filedialog.askdirectory(title="Select image folder")
        if path:
            self.image_dir_var.set(path)
            self.video_path_var.set("")
            self.source_type_var.set("images")
            self._maybe_autofill_scene_from_path(path)
            self._sync_status()
            self._set_status("Input ready. Click Extract to create a new scene.", is_error=False)

    def _require_scene(self) -> Optional[str]:
        if self.app_state.current_scene_id is None:
            self._set_status("Create or select a scene before extracting frames.", is_error=True)
            return None
        return str(self.app_state.current_scene_id)

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

