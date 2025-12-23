"""Frame grid, selection, and thumbnail handling for InputsTab."""

from __future__ import annotations

import io
import time
import tkinter as tk
from tkinter import ttk
from typing import List, Optional

from PIL import Image, ImageTk

from nullsplats.backend.video_frames import ExtractionResult, FrameScore, auto_select_best
from nullsplats.util.threading import run_in_background


class InputsTabGridMixin:
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

