"""Inputs tab UI for NullSplats."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, List

from nullsplats.app_state import AppState, SceneStatus
from nullsplats.backend.io_cache import ensure_scene_dirs


class InputsTab:
    """Inputs tab with basic scene selection UI."""

    def __init__(self, master: tk.Misc, app_state: AppState, on_scene_selected: Callable[[str], None]) -> None:
        self.app_state = app_state
        self.on_scene_selected = on_scene_selected
        self.frame = ttk.Frame(master)

        self._build_header()
        self._build_scene_list()
        self.refresh_scenes()

    def _build_header(self) -> None:
        description = (
            "Manage input scenes. Use New Scene to create cache folders, or select "
            "an existing scene to work with extracted frames."
        )
        ttk.Label(self.frame, text=description, wraplength=500, justify="left").pack(
            anchor="w", padx=10, pady=(10, 6)
        )

        button_row = ttk.Frame(self.frame)
        button_row.pack(fill="x", padx=10, pady=(0, 10))

        self.new_scene_entry = ttk.Entry(button_row)
        self.new_scene_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        ttk.Button(button_row, text="New Scene", command=self._create_scene).pack(side="left")
        ttk.Button(button_row, text="Refresh", command=self.refresh_scenes).pack(side="left", padx=(6, 0))

    def _build_scene_list(self) -> None:
        list_frame = ttk.Frame(self.frame)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(list_frame, text="Scenes discovered in cache:").pack(anchor="w")
        self.scene_list = tk.Listbox(list_frame, height=10)
        self.scene_list.pack(fill="both", expand=True, pady=(4, 0))
        self.scene_list.bind("<<ListboxSelect>>", self._handle_selection)

    def refresh_scenes(self) -> None:
        scenes = self.app_state.refresh_scene_status()
        self._populate_listbox(scenes)

    def _populate_listbox(self, scenes: List[SceneStatus]) -> None:
        self.scene_list.delete(0, tk.END)
        for status in scenes:
            label = f"{status.scene_id} | inputs:{status.has_inputs} sfm:{status.has_sfm} splats:{status.has_splats}"
            self.scene_list.insert(tk.END, label)

    def _handle_selection(self, _: object) -> None:
        if not self.scene_list.curselection():
            return
        index = self.scene_list.curselection()[0]
        selection = self.scene_list.get(index)
        scene_id = selection.split("|", 1)[0].strip()
        normalized = self.app_state.set_current_scene(scene_id)
        self.on_scene_selected(str(normalized))  # type: ignore[arg-type]

    def _create_scene(self) -> None:
        raw_value = self.new_scene_entry.get().strip()
        if not raw_value:
            messagebox.showinfo("New Scene", "Enter a scene ID to create a new scene.")
            return
        try:
            normalized = self.app_state.set_current_scene(raw_value)
        except ValueError as exc:
            messagebox.showerror("Invalid Scene ID", str(exc))
            return
        ensure_scene_dirs(str(normalized), cache_root=self.app_state.config.cache_root)
        messagebox.showinfo("Scene Selected", f"Active scene set to {normalized}")
        self.on_scene_selected(str(normalized))  # type: ignore[arg-type]
        self.refresh_scenes()
