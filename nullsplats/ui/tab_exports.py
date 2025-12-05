"""Exports tab UI for NullSplats."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from nullsplats.app_state import AppState


class ExportsTab:
    """Exports tab placeholder with current-scene display."""

    def __init__(self, master: tk.Misc, app_state: AppState) -> None:
        self.app_state = app_state
        self.frame = ttk.Frame(master)
        self._build_contents()

    def _build_contents(self) -> None:
        ttk.Label(self.frame, text="Exports will be managed here.", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=10, pady=(10, 4)
        )
        self.scene_label = ttk.Label(self.frame, text=self._scene_text(), anchor="w", justify="left")
        self.scene_label.pack(fill="x", padx=10, pady=(0, 10))

    def _scene_text(self) -> str:
        scene = self.app_state.current_scene_id
        if scene is None:
            return "No active scene selected."
        return f"Active scene: {scene}"

    def on_scene_changed(self, scene_id: str | None) -> None:
        if scene_id is not None:
            self.app_state.set_current_scene(scene_id)
        self.scene_label.config(text=self._scene_text())
