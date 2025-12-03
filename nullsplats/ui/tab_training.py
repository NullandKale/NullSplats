"""Training tab UI for NullSplats."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from nullsplats.app_state import AppState


class TrainingTab:
    """Training tab placeholder with current-scene display."""

    def __init__(self, master: tk.Misc, app_state: AppState) -> None:
        self.app_state = app_state
        self.frame = ttk.Frame(master)
        self._build_contents()

    def _build_contents(self) -> None:
        ttk.Label(self.frame, text="Training tasks will appear here.", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=10, pady=(10, 4)
        )
        self.scene_label = ttk.Label(self.frame, text=self._scene_text(), anchor="w", justify="left")
        self.scene_label.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(self.frame, text="Refresh scene status", command=self._refresh_scene).pack(
            anchor="w", padx=10
        )

    def _scene_text(self) -> str:
        scene = self.app_state.current_scene_id
        if scene is None:
            return "No active scene selected."
        return f"Active scene: {scene}"

    def _refresh_scene(self) -> None:
        self.app_state.refresh_scene_status()
        self.scene_label.config(text=self._scene_text())
