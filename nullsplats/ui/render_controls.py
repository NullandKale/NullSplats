"""Shared render settings controls for the GL preview panels."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional


class RenderSettingsPanel(ttk.LabelFrame):
    """User controls for Gaussian render parameters."""

    def __init__(self, master: tk.Misc, viewer_getter: Callable[[], Optional[object]], *, text: str = "Render controls"):
        super().__init__(master, text=text)
        self.viewer_getter = viewer_getter

        self.point_scale_var = tk.DoubleVar(value=1.0)
        self.point_scale_label = tk.StringVar(value="1.00x")
        self.sort_back_to_front_var = tk.BooleanVar(value=True)
        self.background_var = tk.StringVar(value="black")

        point_frame = ttk.Frame(self)
        point_frame.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Label(point_frame, text="Point scale").pack(side="left")
        self._scale = ttk.Scale(point_frame, from_=0.3, to=3.0, orient="horizontal", value=1.0, command=self._on_point_scale)
        self._scale.pack(side="left", fill="x", expand=True, padx=(6, 4))
        ttk.Label(point_frame, textvariable=self.point_scale_label, width=5).pack(side="right")

        sort_frame = ttk.Frame(self)
        sort_frame.pack(fill="x", padx=4, pady=(0, 2))
        self._sort_cb = ttk.Checkbutton(
            sort_frame,
            text="Sort alpha back-to-front",
            variable=self.sort_back_to_front_var,
            command=self._on_sort_toggle,
        )
        self._sort_cb.pack(side="left", anchor="w")

        bg_frame = ttk.Frame(self)
        bg_frame.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(bg_frame, text="Background").pack(side="left")
        self._bg_combo = ttk.Combobox(
            bg_frame,
            textvariable=self.background_var,
            values=["black", "gray", "white"],
            state="readonly",
            width=7,
        )
        self._bg_combo.pack(side="left", padx=(4, 0))
        self._bg_combo.bind("<<ComboboxSelected>>", self._on_background_change)

        self._apply_initial_state()

    def _apply_initial_state(self) -> None:
        self._apply_point_scale(self.point_scale_var.get())
        self._apply_background(self.background_var.get())
        self._apply_sort_order(self.sort_back_to_front_var.get())

    def _get_viewer(self):
        return self.viewer_getter()

    def _on_point_scale(self, value: str) -> None:
        scaled = float(value)
        self.point_scale_label.set(f"{scaled:.2f}x")
        self.point_scale_var.set(scaled)
        self._apply_point_scale(scaled)

    def _apply_point_scale(self, value: float) -> None:
        viewer = self._get_viewer()
        if viewer is None:
            return
        try:
            viewer.set_point_scale(value)
        except Exception:
            pass

    def _on_background_change(self, event: tk.Event | None = None) -> None:
        self._apply_background(self.background_var.get())

    def _apply_background(self, name: str) -> None:
        viewer = self._get_viewer()
        if viewer is None:
            return
        try:
            viewer.set_background_color(name)
        except Exception:
            pass

    def _on_sort_toggle(self) -> None:
        value = self.sort_back_to_front_var.get()
        self._apply_sort_order(value)

    def _apply_sort_order(self, back_to_front: bool) -> None:
        viewer = self._get_viewer()
        if viewer is None:
            return
        try:
            viewer.set_sort_back_to_front(back_to_front)
        except Exception:
            pass
