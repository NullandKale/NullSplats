"""Advanced render controls for the Gaussian splat viewer."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from nullsplats.ui.gl_canvas import CameraView


class AdvancedRenderSettingsPanel(ttk.LabelFrame):
    """Controls for deeper Gaussian-viewer tweaks."""

    def __init__(self, master: tk.Misc, viewer_getter: Callable[[], Optional[object]], *, text: str = "Advanced"):
        super().__init__(master, text=text)
        self.viewer_getter = viewer_getter

        self.sort_mode_var = tk.StringVar(value="back")
        self.debug_mode_var = tk.BooleanVar(value=False)
        self.flat_color_var = tk.BooleanVar(value=False)
        self._yaw_var = tk.DoubleVar(value=0.0)
        self._pitch_var = tk.DoubleVar(value=10.0)
        self._distance_var = tk.DoubleVar(value=3.0)
        self._scale_x_var = tk.DoubleVar(value=0.0)
        self._scale_y_var = tk.DoubleVar(value=0.0)
        self._scale_z_var = tk.DoubleVar(value=0.0)
        self._opacity_bias_var = tk.DoubleVar(value=0.0)

        sort_frame = ttk.Frame(self)
        sort_frame.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Label(sort_frame, text="Alpha sorting:").pack(side="left")
        ttk.Radiobutton(
            sort_frame,
            text="Back-to-front",
            variable=self.sort_mode_var,
            value="back",
            command=self._apply_sort_mode,
        ).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(
            sort_frame,
            text="Front-to-back",
            variable=self.sort_mode_var,
            value="front",
            command=self._apply_sort_mode,
        ).pack(side="left", padx=(4, 0))
        ttk.Button(sort_frame, text="Force re-sort", command=self._force_sort).pack(side="right")

        debug_frame = ttk.Frame(self)
        debug_frame.pack(fill="x", padx=4, pady=(0, 2))
        ttk.Checkbutton(
            debug_frame,
            text="Enable debug mode (larger splats)",
            variable=self.debug_mode_var,
            command=self._apply_debug_mode,
        ).pack(anchor="w")
        ttk.Checkbutton(
            debug_frame,
            text="Flat color shading",
            variable=self.flat_color_var,
            command=self._apply_flat_color_mode,
        ).pack(anchor="w")

        camera_frame = ttk.LabelFrame(self, text="Camera control (degrees/meters)")
        camera_frame.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(camera_frame, text="Yaw").grid(row=0, column=0, sticky="w")
        ttk.Label(camera_frame, text="Pitch").grid(row=0, column=1, sticky="w")
        ttk.Label(camera_frame, text="Distance").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(camera_frame, from_=-180.0, to=180.0, textvariable=self._yaw_var, width=8).grid(row=1, column=0, padx=(0, 4))
        ttk.Spinbox(camera_frame, from_=-89.0, to=89.0, textvariable=self._pitch_var, width=8).grid(
            row=1, column=1, padx=(0, 4)
        )
        ttk.Spinbox(camera_frame, from_=0.1, to=1000.0, increment=0.1, textvariable=self._distance_var, width=8).grid(
            row=1, column=2
        )
        ttk.Button(camera_frame, text="Apply angles", command=self._apply_camera_angles).grid(
            row=2, column=0, columnspan=2, pady=(4, 0), sticky="ew"
        )
        ttk.Button(camera_frame, text="Recenter viewer", command=self._recenter_camera).grid(
            row=2, column=2, pady=(4, 0), sticky="ew"
        )

        self._listener_registered = False
        self.after(10, self._register_camera_listener)

        scale_frame = ttk.LabelFrame(self, text="Scale adjustments (log space)")
        scale_frame.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(scale_frame, text="X").grid(row=0, column=0, padx=(2, 2))
        ttk.Label(scale_frame, text="Y").grid(row=0, column=1, padx=(2, 2))
        ttk.Label(scale_frame, text="Z").grid(row=0, column=2, padx=(2, 2))
        ttk.Spinbox(scale_frame, from_=-3.0, to=3.0, increment=0.01, textvariable=self._scale_x_var, width=6).grid(row=1, column=0, padx=(2, 2))
        ttk.Spinbox(scale_frame, from_=-3.0, to=3.0, increment=0.01, textvariable=self._scale_y_var, width=6).grid(row=1, column=1, padx=(2, 2))
        ttk.Spinbox(scale_frame, from_=-3.0, to=3.0, increment=0.01, textvariable=self._scale_z_var, width=6).grid(row=1, column=2, padx=(2, 2))
        ttk.Button(scale_frame, text="Apply scale bias", command=self._apply_scale_bias).grid(
            row=2, column=0, columnspan=3, pady=(4, 0), sticky="ew"
        )

        opacity_frame = ttk.Frame(self)
        opacity_frame.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(opacity_frame, text="Opacity bias").pack(side="left")
        ttk.Scale(
            opacity_frame,
            from_=-3.0,
            to=3.0,
            variable=self._opacity_bias_var,
            command=lambda value: self._apply_opacity_bias(),
        ).pack(side="left", fill="x", expand=True, padx=(4, 4))
        ttk.Label(opacity_frame, textvariable=self._opacity_bias_var, width=6).pack(side="right")

    def _get_viewer(self):
        return self.viewer_getter()

    def _apply_sort_mode(self):
        viewer = self._get_viewer()
        if viewer is None:
            return
        try:
            viewer.set_sort_back_to_front(self.sort_mode_var.get() == "back")
        except Exception:
            pass

    def _force_sort(self):
        viewer = self._get_viewer()
        if viewer is None:
            return
        try:
            viewer.request_depth_sort()
        except Exception:
            pass

    def _apply_debug_mode(self):
        viewer = self._get_viewer()
        if viewer is None:
            return
        try:
            viewer.set_debug_mode(self.debug_mode_var.get())
        except Exception:
            pass

    def _apply_flat_color_mode(self):
        viewer = self._get_viewer()
        if viewer is None:
            return
        try:
            viewer.set_flat_color_mode(self.flat_color_var.get())
        except Exception:
            pass

    def _apply_scale_bias(self):
        viewer = self._get_viewer()
        if viewer is None:
            return
        bias = (self._scale_x_var.get(), self._scale_y_var.get(), self._scale_z_var.get())
        try:
            viewer.set_scale_bias(bias)
        except Exception:
            pass

    def _apply_opacity_bias(self):
        viewer = self._get_viewer()
        if viewer is None:
            return
        bias = self._opacity_bias_var.get()
        try:
            viewer.set_opacity_bias(bias)
        except Exception:
            pass

    def _apply_camera_angles(self):
        viewer = self._get_viewer()
        if viewer is None:
            return
        try:
            yaw_rad = math.radians(self._yaw_var.get())
            pitch_rad = math.radians(self._pitch_var.get())
            viewer.adjust_camera_angles(yaw=yaw_rad, pitch=pitch_rad, distance=self._distance_var.get())
        except Exception:
            pass

    def _recenter_camera(self):
        viewer = self._get_viewer()
        if viewer is None:
            return
        try:
            viewer.recenter_camera()
        except Exception:
            pass

    def _register_camera_listener(self):
        if self._listener_registered:
            return
        viewer = self._get_viewer()
        if viewer is None:
            self.after(200, self._register_camera_listener)
            return

        def _update_fields(view: CameraView) -> None:
            self._yaw_var.set(math.degrees(view.yaw))
            self._pitch_var.set(math.degrees(view.pitch))
            self._distance_var.set(view.distance)

        viewer.add_camera_listener(_update_fields)
        self._listener_registered = True
