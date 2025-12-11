"""Panel to inspect and reuse COLMAP camera poses for the current scene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import tkinter as tk
from tkinter import ttk

from nullsplats.backend.io_cache import ScenePaths
from nullsplats.util.logging import get_logger
from nullsplats.util.scene_id import SceneId

logger = get_logger("ui.colmap_camera_panel")


@dataclass(frozen=True)
class ColmapCameraPose:
    """Lightweight representation of a COLMAP camera pose."""

    image_id: int
    camera_id: int
    name: str
    position: np.ndarray
    rotation: np.ndarray


class ColmapCameraPanel(ttk.LabelFrame):
    """Show cached COLMAP pose list and push a selected pose to the viewer."""

    def __init__(
        self,
        master: tk.Misc,
        viewer_getter: Callable[[], Optional[object]],
        scene_getter: Callable[[], Optional[SceneId | str]],
        cache_root_getter: Callable[[], Path],
    ):
        super().__init__(master, text="COLMAP cameras")
        self.viewer_getter = viewer_getter
        self.scene_getter = scene_getter
        self.cache_root_getter = cache_root_getter
        self._poses: List[ColmapCameraPose] = []
        self._current_pose_idx: Optional[int] = None

        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Label(toolbar, text="Reveal COLMAP camera poses used during training.").pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(toolbar, text="Refresh cameras", command=self._load_cameras).pack(side="right")

        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        self._tree = ttk.Treeview(
            tree_frame,
            columns=("camera", "position"),
            show="headings",
            selectmode="browse",
            height=6,
        )
        self._tree.heading("camera", text="Image / camera")
        self._tree.heading("position", text="Position (X,Y,Z)")
        self._tree.column("camera", width=180)
        self._tree.column("position", width=220)
        self._tree.pack(side="left", fill="both", expand=True)
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        scrollbar.pack(side="right", fill="y")
        self._tree.config(yscrollcommand=scrollbar.set)

        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(action_frame, text="Previous camera", command=self._prev_pose).pack(side="left")
        ttk.Button(action_frame, text="Use selected pose", command=self._apply_current_pose).pack(side="left", padx=(4, 4))
        ttk.Button(action_frame, text="Next camera", command=self._next_pose).pack(side="left")
        self._status_var = tk.StringVar(value="No COLMAP cameras loaded.")
        ttk.Label(action_frame, textvariable=self._status_var).pack(side="right")
        self.after(0, self._load_cameras)

    def _load_cameras(self) -> None:
        scene_id = self.scene_getter()
        if scene_id is None:
            self._status_var.set("Select a scene before loading cameras.")
            return
        cache_root = self.cache_root_getter()
        paths = ScenePaths(scene_id, cache_root=cache_root)
        images_file = self._find_images_file(paths.sfm_dir)
        if images_file is None:
            self._status_var.set("COLMAP images.txt not found for this scene.")
            self._tree.delete(*self._tree.get_children())
            return

        self._poses = _parse_images_file(images_file)
        self._tree.delete(*self._tree.get_children())
        for idx, pose in enumerate(self._poses):
            label = f"{pose.name} ({pose.image_id})"
            pos_text = f"{pose.position[0]:.2f}, {pose.position[1]:.2f}, {pose.position[2]:.2f}"
            self._tree.insert("", "end", iid=str(idx), values=(label, pos_text))
        self._status_var.set(f"Loaded {len(self._poses)} COLMAP cameras.")

        # Try to pick camera closest to scene center
        center = self._scene_center()
        if center is not None and self._poses:
            self._current_pose_idx = min(
                range(len(self._poses)),
                key=lambda idx: np.linalg.norm(self._poses[idx].position - center),
            )
        elif self._poses:
            self._current_pose_idx = 0
        else:
            self._current_pose_idx = None

        self._select_pose(self._current_pose_idx)
        if self._current_pose_idx is not None:
            self._apply_pose(self._current_pose_idx)

    def _find_images_file(self, sfm_dir: Path) -> Optional[Path]:
        candidates = [
            sfm_dir / "sparse" / "text" / "images.txt",
            sfm_dir / "sparse" / "0" / "images.txt",
            sfm_dir / "sparse" / "images.txt",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _apply_selected_pose(self) -> None:
        if self._current_pose_idx is None:
            self._status_var.set("Load cameras before applying a pose.")
            return
        pose = self._poses[self._current_pose_idx]
        viewer = self.viewer_getter()
        if viewer is None:
            self._status_var.set("Viewer not ready.")
            return
        try:
            viewer.set_camera_pose(pose.position, rotation=pose.rotation)
            self._status_var.set(f"Applied pose: {pose.name}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to apply COLMAP camera pose")
            self._status_var.set(f"Failed to apply pose: {exc}")

    def _apply_current_pose(self) -> None:
        self._apply_selected_pose()

    def _prev_pose(self) -> None:
        if not self._poses:
            return
        if self._current_pose_idx is None:
            self._current_pose_idx = 0
        else:
            self._current_pose_idx = (self._current_pose_idx - 1) % len(self._poses)
        self._select_pose(self._current_pose_idx)
        self._apply_pose(self._current_pose_idx)

    def _next_pose(self) -> None:
        if not self._poses:
            return
        if self._current_pose_idx is None:
            self._current_pose_idx = 0
        else:
            self._current_pose_idx = (self._current_pose_idx + 1) % len(self._poses)
        self._select_pose(self._current_pose_idx)
        self._apply_pose(self._current_pose_idx)

    def _apply_pose(self, idx: int) -> None:
        if idx is None or idx >= len(self._poses):
            return
        pose = self._poses[idx]
        self._tree.selection_set(str(idx))
        self._tree.see(str(idx))
        self._status_var.set(f"Applying pose: {pose.name}")
        self._apply_selected_pose()

    def _select_pose(self, idx: Optional[int]) -> None:
        self._tree.selection_remove(self._tree.selection())
        if idx is not None and str(idx) in self._tree.get_children():
            self._tree.selection_set(str(idx))

    def _scene_center(self) -> Optional[np.ndarray]:
        viewer = self.viewer_getter()
        if viewer is None or not hasattr(viewer, "get_scene_center"):
            return None
        return viewer.get_scene_center()

    def _on_tree_select(self, _event: object) -> None:
        selection = self._tree.selection()
        if not selection:
            return
        try:
            idx = int(selection[0])
        except ValueError:
            return
        self._current_pose_idx = idx

    def refresh(self) -> None:
        """Reload the camera list for the active scene."""
        self._load_cameras()


def _parse_images_file(images_file: Path) -> List[ColmapCameraPose]:
    poses: List[ColmapCameraPose] = []
    with images_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            try:
                image_id = int(parts[0])
            except ValueError:
                continue
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]
            rot_world_to_camera = _quat_to_rotation_matrix(np.array([qw, qx, qy, qz], dtype=np.float64))
            translation = np.array([tx, ty, tz], dtype=np.float64)
            position = -rot_world_to_camera.T @ translation
            poses.append(
                ColmapCameraPose(
                    image_id=image_id,
                    camera_id=camera_id,
                    name=name,
                    position=position.astype(np.float32),
                    rotation=rot_world_to_camera.astype(np.float32),
                )
            )
    return poses


def _quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )
