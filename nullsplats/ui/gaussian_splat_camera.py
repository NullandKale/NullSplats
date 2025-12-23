"""Camera math helpers for the Gaussian splat viewer."""

import numpy as np



def _look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a right-handed look-at view matrix."""
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    true_up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(true_up, eye)
    view[2, 3] = np.dot(forward, eye)
    return view


class Camera:
    """Simple camera class with direct position control."""

    def __init__(self) -> None:
        self.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def set_position_direct(self, x: float, y: float, z: float) -> None:
        """Directly set camera position in world coordinates."""
        self.position = np.array([x, y, z], dtype=np.float32)

    def set_target_direct(self, x: float, y: float, z: float) -> None:
        """Directly set where camera looks."""
        self.target = np.array([x, y, z], dtype=np.float32)

    def get_view_matrix(self) -> np.ndarray:
        """Create look-at view matrix."""
        return _look_at_matrix(self.position, self.target, self.up)
