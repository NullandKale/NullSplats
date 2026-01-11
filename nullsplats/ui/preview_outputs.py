"""Lightweight preview output sink interfaces.

These definitions intentionally avoid importing heavy OpenGL modules. Concrete
sinks can opt-in to live rendering by implementing these hooks; when no sinks
are loaded, the rest of the app should behave identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass
class PreviewFrameInfo:
    """Minimal metadata for a rendered frame."""

    frame_id: int
    timestamp: float


class PreviewOutputSink(Protocol):
    """Pluggable sink for mirroring the live preview output."""

    def on_viewer_ready(self, viewer: Any) -> None:
        """Called once the OpenGL viewer is constructed and ready for hooks."""

    def on_viewer_destroyed(self) -> None:
        """Called when the viewer is being torn down."""

    def on_camera_updated(self, camera_view: Any) -> None:  # camera_view is typically CameraView
        """Receive camera updates so the sink can mirror the main view."""

    def on_frame_rendered(self, frame_info: PreviewFrameInfo) -> None:
        """Called after a frame renders; sinks can submit output elsewhere."""

    def stop(self) -> None:
        """Cleanly release resources and stop output."""

