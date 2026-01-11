"""Bridge SDK session management."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

from nullsplats.util.logging import get_logger

logger = get_logger("plugins.looking_glass.bridge")


class BridgeSession:
    def __init__(self, display_index: int = -1) -> None:
        self.display_index = display_index
        self.api = None
        self.window_handle: Optional[int] = None
        self.PixelFormats = None

    def _patch_sys_path(self) -> None:
        sdk_paths = []
        raw = os.getenv("NULLSPLATS_LKG_SDK_PATH")
        if raw:
            for entry in raw.split(os.pathsep):
                entry = entry.strip()
                if entry:
                    sdk_paths.append(Path(entry))
        program_files = [os.getenv("ProgramFiles"), os.getenv("ProgramFiles(x86)")]
        for base in [p for p in program_files if p]:
            root = Path(base) / "Looking Glass" / "Bridge" / "resources"
            sdk_paths.extend(
                [
                    root / "app" / "LKGBridgeSDK",
                    root / "app.asar.unpacked" / "LKGBridgeSDK",
                ]
            )
        for candidate in sdk_paths:
            try:
                if not candidate.exists():
                    continue
                if candidate.name == "bridge_python_sdk":
                    sdk_root = candidate.parent
                elif (candidate / "bridge_python_sdk").exists():
                    sdk_root = candidate
                else:
                    continue
                if str(sdk_root) not in sys.path:
                    sys.path.insert(0, str(sdk_root))
                    logger.info("Added Bridge SDK path: %s", sdk_root)
            except Exception:
                logger.debug("Bridge SDK path candidate failed: %s", candidate, exc_info=True)
        try:
            import bridge_python_sdk  # type: ignore
        except Exception:
            return
        pkg_dir = Path(bridge_python_sdk.__file__).parent
        if str(pkg_dir) not in sys.path:
            sys.path.insert(0, str(pkg_dir))

    def start(self) -> bool:
        if self.api is not None and self.window_handle is not None:
            return True
        try:
            self._patch_sys_path()
            from bridge_python_sdk.BridgeApi import BridgeAPI  # type: ignore
            from bridge_python_sdk import BridgeDataTypes  # type: ignore
        except Exception as exc:  # noqa: BLE001
            logger.debug("Bridge SDK import failed; Looking Glass output disabled. %s", exc, exc_info=True)
            return False
        try:
            api = BridgeAPI()
            if not api.initialize("NullSplats Looking Glass Preview"):
                logger.info("Bridge initialize returned False; Looking Glass output disabled.")
                return False
            target_indices = []
            if self.display_index >= 0:
                target_indices.append(self.display_index)
            try:
                detected = api.get_displays()
                if detected:
                    for idx in detected:
                        if idx not in target_indices:
                            target_indices.append(idx)
            except Exception:
                target_indices.append(self.display_index)
            wnd = 0
            for idx in target_indices or [-1]:
                try:
                    wnd = api.instance_window_gl(idx)
                except Exception:
                    logger.debug("Bridge instance_window_gl threw for idx=%s", idx, exc_info=True)
                    wnd = 0
                if wnd != 0:
                    self.display_index = idx
                    break
            if wnd == 0:
                logger.info("Bridge instance_window_gl returned 0 for indices=%s; Looking Glass output disabled.", target_indices)
                return False
            self.api = api
            self.window_handle = wnd
            self.PixelFormats = BridgeDataTypes.PixelFormats
            logger.info("Bridge session established display_index=%s window_handle=%s", self.display_index, wnd)
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Bridge session start failed.")
            return False

    def default_quilt(self) -> Optional[Tuple[float, int, int, int, int]]:
        if self.api is None or self.window_handle is None:
            return None
        try:
            return self.api.get_default_quilt_settings(self.window_handle)
        except Exception:  # noqa: BLE001
            logger.exception("Bridge get_default_quilt_settings failed.")
            return None

    def submit_quilt_texture(self, texture: int, *, width: int, height: int, vx: int, vy: int, aspect: float, zoom: float) -> bool:
        if self.api is None or self.window_handle is None or self.PixelFormats is None:
            return False
        try:
            # Bridge examples use RGBA; stick with that to satisfy the API's allowed formats.
            fmt = self.PixelFormats.RGBA
            self.api.draw_interop_quilt_texture_gl(
                self.window_handle,
                texture,
                fmt,
                int(width),
                int(height),
                int(vx),
                int(vy),
                float(aspect),
                float(zoom),
            )
            return True
        except Exception:  # noqa: BLE001
            logger.debug("Bridge quilt submission failed", exc_info=True)
            return False

    def stop(self) -> None:
        self.api = None
        self.window_handle = None
        self.PixelFormats = None
