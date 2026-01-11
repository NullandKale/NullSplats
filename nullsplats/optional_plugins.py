"""Optional plugin loader for preview output sinks."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import List

import logging

from nullsplats.util.logging import get_logger
from nullsplats.ui.preview_outputs import PreviewOutputSink

logger = get_logger("optional_plugins")
logger.propagate = True
# Do not attach a NullHandler here; allow messages to flow to the app logger.

_ENV_ENABLE_LOOKING_GLASS = "NULLSPLATS_ENABLE_LOOKING_GLASS"
_ENV_LKG_SDK_PATH = "NULLSPLATS_LKG_SDK_PATH"

_FALSEY = {"", "0", "false", "no", "off"}
_TRUTHY = {"1", "true", "yes", "on"}
_LKG_AVAILABLE: bool | None = None


def _env_enabled() -> bool:
    raw = os.getenv(_ENV_ENABLE_LOOKING_GLASS)
    if raw is None:
        return False
    val = raw.strip().lower()
    if val in _FALSEY:
        return False
    if val in _TRUTHY:
        return True
    return False


def _env_disabled() -> bool:
    raw = os.getenv(_ENV_ENABLE_LOOKING_GLASS)
    if raw is None:
        return False
    return raw.strip().lower() in _FALSEY


def _candidate_sdk_paths() -> list[Path]:
    paths: list[Path] = []
    raw = os.getenv(_ENV_LKG_SDK_PATH)
    if raw:
        for entry in raw.split(os.pathsep):
            entry = entry.strip()
            if not entry:
                continue
            paths.append(Path(entry))
    program_files = [os.getenv("ProgramFiles"), os.getenv("ProgramFiles(x86)")]
    for base in [p for p in program_files if p]:
        root = Path(base) / "Looking Glass" / "Bridge" / "resources"
        paths.extend(
            [
                root / "app" / "LKGBridgeSDK",
                root / "app.asar.unpacked" / "LKGBridgeSDK",
            ]
        )
    return paths


def _ensure_bridge_sdk_on_path() -> None:
    for candidate in _candidate_sdk_paths():
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


def _detect_looking_glass_display() -> bool:
    """Return True if Bridge SDK is present and at least one display is detected."""
    logger.debug("Looking Glass detection start (env=%s)", os.getenv(_ENV_ENABLE_LOOKING_GLASS))
    _ensure_bridge_sdk_on_path()
    try:
        import bridge_python_sdk  # type: ignore
    except Exception:
        logger.debug("Bridge SDK import failed during detection.")
        return False
    try:
        pkg_dir = Path(bridge_python_sdk.__file__).parent
        if str(pkg_dir) not in sys.path:
            sys.path.insert(0, str(pkg_dir))
        from bridge_python_sdk.BridgeApi import BridgeAPI  # type: ignore
    except Exception:
        logger.debug("Bridge SDK present but import failed during detection.", exc_info=True)
        return False
    try:
        api = BridgeAPI()
        if not api.initialize("NullSplats Looking Glass Detect"):
            logger.debug("Bridge initialize returned False during detection.")
            return False
        displays = api.get_displays()
        if len(displays) > 0:
            logger.info("Detected Looking Glass display(s): %s", displays)
            return True
        logger.info("Bridge detection found no Looking Glass displays.")
        return False
    except Exception:  # noqa: BLE001
        logger.debug("Bridge display detection failed.", exc_info=True)
        return False


def load_preview_sinks(*, app_state: object | None = None) -> List[PreviewOutputSink]:
    """Attempt to load opt-in preview output sinks.

    Returns an empty list unless the user has explicitly enabled Looking Glass
    support via the environment variable or a connected device is detected.
    """
    if _env_disabled():
        logger.info("NULLSPLATS_ENABLE_LOOKING_GLASS explicitly disabled; skipping sinks.")
        return []

    enabled = _env_enabled()
    detected = False
    if not enabled:
        logger.debug("Looking Glass env not enabled; probing for display.")
        detected = _detect_looking_glass_display()
        enabled = detected
    if not enabled:
        logger.debug("Looking Glass sinks not enabled (env disabled and no display detected).")
        return []
    logger.info("Looking Glass sinks enabled (env=%s detected=%s).", _env_enabled(), detected)

    try:
        _ensure_bridge_sdk_on_path()
        module = importlib.import_module("nullsplats_plugins_looking_glass.plugin")
    except ModuleNotFoundError:
        logger.info(
            "Looking Glass plugin not found; skipping optional sink load."
        )
        return []
    except Exception:  # noqa: BLE001
        logger.exception("Failed to import Looking Glass plugin module; skipping optional sink load.")
        return []

    if not hasattr(module, "create_sinks"):
        logger.info("Looking Glass plugin module is missing create_sinks; skipping.")
        return []

    try:
        sinks = module.create_sinks(app_state=app_state)
    except Exception:  # noqa: BLE001
        logger.exception("Looking Glass plugin create_sinks failed; skipping.")
        return []

    if not sinks:
        return []

    # Ensure types line up for callers expecting PreviewOutputSink instances.
    return [sink for sink in sinks if isinstance(sink, object)]


def looking_glass_available(*, refresh: bool = False) -> bool:
    """Return True if a Looking Glass display is available or explicitly enabled."""
    global _LKG_AVAILABLE
    if _LKG_AVAILABLE is not None and not refresh:
        return _LKG_AVAILABLE
    if _env_disabled():
        _LKG_AVAILABLE = False
        return False
    if _env_enabled():
        _LKG_AVAILABLE = True
        return True
    _LKG_AVAILABLE = _detect_looking_glass_display()
    return _LKG_AVAILABLE
