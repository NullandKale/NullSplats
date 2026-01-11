"""Looking Glass preview plugin factory."""

from __future__ import annotations

import os
from typing import List

from nullsplats.ui.preview_outputs import PreviewOutputSink
from nullsplats.util.logging import get_logger

from .config import load_config
from .sink import LookingGlassSink

logger = get_logger("plugins.looking_glass")


def create_sinks(*, app_state: object | None = None) -> List[PreviewOutputSink]:
    """Create Looking Glass preview sinks. Loader handles gating/detection."""
    try:
        config = load_config()
        return [LookingGlassSink(config)]
    except Exception:  # noqa: BLE001
        logger.exception("Looking Glass sink creation failed.")
        return []
