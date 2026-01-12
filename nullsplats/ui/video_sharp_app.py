"""Shim loader for the video SHARP UI implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Optional


def _load_video_sharp_module():
    module_name = "nullsplats_video_sharp_ui"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = Path(__file__).resolve().parents[1] / "video-sharp" / "video_sharp_ui.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Video SHARP UI module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_video_sharp_app(cache_root: Optional[Path] = None, *, input_path: Optional[Path] = None) -> None:
    module = _load_video_sharp_module()
    module.run_video_sharp_app(cache_root, input_path=input_path)


def VideoSharpApp(*args, **kwargs):
    module = _load_video_sharp_module()
    return module.VideoSharpApp(*args, **kwargs)


def VideoSharpTab(*args, **kwargs):
    module = _load_video_sharp_module()
    return module.VideoSharpTab(*args, **kwargs)


__all__ = ["run_video_sharp_app", "VideoSharpApp", "VideoSharpTab"]
