"""Entry point for the NullSplats Tkinter application."""

from __future__ import annotations

import faulthandler
from pathlib import Path
import sys
import threading
import shutil
import argparse
import os

from nullsplats.app_state import AppState
from nullsplats.ui.root import create_root
from nullsplats.ui.viewer_app import run_viewer_app
from nullsplats.ui.video_sharp_app import run_video_sharp_app
from nullsplats.util.logging import setup_logging, get_logger


def _clear_logs() -> None:
    """Remove prior log files at startup to keep logs/app.log small."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return
    try:
        for entry in logs_dir.iterdir():
            if entry.is_file() or entry.is_symlink():
                entry.unlink()
            elif entry.is_dir():
                shutil.rmtree(entry)
    except Exception:
        # Fall back silently; logging setup will still work even if old files remain.
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="NullSplats entrypoint")
    parser.add_argument("--viewer", action="store_true", help="Launch the cache viewer app.")
    parser.add_argument("--cache-root", default="cache", help="Cache root for the viewer app.")
    parser.add_argument("--video-sharp", action="store_true", help="Launch the video SHARP app.")
    parser.add_argument("--input", help="Input video path for the video SHARP app.")
    parser.add_argument("--looking-glass", action="store_true", help="Enable Looking Glass preview output.")
    parser.add_argument("--looking-glass-sdk", help="Path to Bridge SDK (folder containing bridge_python_sdk).")
    parser.add_argument("--looking-glass-display", type=int, help="Looking Glass display index override.")
    args = parser.parse_args()

    if args.looking_glass:
        os.environ.setdefault("NULLSPLATS_ENABLE_LOOKING_GLASS", "1")
    if args.looking_glass_sdk:
        os.environ["NULLSPLATS_LKG_SDK_PATH"] = args.looking_glass_sdk
    if args.looking_glass_display is not None:
        os.environ["NULLSPLATS_LKG_DISPLAY_INDEX"] = str(args.looking_glass_display)

    _clear_logs()
    setup_logging()
    logger = get_logger("main")

    def _log_exception(exc_type, exc_value, exc_traceback) -> None:
        logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    def _thread_exception(args: threading.ExceptHookArgs) -> None:
        logger.exception("Uncaught thread exception (thread=%s)", args.thread.name, exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = _log_exception
    threading.excepthook = _thread_exception  # type: ignore[assignment]

    # Periodic traceback dump to help diagnose freezes.
    trace_log = Path("logs") / "tracebacks.log"
    trace_log.parent.mkdir(parents=True, exist_ok=True)
    trace_handle = trace_log.open("a", encoding="utf-8", buffering=1)
    trace_handle.write("\n=== NullSplats trace logger armed ===\n")
    trace_handle.flush()
    faulthandler.dump_traceback_later(5.0, repeat=True, file=trace_handle)
    faulthandler.enable(file=trace_handle, all_threads=True)
    logger.info("Logging to console and logs/app.log tracebacks=%s", trace_log)
    if args.viewer and args.video_sharp:
        logger.error("Choose either --viewer or --video-sharp, not both.")
        return
    if args.video_sharp:
        logger.info("Starting NullSplats video SHARP app")
        input_path = Path(args.input).expanduser() if args.input else None
        run_video_sharp_app(Path(args.cache_root), input_path=input_path)
        logger.info("Video SHARP app exited")
        return
    if args.viewer:
        logger.info("Starting NullSplats cache viewer")
        run_viewer_app(Path(args.cache_root))
        logger.info("Viewer app exited")
        return

    logger.info("Starting NullSplats UI")
    app_state = AppState()
    root = create_root(app_state)
    logger.info("Entering Tk main loop")
    root.mainloop()
    logger.info("Tk main loop exited")


if __name__ == "__main__":
    main()
