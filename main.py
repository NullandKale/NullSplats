"""Entry point for the NullSplats Tkinter application."""

from __future__ import annotations

import faulthandler
from pathlib import Path
import sys
import threading
import shutil

from nullsplats.app_state import AppState
from nullsplats.ui.root import create_root
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
    logger.info("Starting NullSplats UI")
    app_state = AppState()
    root = create_root(app_state)
    logger.info("Entering Tk main loop")
    root.mainloop()
    logger.info("Tk main loop exited")


if __name__ == "__main__":
    main()
