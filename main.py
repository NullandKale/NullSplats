"""Entry point for the NullSplats Tkinter application."""

from __future__ import annotations

import faulthandler
from pathlib import Path
import sys

from nullsplats.app_state import AppState
from nullsplats.ui.root import create_root
from nullsplats.util.logging import setup_logging, get_logger


def main() -> None:
    setup_logging()
    # Periodic traceback dump to help diagnose freezes.
    trace_log = Path("logs") / "tracebacks.log"
    trace_log.parent.mkdir(parents=True, exist_ok=True)
    trace_handle = trace_log.open("a", encoding="utf-8", buffering=1)
    trace_handle.write("\n=== NullSplats trace logger armed ===\n")
    trace_handle.flush()
    faulthandler.dump_traceback_later(5.0, repeat=True, file=trace_handle)
    faulthandler.enable(file=trace_handle, all_threads=True)
    logger = get_logger("main")
    logger.info("Logging to console and logs/app.log tracebacks=%s", trace_log)
    logger.info("Starting NullSplats UI")
    app_state = AppState()
    root = create_root(app_state)
    logger.info("Entering Tk main loop")
    root.mainloop()
    logger.info("Tk main loop exited")


if __name__ == "__main__":
    main()
