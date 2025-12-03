"""Entry point for the NullSplats Tkinter application."""

from __future__ import annotations

from nullsplats.app_state import AppState
from nullsplats.ui.root import create_root
from nullsplats.util.logging import setup_logging, get_logger


def main() -> None:
    setup_logging()
    logger = get_logger("main")
    logger.info("Logging to console and logs/app.log")
    logger.info("Starting NullSplats UI")
    app_state = AppState()
    root = create_root(app_state)
    logger.info("Entering Tk main loop")
    root.mainloop()
    logger.info("Tk main loop exited")


if __name__ == "__main__":
    main()
