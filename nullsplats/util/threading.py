"""Background task helper for Tkinter applications.

Tasks run on a worker thread, with callbacks routed back onto the Tk event loop
to keep UI updates thread-safe. Logging captures the life-cycle of the task so
background work is visible for debugging.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from nullsplats.util.logging import get_logger


_logger = get_logger("threading")


SuccessCallback = Callable[[Any], None]
ErrorCallback = Callable[[Exception], None]


def run_in_background(
    func: Callable[..., Any],
    /,
    *args: Any,
    tk_root: Any = None,
    on_success: Optional[SuccessCallback] = None,
    on_error: Optional[ErrorCallback] = None,
    thread_name: Optional[str] = None,
    **kwargs: Any,
) -> threading.Thread:
    """Execute ``func`` on a background thread and forward results to Tk.

    Args:
        func: The callable to execute.
        *args: Positional arguments for the callable.
        tk_root: Tk root used to marshal callbacks onto the UI thread.
        on_success: Callback invoked with the result when ``func`` completes.
        on_error: Callback invoked with the exception raised by ``func``.
        thread_name: Optional name for the worker thread.
        **kwargs: Keyword arguments for the callable.

    Returns:
        The launched :class:`threading.Thread` instance.
    """

    def _dispatch_to_ui(callback: Callable[[], None]) -> None:
        if tk_root is None:
            callback()
        else:
            tk_root.after(0, callback)

    def _worker() -> None:
        label = thread_name or func.__name__
        _logger.info("Background task start: %s", label)
        try:
            result = func(*args, **kwargs)
            _logger.info("Background task completed: %s", label)
            if on_success is not None:
                _dispatch_to_ui(lambda: on_success(result))
        except Exception as exc:  # noqa: BLE001 - forward errors to handler
            _logger.exception("Background task failed: %s", label)
            if on_error is not None:
                _dispatch_to_ui(lambda: on_error(exc))
            else:
                raise

    thread = threading.Thread(target=_worker, name=thread_name or func.__name__, daemon=True)
    thread.start()
    return thread


__all__ = ["run_in_background"]
