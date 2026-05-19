"""Ctrl-C / SIGINT state machine for the agent loop.

Single press while a tool is running → set the abort flag. Tool wrappers
running asyncio.run() will see KeyboardInterrupt and may convert it to a
{"cancelled": True} result. The loop, on detecting the abort flag after the
tool returns, posts a synthetic cancelled-tool message and continues.

Two presses within `double_press_window_sec` → exit the REPL.
"""

from __future__ import annotations

import signal
import threading
import time
from typing import Callable


class AbortController:
    """Thread-safe state for SIGINT handling."""

    def __init__(self, double_press_window_sec: float = 1.5):
        self._abort = threading.Event()
        self._exit = threading.Event()
        self._last_press: float = 0.0
        self._window = double_press_window_sec
        self._lock = threading.Lock()

    def is_aborted(self) -> bool:
        return self._abort.is_set()

    def exit_requested(self) -> bool:
        return self._exit.is_set()

    def set_aborted(self) -> None:
        self._abort.set()

    def reset(self) -> None:
        self._abort.clear()

    def handle_sigint(self, when: float | None = None) -> None:
        """Called from the SIGINT signal handler (or in tests with explicit when)."""
        now = when if when is not None else time.monotonic()
        with self._lock:
            if self._abort.is_set() and (now - self._last_press) <= self._window:
                self._exit.set()
            else:
                self._abort.set()
            self._last_press = now


def install_handler(controller: AbortController) -> Callable[[], None]:
    """Install controller as the SIGINT handler. Returns an uninstall fn."""
    prev = signal.getsignal(signal.SIGINT)

    def _handler(signum, frame):
        controller.handle_sigint()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)

    def _uninstall():
        signal.signal(signal.SIGINT, prev)

    return _uninstall
