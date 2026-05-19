"""Tests for chat.abort — SIGINT state machine."""

from __future__ import annotations


def test_abort_controller_initial_state():
    from paper_distiller.chat.abort import AbortController

    ac = AbortController()
    assert ac.is_aborted() is False
    assert ac.exit_requested() is False


def test_abort_set_then_reset():
    from paper_distiller.chat.abort import AbortController

    ac = AbortController()
    ac.set_aborted()
    assert ac.is_aborted() is True
    ac.reset()
    assert ac.is_aborted() is False


def test_double_sigint_within_window_requests_exit():
    """Two interrupts within the configured window → exit_requested True."""
    from paper_distiller.chat.abort import AbortController

    ac = AbortController(double_press_window_sec=2.0)
    ac.handle_sigint(when=100.0)
    assert ac.is_aborted() is True
    assert ac.exit_requested() is False
    ac.handle_sigint(when=101.0)
    assert ac.exit_requested() is True


def test_double_sigint_outside_window_does_not_exit():
    from paper_distiller.chat.abort import AbortController

    ac = AbortController(double_press_window_sec=1.0)
    ac.handle_sigint(when=100.0)
    ac.handle_sigint(when=110.0)
    assert ac.is_aborted() is True
    assert ac.exit_requested() is False


def test_install_returns_uninstall_callable():
    from paper_distiller.chat.abort import AbortController, install_handler

    ac = AbortController()
    uninstall = install_handler(ac)
    try:
        assert callable(uninstall)
    finally:
        uninstall()
