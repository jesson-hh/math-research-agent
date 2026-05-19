"""Tests for cost_estimator + plan_mode."""

from __future__ import annotations


def test_estimate_research_uses_max_cost():
    from paper_distiller.chat.cost_estimator import estimate_tool_cost_cny
    assert estimate_tool_cost_cny("research", {"max_cost_cny": 20.0}) == 20.0
    assert estimate_tool_cost_cny("research", {}) == 15.0


def test_estimate_ask_uses_max_cost():
    from paper_distiller.chat.cost_estimator import estimate_tool_cost_cny
    assert estimate_tool_cost_cny("ask", {"max_cost_cny": 3.0}) == 3.0
    assert estimate_tool_cost_cny("ask", {}) == 5.0


def test_estimate_distill_by_id_scales_with_count():
    import pytest
    from paper_distiller.chat.cost_estimator import estimate_tool_cost_cny
    assert estimate_tool_cost_cny(
        "distill_by_id", {"ids": ["a", "b", "c"]}
    ) == pytest.approx(0.6)


def test_estimate_search_is_cheap():
    from paper_distiller.chat.cost_estimator import estimate_tool_cost_cny
    assert estimate_tool_cost_cny("search", {"topic": "x"}) == 0.05


def test_estimate_show_is_free():
    from paper_distiller.chat.cost_estimator import estimate_tool_cost_cny
    assert estimate_tool_cost_cny("show", {"slug": "x"}) == 0.0


def test_estimate_ask_user_is_free():
    from paper_distiller.chat.cost_estimator import estimate_tool_cost_cny
    assert estimate_tool_cost_cny("ask_user", {"question": "?", "options": []}) == 0.0


def test_should_show_plan_below_threshold():
    from paper_distiller.chat.plan_mode import should_show_plan
    assert not should_show_plan("ask", {"max_cost_cny": 3.0})
    assert not should_show_plan("search", {"topic": "x"})


def test_should_show_plan_above_threshold():
    from paper_distiller.chat.plan_mode import should_show_plan
    assert should_show_plan("research", {"max_cost_cny": 15.0})
    assert should_show_plan("ask", {"max_cost_cny": 20.0})


def test_threshold_env_override(monkeypatch):
    from paper_distiller.chat.plan_mode import should_show_plan
    monkeypatch.setenv("PD_PLAN_THRESHOLD_CNY", "1.0")
    assert should_show_plan("ask", {"max_cost_cny": 5.0})
    assert not should_show_plan("show", {"slug": "x"})


def test_render_plan_card_includes_args():
    from paper_distiller.chat.plan_mode import render_plan_card
    out = render_plan_card(
        "research",
        {"question": "X?", "duration": "2h", "max_cost_cny": 15.0},
        estimated_cost_cny=15.0,
    )
    assert "research" in out
    assert "2h" in out
    assert "15" in out


def test_confirm_plan_proceed_on_enter(monkeypatch):
    from paper_distiller.chat.plan_mode import confirm_plan
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "")
    assert confirm_plan(
        "research",
        {"question": "X?"},
        estimated_cost_cny=15.0,
        countdown_sec=0,
    ) is True


def test_confirm_plan_cancel_on_q(monkeypatch):
    from paper_distiller.chat.plan_mode import confirm_plan
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "q")
    assert confirm_plan(
        "research",
        {"question": "X?"},
        estimated_cost_cny=15.0,
        countdown_sec=0,
    ) is False
