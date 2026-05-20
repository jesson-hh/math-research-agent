"""Tests for FanoutAgent — runtime expansion of one agent into N parallel sub-agents."""
import asyncio
import time
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context, Status
from paper_distiller.agents.dag import DAG
from paper_distiller.agents.orchestrator import Orchestrator
from paper_distiller.agents.fanout import FanoutAgent


class _LeafAgent:
    def __init__(self, name, value):
        self.name = name
        self.deps = []
        self._value = value

    async def run(self, ctx):
        await asyncio.sleep(0.1)
        return {f"leaf_{self._value}": self._value}


class _FanOutOfThree(FanoutAgent):
    name = "fan"
    deps = []

    def expand(self, ctx):
        return [_LeafAgent(f"leaf-{i}", i) for i in range(3)]


def _ctx(**overrides):
    base = dict(
        cfg=MagicMock(), llm=MagicMock(), vault=MagicMock(),
        shared={}, on_status=MagicMock(),
    )
    base.update(overrides)
    return Context(**base)


@pytest.mark.asyncio
async def test_fanout_produces_n_sub_agents_running_in_parallel():
    fan = _FanOutOfThree()
    ctx = _ctx()
    t0 = time.monotonic()
    result = await Orchestrator(DAG([fan]), ctx).run()
    elapsed = time.monotonic() - t0
    assert result == {"leaf_0": 0, "leaf_1": 1, "leaf_2": 2}
    # 3 leaves each sleep 0.1s — parallel total should be < 0.2s
    assert elapsed < 0.25


@pytest.mark.asyncio
async def test_fanout_emits_status_for_each_sub_agent():
    fan = _FanOutOfThree()
    events = []
    ctx = _ctx(on_status=lambda name, status=None, **kw: events.append((name, status)))
    await Orchestrator(DAG([fan]), ctx).run()
    leaf_done = {n for n, s in events if s == Status.DONE and n.startswith("leaf-")}
    assert leaf_done == {"leaf-0", "leaf-1", "leaf-2"}


# v1.7: fanout concurrency cap
class _CountingLeaf:
    """Tracks max concurrent leaves running at the same time."""

    _max_concurrent = 0
    _current = 0
    _lock = asyncio.Lock()

    def __init__(self, name):
        self.name = name
        self.deps = []

    async def run(self, ctx):
        async with self._lock:
            _CountingLeaf._current += 1
            _CountingLeaf._max_concurrent = max(
                _CountingLeaf._max_concurrent, _CountingLeaf._current
            )
        await asyncio.sleep(0.05)
        async with self._lock:
            _CountingLeaf._current -= 1
        return {}


class _BigFanout(FanoutAgent):
    name = "big-fan"
    deps = []

    def __init__(self, n_subagents=10):
        self.n = n_subagents

    def expand(self, ctx):
        return [_CountingLeaf(f"sub-{i}") for i in range(self.n)]


@pytest.mark.asyncio
async def test_fanout_respects_concurrency_cap(monkeypatch):
    """PD_FANOUT_CONCURRENCY=3 → never more than 3 sub-agents running at once
    even when 10 are scheduled."""
    monkeypatch.setenv("PD_FANOUT_CONCURRENCY", "3")
    _CountingLeaf._max_concurrent = 0
    _CountingLeaf._current = 0

    fan = _BigFanout(n_subagents=10)
    ctx = _ctx()
    await Orchestrator(DAG([fan]), ctx).run()

    assert _CountingLeaf._max_concurrent <= 3
    assert _CountingLeaf._max_concurrent >= 1  # at least something ran


@pytest.mark.asyncio
async def test_fanout_default_concurrency_is_five(monkeypatch):
    from paper_distiller.agents.orchestrator import _fanout_concurrency

    monkeypatch.delenv("PD_FANOUT_CONCURRENCY", raising=False)
    assert _fanout_concurrency() == 5


@pytest.mark.asyncio
async def test_fanout_concurrency_clamps_to_safe_range(monkeypatch):
    from paper_distiller.agents.orchestrator import _fanout_concurrency

    monkeypatch.setenv("PD_FANOUT_CONCURRENCY", "0")
    assert _fanout_concurrency() == 1
    monkeypatch.setenv("PD_FANOUT_CONCURRENCY", "99")
    assert _fanout_concurrency() == 20
    monkeypatch.setenv("PD_FANOUT_CONCURRENCY", "not-a-number")
    assert _fanout_concurrency() == 5
