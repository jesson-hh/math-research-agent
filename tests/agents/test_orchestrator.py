"""Tests for paper_distiller.agents.orchestrator — async DAG execution."""
import asyncio
import time
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context, Status
from paper_distiller.agents.dag import DAG
from paper_distiller.agents.orchestrator import Orchestrator, AgentFailed


class _StubAgent:
    def __init__(self, name, deps=None, output=None, sleep=0.0, raises=None):
        self.name = name
        self.deps = deps or []
        self._output = output or {}
        self._sleep = sleep
        self._raises = raises
        self.run_started_at = None
        self.run_finished_at = None

    async def run(self, ctx):
        self.run_started_at = time.monotonic()
        if self._sleep:
            await asyncio.sleep(self._sleep)
        if self._raises:
            raise self._raises
        self.run_finished_at = time.monotonic()
        return self._output


def _ctx(**overrides):
    base = dict(
        cfg=MagicMock(), llm=MagicMock(), vault=MagicMock(),
        shared={}, on_status=MagicMock(),
    )
    base.update(overrides)
    return Context(**base)


@pytest.mark.asyncio
async def test_orchestrator_runs_single_agent():
    a = _StubAgent("a", output={"x": 1})
    ctx = _ctx()
    result = await Orchestrator(DAG([a]), ctx).run()
    assert result["x"] == 1
    assert a.run_finished_at is not None


@pytest.mark.asyncio
async def test_orchestrator_runs_linear_chain():
    a = _StubAgent("a", output={"a_out": 1})
    b = _StubAgent("b", deps=["a"], output={"b_out": 2})
    ctx = _ctx()
    result = await Orchestrator(DAG([a, b]), ctx).run()
    assert result == {"a_out": 1, "b_out": 2}


@pytest.mark.asyncio
async def test_orchestrator_runs_parallel_siblings():
    """Two no-deps agents both sleep 0.2s — total wall time < 0.4s if parallel."""
    a = _StubAgent("a", sleep=0.2)
    b = _StubAgent("b", sleep=0.2)
    ctx = _ctx()
    t0 = time.monotonic()
    await Orchestrator(DAG([a, b]), ctx).run()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.35  # parallel, not 0.4+ sequential


@pytest.mark.asyncio
async def test_orchestrator_emits_status_events():
    a = _StubAgent("a")
    events = []
    ctx = _ctx(on_status=lambda name, status, **kw: events.append((name, status)))
    await Orchestrator(DAG([a]), ctx).run()
    statuses = [s for _, s in events]
    assert Status.RUNNING in statuses
    assert Status.DONE in statuses


@pytest.mark.asyncio
async def test_orchestrator_propagates_agent_error():
    a = _StubAgent("a", raises=RuntimeError("boom"))
    ctx = _ctx()
    with pytest.raises(AgentFailed) as exc_info:
        await Orchestrator(DAG([a]), ctx).run()
    assert exc_info.value.agent_name == "a"
    assert "boom" in str(exc_info.value.__cause__)
