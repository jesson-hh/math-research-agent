"""Async DAG orchestrator. Schedules agents in topological order,
runs parallel siblings concurrently, propagates errors as AgentFailed.

v1.7: fanout sub-agents now go through a Semaphore (default 5 concurrent)
to keep Aliyun / DeepSeek from rate-limiting our LLM calls when distilling
many papers at once.
"""

from __future__ import annotations

import asyncio
import os

from .base import Context, Status
from .dag import DAG


class AgentFailed(RuntimeError):
    def __init__(self, agent_name: str):
        super().__init__(f"agent {agent_name!r} failed")
        self.agent_name = agent_name


def _fanout_concurrency() -> int:
    """How many fanout sub-agents to run concurrently. Default 5.

    Aliyun Bailian allows ~5 concurrent requests per API key without
    throttling. DeepSeek allows ~10. Override via `PD_FANOUT_CONCURRENCY`.
    """
    try:
        n = int(os.getenv("PD_FANOUT_CONCURRENCY", "5"))
        return max(1, min(20, n))
    except ValueError:
        return 5


class Orchestrator:
    def __init__(self, dag: DAG, ctx: Context):
        self.dag = dag
        self.ctx = ctx

    async def run(self) -> dict:
        for name in self.dag.agents:
            self.ctx.on_status(name, Status.QUEUED)

        for level in self.dag.topo_levels():
            await asyncio.gather(*(self._run_one(name) for name in level))
        return self.ctx.shared

    async def _run_one(self, name: str) -> None:
        agent = self.dag.agents[name]
        if hasattr(agent, "expand") and not hasattr(agent, "run"):
            # FanoutAgent: expand and run sub-agents in parallel with
            # bounded concurrency to avoid hammering the LLM provider.
            self.ctx.on_status(name, Status.RUNNING)
            try:
                sub_agents = agent.expand(self.ctx)
                for sub in sub_agents:
                    self.ctx.on_status(sub.name, Status.QUEUED)
                sem = asyncio.Semaphore(_fanout_concurrency())

                async def _bounded(sub):
                    async with sem:
                        await self._run_sub(sub)

                await asyncio.gather(*(_bounded(sub) for sub in sub_agents))
                self.ctx.on_status(name, Status.DONE)
            except Exception as e:
                self.ctx.on_status(name, Status.FAILED, error=e)
                raise AgentFailed(name) from e
            return

        # Regular Agent
        self.ctx.on_status(name, Status.RUNNING)
        try:
            result = await agent.run(self.ctx)
            self.ctx.shared.update(result or {})
            self.ctx.on_status(name, Status.DONE)
        except Exception as e:
            self.ctx.on_status(name, Status.FAILED, error=e)
            raise AgentFailed(name) from e

    async def _run_sub(self, sub) -> None:
        self.ctx.on_status(sub.name, Status.RUNNING)
        try:
            result = await sub.run(self.ctx)
            self.ctx.shared.update(result or {})
            self.ctx.on_status(sub.name, Status.DONE)
        except Exception as e:
            self.ctx.on_status(sub.name, Status.FAILED, error=e)
            raise AgentFailed(sub.name) from e
