"""Agent protocol, shared Context, status enum."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol


class Status(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Context:
    """State passed between agents during a DAG run."""
    cfg: Any                       # paper_distiller.config.Config
    llm: Any                       # paper_distiller.llm.openai_compatible.LLMClient
    vault: Any                     # paper_distiller.vault.store.VaultStore
    shared: dict = field(default_factory=dict)
    on_status: Callable[..., None] = lambda *a, **kw: None


class Agent(Protocol):
    """An agent is a named async unit with declared dependencies."""
    name: str
    deps: list[str]

    async def run(self, ctx: Context) -> dict:
        """Run the agent's work. Returns a dict merged into ctx.shared."""
        ...
