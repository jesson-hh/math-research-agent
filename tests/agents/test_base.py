"""Tests for paper_distiller.agents.base — Agent protocol, Context, Status."""
from dataclasses import is_dataclass
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context, Status


def test_status_enum_values():
    """Status enum has the five required states."""
    assert Status.QUEUED.value == "queued"
    assert Status.RUNNING.value == "running"
    assert Status.DONE.value == "done"
    assert Status.FAILED.value == "failed"
    assert Status.SKIPPED.value == "skipped"


def test_context_is_dataclass():
    """Context is a dataclass with the required fields."""
    assert is_dataclass(Context)


def test_context_construction(tmp_path):
    """Context can be constructed with the required attributes."""
    cfg = MagicMock()
    llm = MagicMock()
    vault = MagicMock()
    on_status = MagicMock()

    ctx = Context(cfg=cfg, llm=llm, vault=vault, shared={}, on_status=on_status)

    assert ctx.cfg is cfg
    assert ctx.llm is llm
    assert ctx.vault is vault
    assert ctx.shared == {}
    assert ctx.on_status is on_status


def test_context_shared_is_mutable():
    """ctx.shared can be mutated by agents."""
    ctx = Context(
        cfg=MagicMock(), llm=MagicMock(), vault=MagicMock(),
        shared={}, on_status=MagicMock(),
    )
    ctx.shared["foo"] = "bar"
    assert ctx.shared == {"foo": "bar"}
