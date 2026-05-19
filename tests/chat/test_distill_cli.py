"""Tests for paper-distiller-chat 'distill' subcommand — end-to-end with mocks."""
from unittest.mock import MagicMock

import pytest


def test_chat_cli_parses_distill_args(monkeypatch):
    """build_parser exposes the distill subcommand with --topic / --n / --vault."""
    from paper_distiller.chat.cli import build_parser
    p = build_parser()
    args = p.parse_args(["distill", "--vault", "/tmp/v", "--topic", "X", "--n", "3"])
    assert args.subcommand == "distill"
    assert args.vault == "/tmp/v"
    assert args.topic == "X"
    assert args.n == 3


def test_chat_cli_dispatches_to_orchestrator(mocker, tmp_path, monkeypatch):
    """`paper-distiller-chat distill ...` builds DAG, runs Orchestrator,
    returns 0."""
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")

    # Patch the orchestrator's run to a no-op coroutine
    async def _fake_run(self):
        return MagicMock()
    mocker.patch("paper_distiller.chat.cli.Orchestrator.run", new=_fake_run)
    # Avoid actually instantiating VaultStore / LLMClient
    mocker.patch("paper_distiller.chat.cli.VaultStore")
    mocker.patch("paper_distiller.chat.cli.LLMClient")

    from paper_distiller.chat.cli import main
    rc = main([
        "distill", "--vault", str(tmp_path), "--topic", "X", "--n", "1",
    ])
    assert rc == 0
