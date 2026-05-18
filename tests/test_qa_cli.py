"""Tests for paper_distiller.qa.cli — argparse + dispatch."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper_distiller.qa.cli import build_parser, main


def test_parser_required_vault_and_question(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "--vault", str(tmp_path), "--question", "why diffusion?",
    ])
    assert args.vault == str(tmp_path)
    assert args.question == "why diffusion?"
    assert args.max_rounds == 5  # default
    assert args.max_articles == 15  # default
    assert args.per_round == 2  # default
    assert args.confidence_threshold == 8
    assert args.interactive is False
    assert args.resume is None


def test_main_dispatches_to_loop(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    mock_run = mocker.patch("paper_distiller.qa.cli.loop_run")
    mock_run.return_value = {
        "session_id": "sid-1", "stop_reason": "llm_done",
        "rounds_completed": 2, "articles_distilled_count": 4,
        "survey_slug": "qa-x-20260518", "cost_cny": 0.5,
        "tokens_in_total": 1000, "tokens_out_total": 500,
    }

    rc = main([
        "--vault", str(tmp_path), "--question", "why?", "--max-rounds", "3",
    ])
    assert rc == 0
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.qa_question == "why?"
    assert cfg.qa_max_rounds == 3
