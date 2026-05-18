from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper_distiller.cli import build_parser, main


def test_parser_required_vault_and_topic_or_author(tmp_path):
    parser = build_parser()
    args = parser.parse_args(["--vault", str(tmp_path), "--topic", "x"])
    assert args.vault == str(tmp_path)
    assert args.topic == "x"
    assert args.author is None
    assert args.n == 5
    assert args.dry_run is False


def test_parser_dry_run(tmp_path):
    parser = build_parser()
    args = parser.parse_args(["--vault", str(tmp_path), "--author", "huang",
                              "--n", "3", "--dry-run"])
    assert args.dry_run is True
    assert args.n == 3


def test_main_dispatches_to_pipeline(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "m")
    mock_run = mocker.patch("paper_distiller.cli.pipeline_run")
    mock_run.return_value = {"distilled": 0, "article_slugs": [],
                             "survey_slug": None, "skipped_dedup": 0,
                             "skipped_failed": 0, "duration_sec": 0,
                             "tokens_in_total": 0, "tokens_out_total": 0}

    main(["--vault", str(tmp_path), "--topic", "x", "--dry-run"])
    mock_run.assert_called_once()
    cfg = mock_run.call_args[0][0]
    assert cfg.topic == "x"
    assert cfg.dry_run is True
