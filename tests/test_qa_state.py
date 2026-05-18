"""Tests for paper_distiller.qa.state — SessionState dataclass + persistence."""
import json
from pathlib import Path

import pytest

from paper_distiller.qa.state import (
    SessionState,
    RoundRecord,
    write_state,
    read_state,
)


def _make_state(question="why diffusion?"):
    return SessionState(
        session_id="20260518-2143-abc12",
        question=question,
        config_snapshot={"max_rounds": 5, "source": "both"},
        started_at="2026-05-18T21:43:00",
    )


def test_session_state_roundtrip(tmp_path: Path):
    """write_state followed by read_state returns equivalent SessionState."""
    vault = tmp_path / "vault"
    vault.mkdir()
    state = _make_state()
    state.rounds_completed = 2
    state.articles_seen_ids = {"2503.04164", "10.1/abc"}
    state.cost_cny = 0.42
    state.tokens_in_total = 5000
    state.history.append(RoundRecord(
        round=1, query="diffusion finance", rationale="seed query",
        candidates_found=10, new_articles=2,
        article_slugs=["a", "b"],
        what_we_know="some", what_is_missing="more",
        confidence=4, timestamp="2026-05-18T21:43:30",
    ))

    write_state(vault, state)
    restored = read_state(vault, state.session_id)
    assert restored.session_id == state.session_id
    assert restored.question == state.question
    assert restored.rounds_completed == 2
    assert restored.articles_seen_ids == {"2503.04164", "10.1/abc"}
    assert restored.cost_cny == 0.42
    assert len(restored.history) == 1
    assert restored.history[0].query == "diffusion finance"


def test_session_state_missing_returns_none(tmp_path: Path):
    """read_state returns None for unknown session_id."""
    vault = tmp_path / "vault"
    vault.mkdir()
    assert read_state(vault, "nonexistent-session") is None


def test_session_state_persists_articles_seen_ids_as_list(tmp_path: Path):
    """The set field is serialized as a JSON list and restored as a set."""
    vault = tmp_path / "vault"
    vault.mkdir()
    state = _make_state()
    state.articles_seen_ids = {"id1", "id2", "id3"}

    write_state(vault, state)
    on_disk_path = vault / ".paper_distiller" / "qa-sessions" / state.session_id / "state.json"
    raw = json.loads(on_disk_path.read_text(encoding="utf-8"))
    assert isinstance(raw["articles_seen_ids"], list)
    assert set(raw["articles_seen_ids"]) == {"id1", "id2", "id3"}

    restored = read_state(vault, state.session_id)
    assert isinstance(restored.articles_seen_ids, set)
    assert restored.articles_seen_ids == {"id1", "id2", "id3"}
