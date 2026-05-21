"""End-to-end integration test for paper-distiller-chat ask — all subsystems mocked.

Tests the full QA loop: 2 rounds of distillation, then synthesis. Vault should
end up with N articles + 1 qa-…md survey + state.json under .paper_distiller/.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper_distiller.distill.article import ArticleResult
from paper_distiller.sources.arxiv import Paper


def _paper(i):
    return Paper(
        source="arxiv", paper_id=f"2501.0000{i}", arxiv_id=f"2501.0000{i}",
        title=f"P{i}", authors=[], abstract=f"abstract {i}",
        pdf_url=f"https://x/{i}.pdf", published="2025-01-01", categories=[],
    )


def test_ask_e2e_writes_qa_survey(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")

    # 3 reflections needed: round 1 continue, round 2 continue, round 3 reflection
    # triggers max_rounds check before continuing.
    reflections = [
        {"is_done": False, "confidence": 4, "what_we_know": "...",
         "what_is_missing": "...", "next_query": "q1",
         "next_query_rationale": "...", "suggest_stop": False},
        {"is_done": False, "confidence": 5, "what_we_know": "...",
         "what_is_missing": "...", "next_query": "q2",
         "next_query_rationale": "...", "suggest_stop": False},
        {"is_done": False, "confidence": 6, "what_we_know": "...",
         "what_is_missing": "...", "next_query": "q3",
         "next_query_rationale": "...", "suggest_stop": False},
    ]
    mocker.patch("paper_distiller.agents.reflector.reflect", side_effect=reflections)
    # Different papers per round so dedup doesn't wipe them
    mocker.patch(
        "paper_distiller.agents.searchers.arxiv_search",
        side_effect=[[_paper(1), _paper(2)], [_paper(3), _paper(4)]],
    )
    mocker.patch("paper_distiller.agents.searchers.ss_search", return_value=[])
    mocker.patch(
        "paper_distiller.agents.curation.rank",
        side_effect=lambda candidates, topic, top_n, llm: candidates[:top_n],
    )
    mocker.patch(
        "paper_distiller.agents.processor.fetch_with_fallback",
        return_value="x" * 600,
    )

    def _make_article(paper, full_text, wiki_index, llm, prior_theorems=None):
        return ArticleResult(
            slug=f"a-{paper.arxiv_id}", title=f"T-{paper.arxiv_id}",
            body=f"body {paper.arxiv_id}", tags=["t"],
            refs=[f"arxiv:{paper.arxiv_id}"], depth="full-pdf",
        )
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        side_effect=_make_article,
    )
    mocker.patch(
        "paper_distiller.agents.processor.load_index",
        return_value=MagicMock(slugs=lambda: set()),
    )
    mocker.patch(
        "paper_distiller.agents.synthesizer.synthesize",
        return_value={
            "title": "QA: 答案", "body": "# answer\n\n...",
            "tags": ["qa"], "cited_slugs": ["a-2501.00001"],
        },
    )

    vault = tmp_path / "vault"
    vault.mkdir()

    from paper_distiller.chat.cli import main
    rc = main([
        "ask", "--vault", str(vault), "--question", "why diffusion?",
        "--max-rounds", "2", "--per-round", "2", "--max-cost-cny", "5",
    ])
    assert rc == 0

    # 2 rounds × 2 papers = 4 articles distilled
    articles_dir = vault / "articles"
    assert len(list(articles_dir.glob("*.md"))) == 4

    # 1 qa-* survey
    surveys_dir = vault / "surveys"
    qa_surveys = list(surveys_dir.glob("qa-*.md"))
    assert len(qa_surveys) == 1

    # state.json present and marks done
    sessions = list((vault / ".paper_distiller" / "qa-sessions").iterdir())
    assert len(sessions) == 1
    state_path = sessions[0] / "state.json"
    assert state_path.exists()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["stop_reason"] == "max_rounds"
    assert data["is_done"] is True
