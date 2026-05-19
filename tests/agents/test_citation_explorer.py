"""Tests for CitationExplorer agent."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.citation_explorer import CitationExplorer
from paper_distiller.sources.arxiv import Paper


def _paper(pid, title="A paper", abstract="abstract"):
    return Paper(
        source="arxiv", paper_id=pid, arxiv_id=pid,
        title=title, authors=[], abstract=abstract,
        pdf_url="...", published="2025-01-01", categories=[],
    )


def _ctx(seeds, seen_ids=None, question="why diffusion?"):
    cfg = SimpleNamespace(qa_question=question, qa_per_round=2, ss_api_key=None)
    qa_state = SimpleNamespace(
        articles_seen_ids=set(seen_ids or []),
        question=question,
    )
    return Context(
        cfg=cfg, llm=MagicMock(), vault=MagicMock(),
        shared={"seed_papers": seeds, "qa_state": qa_state},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_citation_explorer_fetches_refs_for_each_seed(mocker):
    seeds = [_paper("2501.0001", title="diffusion finance")]
    fake_refs = mocker.patch(
        "paper_distiller.agents.citation_explorer.ss_paper_refs",
        return_value=[_paper("2401.0001", title="diffusion theory"),
                      _paper("2401.0002", title="other topic")],
    )
    ctx = _ctx(seeds)
    out = await CitationExplorer().run(ctx)
    fake_refs.assert_called_once()
    assert len(out["citation_expansion_candidates"]) > 0


@pytest.mark.asyncio
async def test_citation_explorer_filters_seen_ids(mocker):
    seeds = [_paper("2501.0001", title="diffusion finance")]
    mocker.patch(
        "paper_distiller.agents.citation_explorer.ss_paper_refs",
        return_value=[_paper("2401.0001", title="seen paper"),
                      _paper("2401.0002", title="new paper")],
    )
    ctx = _ctx(seeds, seen_ids={"2401.0001"})
    out = await CitationExplorer().run(ctx)
    ids = {p.arxiv_id for p in out["citation_expansion_candidates"]}
    assert "2401.0001" not in ids
    assert "2401.0002" in ids


@pytest.mark.asyncio
async def test_citation_explorer_ranks_by_jaccard_relevance(mocker):
    """Higher token overlap with question + seed -> ranked higher."""
    seeds = [_paper("2501.0001", title="diffusion model finance long horizon")]
    mocker.patch(
        "paper_distiller.agents.citation_explorer.ss_paper_refs",
        return_value=[
            _paper("low", title="image recognition cats", abstract="cnn"),
            _paper("high", title="diffusion long horizon time series", abstract="forecasting"),
        ],
    )
    ctx = _ctx(seeds, question="long horizon diffusion")
    out = await CitationExplorer().run(ctx)
    cands = out["citation_expansion_candidates"]
    assert cands[0].arxiv_id == "high"


@pytest.mark.asyncio
async def test_citation_explorer_deps():
    assert CitationExplorer().deps == []
