"""Tests for OpenCLIOpenAlexSearcher — subprocess-mocked."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.opencli_openalex import OpenCLIOpenAlexSearcher, _to_paper


def _ctx(topic="diffusion", source="openalex"):
    cfg = SimpleNamespace(
        topic=topic, author=None, pool=10, source=source,
        ss_api_key=None,
    )
    return Context(
        cfg=cfg, llm=MagicMock(), vault=MagicMock(),
        shared={}, on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_skips_when_source_excludes_openalex():
    """cfg.source = 'arxiv' -> agent returns empty without calling subprocess."""
    ctx = _ctx(source="arxiv")
    out = await OpenCLIOpenAlexSearcher().run(ctx)
    assert out == {"candidates_openalex": []}


@pytest.mark.asyncio
async def test_search_then_detail_fetches(mocker):
    """OpenAlex search -> detail fetch for each result -> Paper list."""
    ctx = _ctx()
    search_results = [
        {"id": "W1", "title": "T1", "doi": "10.48550/arxiv.2401.001", "year": 2024,
         "firstAuthor": "A", "venue": "v", "openAccess": True, "type": "article", "url": "u"},
        {"id": "W2", "title": "T2", "doi": "10.1234/foo", "year": 2024,
         "firstAuthor": "B", "venue": "v2", "openAccess": False, "type": "article", "url": "u2"},
    ]
    detail_W1 = [{"id": "W1", "title": "T1", "doi": "10.48550/arxiv.2401.001",
                  "year": 2024, "date": "2024-01-15",
                  "authors": "Alice, Bob", "venue": "v",
                  "abstract": "abstract 1", "openAccessUrl": "https://arxiv.org/pdf/2401.001"}]
    detail_W2 = [{"id": "W2", "title": "T2", "doi": "10.1234/foo",
                  "year": 2024, "date": "2024-02-10",
                  "authors": "Carol", "venue": "v2",
                  "abstract": "abstract 2", "openAccessUrl": ""}]

    async def _fake_call(args, timeout=60.0):
        if args[1] == "search":
            return search_results
        if args[1] == "work":
            if args[2] == "W1":
                return detail_W1
            if args[2] == "W2":
                return detail_W2
        return []
    mocker.patch(
        "paper_distiller.agents.opencli_openalex._opencli_call",
        side_effect=_fake_call,
    )

    out = await OpenCLIOpenAlexSearcher().run(ctx)
    papers = out["candidates_openalex"]
    assert len(papers) == 2
    # Paper 1: DOI is arxiv-style, so arxiv_id should be extracted
    p1 = next(p for p in papers if p.paper_id == "W1")
    assert p1.arxiv_id == "2401.001"
    assert p1.abstract == "abstract 1"
    assert p1.pdf_url == "https://arxiv.org/pdf/2401.001"
    assert p1.authors == ["Alice", "Bob"]
    # Paper 2: DOI not arxiv-style, so arxiv_id is None
    p2 = next(p for p in papers if p.paper_id == "W2")
    assert p2.arxiv_id is None
    assert p2.doi == "10.1234/foo"


@pytest.mark.asyncio
async def test_empty_query_short_circuits(mocker):
    """If query is empty, agent returns empty without calling opencli."""
    ctx = _ctx(topic="")
    ctx.cfg.author = None
    fake_call = mocker.patch(
        "paper_distiller.agents.opencli_openalex._opencli_call",
        new=AsyncMock(return_value=[]),
    )
    out = await OpenCLIOpenAlexSearcher().run(ctx)
    fake_call.assert_not_called()
    assert out == {"candidates_openalex": []}


@pytest.mark.asyncio
async def test_subprocess_failure_returns_empty(mocker):
    """If opencli search returns empty (subprocess error), agent returns empty without crashing."""
    async def _fake_call(args, timeout=60.0):
        return []  # simulate subprocess failure
    mocker.patch(
        "paper_distiller.agents.opencli_openalex._opencli_call",
        side_effect=_fake_call,
    )
    ctx = _ctx()
    out = await OpenCLIOpenAlexSearcher().run(ctx)
    assert out == {"candidates_openalex": []}


@pytest.mark.asyncio
async def test_deps():
    assert OpenCLIOpenAlexSearcher().deps == []


def test_to_paper_extracts_arxiv_id_from_arxiv_doi():
    work = {
        "id": "W123", "title": "T", "doi": "10.48550/arxiv.2401.12345",
        "year": 2024, "authors": "A, B", "abstract": "abs",
        "openAccessUrl": "u", "date": "2024-01-01",
    }
    p = _to_paper(work)
    assert p.arxiv_id == "2401.12345"
    assert p.doi == "10.48550/arxiv.2401.12345"


def test_to_paper_arxiv_id_none_for_non_arxiv_doi():
    work = {
        "id": "W123", "title": "T", "doi": "10.1234/foo",
        "year": 2024, "authors": "A", "abstract": "abs",
        "openAccessUrl": "u",
    }
    p = _to_paper(work)
    assert p.arxiv_id is None
    assert p.doi == "10.1234/foo"
