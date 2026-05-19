"""Tests for ThemeClusterer agent."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.theme_clusterer import ThemeClusterer
from paper_distiller.distill.article import ArticleResult


def _article(slug, title=None, tags=None):
    return ArticleResult(
        slug=slug, title=title or f"T-{slug}", body="...",
        tags=tags or [], refs=[], depth="full-pdf",
    )


def _ctx(articles):
    qa_state = SimpleNamespace(articles_distilled=articles, question="?")
    return Context(
        cfg=SimpleNamespace(), llm=MagicMock(), vault=MagicMock(),
        shared={"qa_state": qa_state, "all_articles": articles},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_clusterer_returns_themes_from_llm():
    articles = [_article("a"), _article("b"), _article("c")]
    ctx = _ctx(articles)
    ctx.llm.complete.return_value = json.dumps({
        "themes": [
            {"name": "Theory", "slugs": ["a", "b"], "description": "Theoretical work"},
            {"name": "Empirical", "slugs": ["c"], "description": "Experiments"},
        ]
    })
    out = await ThemeClusterer().run(ctx)
    assert len(out["themes"]) == 2
    assert out["themes"][0]["name"] == "Theory"
    assert "a" in out["themes"][0]["slugs"]


@pytest.mark.asyncio
async def test_clusterer_handles_single_article_no_cluster_needed():
    articles = [_article("only")]
    ctx = _ctx(articles)
    out = await ThemeClusterer().run(ctx)
    ctx.llm.complete.assert_not_called()
    assert len(out["themes"]) == 1
    assert out["themes"][0]["slugs"] == ["only"]


@pytest.mark.asyncio
async def test_clusterer_deps():
    assert ThemeClusterer().deps == []
