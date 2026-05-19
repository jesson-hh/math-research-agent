"""Tests for TheoremExtractor agent."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.theorem_extractor import TheoremExtractor
from paper_distiller.distill.article import ArticleResult


def _article(slug):
    return ArticleResult(
        slug=slug, title=f"T-{slug}", body="## 关键结果\n\n证明了 $n^{-1/d}$ 速率。",
        tags=[], refs=[], depth="full-pdf",
    )


def _ctx(articles):
    return Context(
        cfg=SimpleNamespace(), llm=MagicMock(), vault=MagicMock(),
        shared={"all_articles": articles},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_extractor_calls_llm_per_article():
    arts = [_article("a"), _article("b")]
    ctx = _ctx(arts)
    ctx.llm.complete.return_value = json.dumps({
        "theorems": ["Theorem 1"],
        "assumptions": ["Lipschitz"],
        "convergence_rates": ["n^{-1/d}"],
        "key_lemmas": ["Lemma 2"],
    })
    out = await TheoremExtractor().run(ctx)
    assert ctx.llm.complete.call_count == 2
    assert "structured_extractions" in out
    assert len(out["structured_extractions"]) == 2
    for slug, struct in out["structured_extractions"].items():
        assert "Theorem 1" in struct["theorems"]


@pytest.mark.asyncio
async def test_extractor_handles_malformed_response_gracefully():
    arts = [_article("a")]
    ctx = _ctx(arts)
    ctx.llm.complete.return_value = "not json"
    out = await TheoremExtractor().run(ctx)
    assert out["structured_extractions"]["a"] == {
        "theorems": [], "assumptions": [], "convergence_rates": [], "key_lemmas": [],
    }


@pytest.mark.asyncio
async def test_extractor_deps():
    assert TheoremExtractor().deps == []
