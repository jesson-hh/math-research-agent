"""Tests for paper_distiller.qa.answer — LLM answer synthesis."""
import json
from unittest.mock import MagicMock

import pytest

from paper_distiller.qa.answer import synthesize, AnswerError
from paper_distiller.distill.article import ArticleResult


def _article(slug, body="..."):
    return ArticleResult(
        slug=slug, title=f"Title {slug}", body=body,
        tags=["t"], refs=[f"arxiv:x-{slug}"], depth="full-pdf",
    )


def test_synthesize_returns_answer_result():
    """synthesize() returns the parsed JSON answer."""
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "title": "QA: 测试问题",
        "body": "# QA: 测试\n\n答案 [[a]] [[b]]",
        "tags": ["test"],
        "cited_slugs": ["a", "b"],
    })
    result = synthesize(
        question="为什么?",
        articles=[_article("a"), _article("b")],
        llm=llm,
    )
    assert result["title"] == "QA: 测试问题"
    assert "[[a]]" in result["body"]
    assert set(result["cited_slugs"]) == {"a", "b"}


def test_synthesize_strips_invented_wikilinks():
    """[[slug]] in body referencing slugs NOT in the articles list are stripped."""
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "title": "T", "body": "see [[real]] and [[fake|Display]]",
        "tags": [], "cited_slugs": ["real", "fake"],
    })
    result = synthesize(
        question="q",
        articles=[_article("real")],
        llm=llm,
    )
    assert "[[real]]" in result["body"]
    assert "[[fake|Display]]" not in result["body"]
    assert "Display" in result["body"]


def test_synthesize_raises_on_malformed_json():
    """Non-JSON LLM response raises AnswerError after one retry."""
    llm = MagicMock()
    llm.complete.side_effect = ["not json", "also not json"]
    with pytest.raises(AnswerError, match="malformed"):
        synthesize(
            question="q",
            articles=[_article("a")],
            llm=llm,
        )
