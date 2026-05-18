import json
from unittest.mock import MagicMock

from paper_distiller.vault.crosslink import WikiIndex
from paper_distiller.distill.article import ArticleResult
from paper_distiller.distill.survey import compose


def _article(slug):
    return ArticleResult(slug=slug, title=f"Title {slug}", body=f"body of {slug}",
                         tags=["t"], refs=["arxiv:x"], depth="full-pdf")


def test_compose_returns_survey_result():
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "title": "扩散模型综述",
        "body": "# 综述\n[[a]] [[b]]",
        "tags": ["diffusion"],
        "related_articles": ["a", "b"],
    })
    result = compose(
        [_article("a"), _article("b")],
        topic="diffusion",
        wiki_index=WikiIndex(entries=[]),
        llm=llm,
    )
    assert result.title == "扩散模型综述"
    assert "[[a]]" in result.body
    assert result.related_articles == ["a", "b"]
