"""Compose a session survey from N freshly distilled articles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..llm.openai_compatible import LLMClient, LLMError
from ..vault.crosslink import WikiIndex
from ..vault.store import slugify
from .article import ArticleResult

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "survey.md"


@dataclass
class SurveyResult:
    slug: str
    title: str
    body: str
    tags: list
    related_articles: list  # slugs

    def to_save_kwargs(self) -> dict:
        return {
            "title": self.title,
            "body": self.body,
            "tags": self.tags,
            "refs": [f"session:{self.slug}"],
            "slug": self.slug,
        }


def compose(
    articles: list[ArticleResult],
    topic: str,
    wiki_index: WikiIndex,
    llm: LLMClient,
) -> SurveyResult:
    """LLM-compose a cluster survey covering the given articles."""
    if len(articles) < 2:
        raise ValueError("survey requires at least 2 articles")

    articles_block = "\n".join(
        f"- slug: {a.slug}, title: {a.title}\n  body excerpt: {a.body[:500]}..."
        for a in articles
    )
    index_lines = wiki_index.to_prompt_lines() or ["(vault otherwise empty)"]
    wiki_index_block = "\n".join(index_lines[:100])
    related_slugs_json = json.dumps([a.slug for a in articles], ensure_ascii=False)

    prompt = _PROMPT_FILE.read_text(encoding="utf-8").format(
        topic=topic,
        n_articles=len(articles),
        articles_block=articles_block,
        wiki_index_block=wiki_index_block,
        related_slugs_json=related_slugs_json,
    )
    raw = llm.complete(
        [{"role": "user", "content": prompt}],
        temperature=0.6,
        response_format="json",
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"survey returned non-JSON: {raw[:200]}") from e

    title = parsed.get("title", f"Survey: {topic}").strip()
    body = parsed.get("body", "").strip()
    if not body:
        raise LLMError("survey returned empty body")
    tags = parsed.get("tags", []) or []
    related = parsed.get("related_articles", []) or [a.slug for a in articles]

    # Slug includes timestamp so successive surveys on the same topic don't collide
    ts = datetime.now().strftime("%Y%m%d")
    base = slugify(title)
    slug = f"{base}-{ts}"

    return SurveyResult(slug=slug, title=title, body=body, tags=tags,
                        related_articles=related)
