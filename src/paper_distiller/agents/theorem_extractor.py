"""TheoremExtractor — extra LLM pass to add structured frontmatter to articles."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path


_PROMPT_FILE = Path(__file__).parent / "prompts" / "extract.md"
_EMPTY = {"theorems": [], "assumptions": [], "convergence_rates": [], "key_lemmas": []}


async def _extract_one(article, llm) -> dict:
    prompt = _PROMPT_FILE.read_text(encoding="utf-8").format(
        slug=article.slug,
        article_body=(article.body or "")[:8000],
    )
    messages = [{"role": "user", "content": prompt}]
    raw = await asyncio.to_thread(
        llm.complete, messages, temperature=0.2, response_format="json",
    )
    try:
        parsed = json.loads(raw)
        return {
            "theorems": parsed.get("theorems", []) or [],
            "assumptions": parsed.get("assumptions", []) or [],
            "convergence_rates": parsed.get("convergence_rates", []) or [],
            "key_lemmas": parsed.get("key_lemmas", []) or [],
        }
    except (json.JSONDecodeError, KeyError):
        return dict(_EMPTY)


class TheoremExtractor:
    name = "theorem-extractor"
    deps: list[str] = []

    async def run(self, ctx) -> dict:
        articles = ctx.shared.get("all_articles", [])
        if not articles:
            return {"structured_extractions": {}}
        extractions = await asyncio.gather(*[_extract_one(a, ctx.llm) for a in articles])
        return {"structured_extractions": {
            a.slug: ext for a, ext in zip(articles, extractions)
        }}
