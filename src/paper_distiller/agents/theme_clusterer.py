"""ThemeClusterer — LLM clusters articles into 2-5 themes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path


_PROMPT_FILE = Path(__file__).parent / "prompts" / "cluster.md"


def _article_summary_block(articles) -> str:
    lines = []
    for a in articles:
        first_line = (a.body or "").split("\n", 1)[0][:120]
        tags_str = ", ".join(a.tags or [])
        lines.append(f"### {a.slug}\n  title: {a.title}\n  tags: [{tags_str}]\n  summary: {first_line}")
    return "\n\n".join(lines)


class ThemeClusterer:
    name = "theme-clusterer"
    deps: list[str] = []

    async def run(self, ctx) -> dict:
        articles = ctx.shared.get("all_articles", [])
        if not articles:
            return {"themes": []}
        if len(articles) <= 1:
            return {"themes": [{
                "name": "All articles",
                "description": "Single-article cluster",
                "slugs": [a.slug for a in articles],
            }]}
        prompt = _PROMPT_FILE.read_text(encoding="utf-8").format(
            n_articles=len(articles),
            articles_block=_article_summary_block(articles),
        )
        messages = [{"role": "user", "content": prompt}]
        for attempt in (1, 2):
            raw = await asyncio.to_thread(
                ctx.llm.complete, messages, temperature=0.3, response_format="json",
            )
            try:
                parsed = json.loads(raw)
                themes = parsed.get("themes", [])
                if not isinstance(themes, list) or not themes:
                    raise ValueError("no themes")
                return {"themes": themes}
            except (json.JSONDecodeError, ValueError):
                if attempt == 2:
                    return {"themes": [{
                        "name": "Mixed",
                        "description": "Clustering failed; all articles in one bucket",
                        "slugs": [a.slug for a in articles],
                    }]}
                continue
        return {"themes": []}
