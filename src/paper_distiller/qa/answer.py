"""LLM answer-synthesis call for the QA loop.

Given the question + all distilled articles, produces a final markdown
answer with [[wikilink]] citations. Invented slugs (not in the articles
set) are stripped post-LLM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..llm.openai_compatible import LLMClient


class AnswerError(RuntimeError):
    pass


_PROMPT_FILE = Path(__file__).parent / "prompts" / "answer.md"
_LINK_RE = re.compile(r"\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]")
_REQUIRED_KEYS = {"title", "body", "tags", "cited_slugs"}


def _render_prompt(question: str, articles: list) -> str:
    if articles:
        blocks = []
        for a in articles:
            body_capped = (a.body or "")[:12000]
            blocks.append(f"### slug: {a.slug}\n### title: {a.title}\n\n{body_capped}")
        articles_full = "\n\n---\n\n".join(blocks)
    else:
        articles_full = "(没有可用 articles —— 这种情况你应在 body 中说明无法回答)"
    return _PROMPT_FILE.read_text(encoding="utf-8").format(
        question=question,
        n_articles=len(articles),
        articles_full=articles_full,
    )


def _scrub_invented_links(body: str, valid_slugs: set) -> str:
    """Strip [[slug]] / [[slug|Display]] when slug is not in valid_slugs.

    For invented links: if a display text exists, keep it as plain text;
    otherwise keep the bare slug as plain text.
    """
    def repl(m):
        slug = m.group(1).strip()
        display = m.group(2)
        if slug in valid_slugs:
            return m.group(0)
        return display if display else slug
    return _LINK_RE.sub(repl, body)


def _parse_response(raw: str) -> dict:
    parsed = json.loads(raw)
    missing = _REQUIRED_KEYS - set(parsed.keys())
    if missing:
        raise ValueError(f"answer JSON missing keys: {missing}")
    return parsed


def synthesize(question: str, articles: list, llm: LLMClient) -> dict:
    """One answer synthesis call. Retries once on malformed JSON.

    `articles` is a list of ArticleResult-like objects (must have .slug,
    .title, .body). Returns dict with keys: title, body, tags, cited_slugs.
    Invented [[wikilinks]] are scrubbed from body before return.
    """
    prompt = _render_prompt(question, articles)
    messages = [{"role": "user", "content": prompt}]
    valid_slugs = {a.slug for a in articles}

    for attempt in (1, 2):
        raw = llm.complete(messages, temperature=0.5, response_format="json")
        try:
            parsed = _parse_response(raw)
            parsed["body"] = _scrub_invented_links(parsed["body"], valid_slugs)
            return parsed
        except (json.JSONDecodeError, ValueError):
            if attempt == 2:
                raise AnswerError(
                    f"answer synthesis returned malformed JSON twice: {raw[:200]}"
                )
            continue
    raise AnswerError("unreachable")  # pragma: no cover
