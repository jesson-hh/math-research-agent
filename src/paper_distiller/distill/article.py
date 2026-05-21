"""Distill one paper into an ArticleResult.

Three steps:
  1. Render article.md prompt with paper + full_text + wiki_index + retrieved
     prior theorems (from ProofStore) as cross-reference context
  2. Call LLM with response_format=json
  3. Parse, scrub invented [[wikilinks]], return ArticleResult (which also
     carries the extracted proof_sidecar so the agent can persist it).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..llm.openai_compatible import LLMClient, LLMError
from ..proofs.store import ProofSidecar, Theorem
from ..sources.arxiv import ArxivPaper
from ..vault.crosslink import WikiIndex
from ..vault.store import slugify

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "article.md"
_LINK_RE = re.compile(r"\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]")


@dataclass
class ArticleResult:
    slug: str
    title: str
    body: str
    tags: list
    refs: list
    depth: str  # "full-pdf" or "abstract-only"
    proof_sidecar: ProofSidecar = field(default_factory=ProofSidecar)

    def to_save_kwargs(self) -> dict:
        return {
            "title": self.title,
            "body": self.body,
            "tags": self.tags,
            "refs": self.refs,
            "slug": self.slug,
        }


def _format_prior_theorems_for_prompt(theorems: list[Theorem]) -> str:
    """Render retrieved theorems as a markdown block to prepend before paper text.

    Empty list → returns empty string (the prompt block stays clean).
    """
    if not theorems:
        return ""
    lines = ["# 已知相关定理（vault 历史，仅供参考，可对照命名 / 复用证明思路）", ""]
    for thm in theorems:
        lines.append(f"## {thm.name} — from `arxiv:{thm.paper_arxiv_id}`")
        if thm.statement:
            lines.append(f"**Statement**: {thm.statement}")
        if thm.proof_sketch:
            lines.append(f"**Proof sketch**: {thm.proof_sketch}")
        if thm.techniques_used:
            lines.append(f"**Techniques**: {', '.join(thm.techniques_used)}")
        lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _scrub_invented_links(body: str, valid_slugs: set) -> str:
    """Replace [[slug]] / [[slug|Display]] with bare slug or display text when slug is not in valid_slugs."""
    def repl(m):
        slug = m.group(1).strip()
        display = m.group(2)
        if slug in valid_slugs:
            return m.group(0)
        return display if display else slug
    return _LINK_RE.sub(repl, body)


def distill(
    paper: ArxivPaper,
    full_text: str,
    wiki_index: WikiIndex,
    llm: LLMClient,
    prior_theorems: list[Theorem] | None = None,
) -> ArticleResult:
    """LLM-distill one paper into an article entry.

    `prior_theorems` (v1.8): theorems retrieved from the vault's ProofStore
    that are likely relevant to this paper. They get rendered as a markdown
    block prepended before the paper text so the LLM can:
      - reuse consistent notation across the vault
      - cite "[[paper X]]'s Theorem 4.3" when a result is reused
      - flag duplicates / contradictions
    Pass an empty list (or None) on the very first paper or when no proof
    store is available — distill still works (no RAG).
    """
    depth_mode = "full-pdf" if full_text and len(full_text) > 500 else "abstract-only"
    body_input = full_text if depth_mode == "full-pdf" else paper.abstract

    index_lines = wiki_index.to_prompt_lines() or ["(vault is empty — no crosslinks yet)"]
    wiki_index_block = "\n".join(index_lines[:200])  # cap at 200 lines (~10K tokens)

    prior_block = _format_prior_theorems_for_prompt(prior_theorems or [])
    text_with_context = prior_block + body_input if prior_block else body_input

    # qwen3.5-plus / qwen-plus support 128K tokens ≈ 400K chars input.
    # Reserve ~30K chars for prompt header + wiki index + JSON template,
    # leaving ~350K chars for the paper text. Use 250K as a safe ceiling.
    _MAX_TEXT_CHARS = 250_000
    prompt = _PROMPT_FILE.read_text(encoding="utf-8").format(
        paper_title=paper.title,
        paper_authors=", ".join(paper.authors),
        paper_arxiv_id=paper.arxiv_id,
        paper_published=paper.published,
        paper_abstract=paper.abstract,
        depth_mode=depth_mode,
        full_text=text_with_context[:_MAX_TEXT_CHARS],
        wiki_index_block=wiki_index_block,
    )
    raw = llm.complete(
        [{"role": "user", "content": prompt}],
        temperature=0.5,
        response_format="json",
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"article distillation returned non-JSON: {raw[:200]}") from e

    title = parsed.get("title", paper.title).strip()
    body = parsed.get("body", "").strip()
    tags = parsed.get("tags", []) or []
    refs = parsed.get("refs", []) or []
    sidecar_raw = parsed.get("proof_sidecar") or {}

    # Inject canonical ref(s). Priority: arxiv -> doi -> ss_paper_id.
    canonical_refs: list[str] = []
    if paper.arxiv_id:
        canonical_refs.append(f"arxiv:{paper.arxiv_id}")
    if paper.doi:
        canonical_refs.append(f"doi:{paper.doi}")
    if not canonical_refs and paper.ss_paper_id:
        canonical_refs.append(f"ss:{paper.ss_paper_id}")
    refs = canonical_refs + [r for r in refs if r not in canonical_refs]
    if not body:
        raise LLMError("article distillation returned empty body")

    body = _scrub_invented_links(body, wiki_index.slugs())
    slug = slugify(title)

    return ArticleResult(
        slug=slug,
        title=title,
        body=body,
        tags=tags,
        refs=refs,
        depth=depth_mode,
        proof_sidecar=ProofSidecar.from_json(sidecar_raw),
    )
