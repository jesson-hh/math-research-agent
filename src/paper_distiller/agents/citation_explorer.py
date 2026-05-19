"""CitationExplorer — pull refs/cited-by from SS for each seed, rank by Jaccard relevance."""

from __future__ import annotations

import asyncio
import re

from ..sources.semantic_scholar import paper_refs as ss_paper_refs
from .base import Context


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokens(text: str) -> set:
    return set(t.lower() for t in _TOKEN_RE.findall(text or ""))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class CitationExplorer:
    name = "citation-explorer"
    deps: list[str] = []

    async def run(self, ctx: Context) -> dict:
        seeds = ctx.shared.get("seed_papers", [])
        qa_state = ctx.shared.get("qa_state")
        question = getattr(ctx.cfg, "qa_question", "") or ""
        seen = qa_state.articles_seen_ids if qa_state else set()
        per_round = getattr(ctx.cfg, "qa_per_round", 5) or 5
        # Top-K total candidates returned (3x headroom over per_round for downstream rerank)
        top_k = per_round * 3

        # 1. Pull refs for each seed (parallel via asyncio.gather)
        async def _refs_one(paper):
            pid = paper.arxiv_id or paper.doi
            if not pid:
                return []
            return await asyncio.to_thread(
                ss_paper_refs,
                arxiv_id_or_doi=pid,
                max_results=30,
                api_key=ctx.cfg.ss_api_key,
            )

        nested = await asyncio.gather(*[_refs_one(s) for s in seeds])
        all_candidates = [p for sub in nested for p in sub]

        # 2. Dedup against seen + within batch
        deduped = []
        seen_in_batch = set()
        for p in all_candidates:
            pid = p.arxiv_id or p.doi
            if pid and (pid in seen or pid in seen_in_batch):
                continue
            if pid:
                seen_in_batch.add(pid)
            deduped.append(p)

        # 3. Rank by Jaccard relevance against question + seed titles
        seed_text = question + " " + " ".join(s.title for s in seeds)
        seed_toks = _tokens(seed_text)
        ranked = sorted(
            deduped,
            key=lambda p: -_jaccard(seed_toks, _tokens(p.title + " " + (p.abstract or "")[:500])),
        )

        return {"citation_expansion_candidates": ranked[:top_k]}
