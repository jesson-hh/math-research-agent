"""PaperProcessor — fanout agent: one sub-agent per paper. Each does
fetch + extract + distill independently, in parallel.

v1.8: each distillation also extracts a `proof_sidecar` (theorems +
techniques) and persists it into the per-vault ProofStore for future
cross-paper retrieval. Before distilling a paper, we query the store for
likely-relevant prior theorems (based on the paper abstract's mentioned
techniques) and inject them as context.

Per-paper LLM failures are logged + dropped — they don't abort the run.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path

from ..distill.article import distill as distill_article
from ..llm.openai_compatible import LLMError
from ..pipeline import fetch_with_fallback
from ..proofs.store import open_for_vault
from ..vault.crosslink import load_index
from .base import Agent, Context

# Module-level lock — fine because each run() invocation creates a fresh DAG.
# Multiple _DistillOne instances inside one fanout invocation share this lock
# to serialize their ctx.shared["articles"].append() calls.
_articles_lock = asyncio.Lock()
_proof_store_lock = asyncio.Lock()


# Hand-curated list of common math/CS technique names to scan for in the
# paper abstract. Cheap pre-filter that picks "candidate techniques" for
# proof store retrieval — much cheaper than an LLM extraction pass.
_TECHNIQUE_HINTS = [
    "Hölder", "Holder", "Cauchy-Schwarz", "Bernstein", "Hoeffding",
    "Markov inequality", "Chebyshev", "Jensen", "Pinsker",
    "Dudley", "chaining", "Rademacher", "VC dimension",
    "Lipschitz", "Sobolev", "Lipschitz extension",
    "Wasserstein", "optimal transport", "Kantorovich",
    "Talagrand", "concentration", "sub-Gaussian", "sub-Exponential",
    "martingale", "Doob",
    "ReLU", "neural network approximation", "Barron",
    "Bayesian", "posterior", "variational",
    "diffusion", "score-matching", "Langevin", "Stein", "Fisher",
    "gradient descent", "SGD", "Adam", "momentum",
    "convex", "non-convex", "saddle point",
    "RKHS", "kernel", "GP",
    "PAC-Bayes", "minimax", "Le Cam",
    "TV distance", "KL divergence", "f-divergence", "IPM",
]


def _extract_candidate_techniques(paper) -> list[str]:
    """Cheap keyword scan of title + abstract for known technique names.

    Used to pre-fetch potentially relevant theorems from the ProofStore
    before LLM distillation. Best-effort — false negatives are OK (we
    just won't inject as much context); false positives also OK (the LLM
    can ignore irrelevant references).
    """
    haystack = " ".join([
        getattr(paper, "title", "") or "",
        getattr(paper, "abstract", "") or "",
    ]).lower()
    found = []
    for tech in _TECHNIQUE_HINTS:
        if tech.lower() in haystack:
            found.append(tech)
    return found


class _DistillOne:
    def __init__(self, paper, idx, total, tmpdir, wiki_index, proof_store):
        self.name = f"paper-processor[{idx + 1}/{total}]"
        self.deps: list[str] = []
        self._paper = paper
        self._tmpdir = tmpdir
        self._wiki_index = wiki_index
        self._proof_store = proof_store

    async def run(self, ctx: Context) -> dict:
        title_preview = (getattr(self._paper, "title", "") or "")[:50]
        try:
            try:
                ctx.on_status(
                    self.name,
                    activity=f"PDF fetch: {self._paper.arxiv_id}",
                )
            except Exception:
                pass
            full_text = await asyncio.to_thread(
                fetch_with_fallback, self._paper, ctx.cfg, self._tmpdir,
            )

            # Pre-fetch relevant prior theorems from the ProofStore
            prior_theorems = []
            if self._proof_store is not None:
                candidates = _extract_candidate_techniques(self._paper)
                if candidates:
                    try:
                        ctx.on_status(
                            self.name,
                            activity=f"proof RAG: {len(candidates)} candidate techniques",
                        )
                    except Exception:
                        pass
                    try:
                        prior_theorems = await asyncio.to_thread(
                            self._proof_store.retrieve_relevant,
                            candidates,
                        )
                    except Exception:
                        prior_theorems = []

            try:
                ctx.on_status(
                    self.name,
                    activity=(
                        f"LLM distill: {title_preview}"
                        + (f"  (+{len(prior_theorems)} prior theorems)"
                           if prior_theorems else "")
                    ),
                )
            except Exception:
                pass
            article = await asyncio.to_thread(
                distill_article, self._paper, full_text,
                self._wiki_index, ctx.llm, prior_theorems,
            )
        except LLMError:
            if ctx.cfg.verbose:
                print(f"  distill failed for {self._paper.arxiv_id}")
            return {}
        async with _articles_lock:
            current = ctx.shared.get("articles", [])
            current.append(article)
            ctx.shared["articles"] = current

        # Persist sidecar into proof store (serialized via lock)
        if self._proof_store is not None and article.proof_sidecar.theorems:
            try:
                async with _proof_store_lock:
                    await asyncio.to_thread(
                        self._proof_store.ingest_sidecar,
                        article.proof_sidecar,
                        self._paper.arxiv_id or "",
                        article.slug,
                    )
            except Exception:
                pass  # never let proof-store errors abort distillation
        return {}


class PaperProcessor:
    """Fanout agent — produces N _DistillOne sub-agents at runtime.

    Sub-agents MUST have deps=[] — they run as a synthetic single-level
    fanout, not via topological sort.
    """
    name = "paper-processor"
    deps = ["candidate-ranker"]

    def expand(self, ctx: Context) -> list[Agent]:
        # Always setdefault — never clobber. QA-mode accumulates articles
        # across rounds; a round with zero ranked papers must not wipe them.
        ctx.shared.setdefault("articles", [])
        papers = ctx.shared.get("ranked", [])
        if not papers:
            return []
        tmpdir = Path(tempfile.mkdtemp(prefix="paper-distiller-"))
        wiki_index = load_index(ctx.vault)
        # Open the per-vault ProofStore once and share it across sub-agents.
        # _proof_store_lock serializes writes; concurrent reads are safe.
        try:
            proof_store = open_for_vault(ctx.vault.root)
        except Exception:
            proof_store = None
        return [
            _DistillOne(p, i, len(papers), tmpdir, wiki_index, proof_store)
            for i, p in enumerate(papers)
        ]
