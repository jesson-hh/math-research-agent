"""Cross-paper linker: find candidate node pairs and classify their relation.

Phase 5 of the proof-graph build pipeline.
"""

from __future__ import annotations

from paper_distiller.proofs.store import Node, ProofStore


# ---------------------------------------------------------------------------
# Task 5.1 — find_candidates (deterministic, cross-paper only)
# ---------------------------------------------------------------------------


def find_candidates(store: ProofStore, node: Node, k: int = 6) -> list[Node]:
    """Return up to *k* cross-paper candidate nodes for *node*.

    Strategy (deterministic, no LLM):
      1. Technique overlap — for each technique on *node*, query
         ``store.nodes_using_technique(t, limit=k)``.
      2. FTS5 text match — ``store.search_nodes(node.text, limit=k)``.

    Guarantees:
      - Excludes any node whose ``paper_arxiv_id == node.paper_arxiv_id``.
      - Excludes ``node`` itself (same ``id``).
      - Deduplicates by ``id`` (first-seen order: technique matches first).
      - Returns first ``k`` after dedup.
    """
    seen: dict[int, Node] = {}

    def _add(candidate: Node) -> None:
        if candidate.id is None:
            return
        if candidate.paper_arxiv_id == node.paper_arxiv_id:
            return
        if candidate.id == node.id:
            return
        if candidate.id not in seen:
            seen[candidate.id] = candidate

    # Technique overlap (strategy A)
    for technique in node.techniques or []:
        for cand in store.nodes_using_technique(technique, limit=k):
            _add(cand)

    # FTS text match (strategy B)
    for cand in store.search_nodes(node.text, limit=k):
        _add(cand)

    return list(seen.values())[:k]
