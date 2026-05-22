"""Structured informal review of a proof graph.

Walk a paper's proof dependency DAG, label each node with a structured
judgment, propagate early-error taint to descendants, persist labels, and
build a prioritised report.

Review LOCATES suspicious steps / gaps; it does NOT certify correctness.
LLM confidence is down-weighted (capped at 0.7) per the "Proof or Bluff?"
caveat — LLM judges are near-chance on proof soundness.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..proofs.store import ProofStore, Node


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

LABELS = {"ok", "suspicious", "gap", "unsupported", "unstated"}
PROBLEM = {"suspicious", "gap", "unsupported"}


@dataclass
class ReviewResult:
    """Review judgment for one graph node."""

    node_id: int
    label: str
    reason: str
    confidence: float
    tainted_by: list[int] = field(default_factory=list)


@dataclass
class ReviewReport:
    """Aggregated review for a target (paper or subtree)."""

    target: str
    nodes_reviewed: int
    by_label: dict[str, int]
    flagged: list[ReviewResult]
    summary: str


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).parent / "prompts" / "review_node.md"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _fmt_nodes(nodes) -> str:
    if not nodes:
        return "(none)"
    parts = []
    for n in nodes:
        parts.append(
            f"- [{n.kind}] {n.label or ''}: {n.text}\n"
            f"  Quote: {n.source_quote or '(none)'}"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Task 6.1: review_node
# ---------------------------------------------------------------------------

def review_node(store: "ProofStore", node: "Node", llm) -> ReviewResult:
    """Judge one node in local context; return a ReviewResult.

    Never raises. On any parse / LLM error: returns ReviewResult with
    label='unstated', confidence=0.0.
    """
    try:
        # Gather local context ---------------------------------------------------
        dep_rels = {"depends_on", "uses_lemma", "uses_def", "uses_assumption"}
        parents = [
            store.get_node(e.dst_id)
            for e in store.out_edges(node.id)
            if e.rel in dep_rels
        ]
        parents = [p for p in parents if p is not None]

        same_as = [
            store.get_node(e.dst_id)
            for e in store.out_edges(node.id, "same_as")
        ]
        same_as = [n for n in same_as if n is not None]

        # KB exemplars: up to 3 per technique, excluding the node itself
        kb_nodes: list = []
        for t in (node.techniques or []):
            candidates = store.nodes_using_technique(t, limit=4)
            for c in candidates:
                if c.id != node.id and c not in kb_nodes:
                    kb_nodes.append(c)
                    if len(kb_nodes) >= 3 * max(1, len(node.techniques or [])):
                        break

        # Build prompt -----------------------------------------------------------
        template = _load_prompt()
        prompt = (
            template
            .replace("{kind}", node.kind)
            .replace("{label}", node.label or "(unlabeled)")
            .replace("{text}", node.text)
            .replace("{source_quote}", node.source_quote or "(no quote)")
            .replace("{parents_text}", _fmt_nodes(parents))
            .replace("{kb_text}", _fmt_nodes(kb_nodes[:3]))
            .replace("{same_as_text}", _fmt_nodes(same_as))
        )

        # LLM call ---------------------------------------------------------------
        raw = llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format="json",
        )

        # Parse response ---------------------------------------------------------
        parsed = json.loads(raw)
        label = parsed.get("label", "")
        reason = parsed.get("reason", "review inconclusive")
        confidence_raw = parsed.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        if label not in LABELS:
            return ReviewResult(
                node_id=node.id,
                label="unstated",
                reason="review inconclusive",
                confidence=0.0,
            )

        # Down-weight: cap at 0.7
        confidence = min(confidence, 0.7)

        return ReviewResult(
            node_id=node.id,
            label=label,
            reason=str(reason),
            confidence=confidence,
        )

    except Exception:
        return ReviewResult(
            node_id=node.id,
            label="unstated",
            reason="review inconclusive",
            confidence=0.0,
        )


# ---------------------------------------------------------------------------
# Task 6.2: compute_taint
# ---------------------------------------------------------------------------

def compute_taint(
    store: "ProofStore",
    node_ids: list[int],
    label_by_id: dict[int, str],
) -> dict[int, list[int]]:
    """For each node, find problem-label depends_on ancestors transitively.

    Returns a dict mapping node_id → sorted list of ancestor ids whose label
    is in PROBLEM. Only nodes that ARE tainted (have at least one such
    ancestor) appear in the result. The problem node itself is NOT included
    (it's the source, not tainted by an ancestor).

    Cycle-safe via a visited set per walk.
    """
    taint: dict[int, list[int]] = {}
    node_id_set = set(node_ids)

    for nid in node_ids:
        # BFS over depends_on ancestors
        visited: set[int] = {nid}
        queue: deque[int] = deque([nid])
        problem_ancestors: list[int] = []

        while queue:
            cur = queue.popleft()
            for edge in store.out_edges(cur, "depends_on"):
                anc = edge.dst_id
                if anc in visited:
                    continue
                visited.add(anc)
                # Only care about ancestors within our reviewed set
                if anc in node_id_set and label_by_id.get(anc) in PROBLEM:
                    problem_ancestors.append(anc)
                queue.append(anc)

        if problem_ancestors:
            taint[nid] = sorted(problem_ancestors)

    return taint


# ---------------------------------------------------------------------------
# Task 6.3: review_target
# ---------------------------------------------------------------------------

def review_target(
    store: "ProofStore",
    *,
    paper_arxiv_id: str | None = None,
    node_id: int | None = None,
    llm,
) -> ReviewReport:
    """Orchestrate review of a paper or subtree.

    Walks nodes, calls review_node for each, persists statuses, propagates
    taint, and returns a ReviewReport.
    """
    if paper_arxiv_id is not None:
        nodes = store.nodes_by_paper(paper_arxiv_id)
        target_str = paper_arxiv_id
    elif node_id is not None:
        root = store.get_node(node_id)
        if root is None:
            nodes = []
        else:
            nodes = [root] + store.dependency_walk(node_id)
        target_str = f"node:{node_id}"
    else:
        raise ValueError("review_target requires paper_arxiv_id or node_id")

    # Review each node and persist status
    results: list[ReviewResult] = []
    label_by_id: dict[int, str] = {}

    for node in nodes:
        r = review_node(store, node, llm)
        store.set_node_status(node.id, r.label)
        results.append(r)
        label_by_id[node.id] = r.label

    # Compute taint and attach to results
    reviewed_ids = [n.id for n in nodes]
    taint = compute_taint(store, reviewed_ids, label_by_id)
    for r in results:
        r.tainted_by = taint.get(r.node_id, [])

    # Build flagged list: own problems first, then tainted-only
    own_problems = [r for r in results if r.label in PROBLEM]
    tainted_only = [
        r for r in results
        if r.label not in PROBLEM and r.tainted_by
    ]
    flagged = own_problems + tainted_only

    # Count labels
    from collections import Counter
    by_label = dict(Counter(r.label for r in results))

    n_flagged = len(flagged)
    n_total = len(nodes)
    summary = f"{n_flagged} of {n_total} nodes need attention"

    return ReviewReport(
        target=target_str,
        nodes_reviewed=n_total,
        by_label=by_label,
        flagged=flagged,
        summary=summary,
    )
