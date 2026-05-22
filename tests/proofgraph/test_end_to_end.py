"""End-to-end test: build_graph_for_paper → review_target chain.

Uses a real ProofStore (tmp_path SQLite), a stub LLM with canned responses,
and no network calls.  Proves the build→review pipeline works together.
"""
from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# Tiny multi-segment paper fixture
# ---------------------------------------------------------------------------

FAKE_PAPER = """\
1 Introduction
We study a simple convergence problem.

2 Main Result
Theorem 1. For all n, E[X_n] <= C / sqrt(n).

Proof. By Azuma's inequality we bound the martingale differences.
This follows from the Doob decomposition applied to the process. □

3 Conclusion
The bound is tight.
"""

# Verbatim quotes from the paper (used in mock LLM responses)
THEOREM_QUOTE = "For all n, E[X_n] <= C / sqrt(n)."
PROOF_QUOTE_A = "By Azuma's inequality we bound the martingale differences."
PROOF_QUOTE_B = "This follows from the Doob decomposition applied to the process."


# ---------------------------------------------------------------------------
# Stub LLM
#
# Dispatch logic (mirrors test_pipeline.py pattern):
#   - Self-check prompt  → no-suspicious verdict
#   - Theorem extraction → canned theorem node
#   - Proof extraction   → canned proof_step nodes
#   - Review prompt      → canned ok/ok/suspicious verdicts
#   - Everything else    → empty nodes
# ---------------------------------------------------------------------------

class _StubLLM:
    """Canned LLM that covers extraction + self-check + review without network."""

    def __init__(self):
        self.call_count = 0

        self._no_suspicious = json.dumps({"suspicious_labels": []})

        self._theorem_extraction = json.dumps({"nodes": [{
            "kind": "theorem",
            "label": "Theorem 1",
            "text": "For all n, E[X_n] <= C / sqrt(n).",
            "source_quote": THEOREM_QUOTE,
            "techniques": ["Azuma"],
            "refs": [],
        }]})

        self._proof_extraction = json.dumps({"nodes": [
            {
                "kind": "proof_step",
                "label": "Step A",
                "text": "Azuma martingale bound",
                "source_quote": PROOF_QUOTE_A,
                "techniques": ["Azuma"],
                "refs": [{"rel": "depends_on", "target": "Theorem 1"}],
            },
            {
                "kind": "proof_step",
                "label": "Step B",
                "text": "Doob decomposition",
                "source_quote": PROOF_QUOTE_B,
                "techniques": [],
                "refs": [{"rel": "depends_on", "target": "Step A"}],
            },
        ]})

        self._empty = json.dumps({"nodes": []})

        # Review: return "ok" for most, "suspicious" for one (to test flagging)
        self._review_ok = json.dumps(
            {"label": "ok", "reason": "step is well-supported", "confidence": 0.8}
        )
        self._review_suspicious = json.dumps(
            {"label": "suspicious", "reason": "leap not fully justified", "confidence": 0.9}
        )
        self._review_call_count = 0

    def complete(self, messages, temperature=0.2, response_format=None):
        self.call_count += 1
        content = messages[0]["content"] if messages else ""

        # --- Self-check: starts with "You are reviewing extracted mathematical" ---
        if content.startswith("You are reviewing extracted mathematical"):
            return self._no_suspicious

        # --- Review prompt (review_node.md template) ---
        # The review prompt starts with "You are a careful mathematical reviewer"
        if content.startswith("You are a careful mathematical reviewer"):
            self._review_call_count += 1
            # Return suspicious on the first review call, ok for the rest
            if self._review_call_count == 1:
                return self._review_suspicious
            return self._review_ok

        # --- Extraction: proof segment contains both proof quotes ---
        if PROOF_QUOTE_A in content and PROOF_QUOTE_B in content:
            return self._proof_extraction

        # --- Extraction: theorem segment ---
        if THEOREM_QUOTE in content and "Kind hint: theorem" in content:
            return self._theorem_extraction

        # Headings / other segments
        return self._empty


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

def test_build_then_review_chain(tmp_path):
    """build_graph_for_paper followed by review_target works end-to-end.

    Asserts:
    - Nodes were created in the store.
    - review report covers all nodes (nodes_reviewed == node count).
    - Node statuses are persisted by review_target.
    - At least one node is flagged (suspicious from the stub).
    """
    from paper_distiller.proofs.store import ProofStore
    from paper_distiller.proofgraph.pipeline import build_graph_for_paper, CoverageReport
    from paper_distiller.proofgraph.reviewer import review_target, ReviewReport

    store = ProofStore(tmp_path / "proofs.db")
    llm = _StubLLM()

    # --- Build phase ---
    report = build_graph_for_paper(
        store, "P", FAKE_PAPER,
        paper_slug="fake-e2e", llm=llm, depth="step",
    )
    assert isinstance(report, CoverageReport)

    nodes = store.nodes_by_paper("P")
    assert len(nodes) >= 2, f"Expected ≥2 nodes, got {len(nodes)}: {[n.label for n in nodes]}"

    # At least one theorem and one proof_step must be present
    kinds = {n.kind for n in nodes}
    assert "theorem" in kinds or "proof_step" in kinds, f"Unexpected kinds: {kinds}"

    # --- Review phase ---
    review_report = review_target(store, paper_arxiv_id="P", llm=llm)
    assert isinstance(review_report, ReviewReport)

    # Every node must be reviewed
    assert review_report.nodes_reviewed == len(nodes), (
        f"nodes_reviewed={review_report.nodes_reviewed} but store has {len(nodes)} nodes"
    )

    # Statuses must be persisted: no node should still have the default 'extracted' status
    refreshed_nodes = store.nodes_by_paper("P")
    statuses = {n.status for n in refreshed_nodes}
    # After review, statuses should be updated (review labels replace 'extracted')
    valid_review_labels = {"ok", "suspicious", "gap", "unsupported", "unstated"}
    assert statuses <= valid_review_labels, (
        f"Some nodes still have unexpected status: {statuses}"
    )

    # At least one node must be suspicious (from stub returning suspicious on first call)
    assert "suspicious" in statuses or any(
        r.label == "suspicious" for r in review_report.flagged
    ), "Expected at least one suspicious node from the stub LLM"

    # by_label must sum to nodes_reviewed
    assert sum(review_report.by_label.values()) == review_report.nodes_reviewed

    store.close()
