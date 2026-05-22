"""Tests for proofgraph.reviewer — review_node, compute_taint, review_target."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store(tmp_path):
    from paper_distiller.proofs.store import ProofStore
    return ProofStore(tmp_path / "proofs.db")


def _node(paper_arxiv_id="P", kind="proof_step", text="some text", techniques=None):
    from paper_distiller.proofs.store import Node
    return Node(
        paper_arxiv_id=paper_arxiv_id,
        kind=kind,
        text=text,
        label=None,
        source_quote="verbatim span",
        techniques=techniques or [],
    )


class _StubLLM:
    """Returns a fixed JSON string from .complete()."""

    def __init__(self, response: str):
        self._response = response
        self.call_count = 0

    def complete(self, messages, temperature=0.2, response_format=None):
        self.call_count += 1
        return self._response


class _RecordingLLM:
    """Captures the prompt text passed to .complete()."""

    def __init__(self, response='{"label":"ok","reason":"well-formed","confidence":0.5}'):
        self._response = response
        self.last_prompt = ""

    def complete(self, messages, temperature=0.2, response_format=None):
        self.last_prompt = messages[0]["content"]
        return self._response


# ---------------------------------------------------------------------------
# Statement-vs-step prompt handling (eval-driven fix)
# ---------------------------------------------------------------------------

def test_review_prompt_treats_statement_nodes_as_premises(tmp_path):
    """A theorem-kind node's prompt carries its kind + the statement-as-premise rule."""
    from paper_distiller.proofs.store import Node
    from paper_distiller.proofgraph.reviewer import review_node
    store = _store(tmp_path)
    nid = store.add_node(Node(paper_arxiv_id="P", kind="theorem",
                              text="For all x, P(x).", label="Theorem 1",
                              source_quote="For all x, P(x)."))
    node = store.get_node(nid)
    llm = _RecordingLLM()
    review_node(store, node, llm)
    p = llm.last_prompt.lower()
    assert "theorem" in p
    assert "premise" in p  # statement nodes judged as premises, not unsupported steps
    store.close()


def test_review_prompt_scrutinizes_step_nodes(tmp_path):
    """A proof_step node's prompt still instructs 'follows from dependencies' scrutiny."""
    from paper_distiller.proofs.store import Node
    from paper_distiller.proofgraph.reviewer import review_node
    store = _store(tmp_path)
    nid = store.add_node(Node(paper_arxiv_id="P", kind="proof_step",
                              text="By Holder, A<=B.", source_quote="By Holder, A<=B."))
    node = store.get_node(nid)
    llm = _RecordingLLM()
    review_node(store, node, llm)
    p = llm.last_prompt.lower()
    assert "proof_step" in p
    assert "follow" in p  # steps must follow from parents/technique
    store.close()


# ---------------------------------------------------------------------------
# Task 6.1 — review_node
# ---------------------------------------------------------------------------

def test_review_node_suspicious_label_capped(tmp_path):
    """LLM returns suspicious + high confidence → confidence capped at 0.7."""
    from paper_distiller.proofgraph.reviewer import review_node, ReviewResult

    store = _store(tmp_path)

    # Add a parent node
    from paper_distiller.proofs.store import Node, Edge
    parent = _node(text="parent assertion", techniques=[])
    pid = store.add_node(parent)

    # The node under review
    node = _node(text="step claiming A from B", techniques=["Bernstein"])
    nid = store.add_node(node)
    node.id = nid

    # Add a depends_on edge: node -> parent
    store.add_edge(Edge(src_id=nid, dst_id=pid, rel="depends_on"))

    # Add a kb exemplar node using same technique
    kb_node = _node(paper_arxiv_id="P", text="another Bernstein usage", techniques=["Bernstein"])
    store.add_node(kb_node)

    llm = _StubLLM('{"label":"suspicious","reason":"the leap from A to B is unjustified","confidence":0.9}')

    result = review_node(store, node, llm)

    assert isinstance(result, ReviewResult)
    assert result.node_id == nid
    assert result.label == "suspicious"
    assert "unjustified" in result.reason
    # confidence must be capped at 0.7
    assert result.confidence <= 0.7
    assert result.tainted_by == []
    store.close()


def test_review_node_junk_output_abstains(tmp_path):
    """Junk LLM output → label=unstated, confidence=0.0, no crash."""
    from paper_distiller.proofgraph.reviewer import review_node

    store = _store(tmp_path)
    node = _node()
    node.id = store.add_node(node)

    llm = _StubLLM("NOT JSON AT ALL !!!!")

    result = review_node(store, node, llm)

    assert result.label == "unstated"
    assert result.confidence == 0.0
    store.close()


def test_review_node_invalid_label_abstains(tmp_path):
    """LLM returns a label not in LABELS → label=unstated."""
    from paper_distiller.proofgraph.reviewer import review_node

    store = _store(tmp_path)
    node = _node()
    node.id = store.add_node(node)

    llm = _StubLLM('{"label":"nonsense","reason":"whatever","confidence":0.5}')

    result = review_node(store, node, llm)

    assert result.label == "unstated"
    assert result.confidence == 0.0
    store.close()


def test_review_node_ok_label(tmp_path):
    """LLM returns ok → ReviewResult with label ok."""
    from paper_distiller.proofgraph.reviewer import review_node

    store = _store(tmp_path)
    node = _node()
    node.id = store.add_node(node)

    llm = _StubLLM('{"label":"ok","reason":"follows directly","confidence":0.6}')

    result = review_node(store, node, llm)

    assert result.label == "ok"
    assert result.confidence == pytest.approx(0.6)
    store.close()


def test_review_node_confidence_min_float_zero(tmp_path):
    """confidence field missing in LLM output → 0.0."""
    from paper_distiller.proofgraph.reviewer import review_node

    store = _store(tmp_path)
    node = _node()
    node.id = store.add_node(node)

    llm = _StubLLM('{"label":"gap","reason":"missing justification"}')

    result = review_node(store, node, llm)

    assert result.label == "gap"
    assert result.confidence == 0.0
    store.close()


# ---------------------------------------------------------------------------
# Task 6.2 — compute_taint
# ---------------------------------------------------------------------------

def test_compute_taint_transitive(tmp_path):
    """B depends_on A (suspicious), C depends_on B (ok) → both B and C tainted by A."""
    from paper_distiller.proofgraph.reviewer import compute_taint
    from paper_distiller.proofs.store import Edge

    store = _store(tmp_path)

    A = _node(text="A"); A_id = store.add_node(A); A.id = A_id
    B = _node(text="B"); B_id = store.add_node(B); B.id = B_id
    C = _node(text="C"); C_id = store.add_node(C); C.id = C_id

    store.add_edge(Edge(src_id=B_id, dst_id=A_id, rel="depends_on"))
    store.add_edge(Edge(src_id=C_id, dst_id=B_id, rel="depends_on"))

    label_by_id = {A_id: "suspicious", B_id: "ok", C_id: "ok"}

    taint = compute_taint(store, [A_id, B_id, C_id], label_by_id)

    # A is the problem source — not in tainted dict
    assert A_id not in taint
    # B is directly downstream of A
    assert taint[B_id] == [A_id]
    # C is transitively downstream of A
    assert taint[C_id] == [A_id]
    store.close()


def test_compute_taint_source_not_tainted(tmp_path):
    """The problem node itself must NOT appear in the taint dict."""
    from paper_distiller.proofgraph.reviewer import compute_taint
    from paper_distiller.proofs.store import Edge

    store = _store(tmp_path)
    A = _node(text="A"); A_id = store.add_node(A)
    B = _node(text="B"); B_id = store.add_node(B)

    store.add_edge(Edge(src_id=B_id, dst_id=A_id, rel="depends_on"))
    label_by_id = {A_id: "gap", B_id: "ok"}

    taint = compute_taint(store, [A_id, B_id], label_by_id)

    assert A_id not in taint
    assert B_id in taint
    store.close()


def test_compute_taint_no_problem(tmp_path):
    """All ok labels → empty taint dict."""
    from paper_distiller.proofgraph.reviewer import compute_taint
    from paper_distiller.proofs.store import Edge

    store = _store(tmp_path)
    A = _node(text="A"); A_id = store.add_node(A)
    B = _node(text="B"); B_id = store.add_node(B)

    store.add_edge(Edge(src_id=B_id, dst_id=A_id, rel="depends_on"))
    label_by_id = {A_id: "ok", B_id: "ok"}

    taint = compute_taint(store, [A_id, B_id], label_by_id)

    assert taint == {}
    store.close()


def test_compute_taint_cycle_safe(tmp_path):
    """Cycle A→B→A must not loop forever."""
    from paper_distiller.proofgraph.reviewer import compute_taint
    from paper_distiller.proofs.store import Edge

    store = _store(tmp_path)
    A = _node(text="A"); A_id = store.add_node(A)
    B = _node(text="B"); B_id = store.add_node(B)

    store.add_edge(Edge(src_id=A_id, dst_id=B_id, rel="depends_on"))
    store.add_edge(Edge(src_id=B_id, dst_id=A_id, rel="depends_on"))
    label_by_id = {A_id: "suspicious", B_id: "ok"}

    # Must not raise / hang
    taint = compute_taint(store, [A_id, B_id], label_by_id)
    # B depends on A (suspicious), so B tainted
    assert B_id in taint
    store.close()


def test_compute_taint_multiple_problem_ancestors(tmp_path):
    """Node C rests on both A (suspicious) and B (gap) → tainted_by lists both."""
    from paper_distiller.proofgraph.reviewer import compute_taint
    from paper_distiller.proofs.store import Edge

    store = _store(tmp_path)
    A = _node(text="A"); A_id = store.add_node(A)
    B = _node(text="B"); B_id = store.add_node(B)
    C = _node(text="C"); C_id = store.add_node(C)

    store.add_edge(Edge(src_id=C_id, dst_id=A_id, rel="depends_on"))
    store.add_edge(Edge(src_id=C_id, dst_id=B_id, rel="depends_on"))
    label_by_id = {A_id: "suspicious", B_id: "gap", C_id: "ok"}

    taint = compute_taint(store, [A_id, B_id, C_id], label_by_id)

    assert set(taint[C_id]) == {A_id, B_id}
    store.close()


# ---------------------------------------------------------------------------
# Task 6.3 — review_target
# ---------------------------------------------------------------------------

class _LabelCyclerLLM:
    """Returns 'ok' for first call (theorem), 'suspicious' for second (step1), 'ok' for rest."""

    def __init__(self):
        self.call_count = 0

    def complete(self, messages, temperature=0.2, response_format=None):
        self.call_count += 1
        if self.call_count == 2:
            return '{"label":"suspicious","reason":"step 1 is unjustified","confidence":0.8}'
        return '{"label":"ok","reason":"follows directly","confidence":0.5}'


def test_review_target_full_report(tmp_path):
    """Theorem + 2 steps: step1 suspicious → step2 tainted; all statuses persisted."""
    from paper_distiller.proofgraph.reviewer import review_target, ReviewReport
    from paper_distiller.proofs.store import Node, Edge

    store = _store(tmp_path)

    # Seed: theorem + two proof steps where step2 depends_on step1
    theorem = Node(
        paper_arxiv_id="P", kind="theorem",
        text="Main theorem", source_quote="verbatim",
    )
    th_id = store.add_node(theorem)

    step1 = Node(
        paper_arxiv_id="P", kind="proof_step",
        text="First step", source_quote="verbatim step 1",
    )
    s1_id = store.add_node(step1); step1.id = s1_id

    step2 = Node(
        paper_arxiv_id="P", kind="proof_step",
        text="Second step", source_quote="verbatim step 2",
    )
    s2_id = store.add_node(step2); step2.id = s2_id

    # step2 depends_on step1
    store.add_edge(Edge(src_id=s2_id, dst_id=s1_id, rel="depends_on"))

    llm = _LabelCyclerLLM()

    report = review_target(store, paper_arxiv_id="P", llm=llm)

    # Every node's status must be persisted
    assert store.get_node(th_id).status == "ok"
    assert store.get_node(s1_id).status == "suspicious"
    assert store.get_node(s2_id).status == "ok"

    # Report counts
    assert isinstance(report, ReviewReport)
    assert report.nodes_reviewed == 3
    assert report.by_label["suspicious"] == 1
    assert report.by_label["ok"] == 2

    # step1 flagged (own problem); step2 flagged (tainted by step1)
    flagged_ids = [r.node_id for r in report.flagged]
    assert s1_id in flagged_ids
    assert s2_id in flagged_ids

    # Own-problems come before tainted-only
    own_problems = [r for r in report.flagged if r.label in {"suspicious", "gap", "unsupported"}]
    tainted_only = [r for r in report.flagged if r.label not in {"suspicious", "gap", "unsupported"} and r.tainted_by]
    assert report.flagged[:len(own_problems)] == own_problems
    assert report.flagged[len(own_problems):] == tainted_only

    # Tainted node carries the ancestor id
    tainted_result = next(r for r in report.flagged if r.node_id == s2_id)
    assert s1_id in tainted_result.tainted_by

    store.close()


def test_review_target_node_id_mode(tmp_path):
    """review_target with node_id= collects that node + dependency_walk."""
    from paper_distiller.proofgraph.reviewer import review_target
    from paper_distiller.proofs.store import Node, Edge

    store = _store(tmp_path)

    root = Node(paper_arxiv_id="Q", kind="theorem", text="Root theorem", source_quote="q")
    r_id = store.add_node(root); root.id = r_id

    child = Node(paper_arxiv_id="Q", kind="proof_step", text="Child step", source_quote="c")
    c_id = store.add_node(child); child.id = c_id

    store.add_edge(Edge(src_id=r_id, dst_id=c_id, rel="depends_on"))

    llm = _StubLLM('{"label":"ok","reason":"fine","confidence":0.4}')

    report = review_target(store, node_id=r_id, llm=llm)

    # Should review root + its dependency_walk (child)
    assert report.nodes_reviewed >= 1
    assert report.target == f"node:{r_id}"
    store.close()


def test_review_target_no_target_raises(tmp_path):
    """Calling with neither paper_arxiv_id nor node_id must raise ValueError."""
    from paper_distiller.proofgraph.reviewer import review_target

    store = _store(tmp_path)
    llm = _StubLLM('{"label":"ok","reason":"x","confidence":0.1}')

    with pytest.raises(ValueError):
        review_target(store, llm=llm)

    store.close()


def test_review_target_summary_string(tmp_path):
    """report.summary is a non-empty string."""
    from paper_distiller.proofgraph.reviewer import review_target
    from paper_distiller.proofs.store import Node

    store = _store(tmp_path)
    node = Node(paper_arxiv_id="R", kind="theorem", text="T", source_quote="s")
    store.add_node(node)

    llm = _StubLLM('{"label":"ok","reason":"fine","confidence":0.3}')

    report = review_target(store, paper_arxiv_id="R", llm=llm)

    assert isinstance(report.summary, str) and report.summary
    store.close()


# ---------------------------------------------------------------------------
# Task 6.4 — tool_review_proof (tested separately in test_agent_tools.py)
# but ReviewResult / ReviewReport dataclass JSON-serializability tested here
# ---------------------------------------------------------------------------

def test_review_result_is_dict_convertible():
    """ReviewResult can be converted to a plain dict (for JSON serialization)."""
    from paper_distiller.proofgraph.reviewer import ReviewResult
    import dataclasses

    r = ReviewResult(node_id=1, label="ok", reason="fine", confidence=0.5, tainted_by=[])
    d = dataclasses.asdict(r)
    assert d["node_id"] == 1
    assert d["label"] == "ok"
