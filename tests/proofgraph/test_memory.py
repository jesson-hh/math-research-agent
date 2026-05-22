"""Tests for proofgraph.memory — RunningMemory structured carry-forward state."""
from __future__ import annotations


def _make_extracted_node(kind, label=None, refs=None, status="extracted"):
    """Build a minimal ExtractedNode-like object for testing memory.update()."""
    from paper_distiller.proofgraph.extraction_schema import ExtractedNode, ExtractedRef
    node_refs = []
    if refs:
        for rel, target in refs:
            node_refs.append(ExtractedRef(rel=rel, target=target))
    return ExtractedNode(
        kind=kind,
        text="some text",
        source_quote="some quote",
        label=label,
        status=status,
        refs=node_refs,
    )


def test_update_definition_lands_in_definitions():
    from paper_distiller.proofgraph.memory import RunningMemory
    mem = RunningMemory()
    node = _make_extracted_node("definition", label="Definition 2.1")
    mem.update([node], resolved_labels=set())
    assert len(mem.definitions) == 1
    assert mem.definitions[0]["label"] == "Definition 2.1"


def test_update_theorem_lands_in_established():
    from paper_distiller.proofgraph.memory import RunningMemory
    mem = RunningMemory()
    node = _make_extracted_node("theorem", label="Theorem 4.3")
    mem.update([node], resolved_labels=set())
    assert len(mem.established) == 1
    assert mem.established[0]["label"] == "Theorem 4.3"


def test_update_lemma_lands_in_established():
    from paper_distiller.proofgraph.memory import RunningMemory
    mem = RunningMemory()
    node = _make_extracted_node("lemma", label="Lemma 3.1")
    mem.update([node], resolved_labels=set())
    assert len(mem.established) == 1


def test_update_unresolved_ref_in_obligations():
    from paper_distiller.proofgraph.memory import RunningMemory
    mem = RunningMemory()
    node = _make_extracted_node(
        "proof_step",
        refs=[("depends_on", "Lemma 9"), ("uses_def", "Definition 2.1")],
    )
    # Only "Lemma 9" is unresolved; "Definition 2.1" is resolved
    mem.update([node], resolved_labels={"Definition 2.1"})
    assert "Lemma 9" in mem.obligations
    assert "Definition 2.1" not in mem.obligations


def test_obligations_are_deduped():
    from paper_distiller.proofgraph.memory import RunningMemory
    mem = RunningMemory()
    n1 = _make_extracted_node("proof_step", refs=[("depends_on", "Lemma 9")])
    n2 = _make_extracted_node("proof_step", refs=[("depends_on", "Lemma 9")])
    mem.update([n1, n2], resolved_labels=set())
    assert mem.obligations.count("Lemma 9") == 1


def test_render_returns_nonempty_string_with_known_label():
    from paper_distiller.proofgraph.memory import RunningMemory
    mem = RunningMemory()
    node = _make_extracted_node("theorem", label="Theorem 4.3")
    mem.update([node], resolved_labels=set())
    rendered = mem.render()
    assert isinstance(rendered, str) and rendered
    assert "Theorem 4.3" in rendered


def test_render_bounded_with_100_items():
    from paper_distiller.proofgraph.memory import RunningMemory
    mem = RunningMemory()
    for i in range(100):
        node = _make_extracted_node("theorem", label=f"Theorem {i}")
        mem.update([node], resolved_labels=set())
    rendered = mem.render()
    # Should not explode; and length should be reasonable (capped)
    assert isinstance(rendered, str)
    # 100 items * "Theorem X\n" would be thousands of chars — cap enforced means fewer
    # We just check the render completes and the cap is respected at the data level
    assert len(mem.established) <= 20


def test_render_empty_memory():
    from paper_distiller.proofgraph.memory import RunningMemory
    mem = RunningMemory()
    rendered = mem.render()
    assert isinstance(rendered, str)  # empty but valid string is fine
