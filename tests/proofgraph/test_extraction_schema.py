"""Tests for proofgraph.extraction_schema — tolerant JSON parser."""
from __future__ import annotations


def test_parse_valid_node_with_ref():
    from paper_distiller.proofgraph.extraction_schema import parse_extraction, ExtractedRef
    raw = '{"nodes":[{"kind":"proof_step","text":"t","source_quote":"q","refs":[{"rel":"depends_on","target":"L1"}]}]}'
    result = parse_extraction(raw)
    assert len(result) == 1
    node = result[0]
    assert node.kind == "proof_step"
    assert node.text == "t"
    assert node.source_quote == "q"
    assert len(node.refs) == 1
    assert isinstance(node.refs[0], ExtractedRef)
    assert node.refs[0].rel == "depends_on"
    assert node.refs[0].target == "L1"


def test_parse_garbage_returns_empty():
    from paper_distiller.proofgraph.extraction_schema import parse_extraction
    assert parse_extraction("garbage") == []
    assert parse_extraction("not json at all!!!") == []
    assert parse_extraction("{}") == []
    assert parse_extraction('{"nodes": "not a list"}') == []


def test_parse_missing_text_skips_node():
    from paper_distiller.proofgraph.extraction_schema import parse_extraction
    raw = '{"nodes":[{"kind":"proof_step","source_quote":"q"}]}'
    assert parse_extraction(raw) == []


def test_parse_missing_source_quote_skips_node():
    from paper_distiller.proofgraph.extraction_schema import parse_extraction
    raw = '{"nodes":[{"kind":"proof_step","text":"t"}]}'
    assert parse_extraction(raw) == []


def test_parse_non_list_refs_coerced_to_empty():
    from paper_distiller.proofgraph.extraction_schema import parse_extraction
    raw = '{"nodes":[{"kind":"proof_step","text":"t","source_quote":"q","refs":"not a list"}]}'
    result = parse_extraction(raw)
    assert len(result) == 1
    assert result[0].refs == []


def test_parse_non_list_techniques_coerced_to_empty():
    from paper_distiller.proofgraph.extraction_schema import parse_extraction
    raw = '{"nodes":[{"kind":"theorem","text":"t","source_quote":"q","techniques":"Bernstein"}]}'
    result = parse_extraction(raw)
    assert len(result) == 1
    assert result[0].techniques == []


def test_parse_accepts_dict_input():
    from paper_distiller.proofgraph.extraction_schema import parse_extraction
    data = {"nodes": [{"kind": "lemma", "text": "statement", "source_quote": "verbatim span"}]}
    result = parse_extraction(data)
    assert len(result) == 1
    assert result[0].kind == "lemma"


def test_parse_multiple_nodes_mixed_validity():
    from paper_distiller.proofgraph.extraction_schema import parse_extraction
    raw = '{"nodes":[{"kind":"theorem","text":"valid","source_quote":"q"},{"kind":"bad_node"}]}'
    result = parse_extraction(raw)
    assert len(result) == 1
    assert result[0].kind == "theorem"


def test_parse_label_none_when_missing():
    from paper_distiller.proofgraph.extraction_schema import parse_extraction
    raw = '{"nodes":[{"kind":"proof_step","text":"t","source_quote":"q"}]}'
    result = parse_extraction(raw)
    assert result[0].label is None


def test_parse_label_set_when_present():
    from paper_distiller.proofgraph.extraction_schema import parse_extraction
    raw = '{"nodes":[{"kind":"theorem","label":"Theorem 4.3","text":"t","source_quote":"q"}]}'
    result = parse_extraction(raw)
    assert result[0].label == "Theorem 4.3"


def test_parse_techniques_list_of_strings():
    from paper_distiller.proofgraph.extraction_schema import parse_extraction
    raw = '{"nodes":[{"kind":"proof_step","text":"t","source_quote":"q","techniques":["Bernstein","Dudley"]}]}'
    result = parse_extraction(raw)
    assert result[0].techniques == ["Bernstein", "Dudley"]


def test_default_status_is_extracted():
    from paper_distiller.proofgraph.extraction_schema import parse_extraction
    raw = '{"nodes":[{"kind":"proof_step","text":"t","source_quote":"q"}]}'
    result = parse_extraction(raw)
    assert result[0].status == "extracted"


# ---------------------------------------------------------------------------
# Fix 1: per-node local key
# ---------------------------------------------------------------------------

def test_parse_key_present():
    """A node JSON with "key":"n1" must parse into ExtractedNode.key == "n1"."""
    from paper_distiller.proofgraph.extraction_schema import parse_extraction
    raw = '{"nodes":[{"kind":"proof_step","text":"t","source_quote":"q","key":"n1"}]}'
    result = parse_extraction(raw)
    assert len(result) == 1
    assert result[0].key == "n1"


def test_parse_key_missing_defaults_to_none():
    """A node JSON without "key" must parse into ExtractedNode.key == None."""
    from paper_distiller.proofgraph.extraction_schema import parse_extraction
    raw = '{"nodes":[{"kind":"proof_step","text":"t","source_quote":"q"}]}'
    result = parse_extraction(raw)
    assert result[0].key is None
