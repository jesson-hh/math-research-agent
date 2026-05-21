"""Integration test: distill → sidecar parsing → proof store ingest → retrieve."""

from __future__ import annotations

import json
from unittest.mock import MagicMock


def _mock_llm_response_with_sidecar(paper_arxiv_id="2110.12319"):
    """Realistic JSON shape we expect from the article.md prompt."""
    return json.dumps({
        "title": "测试论文",
        "body": "# 测试论文\n\n## TL;DR\nSome content.\n\n## 2. 设定\n$x \\in \\mathbb{R}$",
        "tags": ["test"],
        "refs": [f"arxiv:{paper_arxiv_id}"],
        "proof_sidecar": {
            "theorems": [
                {
                    "name": "Theorem 1",
                    "statement": "For all f, ||f|| <= C",
                    "proof_sketch": "By Bernstein concentration.",
                    "techniques_used": ["Bernstein", "Hölder"],
                },
            ],
            "key_definitions": [],
            "key_techniques": ["Bernstein", "Hölder", "Dudley chaining"],
        },
    })


def test_distill_parses_proof_sidecar(mocker):
    """distill() should populate ArticleResult.proof_sidecar from LLM JSON."""
    from paper_distiller.distill.article import distill
    from paper_distiller.sources.arxiv import Paper
    from paper_distiller.vault.crosslink import WikiIndex

    paper = Paper(
        source="arxiv", paper_id="2110.12319", title="T", authors=["A"],
        abstract="...", published="2021-10", pdf_url="x",
        arxiv_id="2110.12319",
    )
    llm = MagicMock()
    llm.complete.return_value = _mock_llm_response_with_sidecar()
    wiki = WikiIndex(entries=[])

    result = distill(paper, "x" * 1000, wiki, llm)
    assert len(result.proof_sidecar.theorems) == 1
    assert result.proof_sidecar.theorems[0]["name"] == "Theorem 1"
    assert "Bernstein" in result.proof_sidecar.key_techniques


def test_distill_handles_missing_sidecar(mocker):
    """Old prompts / older LLM responses may not include proof_sidecar."""
    from paper_distiller.distill.article import distill
    from paper_distiller.sources.arxiv import Paper
    from paper_distiller.vault.crosslink import WikiIndex

    paper = Paper(
        source="arxiv", paper_id="X", title="T", authors=["A"],
        abstract="...", published="", pdf_url="x", arxiv_id="X",
    )
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "title": "T", "body": "## TL;DR\nbody", "tags": [], "refs": [],
        # no proof_sidecar key
    })
    wiki = WikiIndex(entries=[])
    result = distill(paper, "x" * 1000, wiki, llm)
    # Should default to empty sidecar
    assert result.proof_sidecar.theorems == []
    assert result.proof_sidecar.key_techniques == []


def test_distill_injects_prior_theorems_into_prompt(mocker):
    """When prior_theorems is non-empty, the prompt should include the
    formatted prior-theorems block."""
    from paper_distiller.distill.article import distill
    from paper_distiller.proofs.store import Theorem
    from paper_distiller.sources.arxiv import Paper
    from paper_distiller.vault.crosslink import WikiIndex

    paper = Paper(
        source="arxiv", paper_id="X", title="T", authors=["A"],
        abstract="...", published="", pdf_url="x", arxiv_id="X",
    )
    captured = {}
    llm = MagicMock()
    def _capture(messages, **kw):
        captured["prompt"] = messages[0]["content"]
        return _mock_llm_response_with_sidecar()
    llm.complete.side_effect = _capture

    prior = [Theorem(
        id=1, paper_arxiv_id="2110.12319", paper_slug="bigan",
        name="Theorem 4.3", statement="||f||_inf <= C n^{-1/2}",
        proof_sketch="Bernstein + chaining.",
        techniques_used=["Bernstein", "Dudley"],
    )]
    distill(paper, "x" * 1000, WikiIndex(entries=[]), llm, prior_theorems=prior)
    prompt_text = captured["prompt"]
    assert "已知相关定理" in prompt_text
    assert "Theorem 4.3" in prompt_text
    assert "Bernstein" in prompt_text


def test_ingest_after_distill_round_trips(tmp_path, mocker):
    """End-to-end: distill produces sidecar, ProofStore ingests, then
    retrieve_relevant finds it."""
    from paper_distiller.distill.article import distill
    from paper_distiller.proofs.store import ProofStore
    from paper_distiller.sources.arxiv import Paper
    from paper_distiller.vault.crosslink import WikiIndex

    paper = Paper(
        source="arxiv", paper_id="2110.12319", title="T", authors=["A"],
        abstract="...", published="", pdf_url="x", arxiv_id="2110.12319",
    )
    llm = MagicMock()
    llm.complete.return_value = _mock_llm_response_with_sidecar()
    wiki = WikiIndex(entries=[])

    result = distill(paper, "x" * 1000, wiki, llm)
    store = ProofStore(tmp_path / "proofs.db")
    ingest_result = store.ingest_sidecar(
        result.proof_sidecar, paper.arxiv_id, paper_slug=result.slug,
    )
    assert ingest_result["theorems_inserted"] == 1
    assert store.theorem_count() == 1

    # Retrieve via technique
    found = store.theorems_using_technique("Bernstein")
    assert len(found) == 1
    assert found[0].paper_arxiv_id == "2110.12319"
    store.close()


def test_extract_candidate_techniques_finds_known_names():
    """Cheap keyword scan picks up well-known technique mentions."""
    from paper_distiller.agents.processor import _extract_candidate_techniques
    from paper_distiller.sources.arxiv import Paper

    paper = Paper(
        source="arxiv", paper_id="x", title="Concentration inequalities",
        authors=["A"], abstract=(
            "We apply Bernstein's inequality and Dudley chaining to prove "
            "a Lipschitz extension result for sub-Gaussian random variables. "
            "Our analysis uses Wasserstein distance and PAC-Bayes."
        ),
        published="", pdf_url="x", arxiv_id="x",
    )
    found = _extract_candidate_techniques(paper)
    assert "Bernstein" in found
    assert "Dudley" in found
    assert "Lipschitz" in found
    assert "Wasserstein" in found
    assert "PAC-Bayes" in found


def test_extract_candidate_techniques_empty_when_unrelated():
    """No false positives on a paper that mentions none of the technique names."""
    from paper_distiller.agents.processor import _extract_candidate_techniques
    from paper_distiller.sources.arxiv import Paper

    paper = Paper(
        source="arxiv", paper_id="x", title="Cooking pizza",
        authors=["A"], abstract="We discuss optimal dough hydration.",
        published="", pdf_url="x", arxiv_id="x",
    )
    found = _extract_candidate_techniques(paper)
    assert found == []
