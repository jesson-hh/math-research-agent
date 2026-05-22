"""Tests for chat.agent_tools — JSON schemas, dispatch table, and wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Schema-level tests
# ---------------------------------------------------------------------------

def test_tool_schemas_valid():
    from paper_distiller.chat.agent_tools import TOOL_SCHEMAS

    assert len(TOOL_SCHEMAS) == 8
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert isinstance(fn.get("name"), str) and fn["name"]
        assert isinstance(fn.get("description"), str) and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert isinstance(params.get("properties"), dict)
        assert params["properties"]  # must declare at least one property


def test_tool_schemas_distinct_names():
    from paper_distiller.chat.agent_tools import TOOL_SCHEMAS

    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert set(names) == {
        "search", "distill_by_id", "show", "ask", "research",
        "ask_user", "find_proof", "review_proof",
    }
    assert len(names) == len(set(names))  # no duplicates


def test_tool_schemas_order_matches_spec():
    from paper_distiller.chat.agent_tools import TOOL_SCHEMAS

    names_in_order = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert names_in_order == [
        "search", "distill_by_id", "show", "ask", "research",
        "ask_user", "find_proof", "review_proof",
    ]


# ---------------------------------------------------------------------------
# execute_tool dispatch
# ---------------------------------------------------------------------------

def test_execute_tool_unknown_name_returns_error():
    from paper_distiller.chat.agent_tools import execute_tool

    result = execute_tool("nope", {}, vault_path="/tmp/v")
    assert result == {"error": "unknown tool: nope"}


def test_execute_tool_dispatches_by_name(mocker, tmp_path):
    """execute_tool must call the right TOOL_FUNCTIONS entry by name."""
    import paper_distiller.chat.agent_tools as at

    fake = mocker.Mock(return_value={"ok": True})
    mocker.patch.dict(at.TOOL_FUNCTIONS, {"show": fake}, clear=False)

    result = at.execute_tool(
        "show", {"slug": "x", "category": "articles"}, vault_path=str(tmp_path)
    )
    fake.assert_called_once_with(slug="x", category="articles", vault_path=str(tmp_path))
    assert result == {"ok": True}


def test_execute_tool_bad_kwargs_returns_error(tmp_path):
    """If LLM passes a kwarg the wrapper doesn't accept, return error not crash."""
    from paper_distiller.chat.agent_tools import execute_tool

    # tool_show only knows slug + category — bogus_kwarg must trip TypeError.
    result = execute_tool(
        "show",
        {"slug": "x", "bogus_kwarg": "nope"},
        vault_path=str(tmp_path),
    )
    assert "error" in result
    assert "TypeError" in result["error"]


# ---------------------------------------------------------------------------
# tool_show
# ---------------------------------------------------------------------------

def test_tool_show_missing_entry_returns_error(tmp_path):
    from paper_distiller.chat.agent_tools import tool_show

    result = tool_show("does-not-exist", vault_path=str(tmp_path))
    assert "error" in result
    assert "not found" in result["error"]


def test_tool_show_reads_existing_entry(tmp_path):
    from paper_distiller.chat.agent_tools import tool_show
    from paper_distiller.vault.store import VaultStore

    vault = VaultStore(tmp_path)
    saved = vault.save_entry(
        title="A Test Paper",
        category="articles",
        body="## Summary\n\nThis is the body content of a test article.",
        tags=["test", "fixture"],
        refs=["arxiv:1234.5678"],
        slug="test-paper-slug",
    )
    assert saved["slug"] == "test-paper-slug"

    result = tool_show("test-paper-slug", vault_path=str(tmp_path))
    assert "error" not in result
    assert result["slug"] == "test-paper-slug"
    assert result["title"] == "A Test Paper"
    assert result["category"] == "articles"
    assert "test" in result["tags"]
    assert "arxiv:1234.5678" in result["refs"]
    assert "This is the body content" in result["body"]


def test_tool_show_invalid_category_returns_error(tmp_path):
    from paper_distiller.chat.agent_tools import tool_show

    result = tool_show("anything", category="not-a-category", vault_path=str(tmp_path))
    assert "error" in result


# ---------------------------------------------------------------------------
# tool_search (mocked orchestrator)
# ---------------------------------------------------------------------------

@dataclass
class _FakePaper:
    title: str = ""
    authors: list = field(default_factory=list)
    abstract: str = ""
    published: str = ""
    pdf_url: str = ""
    paper_id: str = ""
    arxiv_id: str | None = None
    doi: str | None = None
    ss_paper_id: str | None = None


def test_tool_search_calls_orchestrator(mocker, tmp_path, monkeypatch):
    """tool_search must drive the search DAG and shape Paper objects into dicts."""
    monkeypatch.setenv("PD_API_KEY", "sk-fake")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")

    # The LLMClient init validates api_key but does no HTTP — leave it alone
    # but pin httpx so accidental network calls would also fail.
    mocker.patch(
        "paper_distiller.chat.agent_tools.LLMClient.__init__",
        return_value=None,
    )

    fake_paper = _FakePaper(
        title="Fake Diffusion Paper",
        authors=["Alice", "Bob", "Carol"],
        abstract="We propose a fake method to test the search wrapper.",
        published="2025-04-11",
        pdf_url="https://arxiv.org/pdf/9999.0001",
        paper_id="9999.0001",
        arxiv_id="9999.0001",
    )

    # Replace Orchestrator with a stub whose run() populates ctx.shared.
    class _StubOrch:
        def __init__(self, dag, ctx):
            self.ctx = ctx

        async def run(self):
            self.ctx.shared["ranked"] = [fake_paper]
            return self.ctx.shared

    mocker.patch("paper_distiller.chat.agent_tools.Orchestrator", _StubOrch)

    from paper_distiller.chat.agent_tools import tool_search

    result = tool_search("diffusion", n=5, vault_path=str(tmp_path))

    assert "error" not in result, result
    assert "candidates" in result
    assert len(result["candidates"]) == 1
    cand = result["candidates"][0]
    assert cand["title"] == "Fake Diffusion Paper"
    assert cand["id"] == "9999.0001"
    assert cand["year"] == "2025"
    assert cand["authors"] == ["Alice", "Bob", "Carol"]
    assert "fake method" in cand["abstract"]


def test_tool_search_missing_topic_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-fake")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")

    from paper_distiller.chat.agent_tools import tool_search

    # Empty topic violates load_config()'s "topic or author" requirement.
    result = tool_search("", vault_path=str(tmp_path))
    assert "error" in result


# ---------------------------------------------------------------------------
# tool_distill_by_id (mocked two-phase orchestrator)
# ---------------------------------------------------------------------------

@dataclass
class _FakeArticle:
    slug: str = ""
    title: str = ""


def _patch_distill_orch(mocker, *, phase_a_candidates, phase_b_articles):
    """Patch Orchestrator with a stateful stub: first call = Phase A,
    second call = Phase B. The stub examines call count, not the DAG."""

    class _TwoPhaseOrch:
        call_count = 0

        def __init__(self, dag, ctx):
            self.ctx = ctx
            _TwoPhaseOrch.call_count += 1
            self._phase = _TwoPhaseOrch.call_count

        async def run(self):
            if self._phase == 1:
                # Phase A: populate candidates pool (and ranked, which the
                # wrapper will overwrite with matched anyway).
                self.ctx.shared["candidates"] = list(phase_a_candidates)
                self.ctx.shared["ranked"] = list(phase_a_candidates)
            else:
                # Phase B: populate distilled articles.
                self.ctx.shared["articles"] = list(phase_b_articles)
            return self.ctx.shared

    mocker.patch("paper_distiller.chat.agent_tools.Orchestrator", _TwoPhaseOrch)
    return _TwoPhaseOrch


def test_tool_distill_by_id_happy_path(mocker, tmp_path, monkeypatch):
    """Happy path: one requested ID, one matching candidate, one distilled article."""
    monkeypatch.setenv("PD_API_KEY", "sk-fake")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    mocker.patch(
        "paper_distiller.chat.agent_tools.LLMClient.__init__",
        return_value=None,
    )

    matching_paper = _FakePaper(
        title="Foo Paper",
        arxiv_id="2401.12345",
        paper_id="2401.12345",
    )
    fake_article = _FakeArticle(slug="foo", title="Foo Paper")

    _patch_distill_orch(
        mocker,
        phase_a_candidates=[matching_paper],
        phase_b_articles=[fake_article],
    )

    from paper_distiller.chat.agent_tools import tool_distill_by_id

    result = tool_distill_by_id(
        ids=["2401.12345"], topic="foo", vault_path=str(tmp_path)
    )

    assert "error" not in result, result
    assert result["matched_count"] == 1
    assert result["requested_count"] == 1
    assert "unmatched" not in result
    assert result["distilled"] == [
        {"slug": "foo", "title": "Foo Paper", "category": "articles"}
    ]


def test_tool_distill_by_id_unmatched_path(mocker, tmp_path, monkeypatch):
    """Unmatched path: requested IDs don't appear in the candidate pool —
    Phase B must not run; unmatched list returned."""
    monkeypatch.setenv("PD_API_KEY", "sk-fake")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    mocker.patch(
        "paper_distiller.chat.agent_tools.LLMClient.__init__",
        return_value=None,
    )

    # Candidate pool contains a paper with a totally different ID.
    decoy = _FakePaper(arxiv_id="0000.0000", paper_id="0000.0000", title="Decoy")

    stub_cls = _patch_distill_orch(
        mocker,
        phase_a_candidates=[decoy],
        phase_b_articles=[],  # never reached
    )

    from paper_distiller.chat.agent_tools import tool_distill_by_id

    result = tool_distill_by_id(
        ids=["9999.9999", "8888.8888"], topic="anything", vault_path=str(tmp_path)
    )

    assert "error" not in result, result
    assert result["matched_count"] == 0
    assert result["requested_count"] == 2
    assert result["unmatched"] == ["9999.9999", "8888.8888"]
    assert result["distilled"] == []
    assert result["survey_slug"] is None
    # Phase B must NOT have run when there's nothing to distill.
    assert stub_cls.call_count == 1


# ---------------------------------------------------------------------------
# tool_ask + tool_research (mocked runners)
# ---------------------------------------------------------------------------

def test_tool_ask_dispatches_to_run_qa_loop(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-fake")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")

    stub_summary = {
        "session_id": "sid-stub",
        "stop_reason": "llm_done",
        "rounds_completed": 1,
        "articles_distilled_count": 2,
        "cost_cny": 0.42,
        "tokens_in_total": 1234,
        "tokens_out_total": 567,
    }
    fake = mocker.patch(
        "paper_distiller.chat.agent_tools.run_qa_loop",
        return_value=stub_summary,
    )

    from paper_distiller.chat.agent_tools import tool_ask

    result = tool_ask(
        "why does X work?",
        max_rounds=2,
        per_round=3,
        max_cost_cny=1.0,
        max_articles=4,
        vault_path=str(tmp_path),
    )

    assert result == stub_summary
    fake.assert_called_once()
    cfg = fake.call_args[0][0]
    assert cfg.qa_question == "why does X work?"
    assert cfg.qa_max_rounds == 2
    assert cfg.qa_per_round == 3


def test_tool_research_dispatches_to_run_research_loop(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-fake")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")

    stub = {
        "session_id": "rs-1",
        "stop_reason": "budget",
        "papers_distilled_count": 12,
        "themes_count": 3,
        "synthesis_count": 3,
        "final_report_slug": "report-x",
        "total_cost_cny": 7.5,
        "total_tokens_in": 100000,
        "total_tokens_out": 50000,
        "iterations_completed": 2,
    }
    fake = mocker.patch(
        "paper_distiller.chat.agent_tools.run_research_loop",
        return_value=stub,
    )

    from paper_distiller.chat.agent_tools import tool_research

    result = tool_research(
        "what is X?",
        duration="30m",
        max_papers=5,
        max_cost_cny=2.0,
        vault_path=str(tmp_path),
    )

    assert result == stub
    fake.assert_called_once()
    cfg = fake.call_args[0][0]
    assert cfg.research_max_papers == 5
    assert cfg.research_max_cost_cny == 2.0
    assert cfg.research_max_duration_sec == 1800


def test_tool_research_invalid_duration_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-fake")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")

    from paper_distiller.chat.agent_tools import tool_research

    result = tool_research("X?", duration="not-a-duration", vault_path=str(tmp_path))
    assert "error" in result


# ---------------------------------------------------------------------------
# _paper_matches_id helper
# ---------------------------------------------------------------------------

def test_paper_matches_id_arxiv():
    from paper_distiller.chat.agent_tools import _paper_matches_id

    p = _FakePaper(arxiv_id="2401.12345", paper_id="2401.12345")
    assert _paper_matches_id(p, "2401.12345")
    assert _paper_matches_id(p, "2401.12345 ")  # extra whitespace
    assert not _paper_matches_id(p, "9999.0000")
    assert not _paper_matches_id(p, "")


def test_paper_matches_id_case_insensitive():
    from paper_distiller.chat.agent_tools import _paper_matches_id

    p = _FakePaper(doi="10.1234/AbC", paper_id="x")
    assert _paper_matches_id(p, "10.1234/abc")
    assert _paper_matches_id(p, "10.1234/ABC")


# ---------------------------------------------------------------------------
# tool_ask_user
# ---------------------------------------------------------------------------

def test_ask_user_schema_present():
    from paper_distiller.chat.agent_tools import TOOL_SCHEMAS

    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert "ask_user" in names


def test_ask_user_returns_selected_label(monkeypatch, tmp_path):
    from paper_distiller.chat.agent_tools import tool_ask_user

    inputs = iter(["1"])
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(inputs))
    result = tool_ask_user(
        question="Pick one",
        options=[
            {"label": "Option A", "description": "a"},
            {"label": "Option B", "description": "b"},
        ],
        vault_path=str(tmp_path),
    )
    assert result == {"selected": ["Option A"], "cancelled": False}


def test_ask_user_multi_select(monkeypatch, tmp_path):
    from paper_distiller.chat.agent_tools import tool_ask_user

    monkeypatch.setattr("builtins.input", lambda *a, **kw: "1,3")
    result = tool_ask_user(
        question="Pick any",
        options=[
            {"label": "A", "description": "a"},
            {"label": "B", "description": "b"},
            {"label": "C", "description": "c"},
        ],
        multi_select=True,
        vault_path=str(tmp_path),
    )
    assert result == {"selected": ["A", "C"], "cancelled": False}


def test_ask_user_cancel_with_q(monkeypatch, tmp_path):
    from paper_distiller.chat.agent_tools import tool_ask_user

    monkeypatch.setattr("builtins.input", lambda *a, **kw: "q")
    result = tool_ask_user(
        question="?",
        options=[
            {"label": "A", "description": "a"},
            {"label": "B", "description": "b"},
        ],
        vault_path=str(tmp_path),
    )
    assert result == {"cancelled": True}


def test_ask_user_invalid_then_valid(monkeypatch, tmp_path):
    from paper_distiller.chat.agent_tools import tool_ask_user

    inputs = iter(["xxx", "1"])
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(inputs))
    result = tool_ask_user(
        question="?",
        options=[
            {"label": "A", "description": "a"},
            {"label": "B", "description": "b"},
        ],
        vault_path=str(tmp_path),
    )
    assert result == {"selected": ["A"], "cancelled": False}


# ---------------------------------------------------------------------------
# tool_find_proof (v1.10 — query the per-vault ProofStore from chat)
# ---------------------------------------------------------------------------

def _seed_proof_store(vault_path):
    """Helper: open the vault's proof store and seed it with 2 theorems."""
    from paper_distiller.proofs.store import open_for_vault, ProofSidecar
    store = open_for_vault(vault_path)
    store.ingest_sidecar(
        ProofSidecar(
            theorems=[{
                "name": "Theorem A1",
                "statement": "By Bernstein concentration, x <= C.",
                "proof_sketch": "Apply MGF bound.",
                "techniques_used": ["Bernstein", "MGF"],
            }],
            key_techniques=["Bernstein", "MGF"],
        ),
        "arxiv:2110.12319", paper_slug="paper-a",
    )
    store.ingest_sidecar(
        ProofSidecar(
            theorems=[{
                "name": "Lemma B1",
                "statement": "Wasserstein-1 duality via Kantorovich.",
                "proof_sketch": "Convex duality.",
                "techniques_used": ["Wasserstein", "Kantorovich"],
            }],
            key_techniques=["Wasserstein", "Kantorovich"],
        ),
        "arxiv:2204.99999", paper_slug="paper-b",
    )
    store.close()


def test_find_proof_stats_on_empty_vault(tmp_path):
    from paper_distiller.chat.agent_tools import tool_find_proof
    result = tool_find_proof("stats", vault_path=str(tmp_path))
    assert result == {"theorems": 0, "techniques": 0, "papers_covered": 0}


def test_find_proof_stats_after_ingest(tmp_path):
    from paper_distiller.chat.agent_tools import tool_find_proof
    _seed_proof_store(tmp_path)
    result = tool_find_proof("stats", vault_path=str(tmp_path))
    assert result["theorems"] == 2
    assert result["techniques"] == 4
    assert result["papers_covered"] == 2


def test_find_proof_list_techniques(tmp_path):
    from paper_distiller.chat.agent_tools import tool_find_proof
    _seed_proof_store(tmp_path)
    result = tool_find_proof("list_techniques", limit=10, vault_path=str(tmp_path))
    names = {t["name"] for t in result["techniques"]}
    assert names == {"Bernstein", "MGF", "Wasserstein", "Kantorovich"}


def test_find_proof_by_technique(tmp_path):
    from paper_distiller.chat.agent_tools import tool_find_proof
    _seed_proof_store(tmp_path)
    result = tool_find_proof(
        "by_technique", query="Bernstein", vault_path=str(tmp_path),
    )
    assert len(result["theorems"]) == 1
    assert result["theorems"][0]["name"] == "Theorem A1"
    assert "Bernstein" in result["theorems"][0]["techniques_used"]


def test_find_proof_by_text(tmp_path):
    from paper_distiller.chat.agent_tools import tool_find_proof
    _seed_proof_store(tmp_path)
    result = tool_find_proof(
        "by_text", query="Kantorovich duality", vault_path=str(tmp_path),
    )
    assert len(result["theorems"]) >= 1
    assert any("Wasserstein" in t["statement"] for t in result["theorems"])


def test_find_proof_by_paper(tmp_path):
    from paper_distiller.chat.agent_tools import tool_find_proof
    _seed_proof_store(tmp_path)
    result = tool_find_proof(
        "by_paper", query="arxiv:2110.12319", vault_path=str(tmp_path),
    )
    assert len(result["theorems"]) == 1
    assert result["theorems"][0]["paper_slug"] == "paper-a"


def test_find_proof_missing_query_returns_error(tmp_path):
    """by_technique / by_text / by_paper all require query."""
    from paper_distiller.chat.agent_tools import tool_find_proof
    _seed_proof_store(tmp_path)
    result = tool_find_proof(
        "by_technique", query=None, vault_path=str(tmp_path),
    )
    assert "error" in result


def test_find_proof_unknown_query_type_returns_error(tmp_path):
    from paper_distiller.chat.agent_tools import tool_find_proof
    _seed_proof_store(tmp_path)
    result = tool_find_proof(
        "by_nothing", query="x", vault_path=str(tmp_path),
    )
    assert "error" in result
    assert "unknown query_type" in result["error"]


def test_find_proof_no_match_returns_empty(tmp_path):
    from paper_distiller.chat.agent_tools import tool_find_proof
    _seed_proof_store(tmp_path)
    result = tool_find_proof(
        "by_technique", query="QuantumGravity",
        vault_path=str(tmp_path),
    )
    assert result == {"theorems": []}


def test_find_proof_in_execute_tool_dispatch(tmp_path):
    """execute_tool should route 'find_proof' correctly."""
    from paper_distiller.chat.agent_tools import execute_tool
    _seed_proof_store(tmp_path)
    result = execute_tool(
        "find_proof", {"query_type": "stats"},
        vault_path=str(tmp_path),
    )
    assert result["theorems"] == 2


def test_ask_user_in_execute_tool_dispatch(mocker, tmp_path):
    """Verify execute_tool can route ask_user too."""
    from paper_distiller.chat.agent_tools import execute_tool

    mocker.patch(
        "builtins.input", return_value="1"
    )
    result = execute_tool(
        "ask_user",
        {
            "question": "?",
            "options": [
                {"label": "A", "description": "a"},
                {"label": "B", "description": "b"},
            ],
        },
        vault_path=str(tmp_path),
    )
    assert result["selected"] == ["A"]


# ---------------------------------------------------------------------------
# tool_review_proof (Task 6.4 — 8th tool)
# ---------------------------------------------------------------------------

def _make_stub_report():
    """Build a minimal ReviewReport for monkeypatching."""
    from paper_distiller.proofgraph.reviewer import ReviewReport, ReviewResult
    return ReviewReport(
        target="2110.1",
        nodes_reviewed=2,
        by_label={"ok": 1, "suspicious": 1},
        flagged=[
            ReviewResult(node_id=1, label="suspicious", reason="bad step", confidence=0.5),
        ],
        summary="1 of 2 nodes need attention",
    )


def test_tool_review_proof_paper_mode(monkeypatch, tmp_path):
    """tool_review_proof monkeypatches review_target; checks returned dict shape."""
    import paper_distiller.chat.agent_tools as at

    stub_report = _make_stub_report()
    monkeypatch.setattr(at, "review_target", lambda *a, **kw: stub_report)
    # Also monkeypatch open_for_vault to avoid needing a real store
    from paper_distiller.proofs.store import ProofStore
    fake_store = ProofStore(tmp_path / "proofs.db")
    monkeypatch.setattr(at, "open_for_vault", lambda p: fake_store)
    # Provide env vars so LLMClient doesn't raise
    monkeypatch.setenv("PD_API_KEY", "test-key")
    monkeypatch.setenv("PD_BASE_URL", "http://localhost")
    monkeypatch.setenv("PD_MODEL", "test-model")

    from paper_distiller.chat.agent_tools import tool_review_proof
    result = tool_review_proof(
        target_type="paper", target="2110.1", vault_path=str(tmp_path)
    )

    assert "target" in result
    assert "nodes_reviewed" in result
    assert "by_label" in result
    assert "flagged" in result
    assert "summary" in result
    # flagged must be a list of dicts (JSON-serializable), not ReviewResult objects
    assert isinstance(result["flagged"], list)
    if result["flagged"]:
        assert isinstance(result["flagged"][0], dict)


def test_tool_review_proof_bad_target_type_returns_error(monkeypatch, tmp_path):
    """Unknown target_type must return {'error': ...}."""
    monkeypatch.setenv("PD_API_KEY", "test-key")
    monkeypatch.setenv("PD_BASE_URL", "http://localhost")
    monkeypatch.setenv("PD_MODEL", "test-model")

    from paper_distiller.chat.agent_tools import tool_review_proof
    result = tool_review_proof(
        target_type="galaxy", target="2110.1", vault_path=str(tmp_path)
    )
    assert "error" in result


def test_tool_review_proof_missing_env_returns_error(monkeypatch, tmp_path):
    """Missing PD_API_KEY returns {'error': ...} without crashing."""
    monkeypatch.delenv("PD_API_KEY", raising=False)
    monkeypatch.delenv("PD_BASE_URL", raising=False)
    monkeypatch.delenv("PD_MODEL", raising=False)

    from paper_distiller.chat.agent_tools import tool_review_proof
    result = tool_review_proof(
        target_type="paper", target="2110.1", vault_path=str(tmp_path)
    )
    assert "error" in result


def test_review_proof_in_tool_schemas():
    """review_proof must appear in TOOL_SCHEMAS."""
    from paper_distiller.chat.agent_tools import TOOL_SCHEMAS
    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert "review_proof" in names


def test_review_proof_in_tool_functions():
    """review_proof must be in TOOL_FUNCTIONS."""
    from paper_distiller.chat.agent_tools import TOOL_FUNCTIONS
    assert "review_proof" in TOOL_FUNCTIONS


def test_review_proof_in_all():
    """tool_review_proof must appear in __all__."""
    from paper_distiller.chat import agent_tools
    assert "tool_review_proof" in agent_tools.__all__


# ---------------------------------------------------------------------------
# tool_find_proof — graph query types (Task 7.1)
# ---------------------------------------------------------------------------

def _seed_graph_store(vault_path):
    """Seed a ProofStore with two nodes + a depends_on edge for graph query tests."""
    from paper_distiller.proofs.store import open_for_vault, Node, Edge
    store = open_for_vault(vault_path)
    parent_id = store.add_node(Node(
        paper_arxiv_id="2301.00001",
        kind="theorem",
        text="Bernstein concentration implies sub-Gaussian tails.",
        label="Theorem 1.1",
        source_quote="Bernstein concentration implies sub-Gaussian tails.",
        techniques=["Bernstein"],
    ))
    child_id = store.add_node(Node(
        paper_arxiv_id="2301.00001",
        kind="proof_step",
        text="Apply Bernstein to bound the moment generating function.",
        label="Step (a)",
        source_quote="Apply Bernstein to bound the MGF.",
        techniques=["Bernstein", "MGF"],
    ))
    store.add_edge(Edge(src_id=child_id, dst_id=parent_id, rel="depends_on"))
    store.close()
    return parent_id, child_id


def test_find_proof_by_step_returns_matching_node(tmp_path):
    """by_step FTS over node text finds the 'Bernstein' node."""
    from paper_distiller.chat.agent_tools import tool_find_proof
    _seed_graph_store(tmp_path)
    result = tool_find_proof("by_step", query="Bernstein", vault_path=str(tmp_path))
    assert "nodes" in result, result
    assert len(result["nodes"]) >= 1
    texts = [n["text"] for n in result["nodes"]]
    assert any("Bernstein" in t for t in texts)


def test_find_proof_by_step_missing_query_returns_error(tmp_path):
    """by_step without query must return {'error': ...}."""
    from paper_distiller.chat.agent_tools import tool_find_proof
    _seed_graph_store(tmp_path)
    result = tool_find_proof("by_step", query=None, vault_path=str(tmp_path))
    assert "error" in result


def test_find_proof_dependency_walk_returns_parent(tmp_path):
    """dependency_walk from child_id must return the parent node."""
    from paper_distiller.chat.agent_tools import tool_find_proof
    parent_id, child_id = _seed_graph_store(tmp_path)
    result = tool_find_proof(
        "dependency_walk", query=str(child_id), vault_path=str(tmp_path)
    )
    assert "nodes" in result, result
    ids = [n["id"] for n in result["nodes"]]
    assert parent_id in ids


def test_find_proof_dependency_walk_missing_query_returns_error(tmp_path):
    """dependency_walk without query must return {'error': ...}."""
    from paper_distiller.chat.agent_tools import tool_find_proof
    _seed_graph_store(tmp_path)
    result = tool_find_proof("dependency_walk", query=None, vault_path=str(tmp_path))
    assert "error" in result


def test_find_proof_node_returns_node_and_edges(tmp_path):
    """node query returns the node dict + its out-edges."""
    from paper_distiller.chat.agent_tools import tool_find_proof
    parent_id, child_id = _seed_graph_store(tmp_path)
    result = tool_find_proof("node", query=str(child_id), vault_path=str(tmp_path))
    assert "node" in result, result
    assert "edges" in result, result
    assert result["node"]["id"] == child_id
    edge_dsts = [e["dst_id"] for e in result["edges"]]
    assert parent_id in edge_dsts


def test_find_proof_node_missing_query_returns_error(tmp_path):
    """node query without query must return {'error': ...}."""
    from paper_distiller.chat.agent_tools import tool_find_proof
    _seed_graph_store(tmp_path)
    result = tool_find_proof("node", query=None, vault_path=str(tmp_path))
    assert "error" in result


def test_find_proof_node_nonexistent_id_returns_error(tmp_path):
    """node query with non-existent id must return {'error': ...}."""
    from paper_distiller.chat.agent_tools import tool_find_proof
    _seed_graph_store(tmp_path)
    result = tool_find_proof("node", query="99999", vault_path=str(tmp_path))
    assert "error" in result


def test_find_proof_graph_query_types_in_schema(tmp_path):
    """The _FIND_PROOF_SCHEMA enum must include by_step, dependency_walk, node."""
    from paper_distiller.chat.agent_tools import _FIND_PROOF_SCHEMA
    enum_vals = (
        _FIND_PROOF_SCHEMA["function"]["parameters"]
        ["properties"]["query_type"]["enum"]
    )
    assert "by_step" in enum_vals
    assert "dependency_walk" in enum_vals
    assert "node" in enum_vals
