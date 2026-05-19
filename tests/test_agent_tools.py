"""Tests for chat.agent_tools — JSON schemas, dispatch table, and wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Schema-level tests
# ---------------------------------------------------------------------------

def test_tool_schemas_valid():
    from paper_distiller.chat.agent_tools import TOOL_SCHEMAS

    assert len(TOOL_SCHEMAS) == 5
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
    assert set(names) == {"search", "distill_by_id", "show", "ask", "research"}
    assert len(names) == len(set(names))  # no duplicates


def test_tool_schemas_order_matches_spec():
    from paper_distiller.chat.agent_tools import TOOL_SCHEMAS

    names_in_order = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert names_in_order == ["search", "distill_by_id", "show", "ask", "research"]


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
