"""Tests for chat.research_runner — the 5-phase deep-research driver."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper_distiller.distill.article import ArticleResult
from paper_distiller.sources.arxiv import Paper


def _paper(i):
    return Paper(
        source="arxiv", paper_id=f"2501.000{i}", arxiv_id=f"2501.000{i}",
        title=f"P{i}", authors=[], abstract=f"abs {i}",
        pdf_url=f"https://x/{i}.pdf", published="2025-01-01", categories=[],
    )


def _cfg(tmp_path, max_papers=8, max_cost=10.0, max_duration_sec=3600):
    from paper_distiller.config import Config
    return Config(
        vault_path=tmp_path / "vault",
        topic=None, author=None, top_n=2, pool=10,
        force=False, dry_run=False, verbose=False,
        api_key="sk-test", base_url="https://x/v1", model="qwen-plus",
        provider_name="test", pdf_timeout_sec=60, min_papers_for_survey=2,
        source="arxiv", ss_api_key=None,
        qa_max_rounds=2, qa_max_articles=max_papers,
        qa_max_cost_cny=max_cost, qa_confidence_threshold=8,
        qa_per_round=2, qa_interactive=False,
        qa_resume_session_id=None,
        qa_question="why diffusion?",
        research_max_papers=max_papers,
        research_max_cost_cny=max_cost,
        research_max_duration_sec=max_duration_sec,
    )


def _common_mocks(mocker, gap_continues_first=False, gap_continues_then_stop=False):
    """Mock all subsystems. Default: GapDetector says STOP immediately (1 iteration)."""
    fake_llm_class = mocker.patch("paper_distiller.chat.research_runner.LLMClient")
    llm_instance = fake_llm_class.return_value
    llm_instance.total_tokens_in = 100
    llm_instance.total_tokens_out = 50

    # Phase 1 reflection (says done immediately)
    mocker.patch(
        "paper_distiller.agents.reflector.reflect",
        return_value={
            "is_done": True, "confidence": 9,
            "what_we_know": "all clear", "what_is_missing": "",
            "next_query": "diffusion", "next_query_rationale": "",
            "suggest_stop": False,
        },
    )
    # Sources
    mocker.patch(
        "paper_distiller.agents.searchers.arxiv_search",
        return_value=[_paper(1), _paper(2)],
    )
    mocker.patch("paper_distiller.agents.searchers.ss_search", return_value=[])
    # Ranker
    mocker.patch(
        "paper_distiller.agents.curation.rank",
        side_effect=lambda c, t, n, llm: c[:n],
    )
    # Fetch + distill
    mocker.patch(
        "paper_distiller.agents.processor.fetch_with_fallback",
        return_value="x" * 600,
    )
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        side_effect=lambda p, ft, wi, llm, prior_theorems=None: ArticleResult(
            slug=f"a-{p.arxiv_id}", title=f"T-{p.arxiv_id}",
            body=f"# T-{p.arxiv_id}\n\nBody.", tags=["t"],
            refs=[f"arxiv:{p.arxiv_id}"], depth="full-pdf",
        ),
    )
    mocker.patch(
        "paper_distiller.agents.processor.load_index",
        return_value=MagicMock(slugs=lambda: set()),
    )
    # CitationExplorer (returns empty by default)
    mocker.patch(
        "paper_distiller.agents.citation_explorer.ss_paper_refs",
        return_value=[],
    )
    # SurveyComposer's underlying compose function
    from types import SimpleNamespace
    mocker.patch(
        "paper_distiller.chat.research_runner.compose_survey",
        return_value=SimpleNamespace(
            slug="synth", title="Synthesis title",
            body="synthesis body", tags=["synth"],
            related_articles=[],
        ),
    )
    # llm.complete handles: theme cluster + theorem extract + gap detect + synthesizer
    def _llm_complete(messages, **kw):
        content = messages[0]["content"]
        if "聚类" in content or "themes" in content.lower():
            return json.dumps({"themes": [
                {"name": "All", "description": "single bucket",
                 "slugs": ["a-2501.0001", "a-2501.0002"]},
            ]})
        if "theorems" in content.lower() or "假设" in content:
            return json.dumps({
                "theorems": ["T1"], "assumptions": ["A1"],
                "convergence_rates": [], "key_lemmas": [],
            })
        if "覆盖度" in content or "gap" in content.lower():
            # Default: stop
            return json.dumps({
                "should_continue": False,
                "missing_aspects": [],
                "next_query": "",
                "rationale": "Coverage sufficient.",
            })
        # Unknown — empty JSON
        return "{}"
    llm_instance.complete.side_effect = _llm_complete


def test_research_terminates_on_gap_detector_stop(tmp_path, mocker):
    cfg = _cfg(tmp_path)
    cfg.vault_path.mkdir()
    _common_mocks(mocker)
    from paper_distiller.chat.research_runner import run_research_loop
    summary = run_research_loop(cfg)
    assert summary["stop_reason"] in ("all_themes_synthesized", "max_papers")
    assert summary["papers_distilled_count"] >= 1


def test_research_terminates_on_max_papers(tmp_path, mocker):
    cfg = _cfg(tmp_path, max_papers=2)
    cfg.vault_path.mkdir()
    _common_mocks(mocker)
    from paper_distiller.chat.research_runner import run_research_loop
    summary = run_research_loop(cfg)
    assert summary["papers_distilled_count"] <= cfg.research_max_papers + 1


def test_research_persists_state(tmp_path, mocker):
    cfg = _cfg(tmp_path)
    cfg.vault_path.mkdir()
    _common_mocks(mocker)
    from paper_distiller.chat.research_runner import run_research_loop
    summary = run_research_loop(cfg)
    sid = summary["session_id"]
    state_path = cfg.vault_path / ".paper_distiller" / "research-sessions" / sid / "state.json"
    assert state_path.exists()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["is_done"] is True


def test_research_writes_synthesis_docs(tmp_path, mocker):
    cfg = _cfg(tmp_path)
    cfg.vault_path.mkdir()
    _common_mocks(mocker)
    from paper_distiller.chat.research_runner import run_research_loop
    summary = run_research_loop(cfg)
    surveys_dir = cfg.vault_path / "surveys"
    surveys = list(surveys_dir.glob("*.md"))
    # At least one survey written (synthesis + final report)
    assert len(surveys) >= 1


def test_research_summary_keys(tmp_path, mocker):
    cfg = _cfg(tmp_path)
    cfg.vault_path.mkdir()
    _common_mocks(mocker)
    from paper_distiller.chat.research_runner import run_research_loop
    summary = run_research_loop(cfg)
    expected_keys = {
        "session_id", "stop_reason", "papers_distilled_count",
        "themes_count", "synthesis_count", "final_report_slug",
        "total_cost_cny", "total_tokens_in", "total_tokens_out",
        "iterations_completed",
    }
    assert expected_keys.issubset(summary.keys())
