import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paper_distiller.config import Config
from paper_distiller.sources.arxiv import ArxivPaper
from paper_distiller.distill.article import ArticleResult
from paper_distiller.distill.survey import SurveyResult


def _config(tmp_path):
    return Config(
        vault_path=tmp_path / "vault",
        topic="diffusion",
        author=None,
        top_n=2, pool=10, force=False, dry_run=False, verbose=False,
        api_key="sk-test", base_url="https://x/v1", model="qwen-plus",
        provider_name="test", pdf_timeout_sec=60, min_papers_for_survey=2,
    )


def _paper(i):
    return ArxivPaper(
        source="arxiv", paper_id=f"2501.0000{i}",
        arxiv_id=f"2501.0000{i}", title=f"Paper {i}", authors=["A"],
        abstract=f"abstract {i}", pdf_url=f"https://arxiv.org/pdf/2501.0000{i}.pdf",
        published="2025-01-01", categories=["math.AT"],
    )


def test_pipeline_dry_run_makes_no_external_calls(tmp_path, mocker):
    from paper_distiller.pipeline import run
    cfg = _config(tmp_path); cfg.dry_run = True
    cfg.vault_path.mkdir()

    mock_search = mocker.patch("paper_distiller.pipeline.arxiv_search")
    mock_llm_cls = mocker.patch("paper_distiller.pipeline.LLMClient")
    run(cfg)
    mock_search.assert_not_called()
    mock_llm_cls.assert_not_called()


def test_pipeline_happy_path(tmp_path, mocker):
    from paper_distiller.pipeline import run
    cfg = _config(tmp_path)
    cfg.vault_path.mkdir()

    mocker.patch("paper_distiller.pipeline.arxiv_search",
                 return_value=[_paper(1), _paper(2), _paper(3)])
    mocker.patch("paper_distiller.pipeline.rank",
                 return_value=[_paper(1), _paper(2)])
    mocker.patch("paper_distiller.pipeline.download_pdf_from_url",
                 side_effect=lambda url, dest_dir, filename, timeout: Path(dest_dir) / filename)
    mocker.patch("paper_distiller.pipeline.extract_text",
                 return_value="x" * 1000)  # > 500 -> full-pdf depth

    def fake_distill(paper, full_text, wiki_index, llm):
        return ArticleResult(
            slug=f"paper-{paper.arxiv_id}",
            title=f"Title {paper.arxiv_id}",
            body=f"# {paper.title}\n\nbody",
            tags=["t"], refs=[f"arxiv:{paper.arxiv_id}"],
            depth="full-pdf",
        )
    mocker.patch("paper_distiller.pipeline.distill_article", side_effect=fake_distill)
    mocker.patch("paper_distiller.pipeline.compose_survey",
                 return_value=SurveyResult(
                     slug="diffusion-survey-20260502",
                     title="Diffusion Survey",
                     body="# Survey\n\n[[paper-2501.00001]] [[paper-2501.00002]]",
                     tags=["survey"], related_articles=["paper-2501.00001", "paper-2501.00002"],
                 ))
    mocker.patch("paper_distiller.pipeline.LLMClient")

    run(cfg)

    arts = sorted((cfg.vault_path / "articles").glob("*.md"))
    surveys = sorted((cfg.vault_path / "surveys").glob("*.md"))
    assert len(arts) == 2
    assert len(surveys) == 1
    runs_log = cfg.vault_path / ".paper_distiller" / "runs.jsonl"
    assert runs_log.exists()
    line = json.loads(runs_log.read_text().strip().split("\n")[-1])
    assert line["distilled"] == 2
    assert line["survey_slug"] == "diffusion-survey-20260502"


def test_pipeline_dedup_skips_existing(tmp_path, mocker):
    from paper_distiller.pipeline import run
    from paper_distiller.vault.store import VaultStore
    cfg = _config(tmp_path); cfg.vault_path.mkdir()
    # Pre-populate one article
    store = VaultStore(cfg.vault_path)
    store.save_entry(title="Title 2501.00001", category="articles",
                     body="x", slug="paper-2501.00001")

    mocker.patch("paper_distiller.pipeline.arxiv_search",
                 return_value=[_paper(1), _paper(2)])
    mocker.patch("paper_distiller.pipeline.rank",
                 return_value=[_paper(1), _paper(2)])
    mocker.patch("paper_distiller.pipeline.download_pdf_from_url",
                 side_effect=lambda url, dest_dir, filename, timeout: Path(dest_dir) / filename)
    mocker.patch("paper_distiller.pipeline.extract_text", return_value="x" * 1000)

    def fake_distill(paper, full_text, wiki_index, llm):
        return ArticleResult(slug=f"paper-{paper.arxiv_id}",
                             title=f"Title {paper.arxiv_id}",
                             body="b", tags=[], refs=[], depth="full-pdf")
    mocker.patch("paper_distiller.pipeline.distill_article", side_effect=fake_distill)
    mocker.patch("paper_distiller.pipeline.compose_survey")
    mocker.patch("paper_distiller.pipeline.LLMClient")

    run(cfg)

    line = json.loads((cfg.vault_path / ".paper_distiller" / "runs.jsonl").read_text().strip().split("\n")[-1])
    assert line["skipped_dedup"] == 1
    assert line["distilled"] == 1


def test_pipeline_arxiv_id_dedup_skips_existing(tmp_path, mocker):
    """If the vault already has an article with refs containing this arxiv id,
    skip — even if the slug pattern doesn't match. Fixes the v0.1 issue where
    cofindiff.md and cofindiff-controllable-financial-diffusion.md could both
    exist for the same arxiv paper."""
    from paper_distiller.pipeline import run
    from paper_distiller.vault.store import VaultStore
    cfg = _config(tmp_path); cfg.vault_path.mkdir()
    store = VaultStore(cfg.vault_path)
    # Pre-populate with a hand-written-style entry: slug doesn't match arxiv pattern,
    # but refs contains the arxiv id of the candidate we'll search for.
    store.save_entry(
        title="CoFinDiff (hand-written)",
        category="articles",
        body="pre-existing hand-written content",
        refs=["arxiv:2501.00001"],
        slug="cofindiff-handwritten",
    )

    mocker.patch("paper_distiller.pipeline.arxiv_search",
                 return_value=[_paper(1)])  # _paper(1) has arxiv_id "2501.00001"
    mocker.patch("paper_distiller.pipeline.rank",
                 return_value=[_paper(1)])
    mock_distill = mocker.patch("paper_distiller.pipeline.distill_article")
    mocker.patch("paper_distiller.pipeline.compose_survey")
    mocker.patch("paper_distiller.pipeline.LLMClient")

    run(cfg)

    log = (cfg.vault_path / ".paper_distiller" / "runs.jsonl").read_text()
    line = json.loads(log.strip().split("\n")[-1])
    assert line["skipped_dedup"] == 1
    assert line["distilled"] == 0
    # Critically: distill_article was never called — the skip happened upstream
    mock_distill.assert_not_called()


def test_pipeline_force_overrides_arxiv_id_dedup(tmp_path, mocker):
    """--force bypasses arxiv-id dedup — same behavior as for slug-based dedup."""
    from paper_distiller.pipeline import run
    from paper_distiller.vault.store import VaultStore
    cfg = _config(tmp_path); cfg.vault_path.mkdir()
    cfg.force = True  # the only difference from the dedup-skip test above
    store = VaultStore(cfg.vault_path)
    store.save_entry(
        title="CoFinDiff (hand-written)",
        category="articles",
        body="pre-existing hand-written content",
        refs=["arxiv:2501.00001"],
        slug="cofindiff-handwritten",
    )

    mocker.patch("paper_distiller.pipeline.arxiv_search",
                 return_value=[_paper(1)])
    mocker.patch("paper_distiller.pipeline.rank",
                 return_value=[_paper(1)])
    mocker.patch("paper_distiller.pipeline.download_pdf_from_url",
                 side_effect=lambda url, dest_dir, filename, timeout: Path(dest_dir) / filename)
    mocker.patch("paper_distiller.pipeline.extract_text", return_value="x" * 1000)

    def fake_distill(paper, full_text, wiki_index, llm):
        return ArticleResult(
            slug=f"forced-{paper.arxiv_id}",
            title=f"Forced {paper.arxiv_id}",
            body="b", tags=[], refs=[f"arxiv:{paper.arxiv_id}"],
            depth="full-pdf",
        )
    mock_distill = mocker.patch("paper_distiller.pipeline.distill_article",
                                 side_effect=fake_distill)
    mocker.patch("paper_distiller.pipeline.compose_survey")
    mocker.patch("paper_distiller.pipeline.LLMClient")

    run(cfg)

    log = (cfg.vault_path / ".paper_distiller" / "runs.jsonl").read_text()
    line = json.loads(log.strip().split("\n")[-1])
    assert line["skipped_dedup"] == 0
    assert line["distilled"] == 1
    mock_distill.assert_called_once()


def test_pipeline_source_arxiv_only(tmp_path, mocker):
    """--source arxiv: SS search is NOT called; only arxiv candidates rank."""
    from paper_distiller.pipeline import run
    cfg = _config(tmp_path); cfg.vault_path.mkdir()
    cfg.source = "arxiv"

    mock_arxiv = mocker.patch("paper_distiller.pipeline.arxiv_search",
                               return_value=[_paper(1)])
    mock_ss_search = mocker.patch("paper_distiller.pipeline.ss_search")
    mocker.patch("paper_distiller.pipeline.rank", return_value=[_paper(1)])
    mocker.patch("paper_distiller.pipeline.download_pdf_from_url",
                 side_effect=lambda url, dest_dir, filename, timeout: Path(dest_dir) / filename)
    mocker.patch("paper_distiller.pipeline.extract_text", return_value="x" * 1000)

    def fake_distill(paper, full_text, wiki_index, llm):
        return ArticleResult(slug=f"art-{paper.paper_id}",
                              title="T", body="b", tags=[],
                              refs=[f"arxiv:{paper.arxiv_id}"], depth="full-pdf")
    mocker.patch("paper_distiller.pipeline.distill_article", side_effect=fake_distill)
    mocker.patch("paper_distiller.pipeline.compose_survey")
    mocker.patch("paper_distiller.pipeline.LLMClient")

    run(cfg)
    mock_arxiv.assert_called_once()
    mock_ss_search.assert_not_called()


def test_pipeline_source_both_merges_and_dedups(tmp_path, mocker):
    """--source both: candidates from both APIs are deduped by arxiv_id."""
    from paper_distiller.pipeline import run
    from paper_distiller.sources.arxiv import Paper
    cfg = _config(tmp_path); cfg.vault_path.mkdir()
    cfg.source = "both"

    # Same paper from both sources (matching arxiv_id) — must dedup to 1
    arxiv_paper = Paper(
        source="arxiv", paper_id="2501.00001", arxiv_id="2501.00001",
        title="P1 (arxiv)", authors=[], abstract="a",
        pdf_url="https://arxiv.org/pdf/2501.00001.pdf",
        published="2025-01-01", categories=["math.AT"],
    )
    ss_paper_duplicate = Paper(
        source="semanticscholar", paper_id="ss-xyz",
        arxiv_id="2501.00001", doi="10.1/dup", ss_paper_id="ss-xyz",
        title="P1 (ss view)", authors=[], abstract="a",
        pdf_url="https://example.com/ss.pdf",
        published="2025",
    )
    ss_only_paper = Paper(
        source="semanticscholar", paper_id="ss-abc",
        doi="10.2/unique", ss_paper_id="ss-abc",
        title="P2 (ss only)", authors=[], abstract="b",
        pdf_url="https://example.com/ss2.pdf",
        published="2025",
    )

    mocker.patch("paper_distiller.pipeline.arxiv_search",
                 return_value=[arxiv_paper])
    mocker.patch("paper_distiller.pipeline.ss_search",
                 return_value=[ss_paper_duplicate, ss_only_paper])

    captured_candidates = []
    def capture_rank(candidates, *args, **kwargs):
        captured_candidates.append(list(candidates))
        return candidates[:2]
    mocker.patch("paper_distiller.pipeline.rank", side_effect=capture_rank)

    mocker.patch("paper_distiller.pipeline.download_pdf_from_url",
                 side_effect=lambda url, dest_dir, filename, timeout: Path(dest_dir) / filename)
    mocker.patch("paper_distiller.pipeline.extract_text", return_value="x" * 1000)

    def fake_distill(paper, full_text, wiki_index, llm):
        return ArticleResult(slug=f"art-{paper.paper_id}",
                              title="T", body="b", tags=[], refs=[],
                              depth="full-pdf")
    mocker.patch("paper_distiller.pipeline.distill_article", side_effect=fake_distill)
    mocker.patch("paper_distiller.pipeline.compose_survey",
                 return_value=SurveyResult(
                     slug="s", title="S", body="b",
                     tags=[], related_articles=[]))
    mocker.patch("paper_distiller.pipeline.LLMClient")

    run(cfg)

    # 2 unique papers after dedup (arxiv copy of P1 wins; ss-only P2 kept)
    assert len(captured_candidates[0]) == 2
    p1 = [p for p in captured_candidates[0] if p.arxiv_id == "2501.00001"][0]
    assert p1.source == "arxiv"
    assert any(p.doi == "10.2/unique" for p in captured_candidates[0])


def test_pipeline_pdf_fallback_to_ss(tmp_path, mocker):
    """When arxiv PDF fetch fails for an arxiv-sourced paper with arxiv_id,
    pipeline should call ss_lookup_by_arxiv_id and try its open_access_pdf_url."""
    from paper_distiller.pipeline import run
    from paper_distiller.sources.arxiv import Paper
    cfg = _config(tmp_path); cfg.vault_path.mkdir()
    cfg.source = "arxiv"

    arxiv_paper = Paper(
        source="arxiv", paper_id="2501.00001", arxiv_id="2501.00001",
        title="P1", authors=[], abstract="abstract content " * 50,
        pdf_url="https://arxiv.org/pdf/2501.00001.pdf",
        published="2025-01-01", categories=["math.AT"],
    )
    mocker.patch("paper_distiller.pipeline.arxiv_search",
                 return_value=[arxiv_paper])
    mocker.patch("paper_distiller.pipeline.rank", return_value=[arxiv_paper])

    ss_record = Paper(
        source="semanticscholar", paper_id="ss-1",
        arxiv_id="2501.00001", title="P1 (via SS)", authors=[],
        abstract="x", pdf_url="https://mirror.example.com/p1.pdf",
        published="2025",
        open_access_pdf_url="https://mirror.example.com/p1.pdf",
    )
    mock_ss_lookup = mocker.patch("paper_distiller.pipeline.ss_lookup_by_arxiv_id",
                                    return_value=ss_record)

    # First call (arxiv URL) fails; second call (SS URL) succeeds
    call_count = [0]
    def download_side_effect(url, dest_dir, filename, timeout):
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("HTTP 503")
        return Path(dest_dir) / filename
    mock_download = mocker.patch("paper_distiller.pipeline.download_pdf_from_url",
                                   side_effect=download_side_effect)

    mocker.patch("paper_distiller.pipeline.extract_text", return_value="x" * 1000)

    def fake_distill(paper, full_text, wiki_index, llm):
        return ArticleResult(slug="ok", title="T", body="b", tags=[],
                              refs=[], depth="full-pdf")
    mocker.patch("paper_distiller.pipeline.distill_article", side_effect=fake_distill)
    mocker.patch("paper_distiller.pipeline.compose_survey")
    mocker.patch("paper_distiller.pipeline.LLMClient")

    run(cfg)

    mock_ss_lookup.assert_called_once_with("2501.00001", api_key=None)
    # download_pdf_from_url called twice: once for arxiv URL (failed), once for SS URL
    assert mock_download.call_count == 2
    # Second call should be SS URL
    second_call_kwargs = mock_download.call_args_list[1].kwargs
    url_arg = second_call_kwargs.get("url") or mock_download.call_args_list[1].args[0]
    assert url_arg == "https://mirror.example.com/p1.pdf"
