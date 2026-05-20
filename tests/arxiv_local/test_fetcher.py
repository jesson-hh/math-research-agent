"""Tests for arxiv_local.fetcher — Fetcher abstraction + LocalFirstFetcher."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_local_fetcher_uses_local_store(tmp_path):
    from paper_distiller.arxiv_local.fetcher import LocalFetcher
    from paper_distiller.arxiv_local.store import Store, PaperRow

    store = Store(tmp_path / "arxiv.db")
    store.upsert_many([PaperRow(
        arxiv_id="2401.0", title="Test Paper", authors=["A"],
        abstract="diffusion", categories=["cs.LG"], primary_category="cs.LG",
        published="2024-01-01", updated=None, doi=None, comment=None,
        journal_ref=None, source="bootstrap",
    )])
    f = LocalFetcher(store)
    results = f.search("diffusion", n=10, sort="relevance")
    assert len(results) == 1
    assert f.is_available() is True
    store.close()


def test_local_fetcher_is_unavailable_when_db_empty(tmp_path):
    from paper_distiller.arxiv_local.fetcher import LocalFetcher
    from paper_distiller.arxiv_local.store import Store

    store = Store(tmp_path / "arxiv.db")
    f = LocalFetcher(store)
    assert f.is_available() is False
    store.close()


def test_local_first_fetcher_prefers_local(tmp_path):
    from paper_distiller.arxiv_local.fetcher import LocalFirstFetcher
    from paper_distiller.sources.arxiv import Paper

    local = MagicMock()
    local.is_available.return_value = True
    local.search.return_value = [
        Paper(source="arxiv", paper_id=f"2401.{i}", title=f"T{i}", authors=[],
              abstract="", pdf_url="", published="2024-01-01", arxiv_id=f"2401.{i}")
        for i in range(5)
    ]
    live = MagicMock()

    f = LocalFirstFetcher(local, live, min_local_ratio=0.5)
    results = f.search("x", n=4, sort="relevance")
    assert len(results) == 4
    live.search.assert_not_called()


def test_local_first_fetcher_falls_through_when_local_empty(tmp_path):
    from paper_distiller.arxiv_local.fetcher import LocalFirstFetcher
    from paper_distiller.sources.arxiv import Paper

    local = MagicMock()
    local.is_available.return_value = False
    local.search.return_value = []

    live_results = [
        Paper(source="arxiv", paper_id=f"2401.{i}", title=f"L{i}", authors=[],
              abstract="", pdf_url="", published="2024-01-01", arxiv_id=f"2401.{i}")
        for i in range(3)
    ]
    live = MagicMock()
    live.is_available.return_value = True
    live.search.return_value = live_results

    f = LocalFirstFetcher(local, live, allow_live_topup=True)
    results = f.search("x", n=3, sort="relevance")
    assert len(results) == 3
    live.search.assert_called_once()


def test_local_first_fetcher_dedupes_when_topping_up(tmp_path):
    from paper_distiller.arxiv_local.fetcher import LocalFirstFetcher
    from paper_distiller.sources.arxiv import Paper

    overlap = Paper(source="arxiv", paper_id="dup", title="dup", authors=[],
                    abstract="", pdf_url="", published="", arxiv_id="dup")
    new = Paper(source="arxiv", paper_id="new", title="new", authors=[],
                abstract="", pdf_url="", published="", arxiv_id="new")

    local = MagicMock()
    local.is_available.return_value = True
    local.search.return_value = [overlap]  # only 1 result

    live = MagicMock()
    live.is_available.return_value = True
    live.search.return_value = [overlap, new]  # duplicate + new

    f = LocalFirstFetcher(local, live, min_local_ratio=0.5)
    # n=4, threshold=2, local has 1 → triggers topup
    results = f.search("x", n=4, sort="relevance")
    ids = [p.arxiv_id for p in results]
    # No dupes; both unique results present
    assert ids == ["dup", "new"]


def test_local_only_mode_skips_live(tmp_path, monkeypatch):
    from paper_distiller.arxiv_local.fetcher import LocalFirstFetcher

    monkeypatch.setenv("PD_ARXIV_LOCAL_ONLY", "1")
    local = MagicMock()
    local.is_available.return_value = False
    local.search.return_value = []
    live = MagicMock()
    live.is_available.return_value = True
    live.search.return_value = ["should not be called"]

    f = LocalFirstFetcher(local, live)
    results = f.search("x", n=3, sort="relevance")
    assert results == []
    live.search.assert_not_called()


def test_local_first_skips_live_when_live_unavailable(tmp_path):
    from paper_distiller.arxiv_local.fetcher import LocalFirstFetcher

    local = MagicMock()
    local.is_available.return_value = False
    local.search.return_value = []
    live = MagicMock()
    live.is_available.return_value = False  # arxiv in cooldown
    live.search.return_value = ["nope"]

    f = LocalFirstFetcher(local, live, allow_live_topup=True)
    results = f.search("x", n=3, sort="relevance")
    assert results == []
    live.search.assert_not_called()
