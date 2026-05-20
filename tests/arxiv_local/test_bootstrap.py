"""Tests for arxiv_local.bootstrap — JSONL parsing, ingestion, source chain."""

from __future__ import annotations

import gzip
import json


_SAMPLE_RECORD = {
    "id": "2401.12345",
    "title": "A Diffusion Model for Tests",
    "authors_parsed": [["Smith", "Alice", ""], ["Doe", "Bob", ""]],
    "abstract": "We propose a fake method.",
    "categories": "cs.LG stat.ML",
    "doi": None,
    "comments": None,
    "journal-ref": None,
    "update_date": "2024-01-15",
    "versions": [{"created": "Mon, 15 Jan 2024 00:00:00 GMT", "version": "v1"}],
}


def test_parse_kaggle_record():
    from paper_distiller.arxiv_local.bootstrap import parse_kaggle_record

    paper = parse_kaggle_record(_SAMPLE_RECORD)
    assert paper.arxiv_id == "2401.12345"
    assert paper.title.startswith("A Diffusion")
    assert paper.authors == ["Alice Smith", "Bob Doe"]
    assert paper.categories == ["cs.LG", "stat.ML"]
    assert paper.primary_category == "cs.LG"
    assert paper.published == "2024-01-15"
    assert paper.source == "bootstrap"


def test_parse_handles_missing_optional_fields():
    from paper_distiller.arxiv_local.bootstrap import parse_kaggle_record

    minimal = {
        "id": "x", "title": "T", "categories": "cs.LG",
        "abstract": "a", "update_date": "2024-01-01",
    }
    p = parse_kaggle_record(minimal)
    assert p.arxiv_id == "x"
    assert p.authors == []
    assert p.doi is None


def test_ingest_jsonl_file(tmp_path):
    from paper_distiller.arxiv_local.bootstrap import ingest_jsonl
    from paper_distiller.arxiv_local.store import Store

    jsonl_path = tmp_path / "sample.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for i in range(5):
            rec = dict(_SAMPLE_RECORD)
            rec["id"] = f"2401.0{i}"
            f.write(json.dumps(rec) + "\n")

    db_path = tmp_path / "arxiv.db"
    store = Store(db_path)
    result = ingest_jsonl(jsonl_path, store, batch_size=2)
    store.close()

    assert result.n_inserted == 5
    store = Store(db_path)
    assert store.paper_count() == 5
    store.close()


def test_ingest_jsonl_gzipped(tmp_path):
    """Bootstrap dumps come .json.gz — handle transparently."""
    from paper_distiller.arxiv_local.bootstrap import ingest_jsonl
    from paper_distiller.arxiv_local.store import Store

    jsonl_path = tmp_path / "sample.json.gz"
    with gzip.open(jsonl_path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(_SAMPLE_RECORD) + "\n")
    store = Store(tmp_path / "arxiv.db")
    result = ingest_jsonl(jsonl_path, store)
    assert result.n_inserted == 1
    store.close()


def test_ingest_skips_malformed_lines(tmp_path):
    from paper_distiller.arxiv_local.bootstrap import ingest_jsonl
    from paper_distiller.arxiv_local.store import Store

    jsonl_path = tmp_path / "bad.jsonl"
    rec_a = dict(_SAMPLE_RECORD, id="aaa")
    rec_b = dict(_SAMPLE_RECORD, id="bbb")
    jsonl_path.write_text(
        json.dumps(rec_a) + "\nthis is not json\n" + json.dumps(rec_b) + "\n",
        encoding="utf-8",
    )
    store = Store(tmp_path / "arxiv.db")
    result = ingest_jsonl(jsonl_path, store)
    assert result.n_inserted == 2
    assert result.n_malformed == 1
    store.close()


def test_ingest_skips_records_without_id_or_title(tmp_path):
    from paper_distiller.arxiv_local.bootstrap import ingest_jsonl
    from paper_distiller.arxiv_local.store import Store

    jsonl_path = tmp_path / "incomplete.jsonl"
    lines = [
        json.dumps({"id": "good", "title": "T", "categories": "cs.LG",
                    "abstract": "a", "update_date": "2024-01-01"}),
        json.dumps({"title": "missing-id", "categories": "cs.LG",
                    "abstract": "a", "update_date": "2024-01-01"}),
        json.dumps({"id": "missing-title", "categories": "cs.LG",
                    "abstract": "a", "update_date": "2024-01-01"}),
    ]
    jsonl_path.write_text("\n".join(lines), encoding="utf-8")

    store = Store(tmp_path / "arxiv.db")
    result = ingest_jsonl(jsonl_path, store)
    assert result.n_inserted == 1
    assert result.n_skipped == 2
    store.close()


def test_bootstrap_auto_uses_internet_archive_first(tmp_path, mocker):
    from paper_distiller.arxiv_local import bootstrap as bs
    from paper_distiller.arxiv_local.store import Store

    def _fake_download(url, dest, timeout=600.0):
        with open(dest, "w", encoding="utf-8") as f:
            f.write(json.dumps(_SAMPLE_RECORD) + "\n")

    mocker.patch(
        "paper_distiller.arxiv_local.bootstrap._download_to_file",
        side_effect=_fake_download,
    )

    store = Store(tmp_path / "arxiv.db")
    # Force .gz suffix so ingest_jsonl uses gzip; OR write plain and use .json:
    # Easier: monkey-patch the dump filename. Instead, use a plain JSONL path.
    # We need the dump path to NOT end in .gz for our fake to work; the helper
    # writes to arxiv-metadata.json.gz so gzip read will fail. Patch ingest path.
    def _ia_with_plain(work_dir, store):
        from paper_distiller.arxiv_local.bootstrap import ingest_jsonl
        plain = work_dir / "arxiv-metadata.json"
        with open(plain, "w", encoding="utf-8") as f:
            f.write(json.dumps(_SAMPLE_RECORD) + "\n")
        return ingest_jsonl(plain, store)

    mocker.patch(
        "paper_distiller.arxiv_local.bootstrap._bootstrap_internet_archive",
        side_effect=_ia_with_plain,
    )

    result = bs.bootstrap(store, source="auto", work_dir=tmp_path / "work")
    assert result.n_inserted >= 1
    state = store.load_state()
    assert state["bootstrap_source"] == "internet_archive"
    store.close()


def test_bootstrap_falls_back_on_internet_archive_failure(tmp_path, mocker):
    from paper_distiller.arxiv_local import bootstrap as bs
    from paper_distiller.arxiv_local.store import Store

    def _ia_fails(work_dir, store):
        raise bs.BootstrapError("simulated IA failure")

    def _fake_kaggle(work_dir, store):
        store.upsert_many([bs.parse_kaggle_record(_SAMPLE_RECORD)])
        return bs.IngestResult(n_inserted=1)

    mocker.patch(
        "paper_distiller.arxiv_local.bootstrap._bootstrap_internet_archive",
        side_effect=_ia_fails,
    )
    mocker.patch(
        "paper_distiller.arxiv_local.bootstrap._bootstrap_kaggle",
        side_effect=_fake_kaggle,
    )

    store = Store(tmp_path / "arxiv.db")
    result = bs.bootstrap(store, source="auto", work_dir=tmp_path / "work")
    assert result.n_inserted == 1
    assert store.load_state()["bootstrap_source"] == "kaggle"
    store.close()


def test_bootstrap_explicit_unknown_source_raises(tmp_path):
    from paper_distiller.arxiv_local import bootstrap as bs
    from paper_distiller.arxiv_local.store import Store

    store = Store(tmp_path / "arxiv.db")
    import pytest
    with pytest.raises(bs.BootstrapError):
        bs.bootstrap(store, source="nope", work_dir=tmp_path / "work")
    store.close()


def test_bootstrap_all_sources_fail_raises(tmp_path, mocker):
    from paper_distiller.arxiv_local import bootstrap as bs
    from paper_distiller.arxiv_local.store import Store

    mocker.patch(
        "paper_distiller.arxiv_local.bootstrap._bootstrap_internet_archive",
        side_effect=bs.BootstrapError("ia down"),
    )
    mocker.patch(
        "paper_distiller.arxiv_local.bootstrap._bootstrap_kaggle",
        side_effect=bs.BootstrapError("kaggle down"),
    )

    store = Store(tmp_path / "arxiv.db")
    import pytest
    with pytest.raises(bs.BootstrapError, match="exhausted"):
        bs.bootstrap(store, source="auto", work_dir=tmp_path / "work")
    store.close()
