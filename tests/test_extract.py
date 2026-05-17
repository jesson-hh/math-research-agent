from pathlib import Path

import pytest

from paper_distiller.extract.pymupdf_extractor import extract_text


FIXTURE = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture PDF missing")
def test_extract_text_returns_nonempty():
    text = extract_text(FIXTURE)
    assert isinstance(text, str)
    assert len(text) > 100  # any real paper has > 100 chars of text


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture PDF missing")
def test_extract_text_contains_some_words():
    text = extract_text(FIXTURE).lower()
    # arxiv papers virtually always have these
    assert any(w in text for w in ("abstract", "introduction", "the", "we"))


def test_extract_text_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_text(tmp_path / "does-not-exist.pdf")
