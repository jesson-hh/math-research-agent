"""T2.1 — pdf_cache module tests.

All tests mock the httpx transport; no real network calls are made.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from paper_distiller.web.pdf_cache import get_or_download_pdf

# Minimal PDF byte string for cache-hit / fake-download tests
DUMMY_PDF = b"%PDF-1.4\n1 0 obj\n<</Type /Catalog>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<</Size 1/Root 1 0 R>>\nstartxref\n9\n%%EOF"


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_mock_transport(status: int, content: bytes = b"") -> httpx.MockTransport:
    """Return a MockTransport that always responds with *status* and *content*."""

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=content)

    return httpx.MockTransport(_handler)


def _make_mock_client(status: int, content: bytes = b"") -> httpx.Client:
    return httpx.Client(transport=_make_mock_transport(status, content))


# ── ID-validation tests ───────────────────────────────────────────────────────

class TestIdValidation:
    def test_rejects_dotdot(self, tmp_path):
        with pytest.raises(ValueError, match="forbidden substring"):
            get_or_download_pdf("../etc/passwd", tmp_path)

    def test_rejects_leading_dotdot_variant(self, tmp_path):
        with pytest.raises(ValueError, match="forbidden substring|expected pattern"):
            get_or_download_pdf("..2110.05948", tmp_path)

    def test_rejects_slash(self, tmp_path):
        with pytest.raises(ValueError, match="forbidden substring"):
            get_or_download_pdf("2110/05948", tmp_path)

    def test_rejects_backslash(self, tmp_path):
        with pytest.raises(ValueError, match="forbidden substring"):
            get_or_download_pdf("2110\\05948", tmp_path)

    def test_rejects_percent(self, tmp_path):
        with pytest.raises(ValueError, match="forbidden substring"):
            get_or_download_pdf("2110%2F05948", tmp_path)

    def test_rejects_bare_four_digits(self, tmp_path):
        with pytest.raises(ValueError, match="expected pattern"):
            get_or_download_pdf("1234", tmp_path)

    def test_rejects_letters(self, tmp_path):
        with pytest.raises(ValueError, match="expected pattern"):
            get_or_download_pdf("abc.defgh", tmp_path)

    def test_rejects_too_few_decimal_digits(self, tmp_path):
        with pytest.raises(ValueError, match="expected pattern"):
            get_or_download_pdf("2110.123", tmp_path)

    def test_accepts_four_decimal_digits(self, tmp_path):
        """Should not raise — just call with a mock client."""
        client = _make_mock_client(200, DUMMY_PDF)
        result = get_or_download_pdf("2110.05948", tmp_path, _client=client)
        assert result.exists()

    def test_accepts_version_suffix(self, tmp_path):
        client = _make_mock_client(200, DUMMY_PDF)
        result = get_or_download_pdf("2110.05948v3", tmp_path, _client=client)
        assert result.exists()

    def test_accepts_six_decimal_digits(self, tmp_path):
        client = _make_mock_client(200, DUMMY_PDF)
        result = get_or_download_pdf("2110.123456", tmp_path, _client=client)
        assert result.exists()


# ── Cache-hit test ────────────────────────────────────────────────────────────

class TestCacheHit:
    def test_returns_existing_path_without_network(self, tmp_path):
        """If PDF already cached, no HTTP call should be made."""
        pdfs_dir = tmp_path / ".pdfs"
        pdfs_dir.mkdir()
        cached = pdfs_dir / "2110.05948.pdf"
        cached.write_bytes(DUMMY_PDF)

        # Pass a client that raises if called
        def _should_not_be_called(request):
            raise AssertionError("should not hit the network on a cache hit")

        client = httpx.Client(transport=httpx.MockTransport(_should_not_be_called))
        result = get_or_download_pdf("2110.05948", tmp_path, _client=client)
        assert result == cached.resolve()

    def test_returns_the_cached_bytes_unchanged(self, tmp_path):
        pdfs_dir = tmp_path / ".pdfs"
        pdfs_dir.mkdir()
        cached = pdfs_dir / "2110.05948.pdf"
        cached.write_bytes(DUMMY_PDF)
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        result = get_or_download_pdf("2110.05948", tmp_path, _client=client)
        assert result.read_bytes() == DUMMY_PDF


# ── Cache-miss + successful download ─────────────────────────────────────────

class TestCacheMiss:
    def test_downloads_and_writes_pdf(self, tmp_path):
        client = _make_mock_client(200, DUMMY_PDF)
        result = get_or_download_pdf("2110.05948", tmp_path, _client=client)
        assert result.exists()
        assert result.read_bytes() == DUMMY_PDF

    def test_creates_pdfs_dir_automatically(self, tmp_path):
        client = _make_mock_client(200, DUMMY_PDF)
        get_or_download_pdf("2110.05948", tmp_path, _client=client)
        assert (tmp_path / ".pdfs").is_dir()

    def test_stored_at_correct_path(self, tmp_path):
        client = _make_mock_client(200, DUMMY_PDF)
        result = get_or_download_pdf("2110.05948", tmp_path, _client=client)
        expected = (tmp_path / ".pdfs" / "2110.05948.pdf").resolve()
        assert result == expected


# ── Cache-miss + download failure ────────────────────────────────────────────

class TestDownloadFailure:
    def test_404_raises_runtime_error(self, tmp_path):
        client = _make_mock_client(404)
        with pytest.raises(RuntimeError, match="404"):
            get_or_download_pdf("2110.05948", tmp_path, _client=client)

    def test_503_raises_runtime_error(self, tmp_path):
        client = _make_mock_client(503)
        with pytest.raises(RuntimeError, match="503"):
            get_or_download_pdf("2110.05948", tmp_path, _client=client)

    def test_network_error_raises_runtime_error(self, tmp_path):
        def _fail(request):
            raise httpx.ConnectError("simulated network error")

        client = httpx.Client(transport=httpx.MockTransport(_fail))
        with pytest.raises(RuntimeError, match="Network error"):
            get_or_download_pdf("2110.05948", tmp_path, _client=client)

    def test_failed_download_does_not_write_file(self, tmp_path):
        client = _make_mock_client(404)
        try:
            get_or_download_pdf("2110.05948", tmp_path, _client=client)
        except RuntimeError:
            pass
        assert not (tmp_path / ".pdfs" / "2110.05948.pdf").exists()
