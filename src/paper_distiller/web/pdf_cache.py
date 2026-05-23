"""PDF cache — lazy download from arXiv per vault.

Usage:
    from paper_distiller.web.pdf_cache import get_or_download_pdf

    path = get_or_download_pdf("2110.05948", Path("/my/vault"))

The PDF is cached at ``<vault>/.pdfs/<arxiv_id>.pdf``.

Raises:
    ValueError  — invalid arxiv_id (fails regex or contains dangerous chars)
    RuntimeError — network error or non-200 from arXiv
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx

# Strict regex: modern arXiv IDs only (e.g. 2110.05948, 2110.05948v2)
_ARXIV_ID_RE = re.compile(r"^[0-9]{4}\.[0-9]{4,6}(v[0-9]+)?$")

# Extra safeguard — any of these chars in an ID must be rejected
_BANNED_CHARS = frozenset({"/", "\\", "..", "%"})

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
DOWNLOAD_TIMEOUT = 30.0


def _validate_arxiv_id(arxiv_id: str) -> None:
    """Raise ValueError if arxiv_id looks unsafe or malformed."""
    if not isinstance(arxiv_id, str):
        raise ValueError(f"arxiv_id must be a str, got {type(arxiv_id)!r}")
    # Check for banned substrings before the regex
    for bad in _BANNED_CHARS:
        if bad in arxiv_id:
            raise ValueError(f"arxiv_id contains forbidden substring {bad!r}: {arxiv_id!r}")
    if not _ARXIV_ID_RE.match(arxiv_id):
        raise ValueError(
            f"arxiv_id {arxiv_id!r} does not match expected pattern "
            r"^[0-9]{4}\.[0-9]{4,6}(v[0-9]+)?$"
        )


def _cache_dir(vault_path: Path) -> Path:
    """Return (and create) ``<vault>/.pdfs/``."""
    d = vault_path / ".pdfs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_cache_path(arxiv_id: str, vault_path: Path) -> Path:
    """Return the resolved cache path and verify it stays under .pdfs/."""
    cache_dir = _cache_dir(vault_path)
    candidate = (cache_dir / f"{arxiv_id}.pdf").resolve()
    # Defense in depth: ensure we didn't escape .pdfs/
    if not candidate.is_relative_to(cache_dir.resolve()):
        raise ValueError(
            f"Resolved cache path {candidate} escapes the .pdfs directory"
        )
    return candidate


def get_or_download_pdf(
    arxiv_id: str,
    vault_path: Path,
    *,
    _client: httpx.Client | None = None,
) -> Path:
    """Return local path to the cached PDF, downloading from arXiv if needed.

    Parameters
    ----------
    arxiv_id:
        A validated arXiv ID such as ``2110.05948`` or ``2110.05948v2``.
    vault_path:
        Root of the user's vault.  The PDF is stored at
        ``<vault_path>/.pdfs/<arxiv_id>.pdf``.
    _client:
        Optional ``httpx.Client`` injected by tests (avoids real network).

    Returns
    -------
    Path
        Absolute path to the PDF file (guaranteed to exist on return).

    Raises
    ------
    ValueError
        If *arxiv_id* fails validation.
    RuntimeError
        If the download fails (non-2xx response or network error).
    """
    _validate_arxiv_id(arxiv_id)
    dest = _safe_cache_path(arxiv_id, vault_path)

    if dest.exists():
        return dest

    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)

    def _do_download(client: httpx.Client) -> None:
        try:
            resp = client.get(url, follow_redirects=True, timeout=DOWNLOAD_TIMEOUT)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Network error fetching {url}: {exc}") from exc
        if resp.status_code != 200:
            raise RuntimeError(
                f"arXiv returned HTTP {resp.status_code} for {url}"
            )
        dest.write_bytes(resp.content)

    if _client is not None:
        _do_download(_client)
    else:
        with httpx.Client() as client:
            _do_download(client)

    return dest
