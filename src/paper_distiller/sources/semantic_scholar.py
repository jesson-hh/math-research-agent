"""Semantic Scholar Graph API client.

Free tier: 100 req / 5 min, ~1 req/s. Optional `api_key` parameter sent as
x-api-key header to raise rate limits (apply at https://www.semanticscholar.org/product/api).

Endpoints used:
  GET /paper/search?query=...&limit=...&fields=...    — keyword search
  GET /paper/ARXIV:<id>?fields=...                    — lookup by arxiv id
  GET /paper/DOI:<doi>?fields=...                     — lookup by DOI
"""

from __future__ import annotations

import httpx

from .arxiv import Paper


class SSError(RuntimeError):
    """Raised when SS API returns an error other than 404."""


_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,abstract,authors,year,externalIds,openAccessPdf,venue"
_TIMEOUT = 30.0


def _headers(api_key: str | None) -> dict:
    if api_key:
        return {"x-api-key": api_key}
    return {}


def _record_to_paper(record: dict) -> Paper:
    """Convert an SS API record into the unified Paper dataclass."""
    external = record.get("externalIds") or {}
    open_access = record.get("openAccessPdf") or {}
    authors_field = record.get("authors") or []

    pdf_url_val = open_access.get("url") or ""
    return Paper(
        source="semanticscholar",
        paper_id=record["paperId"],
        title=(record.get("title") or "").strip(),
        authors=[a.get("name", "") for a in authors_field[:10]],
        abstract=(record.get("abstract") or "").strip(),
        published=str(record.get("year") or ""),
        pdf_url=pdf_url_val,
        arxiv_id=external.get("ArXiv"),
        doi=external.get("DOI"),
        ss_paper_id=record["paperId"],
        venue=record.get("venue"),
        open_access_pdf_url=open_access.get("url"),
    )


def search(query: str, max_results: int = 30,
           api_key: str | None = None) -> list[Paper]:
    """Free-text search against Semantic Scholar. Returns up to max_results papers."""
    url = f"{_BASE_URL}/paper/search"
    params = {"query": query, "limit": max_results, "fields": _FIELDS}
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.get(url, params=params, headers=_headers(api_key))
            r.raise_for_status()
    except httpx.HTTPError as e:
        raise SSError(f"SS search failed: {e}") from e

    data = r.json()
    raw_records = data.get("data") or []
    papers = []
    for record in raw_records:
        if not record.get("paperId"):
            continue
        try:
            papers.append(_record_to_paper(record))
        except Exception:
            continue
    return papers


def lookup_by_arxiv_id(arxiv_id: str,
                        api_key: str | None = None) -> Paper | None:
    """Resolve an arxiv id via SS. Returns None on 404 / missing record."""
    return _lookup(f"ARXIV:{arxiv_id}", api_key)


def lookup_by_doi(doi: str, api_key: str | None = None) -> Paper | None:
    """Resolve a DOI via SS. Returns None on 404 / missing record."""
    return _lookup(f"DOI:{doi}", api_key)


def paper_refs(arxiv_id_or_doi: str, max_results: int = 30,
               api_key: str | None = None) -> list[Paper]:
    """Fetch references + cited-by for a given paper ID. Returns combined list of Paper.

    arxiv_id_or_doi: e.g. "2501.00001" or "10.1234/foo". Prefix-aware: if the
    caller already passed "arxiv:..." or "doi:..." it is kept as-is.
    """
    headers = {"x-api-key": api_key} if api_key else {}
    key = arxiv_id_or_doi
    if "/" in key and not key.startswith(("arxiv:", "doi:")):
        key = f"doi:{key}"
    elif "." in key and not key.startswith(("arxiv:", "doi:")):
        key = f"arxiv:{key}"

    results = []
    for endpoint in (f"paper/{key}/references", f"paper/{key}/citations"):
        url = f"https://api.semanticscholar.org/graph/v1/{endpoint}"
        params = {
            "limit": max_results // 2,
            "fields": "title,abstract,authors,year,externalIds,openAccessPdf",
        }
        try:
            r = httpx.get(url, params=params, headers=headers, timeout=30.0)
            r.raise_for_status()
        except httpx.HTTPError:
            continue
        for item in r.json().get("data", []):
            inner = item.get("citedPaper") or item.get("citingPaper") or item
            if not inner:
                continue
            ext = inner.get("externalIds") or {}
            arxiv_id = ext.get("ArXiv")
            doi = ext.get("DOI")
            if not arxiv_id and not doi:
                continue
            results.append(Paper(
                source="ss",
                paper_id=inner.get("paperId", ""),
                arxiv_id=arxiv_id,
                doi=doi,
                title=inner.get("title") or "",
                authors=[a.get("name", "") for a in inner.get("authors", [])],
                abstract=inner.get("abstract") or "",
                pdf_url=(inner.get("openAccessPdf") or {}).get("url", ""),
                published=str(inner.get("year") or ""),
                categories=[],
            ))
    return results


def _lookup(id_with_prefix: str, api_key: str | None) -> Paper | None:
    url = f"{_BASE_URL}/paper/{id_with_prefix}"
    params = {"fields": _FIELDS}
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.get(url, params=params, headers=_headers(api_key))
            if r.status_code == 404:
                return None
            r.raise_for_status()
    except httpx.HTTPError as e:
        if "404" in str(e):
            return None
        raise SSError(f"SS lookup {id_with_prefix} failed: {e}") from e

    record = r.json()
    if not record.get("paperId"):
        return None
    try:
        return _record_to_paper(record)
    except Exception:
        return None
