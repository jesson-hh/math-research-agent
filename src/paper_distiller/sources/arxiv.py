"""arxiv.org search + PDF download."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import arxiv
import httpx


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: list
    abstract: str
    pdf_url: str
    published: str
    categories: list = field(default_factory=list)


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)
    return _client


def search(query: str, max_results: int = 30) -> list[ArxivPaper]:
    """Search arxiv.org. Returns up to max_results papers ranked by relevance."""
    s = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = _get_client()
    papers = []
    for result in client.results(s):
        arxiv_id = result.entry_id.rsplit("/", 1)[-1].split("v")[0]
        papers.append(ArxivPaper(
            arxiv_id=arxiv_id,
            title=result.title.strip(),
            authors=[a.name for a in result.authors[:10]],
            abstract=result.summary.strip(),
            pdf_url=result.pdf_url,
            published=result.published.isoformat()[:10],
            categories=list(result.categories),
        ))
    return papers


def download_pdf(paper: ArxivPaper, dest_dir: Path, timeout: float = 60.0) -> Path:
    """Download paper PDF to dest_dir / <arxiv_id>.pdf. Returns the path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{paper.arxiv_id}.pdf"
    with httpx.stream("GET", paper.pdf_url, timeout=timeout, follow_redirects=True) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return dest
