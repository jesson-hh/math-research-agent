"""Local FTS5 search over the arxiv mirror."""

from __future__ import annotations

import json

from ..sources.arxiv import Paper
from .store import Store


_PDF_URL_FMT = "https://arxiv.org/pdf/{arxiv_id}"


def _normalize_fts_query(q: str) -> str:
    """Escape FTS5 syntax in a user query, preserving simple OR/AND/NOT usage.

    Strategy: split on whitespace; uppercase boolean tokens pass through;
    other tokens are wrapped in double quotes so any embedded FTS operators
    are treated as literals.
    """
    tokens = q.strip().split()
    out = []
    for t in tokens:
        if t.upper() in ("AND", "OR", "NOT"):
            out.append(t.upper())
        else:
            clean = t.replace('"', "")
            out.append(f'"{clean}"')
    return " ".join(out)


def search(
    store: Store,
    query: str,
    n: int = 30,
    sort: str = "relevance",
    primary_category: str | None = None,
    since: str | None = None,
) -> list:
    """Local FTS5 search. Returns sources.arxiv.Paper objects."""
    if not query.strip():
        return []

    fts_query = _normalize_fts_query(query)

    sql_parts = [
        "SELECT p.* FROM papers p",
        "JOIN papers_fts ON papers_fts.rowid = p.rowid",
        "WHERE papers_fts MATCH ?",
    ]
    params: list = [fts_query]

    if primary_category:
        sql_parts.append("AND p.primary_category = ?")
        params.append(primary_category)

    if since:
        sql_parts.append("AND p.published >= ?")
        params.append(since)

    if sort == "date":
        sql_parts.append("ORDER BY p.published DESC")
    else:
        sql_parts.append("ORDER BY bm25(papers_fts)")

    sql_parts.append("LIMIT ?")
    params.append(n)

    sql = " ".join(sql_parts)
    rows = store._conn.execute(sql, params).fetchall()

    out: list = []
    for row in rows:
        out.append(Paper(
            source="arxiv",
            paper_id=row["arxiv_id"],
            title=row["title"],
            authors=json.loads(row["authors"]),
            abstract=row["abstract"],
            pdf_url=_PDF_URL_FMT.format(arxiv_id=row["arxiv_id"]),
            published=row["published"],
            categories=json.loads(row["categories"]),
            arxiv_id=row["arxiv_id"],
            doi=row["doi"] or None,
        ))
    return out
