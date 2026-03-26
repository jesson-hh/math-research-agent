import time
import arxiv

from config import ARXIV_DELAY, ARXIV_RETRIES

# Simple in-memory cache with TTL
_cache = {}
_CACHE_TTL = 300  # 5 minutes
_CACHE_MAX = 100


def _cache_get(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry["time"]) < _CACHE_TTL:
        return entry["data"]
    return None


def _cache_set(key: str, data):
    _cache[key] = {"data": data, "time": time.time()}
    if len(_cache) > _CACHE_MAX:
        oldest = min(_cache, key=lambda k: _cache[k]["time"])
        del _cache[oldest]

CATEGORY_MAP = {
    "algebraic topology": "math.AT",
    "number theory": "math.NT",
    "differential geometry": "math.DG",
    "complex analysis": "math.CV",
    "combinatorics": "math.CO",
    "representation theory": "math.RT",
    "functional analysis": "math.FA",
    "category theory": "math.CT",
    "probability theory": "math.PR",
    "partial differential equations": "math.AP",
    "algebraic geometry": "math.AG",
    "logic": "math.LO",
}


def arxiv_search(
    query: str,
    domain: str = "",
    max_results: int = 8,
    sort_by: str = "relevance",
) -> dict:
    # Resolve English domain name to arxiv category code
    cat = CATEGORY_MAP.get(domain.lower().strip(), domain.strip())

    # Build query with category filter
    if cat:
        full_query = f"({query}) AND cat:{cat}"
    else:
        full_query = f"({query}) AND cat:math*"

    sort_criterion = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
    }.get(sort_by, arxiv.SortCriterion.Relevance)

    limit = min(max(1, max_results), 20)

    cache_key = f"{full_query}|{limit}|{sort_by}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    client = arxiv.Client(page_size=limit, delay_seconds=ARXIV_DELAY, num_retries=ARXIV_RETRIES)
    search = arxiv.Search(
        query=full_query,
        max_results=limit,
        sort_by=sort_criterion,
    )

    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors[:5]],
            "abstract": result.summary[:800],
            "arxiv_id": result.entry_id.split("/")[-1],
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "published": result.published.strftime("%Y-%m-%d") if result.published else "unknown",
            "updated": result.updated.strftime("%Y-%m-%d") if result.updated else "unknown",
            "categories": result.categories,
            "journal_ref": result.journal_ref or "",
        })

    result = {
        "query": full_query,
        "total_found": len(papers),
        "domain_category": cat or "all math",
        "papers": papers,
    }
    _cache_set(cache_key, result)
    return result


def arxiv_author_search(
    author: str,
    max_results: int = 15,
) -> dict:
    """Search arXiv papers by author name. Returns full abstracts and all authors."""
    full_query = f'au:"{author}"'
    limit = min(max(1, max_results), 30)
    client = arxiv.Client(page_size=limit, delay_seconds=ARXIV_DELAY, num_retries=ARXIV_RETRIES)
    search = arxiv.Search(
        query=full_query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "abstract": result.summary,
            "arxiv_id": result.entry_id.split("/")[-1],
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "published": result.published.strftime("%Y-%m-%d") if result.published else "unknown",
            "updated": result.updated.strftime("%Y-%m-%d") if result.updated else "unknown",
            "categories": result.categories,
            "journal_ref": result.journal_ref or "",
        })

    return {
        "query": full_query,
        "author": author,
        "total_found": len(papers),
        "papers": papers,
    }
