"""Fetcher abstraction — Local / Live / Hybrid sources for the same Paper shape."""

from __future__ import annotations

import os
from typing import Protocol

from . import search as local_search
from .store import Store


class Fetcher(Protocol):
    def search(self, query: str, n: int, sort: str) -> list: ...
    def is_available(self) -> bool: ...


class LocalFetcher:
    """Backed by the SQLite store via FTS5."""

    def __init__(self, store: Store):
        self.store = store

    def search(self, query: str, n: int, sort: str) -> list:
        return local_search.search(self.store, query, n=n, sort=sort)

    def is_available(self) -> bool:
        return self.store.paper_count() > 0


class LiveFetcher:
    """Backed by the existing arxiv Python lib (live API call, throttled)."""

    def search(self, query: str, n: int, sort: str) -> list:
        from ..sources.arxiv import search as arxiv_search
        return arxiv_search(query, max_results=n, sort=sort)

    def is_available(self) -> bool:
        from ..agents.rate_limit import ARXIV_LIMITER
        return not ARXIV_LIMITER.is_cooling_down()


class LocalFirstFetcher:
    """Try local; fall through to live if local lacks coverage.

    `min_local_ratio`: if local returns >= n * ratio results, we skip the live
    top-up. Default 0.5 — half coverage is enough since the LLM ranker
    re-ranks anyway.
    `allow_live_topup`: master switch. Env `PD_ARXIV_LOCAL_ONLY=1` overrides.
    """

    def __init__(
        self,
        local,
        live,
        allow_live_topup: bool = True,
        min_local_ratio: float = 0.5,
    ):
        self.local = local
        self.live = live
        self.allow_live_topup = allow_live_topup
        self.min_local_ratio = min_local_ratio

    def search(self, query: str, n: int, sort: str) -> list:
        local_results = (
            self.local.search(query, n, sort)
            if self.local.is_available()
            else []
        )

        local_only_env = os.getenv("PD_ARXIV_LOCAL_ONLY", "0") not in (
            "", "0", "false", "False",
        )
        if local_only_env:
            return local_results[:n]

        threshold = max(1, int(n * self.min_local_ratio))
        if (
            len(local_results) >= threshold
            or not self.allow_live_topup
            or not self.live.is_available()
        ):
            return local_results[:n]

        live_results = self.live.search(query, n - len(local_results), sort)
        seen_ids = {p.arxiv_id for p in local_results}
        merged = list(local_results)
        for p in live_results:
            if p.arxiv_id not in seen_ids:
                merged.append(p)
                seen_ids.add(p.arxiv_id)
        return merged[:n]

    def is_available(self) -> bool:
        return self.local.is_available() or self.live.is_available()
