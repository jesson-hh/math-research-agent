"""Diagnostic: run ArxivSearcher.run() exactly as the agent does and time it.

If LocalFirstFetcher + LocalFetcher is wired correctly, this should return
in <10ms. If it falls through to live API, expect 10-90s.
"""

import asyncio
import os
import time

from paper_distiller.agents.searchers import ArxivSearcher, _build_arxiv_fetcher
from paper_distiller.agents.base import Context
from paper_distiller.config import load_config
from paper_distiller.llm.openai_compatible import LLMClient
from paper_distiller.vault.store import VaultStore


# Minimal config (LLM client is constructed but never called since
# ArxivSearcher doesn't invoke the LLM directly).
os.environ.setdefault("PD_API_KEY", "diag-no-call")
os.environ.setdefault("PD_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("PD_MODEL", "qwen3.5-plus")

# Smoke check: can we build the fetcher?
print("== fetcher build ==")
store, fetcher = _build_arxiv_fetcher()
print(f"  local available: {fetcher.local.is_available()}")
print(f"  local paper count: {store.paper_count():,}")
print(f"  db path: {store.path}")
store.close()
print()

# Full agent path
cfg = load_config(
    vault_path=r"G:\Math research Agent\wiki",
    topic="diffusion models",
    n=10,
    pool=30,
    source="arxiv",
)
vault = VaultStore(cfg.vault_path)
llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model)
ctx = Context(cfg=cfg, llm=llm, vault=vault, shared={"arxiv_sort": "date"})

print("== ArxivSearcher.run() ==")
t0 = time.time()
result = asyncio.run(ArxivSearcher().run(ctx))
elapsed = time.time() - t0
papers = result.get("candidates_arxiv", []) or []
print(f"  elapsed: {elapsed * 1000:.0f} ms")
print(f"  candidates: {len(papers)}")
for p in papers[:5]:
    print(f"    {p.arxiv_id}  {p.published}  {p.title[:60]}")
