"""End-to-end orchestrator.

Wire-up only. Business logic lives in the subsystems.
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .config import Config
from .distill.article import distill as distill_article
from .distill.filter import rank
from .distill.survey import compose as compose_survey
from .extract.pymupdf_extractor import extract_text
from .llm.openai_compatible import LLMClient, LLMError
from .sources.arxiv import (
    Paper,
    search as arxiv_search,
    download_pdf,
    download_pdf_from_url,
)
from .sources.semantic_scholar import (
    search as ss_search,
    lookup_by_arxiv_id as ss_lookup_by_arxiv_id,
    lookup_by_doi as ss_lookup_by_doi,
)
from .vault.crosslink import load_index
from .vault.store import VaultStore, slugify


def _query_for(cfg: Config) -> str:
    if cfg.topic:
        return cfg.topic
    return f"author:{cfg.author}"


def _emit_summary(run_record: dict, vault_path: Path) -> None:
    log_dir = vault_path / ".paper_distiller"
    log_dir.mkdir(exist_ok=True)
    (log_dir / "runs.jsonl").open("a", encoding="utf-8").write(
        json.dumps(run_record, ensure_ascii=False) + "\n"
    )


def _gather_candidates(cfg: Config) -> list[Paper]:
    """Search the configured source(s); merge and dedupe."""
    query = _query_for(cfg)
    arxiv_results: list[Paper] = []
    ss_results: list[Paper] = []

    if cfg.source in ("arxiv", "both"):
        try:
            arxiv_results = arxiv_search(query, max_results=cfg.pool)
        except Exception as e:
            if cfg.verbose:
                print(f"  arxiv search failed: {e}")
            if cfg.source == "arxiv":
                raise

    if cfg.source in ("ss", "both"):
        try:
            ss_results = ss_search(query, max_results=cfg.pool,
                                     api_key=cfg.ss_api_key)
        except Exception as e:
            if cfg.verbose:
                print(f"  SS search failed: {e}")
            if cfg.source == "ss":
                raise

    return merge_candidates(arxiv_results, ss_results)


def merge_candidates(arxiv_papers: list[Paper],
                      ss_papers: list[Paper]) -> list[Paper]:
    """Dedupe across sources. Arxiv-sourced papers win when both have the same paper."""
    seen_keys: set[str] = set()

    def keys_for(p: Paper) -> list[str]:
        ks = []
        if p.arxiv_id:
            ks.append(f"arxiv:{p.arxiv_id}")
        if p.doi:
            ks.append(f"doi:{p.doi}")
        if not ks:
            ks.append(f"{p.source}:{p.paper_id}")
        return ks

    out: list[Paper] = []
    for p in list(arxiv_papers) + list(ss_papers):
        ks = keys_for(p)
        if any(k in seen_keys for k in ks):
            continue
        out.append(p)
        for k in ks:
            seen_keys.add(k)
    return out


def _fetch_with_fallback(paper: Paper, cfg: Config, tmpdir: Path) -> str:
    """Try paper.pdf_url; on failure, fall through to SS open-access; finally to ''."""
    pdf_path = None

    # Step 1: primary URL
    if paper.pdf_url:
        try:
            pdf_path = download_pdf_from_url(
                url=paper.pdf_url,
                dest_dir=tmpdir,
                filename=f"{paper.paper_id}.pdf",
                timeout=cfg.pdf_timeout_sec,
            )
        except Exception as e:
            if cfg.verbose:
                print(f"  primary PDF fetch failed for {paper.paper_id}: {e}")

    # Step 2: SS open-access fallback (only when primary failed AND paper was arxiv-sourced
    # with cross-source identity available)
    if pdf_path is None and paper.source == "arxiv" and (paper.arxiv_id or paper.doi):
        try:
            ss_record = None
            if paper.arxiv_id:
                ss_record = ss_lookup_by_arxiv_id(paper.arxiv_id,
                                                    api_key=cfg.ss_api_key)
            elif paper.doi:
                ss_record = ss_lookup_by_doi(paper.doi,
                                              api_key=cfg.ss_api_key)
            if ss_record and ss_record.open_access_pdf_url:
                if cfg.verbose:
                    print(f"  trying SS open-access PDF for {paper.paper_id}")
                pdf_path = download_pdf_from_url(
                    url=ss_record.open_access_pdf_url,
                    dest_dir=tmpdir,
                    filename=f"{paper.paper_id}-ss.pdf",
                    timeout=cfg.pdf_timeout_sec,
                )
                if cfg.verbose:
                    print(f"  SS open-access PDF fetched OK for {paper.paper_id}")
        except Exception as e:
            if cfg.verbose:
                print(f"  SS PDF fallback also failed: {e}")

    if pdf_path is None:
        return ""

    try:
        return extract_text(pdf_path)
    except Exception as e:
        if cfg.verbose:
            print(f"  PDF parse failed for {paper.paper_id}: {e}")
        return ""


def run(cfg: Config) -> dict:
    """Execute one L2 pipeline run. Returns the run summary dict."""
    start = time.time()
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "topic": cfg.topic or f"author:{cfg.author}",
        "n_requested": cfg.top_n,
        "candidates_found": 0,
        "after_filter": 0,
        "distilled": 0,
        "skipped_dedup": 0,
        "skipped_failed": 0,
        "depth_breakdown": {"full-pdf": 0, "abstract-only": 0},
        "article_slugs": [],
        "survey_slug": None,
        "duration_sec": 0,
        "tokens_in_total": 0,
        "tokens_out_total": 0,
    }

    if cfg.dry_run:
        print(f"[DRY-RUN] Would search arxiv for '{_query_for(cfg)}' "
              f"(pool={cfg.pool}, top_n={cfg.top_n}), distill PDFs, "
              f"and write to {cfg.vault_path}.")
        return summary

    store = VaultStore(cfg.vault_path)
    wiki_index = load_index(store)
    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model)

    # 3. gather candidates from configured source(s)
    candidates = _gather_candidates(cfg)
    summary["candidates_found"] = len(candidates)
    if not candidates:
        summary["duration_sec"] = round(time.time() - start, 1)
        _emit_summary(summary, cfg.vault_path)
        return summary

    # 4. filter
    top = rank(candidates, cfg.topic or cfg.author, cfg.top_n, llm)
    summary["after_filter"] = len(top)

    # 5. per-paper loop
    articles = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for paper in top:
            # ID-based dedup (arxiv_id first, then doi). Skips both if --force.
            if not cfg.force:
                existing = None
                if paper.arxiv_id:
                    existing = store.find_by_arxiv_id(paper.arxiv_id)
                    if existing is not None and cfg.verbose:
                        print(f"  skipping arxiv:{paper.arxiv_id} — "
                              f"already in articles/{existing.slug}.md")
                if existing is None and paper.doi:
                    existing = store.find_by_doi(paper.doi)
                    if existing is not None and cfg.verbose:
                        print(f"  skipping doi:{paper.doi} — "
                              f"already in articles/{existing.slug}.md")
                if existing is not None:
                    summary["skipped_dedup"] += 1
                    continue

            # Slug-based fallback (legacy entries without refs)
            from .vault.store import slugify
            if paper.arxiv_id:
                arxiv_slug = f"paper-{paper.arxiv_id}"
            else:
                arxiv_slug = f"paper-{paper.paper_id}"
            title_slug = slugify(paper.title)
            if (
                store.slug_exists("articles", arxiv_slug)
                or store.slug_exists("articles", title_slug)
            ) and not cfg.force:
                summary["skipped_dedup"] += 1
                continue

            full_text = _fetch_with_fallback(paper, cfg, Path(tmpdir))

            try:
                article = distill_article(paper, full_text, wiki_index, llm)
            except LLMError as e:
                summary["skipped_failed"] += 1
                if cfg.verbose:
                    print(f"  LLM distill failed for {paper.arxiv_id}: {e}")
                continue

            saved = store.save_entry(
                category="articles",
                **article.to_save_kwargs(),
            )
            articles.append(article)
            summary["distilled"] += 1
            summary["article_slugs"].append(saved["slug"])
            summary["depth_breakdown"][article.depth] += 1

    # 6. survey
    if len(articles) >= cfg.min_papers_for_survey:
        try:
            survey = compose_survey(articles, cfg.topic or cfg.author, wiki_index, llm)
            store.save_entry(category="surveys", **survey.to_save_kwargs())
            summary["survey_slug"] = survey.slug
        except LLMError as e:
            if cfg.verbose:
                print(f"  Survey composition failed: {e}")

    summary["duration_sec"] = round(time.time() - start, 1)
    try:
        summary["tokens_in_total"] = int(llm.total_tokens_in)
        summary["tokens_out_total"] = int(llm.total_tokens_out)
    except (TypeError, ValueError):
        # llm.total_tokens_* may be non-numeric in tests with a mocked LLMClient.
        summary["tokens_in_total"] = 0
        summary["tokens_out_total"] = 0

    _emit_summary(summary, cfg.vault_path)
    return summary
