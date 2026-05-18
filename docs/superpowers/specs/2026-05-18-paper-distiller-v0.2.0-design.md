# paper-distiller v0.2.0 — Design

**Date**: 2026-05-18
**Author**: brainstorm session (post-v0.1.0 ship)
**Status**: design (pending implementation plan)
**Target**: `G:\paper-distiller\` (existing repo, `main` branch, baseline tag `v0.1.0` @ `e77cb60`)

---

## 1. Goal

Close two known v0.1.0 gaps that the first real smoke test exposed:

1. **arxiv-id-based dedup**: pipeline currently checks two slug patterns (`paper-<arxiv_id>` and `slugify(paper.title)`), neither of which catches the case where the same paper already exists in the vault under a slug derived from a different title (e.g. a hand-written `cofindiff-controllable-financial-diffusion.md` vs. agent-written `cofindiff.md` — both referencing arxiv:2503.04164). Result: duplicate entries for the same paper.

2. **500-char full-pdf threshold**: `distill.article.distill()` currently uses a truthy check (any non-empty `full_text` → `full-pdf` mode). The original design intent was `len(full_text) > 500` so corrupt/scanned PDFs that yield only a few junk bytes fall back to abstract-only mode with the ⚠️ callout. The v0.1 plan's spec text said 500-char; the test fixture used 9-char — implementer aligned to the test. v0.2 restores the threshold + fixes the test fixture.

This is a small, focused release. No new dependencies, no architectural changes.

---

## 2. Context

- v0.1.0 (commit `e77cb60`, tag `v0.1.0`) shipped 2026-05-18. 17 commits, 45 unit + 3 integration tests passing.
- Real smoke test (CoFinDiff paper, arxiv:2503.04164) produced high-quality output: 4 valid `[[wikilinks]]`, ¥0.15, 147 seconds.
- **Issue surfaced**: agent wrote `wiki/articles/cofindiff.md` despite a hand-written `wiki/articles/cofindiff-controllable-financial-diffusion.md` already existing for the same arxiv paper. Current dedup did not catch this because the slugs differ and the agent doesn't yet check the `refs` frontmatter field.
- Code reviewer at Task 9 (commit `56f36d9`) had flagged "arxiv-id-based dedup" as a v0.2 follow-up. Reality validated.
- Task 9 plan/test inconsistency: implementer dropped the 500-char check to match the test. Now correcting both.

## 3. Out of scope for v0.2.0

Deferred to later v0.x releases (already on roadmap in README):

| Not doing in v0.2.0 | Reason / target |
|---|---|
| Semantic Scholar second source | v0.3.0 |
| LEANN-backed crosslink retrieval | v0.4.0 (wiki not yet big enough) |
| L3 multi-round research loop | v0.5.0 |
| Per-vault `paper-distiller.toml` schema override | v0.6.0 (or later) |
| Changing the existing CoFinDiff duplicate in `G:\Math research Agent\wiki\` | User's decision — v0.2.0 only prevents *new* duplicates |

## 4. Architectural change summary

**Two surgical changes to existing modules. Zero new files in `src/`.**

| File | Change | LOC delta |
|---|---|---|
| `src/paper_distiller/vault/store.py` | Add `VaultStore.find_by_arxiv_id(arxiv_id) -> Entry \| None` | +12 |
| `src/paper_distiller/pipeline.py` | Per-paper loop checks `find_by_arxiv_id` before existing slug-based check; increments `skipped_dedup` and (verbose) logs the existing slug | +6 |
| `src/paper_distiller/distill/article.py` | `depth_mode = "full-pdf" if full_text and len(full_text) > 500 else "abstract-only"` | +0 (1 modified line) |
| `tests/test_vault_store.py` | New tests: `test_find_by_arxiv_id_hit`, `test_find_by_arxiv_id_miss`, `test_find_by_arxiv_id_only_articles` | +25 |
| `tests/test_pipeline.py` | New test: `test_pipeline_arxiv_id_dedup_skips_existing` | +30 |
| `tests/test_distill_article.py` | Update existing `test_distill_returns_article_result` fixture (9→600 chars); add `test_distill_falls_back_to_abstract_for_short_extract` | +12 |

## 5. Detailed design — arxiv-id dedup

### 5.1 `VaultStore.find_by_arxiv_id()`

```python
def find_by_arxiv_id(self, arxiv_id: str) -> Entry | None:
    """Find an article whose `refs` frontmatter contains `arxiv:<arxiv_id>`.
    Returns the first matching Entry, or None if no match.

    Only scans the `articles/` subdirectory — other categories use different ref
    conventions (e.g. `session:<slug>` for surveys) and won't be polluted.
    """
    target_ref = f"arxiv:{arxiv_id}"
    folder = self.root / "articles"
    if not folder.exists():
        return None
    for f in folder.glob("*.md"):
        try:
            meta, body = parse_frontmatter(f.read_text(encoding="utf-8"))
            if target_ref in (meta.get("refs") or []):
                return Entry(
                    slug=meta.get("slug", f.stem),
                    category=meta.get("category", "articles"),
                    title=meta.get("title", f.stem),
                    tags=meta.get("tags") or [],
                    refs=meta.get("refs") or [],
                    links=meta.get("links") or [],
                    created=meta.get("created", ""),
                    updated=meta.get("updated", ""),
                    body=body,
                )
        except Exception:
            continue
    return None
```

**Why not memoize?** A vault scan over 50 articles is <10 ms. A run only does one scan (or a few in a loop with N≤5). YAGNI on caching.

**Why only `articles/`?** The dedup question is "does this paper already have a distilled note?" — answered exclusively by `articles/`. A survey's `refs` like `session:diffusion-20260518` could falsely match if we ever stored arxiv ids there; scoping to `articles/` is defensive.

### 5.2 Pipeline integration

```python
# pipeline.py — per-paper loop, REPLACE current dedup block
for paper in top:
    # NEW: arxiv-id-based dedup (precise, frontmatter-driven)
    if not cfg.force:
        existing = store.find_by_arxiv_id(paper.arxiv_id)
        if existing is not None:
            summary["skipped_dedup"] += 1
            if cfg.verbose:
                print(f"  skipping arxiv:{paper.arxiv_id} — already in articles/{existing.slug}.md")
            continue

    # EXISTING: slug-based fallback (catches edge cases where refs not set)
    from .vault.store import slugify
    arxiv_slug = f"paper-{paper.arxiv_id}"
    title_slug = slugify(paper.title)
    if (
        store.slug_exists("articles", arxiv_slug)
        or store.slug_exists("articles", title_slug)
    ) and not cfg.force:
        summary["skipped_dedup"] += 1
        continue

    # ... rest unchanged: PDF download, extract, distill, save
```

`--force` short-circuits both checks. Slug-based check remains as belt-and-suspenders for legacy entries without refs.

### 5.3 Edge cases

| Case | Behavior |
|---|---|
| Existing article has empty/missing `refs` | `find_by_arxiv_id` returns None → slug-based fallback catches it (if slug pattern matches) → otherwise distills and creates new article. Acceptable; user can manually add refs to old entries to get future protection. |
| arxiv id with version suffix (e.g. `2503.04164v2`) | `sources/arxiv.py` already strips version via `.split("v")[0]`. Refs are always base id. Lookup is symmetric. |
| Two articles in vault both reference the same arxiv id | `find_by_arxiv_id` returns the first one found (filesystem-glob order, unspecified but deterministic per filesystem). v0.2 doesn't try to fix existing duplicates — it prevents new ones. |
| Corrupted frontmatter on a single article file | `try/except` silently skips; dedup may miss that file's arxiv id, falling through to slug check. Acceptable. |

### 5.4 Tests

```python
# tests/test_vault_store.py — additions

def test_find_by_arxiv_id_hit(tmp_vault: Path):
    store = VaultStore(tmp_vault)
    store.save_entry(title="CoFinDiff", category="articles", body="x",
                     refs=["arxiv:2503.04164"], slug="cofindiff-controllable")
    found = store.find_by_arxiv_id("2503.04164")
    assert found is not None
    assert found.slug == "cofindiff-controllable"

def test_find_by_arxiv_id_miss(tmp_vault: Path):
    store = VaultStore(tmp_vault)
    store.save_entry(title="X", category="articles", body="x", refs=["arxiv:9999.99999"])
    assert store.find_by_arxiv_id("0000.00000") is None

def test_find_by_arxiv_id_only_articles(tmp_vault: Path):
    """A survey or other category with arxiv refs (unlikely but possible) is not matched."""
    store = VaultStore(tmp_vault)
    store.save_entry(title="S", category="surveys", body="x", refs=["arxiv:2503.04164"])
    assert store.find_by_arxiv_id("2503.04164") is None  # only articles/ counts
```

```python
# tests/test_pipeline.py — addition

def test_pipeline_arxiv_id_dedup_skips_existing(tmp_path, mocker):
    """If vault has an article with refs containing this arxiv id, skip — even if slug differs."""
    from paper_distiller.pipeline import run
    from paper_distiller.vault.store import VaultStore
    cfg = _config(tmp_path); cfg.vault_path.mkdir()
    store = VaultStore(cfg.vault_path)
    # Pre-populate with hand-written-style article: different slug, but refs match
    store.save_entry(
        title="CoFinDiff (hand-written)",
        category="articles",
        body="hand-written content",
        refs=["arxiv:2501.00001"],
        slug="cofindiff-handwritten",
    )

    mocker.patch("paper_distiller.pipeline.arxiv_search",
                 return_value=[_paper(1)])
    mocker.patch("paper_distiller.pipeline.rank",
                 return_value=[_paper(1)])
    mock_distill = mocker.patch("paper_distiller.pipeline.distill_article")
    mocker.patch("paper_distiller.pipeline.LLMClient")

    run(cfg)

    line = json.loads((cfg.vault_path / ".paper_distiller" / "runs.jsonl").read_text().strip().split("\n")[-1])
    assert line["skipped_dedup"] == 1
    assert line["distilled"] == 0
    mock_distill.assert_not_called()  # never reached the distill stage
```

## 6. Detailed design — 500-char threshold

### 6.1 Code change

`distill/article.py:62` — one-line change:

```python
# BEFORE (v0.1)
depth_mode = "full-pdf" if full_text else "abstract-only"

# AFTER (v0.2)
depth_mode = "full-pdf" if full_text and len(full_text) > 500 else "abstract-only"
```

Rationale: aligns with the spec's original intent. Defends against PyMuPDF returning 50–200 bytes of garbage from scanned or corrupt PDFs.

### 6.2 Tests

```python
# tests/test_distill_article.py — UPDATE existing test
def test_distill_returns_article_result():
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "title": "测试论文",
        "body": "# 测试\n\n## 一句话\n这是测试。",
        "tags": ["test"],
        "refs": ["arxiv:2501.00001"],
    })
    # CHANGED: 9 chars → 600 chars (above 500-char threshold)
    result = distill(_paper(), "x" * 600, _index_with([]), llm)
    assert result.slug == "ce-shi-lun-wen" or result.slug.startswith("entry-")
    assert result.title == "测试论文"
    assert "测试" in result.body
    assert result.tags == ["test"]
    assert result.refs == ["arxiv:2501.00001"]
    assert result.depth == "full-pdf"


# tests/test_distill_article.py — NEW test (the gap that motivated the threshold)
def test_distill_falls_back_to_abstract_for_short_extract():
    """If full_text < 500 chars (e.g. PyMuPDF returned scanned-PDF garbage), use abstract."""
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "title": "T", "body": "# T\nbody", "tags": [], "refs": [],
    })
    result = distill(
        _paper(),
        full_text="50 chars of garbage from corrupt PDF",  # 36 chars, definitely < 500
        wiki_index=_index_with([]),
        llm=llm,
    )
    assert result.depth == "abstract-only"
```

The existing `test_distill_marks_abstract_only_when_no_full_text` (empty string input) stays unchanged.

## 7. Acceptance criteria

- [ ] `VaultStore.find_by_arxiv_id` returns Entry on hit, None on miss
- [ ] Only scans `articles/` (not surveys/etc.)
- [ ] Pipeline skips paper when vault has matching arxiv ref, regardless of slug
- [ ] `--force` overrides arxiv-id dedup (same as slug-based)
- [ ] Verbose mode prints which existing entry caused the skip
- [ ] `runs.jsonl` `skipped_dedup` count increments correctly
- [ ] `distill/article.py` uses `> 500` threshold
- [ ] Short non-empty `full_text` falls back to abstract-only mode
- [ ] All v0.1 tests still pass; new tests pass; total: 45 (v0.1) + 4 (3 new vault + 1 new pipeline) + 1 (new article) = **50 passing** after changes (5 new tests + 1 modified test still passes = net +4 tests)

Wait — count check: v0.1 had 45 tests. v0.2.0 adds:
- 3 new in test_vault_store.py
- 1 new in test_pipeline.py
- 1 new in test_distill_article.py (the short-extract fallback)
- 0 net change in test_distill_article.py's existing test (it's modified, not added)

Total after v0.2.0: 45 + 5 = **50 tests**.

## 8. Migration / release path

1. Implement on `main` branch directly (no feature branch needed — surgical changes)
2. Each fix in its own commit (`fix(vault): ...`, `fix(distill): ...`, `feat(pipeline): ...`)
3. Update `CHANGELOG.md` with v0.2.0 entry
4. Bump `__version__` and `pyproject.toml` to `0.2.0`
5. Tag `v0.2.0`

No new dependencies, no `.env` changes, no user action required to upgrade.

## 9. Implementation roadmap (for writing-plans skill)

Approximate task decomposition (~4–6 tasks):

1. Add `VaultStore.find_by_arxiv_id` + 3 unit tests (TDD)
2. Integrate arxiv-id dedup in pipeline.py + integration test
3. Restore 500-char threshold in distill/article.py + update/add tests
4. Bump version to 0.2.0 in `__init__.py` and `pyproject.toml`
5. Update CHANGELOG with v0.2.0 entry
6. Tag `v0.2.0`

Estimated total effort: 1–2 hours.
