# paper-distiller v0.2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship v0.2.0 — arxiv-id-based dedup (prevents duplicate articles for the same paper under different slugs) + restore the 500-char full-pdf threshold (defends against corrupt-PDF garbage extraction).

**Architecture:** Two surgical changes to existing modules. Add `VaultStore.find_by_arxiv_id()` and wire it into pipeline's per-paper loop ahead of the slug-based fallback. Change one truthy check in `distill/article.py` to a length-thresholded check. Zero new files in `src/`, zero new dependencies.

**Tech Stack:** Same as v0.1 — Python 3.10+, httpx, arxiv, pymupdf, python-dotenv, pytest. No additions.

**Spec:** [docs/superpowers/specs/2026-05-18-paper-distiller-v0.2.0-design.md](../specs/2026-05-18-paper-distiller-v0.2.0-design.md)

---

## File Structure

| File | Action | LOC delta |
|---|---|---|
| `src/paper_distiller/vault/store.py` | Add `find_by_arxiv_id` method | +21 |
| `src/paper_distiller/pipeline.py` | Insert arxiv-id check ahead of slug fallback | +6 |
| `src/paper_distiller/distill/article.py` | Change truthy check → `> 500` threshold | 1 line modified |
| `src/paper_distiller/__init__.py` | `__version__ = "0.2.0"` | 1 line modified |
| `pyproject.toml` | `version = "0.2.0"` | 1 line modified |
| `CHANGELOG.md` | Add v0.2.0 entry | ~25 |
| `tests/test_vault_store.py` | 3 new tests for `find_by_arxiv_id` | +30 |
| `tests/test_pipeline.py` | 1 new integration test for arxiv-id dedup | +30 |
| `tests/test_distill_article.py` | Update 1 fixture + add 1 new test | +12 |

**Test count**: v0.1 had 45 passing. v0.2 adds 5 new (3 vault + 1 pipeline + 1 article). Total after v0.2: **50 passing**.

**Working directory throughout this plan: `G:\paper-distiller\`**

---

## Task 1: Add `VaultStore.find_by_arxiv_id()` (TDD)

**Files:**
- Modify: `src/paper_distiller/vault/store.py` (add method to `VaultStore` class)
- Modify: `tests/test_vault_store.py` (append 3 new tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vault_store.py` (after the existing `test_slug_path_traversal_rejected` parametrize block):

```python
def test_find_by_arxiv_id_hit(tmp_vault: Path):
    """find_by_arxiv_id returns the Entry whose refs contains the matching arxiv: ref."""
    store = VaultStore(tmp_vault)
    store.save_entry(
        title="CoFinDiff",
        category="articles",
        body="x",
        refs=["arxiv:2503.04164"],
        slug="cofindiff-controllable",
    )
    found = store.find_by_arxiv_id("2503.04164")
    assert found is not None
    assert found.slug == "cofindiff-controllable"
    assert "arxiv:2503.04164" in found.refs


def test_find_by_arxiv_id_miss(tmp_vault: Path):
    """find_by_arxiv_id returns None when no article references the given arxiv id."""
    store = VaultStore(tmp_vault)
    store.save_entry(title="X", category="articles", body="x",
                     refs=["arxiv:9999.99999"])
    assert store.find_by_arxiv_id("0000.00000") is None


def test_find_by_arxiv_id_only_scans_articles(tmp_vault: Path):
    """Non-articles categories (e.g. surveys with arxiv refs) must not match — dedup
    scope is paper notes only."""
    store = VaultStore(tmp_vault)
    store.save_entry(title="A survey", category="surveys", body="x",
                     refs=["arxiv:2503.04164"])
    assert store.find_by_arxiv_id("2503.04164") is None
```

- [ ] **Step 2: Run tests to verify failure**

```bash
.venv\Scripts\python.exe -m pytest tests/test_vault_store.py::test_find_by_arxiv_id_hit tests/test_vault_store.py::test_find_by_arxiv_id_miss tests/test_vault_store.py::test_find_by_arxiv_id_only_scans_articles -v
```

Expected: 3 failures with `AttributeError: 'VaultStore' object has no attribute 'find_by_arxiv_id'`.

- [ ] **Step 3: Implement `find_by_arxiv_id`**

In `src/paper_distiller/vault/store.py`, add this method to the `VaultStore` class. Place it AFTER `slug_exists` and BEFORE `list_entries`:

```python
    def find_by_arxiv_id(self, arxiv_id: str) -> Entry | None:
        """Find an article whose `refs` frontmatter contains `arxiv:<arxiv_id>`.

        Returns the first matching Entry, or None if no match.

        Only scans the `articles/` subdirectory — other categories use different
        ref conventions (e.g. `session:<slug>` for surveys) and would create
        false-positive matches if scanned.
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

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv\Scripts\python.exe -m pytest tests/test_vault_store.py -v
```

Expected: all vault tests pass (the 3 new + the existing 16 = 19 total in `test_vault_store.py`).

- [ ] **Step 5: Run full suite as a sanity check**

```bash
.venv\Scripts\python.exe -m pytest -v
```

Expected: 48 passed (45 v0.1 + 3 new).

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/vault/store.py tests/test_vault_store.py
git commit -m "feat(vault): add find_by_arxiv_id for arxiv-id-based dedup lookup"
```

---

## Task 2: Wire arxiv-id dedup into pipeline (TDD)

**Files:**
- Modify: `src/paper_distiller/pipeline.py` (insert dedup block in per-paper loop)
- Modify: `tests/test_pipeline.py` (add 1 new integration test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py` (after the existing `test_pipeline_dedup_skips_existing`):

```python
def test_pipeline_arxiv_id_dedup_skips_existing(tmp_path, mocker):
    """If the vault already has an article with refs containing this arxiv id,
    skip — even if the slug pattern doesn't match. Fixes the v0.1 issue where
    cofindiff.md and cofindiff-controllable-financial-diffusion.md could both
    exist for the same arxiv paper."""
    from paper_distiller.pipeline import run
    from paper_distiller.vault.store import VaultStore
    cfg = _config(tmp_path); cfg.vault_path.mkdir()
    store = VaultStore(cfg.vault_path)
    # Pre-populate with a hand-written-style entry: slug doesn't match arxiv pattern,
    # but refs contains the arxiv id of the candidate we'll search for.
    store.save_entry(
        title="CoFinDiff (hand-written)",
        category="articles",
        body="pre-existing hand-written content",
        refs=["arxiv:2501.00001"],
        slug="cofindiff-handwritten",
    )

    mocker.patch("paper_distiller.pipeline.arxiv_search",
                 return_value=[_paper(1)])  # _paper(1) has arxiv_id "2501.00001"
    mocker.patch("paper_distiller.pipeline.rank",
                 return_value=[_paper(1)])
    mock_distill = mocker.patch("paper_distiller.pipeline.distill_article")
    mocker.patch("paper_distiller.pipeline.compose_survey")
    mocker.patch("paper_distiller.pipeline.LLMClient")

    run(cfg)

    log = (cfg.vault_path / ".paper_distiller" / "runs.jsonl").read_text()
    line = json.loads(log.strip().split("\n")[-1])
    assert line["skipped_dedup"] == 1
    assert line["distilled"] == 0
    # Critically: distill_article was never called — the skip happened upstream
    mock_distill.assert_not_called()
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv\Scripts\python.exe -m pytest tests/test_pipeline.py::test_pipeline_arxiv_id_dedup_skips_existing -v
```

Expected: FAIL. Because the existing pipeline only checks `paper-<arxiv_id>` and `slugify(paper.title)` slug patterns, neither of which matches `cofindiff-handwritten`. The test will see `distilled == 1` and `mock_distill.assert_not_called()` will raise.

- [ ] **Step 3: Implement the dedup integration**

In `src/paper_distiller/pipeline.py`, locate the per-paper loop (currently around line 80–95). The existing block looks like:

```python
    for paper in top:
        from .vault.store import slugify
        arxiv_slug = f"paper-{paper.arxiv_id}"
        title_slug = slugify(paper.title)
        if (
            store.slug_exists("articles", arxiv_slug)
            or store.slug_exists("articles", title_slug)
        ) and not cfg.force:
            summary["skipped_dedup"] += 1
            continue
        ...
```

Replace with this (adds the arxiv-id check BEFORE the existing slug fallback):

```python
    for paper in top:
        # arxiv-id-based dedup: search vault for any article whose `refs` contains
        # this arxiv id. Catches duplicates where slug pattern differs from our
        # conventions (e.g. user hand-wrote the entry with a longer descriptive slug).
        if not cfg.force:
            existing = store.find_by_arxiv_id(paper.arxiv_id)
            if existing is not None:
                summary["skipped_dedup"] += 1
                if cfg.verbose:
                    print(f"  skipping arxiv:{paper.arxiv_id} — already in "
                          f"articles/{existing.slug}.md")
                continue

        # Slug-based fallback: legacy entries without refs metadata, or future
        # callers that bypass our save_entry conventions.
        from .vault.store import slugify
        arxiv_slug = f"paper-{paper.arxiv_id}"
        title_slug = slugify(paper.title)
        if (
            store.slug_exists("articles", arxiv_slug)
            or store.slug_exists("articles", title_slug)
        ) and not cfg.force:
            summary["skipped_dedup"] += 1
            continue
        ...
```

The `...` is everything after the existing dedup block (PDF fetch, extract, distill, save) — leave that completely untouched.

- [ ] **Step 4: Run the new test to verify it passes**

```bash
.venv\Scripts\python.exe -m pytest tests/test_pipeline.py::test_pipeline_arxiv_id_dedup_skips_existing -v
```

Expected: PASS.

- [ ] **Step 5: Run the full pipeline test file to confirm no regression**

```bash
.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v
```

Expected: 4 passed (3 v0.1 + 1 new).

- [ ] **Step 6: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -v
```

Expected: 49 passed (48 prior + 1 new).

- [ ] **Step 7: Commit**

```bash
git add src/paper_distiller/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): arxiv-id dedup ahead of slug-based fallback

Prevents duplicate articles when a paper already exists under a different
slug pattern (e.g. hand-written 'cofindiff-controllable-financial-diffusion'
already references arxiv:2503.04164, so a new run that would write
'cofindiff' is now skipped).

The slug-based check remains as a fallback for legacy entries whose
frontmatter does not include 'arxiv:<id>' in refs."
```

---

## Task 3: Restore 500-char full-pdf threshold (TDD)

**Files:**
- Modify: `src/paper_distiller/distill/article.py` (one line)
- Modify: `tests/test_distill_article.py` (update 1 existing test fixture, add 1 new test)

- [ ] **Step 1: Update existing test fixture + add new test**

In `tests/test_distill_article.py`:

**A) UPDATE** the existing `test_distill_returns_article_result` to use a 600-character `full_text` (above the 500-char threshold). Find:

```python
def test_distill_returns_article_result():
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "title": "测试论文",
        "body": "# 测试\n\n## 一句话\n这是测试。",
        "tags": ["test"],
        "refs": ["arxiv:2501.00001"],
    })
    result = distill(_paper(), "full text", _index_with([]), llm)
    assert result.slug == "ce-shi-lun-wen" or result.slug.startswith("entry-")  # CJK fallback
    assert result.title == "测试论文"
    assert "测试" in result.body
    assert result.tags == ["test"]
    assert result.refs == ["arxiv:2501.00001"]
    assert result.depth == "full-pdf"
```

Replace the line `result = distill(_paper(), "full text", _index_with([]), llm)` with:

```python
    result = distill(_paper(), "x" * 600, _index_with([]), llm)  # 600 chars > 500 threshold
```

**B) ADD** a new test below `test_distill_marks_abstract_only_when_no_full_text` (and before `test_scrub_invented_links_strips_unknown_slugs`):

```python
def test_distill_falls_back_to_abstract_for_short_extract():
    """If full_text is shorter than 500 chars (e.g. PyMuPDF returned garbage
    from a scanned/corrupt PDF), depth_mode should be 'abstract-only' so the
    article body gets the ⚠️ callout and the LLM uses the abstract instead."""
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "title": "T", "body": "# T\nbody", "tags": [], "refs": [],
    })
    result = distill(
        _paper(),
        full_text="50 chars of garbage from a corrupt PDF",  # 38 chars, well below 500
        wiki_index=_index_with([]),
        llm=llm,
    )
    assert result.depth == "abstract-only"
```

- [ ] **Step 2: Run tests to verify failure**

```bash
.venv\Scripts\python.exe -m pytest tests/test_distill_article.py -v
```

Expected: `test_distill_returns_article_result` still passes (600 > current "any truthy" rule), but `test_distill_falls_back_to_abstract_for_short_extract` FAILS — because the current code (`depth_mode = "full-pdf" if full_text else "abstract-only"`) treats the 38-char garbage as full-pdf.

- [ ] **Step 3: Apply the one-line code change**

In `src/paper_distiller/distill/article.py`, find this line (currently at line 62 in the `distill` function):

```python
    depth_mode = "full-pdf" if full_text else "abstract-only"
```

Replace with:

```python
    depth_mode = "full-pdf" if full_text and len(full_text) > 500 else "abstract-only"
```

(One added clause: `and len(full_text) > 500`.)

- [ ] **Step 4: Run tests to verify both pass**

```bash
.venv\Scripts\python.exe -m pytest tests/test_distill_article.py -v
```

Expected: 4 passed (3 existing + 1 new). The updated `test_distill_returns_article_result` still passes (600 > 500); `test_distill_falls_back_to_abstract_for_short_extract` now passes (38 < 500 → abstract-only).

- [ ] **Step 5: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -v
```

Expected: 50 passed (49 prior + 1 new).

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/distill/article.py tests/test_distill_article.py
git commit -m "fix(distill): restore 500-char full-pdf threshold

v0.1 dropped the 500-char check (matching the test fixture's 9-char input
that called itself 'full text'). Now both the test and the code use the
intended threshold from the original design — protects against PyMuPDF
returning a tiny amount of junk text from scanned or corrupt PDFs."
```

---

## Task 4: Bump version + update CHANGELOG

**Files:**
- Modify: `src/paper_distiller/__init__.py`
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Bump `__version__`**

In `src/paper_distiller/__init__.py`, change:

```python
__version__ = "0.1.0"
```

to:

```python
__version__ = "0.2.0"
```

- [ ] **Step 2: Bump `pyproject.toml` version**

In `pyproject.toml`, change:

```toml
version = "0.1.0"
```

to:

```toml
version = "0.2.0"
```

- [ ] **Step 3: Update `CHANGELOG.md`**

Prepend a new `[0.2.0]` section above the existing `[0.1.0]` entry:

```markdown
## [0.2.0] — 2026-05-18

### Added
- `VaultStore.find_by_arxiv_id(arxiv_id)` — look up an article by its arxiv ref. Used by the pipeline for precise dedup.
- Pipeline: arxiv-id-based dedup runs ahead of the slug-based fallback. Prevents creating a sibling article (e.g. `cofindiff.md`) when one already exists for the same arxiv paper under a different slug (e.g. hand-written `cofindiff-controllable-financial-diffusion.md` with `refs: ["arxiv:2503.04164"]`).
- Verbose mode now logs which existing entry caused a dedup skip.

### Fixed
- `distill/article.py` now uses `len(full_text) > 500` as the threshold for "full-pdf" mode. v0.1's truthy check would tag a 50-byte garbage extraction from a scanned PDF as full-pdf and feed it to the LLM as the paper's content. Now such cases correctly fall back to abstract-only with the ⚠️ callout.

### Internal
- 5 new unit/integration tests; total now 50 (was 45 in v0.1.0).
- No new runtime dependencies.
```

- [ ] **Step 4: Verify version consistency**

```bash
findstr /R "^version" pyproject.toml
findstr "__version__" src\paper_distiller\__init__.py
```

Both should now show `0.2.0`.

- [ ] **Step 5: Run full suite one more time**

```bash
.venv\Scripts\python.exe -m pytest -v
```

Expected: 50 passed.

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/__init__.py pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 0.2.0 + changelog"
```

---

## Task 5: Tag v0.2.0

- [ ] **Step 1: Create the annotated tag**

```bash
git tag -a v0.2.0 -m "v0.2.0 — arxiv-id dedup + 500-char full-pdf threshold"
```

- [ ] **Step 2: Verify the tag**

```bash
git tag --list -n
git show v0.2.0 --stat | head -20
git log --oneline | head -10
```

Expected:
- `v0.2.0` appears in tag list with message
- `git show` displays the annotated message and the most recent commit
- Recent log shows v0.2.0 chore commit at HEAD, plus the 4 fix/feat commits above it, plus the v0.1.0 chore

- [ ] **Step 3: (Optional) print final summary**

No commit needed for this step. Just verify the suite still passes:

```bash
.venv\Scripts\python.exe -m pytest --tb=no -q
```

Expected: `50 passed`.

---

## Acceptance criteria (from spec §7)

After all 5 tasks complete and v0.2.0 is tagged:

- [ ] `pytest -v` from `G:\paper-distiller\`: 50 tests pass
- [ ] `VaultStore.find_by_arxiv_id` returns Entry on hit, None on miss, only scans `articles/`
- [ ] Pipeline skips paper when vault has matching arxiv ref, regardless of slug
- [ ] `--force` overrides arxiv-id dedup (same precedence as slug dedup)
- [ ] Verbose mode prints which existing entry caused the skip
- [ ] `runs.jsonl` `skipped_dedup` counts the arxiv-id skips
- [ ] `distill/article.py` uses `> 500` threshold
- [ ] Short non-empty `full_text` (e.g. 38 chars) falls back to abstract-only mode
- [ ] `__version__` and `pyproject.toml` both show `0.2.0`
- [ ] `CHANGELOG.md` has a `[0.2.0]` section
- [ ] Annotated tag `v0.2.0` exists

## Self-review notes

**Spec coverage**: Task 1 implements §5.1 + §5.4 (3 tests). Task 2 implements §5.2 + §5.4 (1 integration test) + §5.3 edge cases (covered via existing fallback chain). Task 3 implements §6.1 + §6.2. Tasks 4–5 implement §8 release path.

**No placeholders detected**. All code in steps is complete and runnable.

**Type/name consistency**:
- `find_by_arxiv_id(arxiv_id: str) -> Entry | None` — signature consistent across Task 1 definition, Task 2 caller (`store.find_by_arxiv_id(paper.arxiv_id)`).
- `summary["skipped_dedup"]` increment — same key used in Task 2's new block and existing slug-based block.
- `Entry` dataclass — used directly from `vault/store.py`, same fields as v0.1.

**Estimated total effort**: 1–2 hours. Five sequential tasks, all narrow.
