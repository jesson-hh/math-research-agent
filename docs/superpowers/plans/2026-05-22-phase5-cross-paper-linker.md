# Phase 5: Cross-Paper Linker

> TDD, one commit per task, branch `feat/proof-graph-phase-1-2`.

**Goal:** Connect related nodes ACROSS papers with `cross_paper=1` edges (`same_as` / `uses_lemma` / `specializes` / `generalizes` / `contradicts`), making the graph plug-and-play ("续上"). Token-efficient: cheap deterministic candidate generation, then a small-context LLM classification per candidate pair (no O(N²) all-pairs).

**Architecture:** New `proofgraph/linker.py`. Reuse the store graph API: `nodes_by_paper`, `nodes_using_technique`, `search_nodes`, `add_edge`, `Edge`. LLM mocked in tests.

**Spec:** `docs/superpowers/specs/2026-05-21-deep-distill-proof-graph-design.md` §6.

---

## File Structure
- **Create** `src/paper_distiller/proofgraph/linker.py` — `find_candidates`, `classify_pair`, `LinkReport`, `link_paper`.
- **Create** `src/paper_distiller/proofgraph/prompts/link_classify.md`.
- **Create** `tests/proofgraph/test_linker.py`.

Run: `python -m pytest tests/proofgraph/ -q` then `python -m pytest -q` (508 before this phase).

## Data contracts
- `LinkReport`: `pairs_considered: int`, `edges_created: int`, `by_rel: dict[str,int]`.
- Valid relations: `{"same_as","specializes","generalizes","uses_lemma","contradicts"}` (the LLM may also answer `"none"` → no edge).

---

## Task 5.1: `find_candidates` (deterministic)

**Files:** Create `linker.py`; Test `test_linker.py`.

- [ ] **Test first:** seed a `ProofStore(tmp_path/"proofs.db")` with paper "A" node (kind theorem, text "Bound via Bernstein concentration", techniques ["Bernstein"]) and paper "B" node (text "We use Bernstein concentration to bound the tail", techniques ["Bernstein"]) and an unrelated paper "C" node (text "convex optimization", techniques ["SGD"]). `find_candidates(store, node_A, k=6)` returns a list of `Node` containing B's node, NOT A's own node, NOT C's node. (B is found via technique overlap and/or FTS text match.)
- [ ] **Run → fail.**
- [ ] **Implement** `find_candidates(store, node, k=6) -> list[Node]`: gather `store.nodes_using_technique(t, limit=k)` for each `t in node.techniques`, plus `store.search_nodes(node.text, limit=k)`; exclude any candidate whose `paper_arxiv_id == node.paper_arxiv_id` or whose `id == node.id`; dedup by `id` (preserve first-seen order: technique matches before text matches); return first `k`.
- [ ] **Run → pass.** Commit: `feat(proofgraph): linker find_candidates (technique + FTS, cross-paper only)`.

## Task 5.2: `classify_pair` (LLM)

**Files:** Modify `linker.py`; Create `prompts/link_classify.md`; Test `test_linker.py`.

- [ ] **Test first** (mock llm): a stub `.complete` returning `'{"rel":"same_as","justification":"both state the same Bernstein bound"}'` → `classify_pair(a, b, llm)` returns `("same_as", "both state...")`. A stub returning `'{"rel":"none","justification":"unrelated"}'` → returns `(None, "unrelated")`. Garbage / invalid `rel` → returns `(None, ...)` (abstain, never invents an edge).
- [ ] **Run → fail.**
- [ ] **Implement** `classify_pair(node_a, node_b, llm) -> tuple[str | None, str]`: load `prompts/link_classify.md`, format with both nodes' `text` + `source_quote`; call `llm.complete(..., response_format="json")`; parse JSON; if `rel` in the valid set → return `(rel, justification)`, else → `(None, justification or "")`. Tolerate junk (try/except → `(None, "")`). The prompt instructs: pick the single best relation or `"none"`; cite which spans justify it; abstain (`none`) when unsure.
- [ ] **Run → pass.** Commit: `feat(proofgraph): linker classify_pair (small-context LLM, abstains)`.

## Task 5.3: `link_paper` (orchestrate + write cross-paper edges)

**Files:** Modify `linker.py`; Test `test_linker.py`.

- [ ] **Test first** (mock llm returning `same_as` for the A↔B pair): seed the 3-paper store from 5.1; `report = link_paper(store, "A", llm, k=6)`; assert a `cross_paper=1` edge with `rel="same_as"` exists from A's node to B's node (`store.out_edges(node_A.id)` includes it); `report.edges_created == 1`, `report.by_rel["same_as"] == 1`, `report.pairs_considered >= 1`. Re-running `link_paper` does not duplicate the edge (the store's `add_edge` is idempotent via `UNIQUE(src,dst,rel)`).
- [ ] **Run → fail.**
- [ ] **Implement** `link_paper(store, paper_arxiv_id, llm, *, k=6) -> LinkReport`: for each `node in store.nodes_by_paper(paper_arxiv_id)`: `cands = find_candidates(store, node, k)`; for each `cand`: `rel, just = classify_pair(node, cand, llm)`; `pairs_considered += 1`; if `rel`: `store.add_edge(Edge(src_id=node.id, dst_id=cand.id, rel=rel, justification=just, cross_paper=1))`; tally `edges_created` + `by_rel[rel]`. Return `LinkReport`.
- [ ] **Run → pass.** Commit: `feat(proofgraph): link_paper writes cross-paper edges + LinkReport`.

## Task 5.4: full suite + ruff
- [ ] `python -m pytest -q` (508 + new, green); `python -m ruff check src/paper_distiller/proofgraph/linker.py`. Clean.

---

## Notes
- Token efficiency = the point: only top-`k` candidates per node get an LLM call (small two-node context), not all pairs.
- Citation-based candidates (via `cites` edges) are a future enhancement — technique-overlap + FTS are sufficient here.
- Wiring `link_paper` into the batch distill flow (call it after all papers ingested) is part of phase 6/usage wiring, not this plan — phase 5 delivers the linker as a library function with tests.
