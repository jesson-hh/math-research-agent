# Phase 7: Graph Queries + Docs Refresh + Eval Scaffold (closeout)

> TDD for code; docs are prose. One commit per task. Branch `feat/proof-graph-phase-1-2`.

**Goal:** Finish the user-facing surface (extend `find_proof` with graph queries), refresh the stale docs to reflect the proof-graph feature (and fix the README's known broken links), and scaffold a manual reviewer-quality eval.

**Spec:** `docs/superpowers/specs/2026-05-21-deep-distill-proof-graph-design.md` §8/§10.

---

## Task 7.1: extend `find_proof` with graph query types

**Files:** Modify `src/paper_distiller/chat/agent_tools.py`; Test `tests/test_agent_tools.py`.

Current `tool_find_proof(query_type, query=None, limit=10, *, vault_path)` supports `stats|list_techniques|by_technique|by_text|by_paper` over the theorem layer. Add graph query types over the node/edge layer:
- `by_step` — FTS over node text/source_quote via `store.search_nodes(query, limit)`; return node dicts.
- `dependency_walk` — `query` = a node id (int-as-string); return `store.dependency_walk(int(query))` node dicts (what the node rests on).
- `node` — `query` = node id; return that node + its out-edges (`store.get_node` + `store.out_edges`).

- [ ] **Test first** (seed a `ProofStore(tmp_path/"proofs.db")` with a couple nodes + a `depends_on` edge): `tool_find_proof("by_step", "Bernstein", vault_path=str(tmp_path))` returns `{"nodes":[...]}` with the matching node; `tool_find_proof("dependency_walk", str(child_id), vault_path=...)` returns the parent node; `tool_find_proof("node", str(nid), vault_path=...)` returns the node + its edges; bad/missing `query` for these → `{"error": ...}`.
- [ ] **Run → fail.**
- [ ] **Implement:** extend `_FIND_PROOF_SCHEMA`'s `query_type` enum + description with the three new modes; in `tool_find_proof`, add branches calling `store.search_nodes` / `store.dependency_walk` / `store.get_node`+`store.out_edges`, returning JSON-able dicts (node → `{id,kind,label,text,status,paper_arxiv_id,techniques}`; edge → `{src_id,dst_id,rel}`). Keep existing branches unchanged.
- [ ] **Run → pass.** Commit: `feat(chat): find_proof graph queries (by_step / dependency_walk / node)`.

## Task 7.2: refresh docs (ARCHITECTURE.md + README) — fix broken links

**Files:** Rewrite `docs/ARCHITECTURE.md`; Modify `README.md`. No tests (prose).

- [ ] **`docs/ARCHITECTURE.md`** — it is frozen at v1.0 ("one CLI, 11 agents") and wrong. Read the current code and rewrite to reflect reality: (1) two console scripts — `paper-distiller-chat` (AgentLoop = conversational brain with **8 LLM tools**: search, distill_by_id, show, ask, research, ask_user, find_proof, **review_proof**; plus one-shot subcommands) and `paper-distiller-arxiv` (local mirror); (2) the `agents/` async-DAG framework (Agent/Context/DAG/Orchestrator + fanout); (3) the **`proofgraph/` subsystem** (reader = segment + grounding gate; extractor; pipeline = build_graph_for_paper; linker; reviewer) and how `_DistillOne` calls `maybe_build_graph` when `PD_GRAPH_DEPTH` is set; (4) the `proofs/store.py` graph schema (theorems layer + nodes/edges/node_techniques/nodes_fts, SCHEMA_VERSION 2); (5) vault format. Keep it accurate and current; trim obsolete v1.0 detail.
- [ ] **`README.md`** — (a) add a "Proof graph & review" bullet to Features; add `review_proof` as the 8th row of the tools table; (b) document `PD_GRAPH_DEPTH` (off/theorem/step, default off) in the env-var table and the optional `pip install paper-distiller[compress]` extra (LLMLingua) in Install; (c) **FIX broken links**: the links to `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff` point to non-existent files — remove those links/sentences (keep a plain "Issues: GitHub Issues" line) until those files exist; the links to `docs/configuration.md` and `docs/tools.md` (non-existent) — repoint to the README's own Configure / tools-table sections or remove; update the tests badge number to the current passing count.
- [ ] Commit: `docs: refresh ARCHITECTURE.md to current state + README proof-graph + fix broken links`.

## Task 7.3: reviewer quality-eval scaffold (manual, not CI)

**Files:** Create `scripts/eval_reviewer.py` + `scripts/eval_fixtures/README.md` (or a small JSON fixture); Test: a tiny import smoke in `tests/`.

- [ ] **Implement** `scripts/eval_reviewer.py`: loads a small hand-labeled fixture (a JSON list of `{statement, source_quote, parents:[...], technique, gold_label}` where gold_label ∈ {ok, problem}), builds an in-memory `ProofStore` (`:memory:` or tmp), runs `review_node` against a REAL `LLMClient` (env-configured), and prints precision/recall/F1 of PROBLEM detection (treating {suspicious,gap,unsupported} as "problem"). Guard `if __name__ == "__main__":`. Include a 3-5 item starter fixture and a docstring explaining: this is run manually (`python scripts/eval_reviewer.py`), needs PD_API_KEY etc., is NOT part of CI, and exists because LLM proof-judgment quality can't be unit-tested (see "Proof or Bluff?").
- [ ] **Test:** a CI-safe test `tests/test_eval_reviewer_imports.py` that imports the script module and asserts the fixture parses + the metric function computes correct precision/recall on a synthetic confusion (NO LLM call). Keep it pure.
- [ ] **Run → pass.** Commit: `feat(eval): manual reviewer precision/recall harness + fixture`.

## Task 7.4: full suite + ruff
- [ ] `python -m pytest -q` (547 + new, green). `python -m ruff check src/paper_distiller/chat/agent_tools.py scripts/eval_reviewer.py`. Clean. (Add `scripts/` to ruff scope only if it's already configured; otherwise lint the file directly.)

---

## Notes
- The `review` one-shot subcommand is intentionally NOT in scope — the `review_proof` LLM tool already exposes the capability through the chat agent. Note it as a possible future addition.
- Docs must be ACCURATE to the current code — read it, don't guess. If you find a doc claim you can't verify in code, leave it out.
