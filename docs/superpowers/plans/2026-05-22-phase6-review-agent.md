# Phase 6: Review Agent + `review_proof` tool

> TDD, one commit per task, branch `feat/proof-graph-phase-1-2`.

**Goal:** Structured informal review — walk a proof's dependency DAG, label each node `{ok|suspicious|gap|unsupported|unstated}` with a grounded reason, propagate early-error taint to descendants, persist labels, and build a prioritized report. Expose it as a new `review_proof` LLM tool (the agent's 8th). It **locates suspicious steps / gaps; it does NOT certify correctness** (LLM confidence is down-weighted).

**Architecture:** New `proofgraph/reviewer.py`. Reuse store graph API: `nodes_by_paper`, `get_node`, `out_edges`, `nodes_using_technique`, `set_node_status`, `dependency_walk`. New tool wrapper in `chat/agent_tools.py`. LLM mocked in tests.

**Spec:** `docs/superpowers/specs/2026-05-21-deep-distill-proof-graph-design.md` §7.

---

## File Structure
- **Create** `src/paper_distiller/proofgraph/reviewer.py` — `ReviewResult`, `ReviewReport`, `review_node`, `compute_taint`, `review_target`.
- **Create** `src/paper_distiller/proofgraph/prompts/review_node.md`.
- **Modify** `src/paper_distiller/chat/agent_tools.py` — add `_REVIEW_PROOF_SCHEMA`, `tool_review_proof`, register in `TOOL_SCHEMAS` + `TOOL_FUNCTIONS` + `__all__`.
- **Modify** tests asserting tool count (find them: `grep -rn "TOOL_SCHEMAS" tests/` — `test_agent_tools.py` and `test_agent_loop.py` assert the count is 7; bump to **8**).
- **Create** `tests/proofgraph/test_reviewer.py`; **Modify** `tests/proofs`/agent_tools tests as needed.

Run: `python -m pytest -q` (526 before this phase).

## Data contracts
- `LABELS = {"ok","suspicious","gap","unsupported","unstated"}`; `PROBLEM = {"suspicious","gap","unsupported"}`.
- `ReviewResult`: `node_id:int`, `label:str`, `reason:str`, `confidence:float`, `tainted_by:list[int]` (default `[]`).
- `ReviewReport`: `target:str`, `nodes_reviewed:int`, `by_label:dict[str,int]`, `flagged:list[ReviewResult]`, `summary:str`.

---

## Task 6.1: `review_node` (local-context LLM judgment)

**Files:** Create `reviewer.py` + `prompts/review_node.md`; Test `test_reviewer.py`.

- [ ] **Test first** (seed a store; mock llm): a node with one `depends_on` parent and a technique. Stub `.complete` returns `'{"label":"suspicious","reason":"the leap from A to B is unjustified","confidence":0.9}'` → `review_node(store, node, llm)` returns `ReviewResult(node_id=node.id, label="suspicious", reason="...", confidence<=0.7)` (confidence DOWN-WEIGHTED, capped at 0.7). Junk LLM output → `label="unstated"`, `confidence==0.0` (abstain, no crash). Invalid label → `"unstated"`.
- [ ] **Run → fail.**
- [ ] **Implement** `review_node(store, node, llm) -> ReviewResult`:
  - Gather local context: parents = `[store.get_node(e.dst_id) for e in store.out_edges(node.id) if e.rel in {"depends_on","uses_lemma","uses_def","uses_assumption"}]`; kb = up to 3 nodes from `store.nodes_using_technique(t)` per technique `t in node.techniques` excluding `node.id`; same_as = `[store.get_node(e.dst_id) for e in store.out_edges(node.id, "same_as")]`.
  - Build prompt from `prompts/review_node.md` (node text+source_quote, the parents' texts, the kb exemplars, instructions: judge whether the node follows from its cited parents + technique AS STATED; pick one label; give a grounded reason citing the node's quote; abstain to `unstated` if you can't tell; do NOT certify correctness).
  - `llm.complete(..., response_format="json")`; parse `{label, reason, confidence}`; if `label not in LABELS` → `"unstated"`; `confidence = min(float(confidence or 0.0), 0.7)`; on any parse error → `ReviewResult(node.id, "unstated", "review inconclusive", 0.0, [])`.
- [ ] **Run → pass.** Commit: `feat(proofgraph): review_node local-context grounded labeling`.

## Task 6.2: `compute_taint` (error propagation)

**Files:** Modify `reviewer.py`; Test `test_reviewer.py`.

- [ ] **Test first:** seed nodes A,B,C in one paper with edges `B -depends_on-> A`, `C -depends_on-> B`. `label_by_id = {A:"suspicious", B:"ok", C:"ok"}`. `compute_taint(store, [A.id,B.id,C.id], label_by_id)` → `{B.id:[A.id], C.id:[A.id]}` (both transitively rest on the problematic A); `A.id` absent (it's the source, not tainted by an ancestor). Cycle-safe.
- [ ] **Run → fail.**
- [ ] **Implement** `compute_taint(store, node_ids, label_by_id) -> dict[int, list[int]]`: for each node id, walk its `depends_on` ancestors transitively (follow `out_edges(nid, "depends_on")` to `dst_id`, BFS, visited-set for cycle safety, capped); collect ancestors whose `label_by_id.get(anc) in PROBLEM`; if non-empty, map `nid -> sorted(those ancestor ids)`. Only include nodes that ARE tainted.
- [ ] **Run → pass.** Commit: `feat(proofgraph): compute_taint propagates problem labels down depends_on`.

## Task 6.3: `review_target` (orchestrate + persist + report)

**Files:** Modify `reviewer.py`; Test `test_reviewer.py`.

- [ ] **Test first** (mock llm returning `suspicious` for an early step, `ok` for the rest): seed a paper with a theorem + 2 proof_steps where step2 `depends_on` step1 and step1 is judged suspicious. `report = review_target(store, paper_arxiv_id="P", llm=llm)`. Assert: every reviewed node's status is persisted (`store.get_node(step1.id).status == "suspicious"`); `report.nodes_reviewed == 3`; `report.by_label` counts match; `report.flagged` contains step1 (own problem) AND step2 (tainted_by step1); flagged is ordered own-problems-before-tainted.
- [ ] **Run → fail.**
- [ ] **Implement** `review_target(store, *, paper_arxiv_id=None, node_id=None, llm) -> ReviewReport`:
  - Collect nodes: if `paper_arxiv_id` → `store.nodes_by_paper(paper_arxiv_id)`; elif `node_id` → `[store.get_node(node_id)] + store.dependency_walk(node_id)`; else raise `ValueError`.
  - For each node: `r = review_node(store, node, llm)`; `store.set_node_status(node.id, r.label)`; collect results + `label_by_id[node.id]=r.label`.
  - `taint = compute_taint(store, [n.id for n in nodes], label_by_id)`; for each result, set `result.tainted_by = taint.get(result.node_id, [])`.
  - `flagged` = results where `label in PROBLEM` OR `tainted_by` non-empty; sort own-problems first (those with `label in PROBLEM`), then tainted-only.
  - `by_label` = Counter of labels; `summary` = short string (e.g. f"{len(flagged)} of {n} nodes need attention").
  - Return `ReviewReport(target=paper_arxiv_id or f"node:{node_id}", nodes_reviewed=len(nodes), by_label=..., flagged=flagged, summary=...)`.
- [ ] **Run → pass.** Commit: `feat(proofgraph): review_target walks DAG, persists labels, builds report`.

## Task 6.4: `review_proof` agent tool (7 → 8 tools)

**Files:** Modify `chat/agent_tools.py`; update tool-count tests.

- [ ] **Test first:** (a) Update the existing assertions that `len(TOOL_SCHEMAS) == 7` to `== 8` in `test_agent_tools.py` and `test_agent_loop.py` (grep for them). (b) Add `test_tool_review_proof`: monkeypatch `paper_distiller.chat.agent_tools.review_target` with a stub returning a `ReviewReport`; call `tool_review_proof(target_type="paper", target="2110.1", vault_path=str(tmp_path))`; assert the returned dict has keys `target`, `nodes_reviewed`, `by_label`, `flagged`, `summary`; and that a bad `target_type` returns `{"error": ...}`.
- [ ] **Run → fail.**
- [ ] **Implement:**
  - `_REVIEW_PROOF_SCHEMA` (OpenAI tools format): name `review_proof`, description ("Structured review of a distilled proof: walks the proof graph, flags suspicious steps / logic gaps with grounded reasons + error propagation. LOCATES issues; does not certify correctness. Needs papers already distilled with PD_GRAPH_DEPTH set."), params: `target_type` (enum `["paper","node"]`), `target` (string: arxiv_id or node id), required `["target_type","target"]`.
  - `tool_review_proof(target_type, target, *, vault_path) -> dict`: validate `target_type`; `from ..proofgraph.reviewer import review_target`; `from ..proofs.store import open_for_vault`; build an `LLMClient` from env (`os.getenv("PD_API_KEY"/"PD_BASE_URL"/"PD_MODEL")`, mirroring `chat/cli.py::_run_agent`; if missing → `{"error": "LLM env not set"}`); open store; `report = review_target(store, paper_arxiv_id=target if target_type=="paper" else None, node_id=int(target) if target_type=="node" else None, llm=llm)`; `store.close()`; return a JSON-able dict (`flagged` → list of dicts). Wrap in try/except → `_error(e)`.
  - Register in `TOOL_SCHEMAS`, `TOOL_FUNCTIONS["review_proof"]`, and `__all__`.
- [ ] **Run → pass.** Commit: `feat(chat): review_proof tool exposes the review agent (8th tool)`.

## Task 6.5: full suite + ruff
- [ ] `python -m pytest -q` (526 + new, all green — including the bumped tool-count tests). `python -m ruff check src/paper_distiller/proofgraph/reviewer.py src/paper_distiller/chat/agent_tools.py`. Clean.

---

## Notes
- Down-weight LLM confidence (cap 0.7) and abstain to `unstated` on uncertainty — per spec §7 + the "Proof or Bluff?" caveat. Review LOCATES, never certifies.
- `tool_review_proof` builds its LLMClient from env like `_run_agent` does; tests monkeypatch `review_target` so no real LLM/env is needed.
- Deferred to phase 7: extending `find_proof` with graph queries (`dependency_walk`/`by_step`) and a `review` one-shot subcommand.
