# Design: Deep Distillation → Proof Graph (knowledge-asset substrate + structured review)

Status: **design approved (sections §1–§7), pending spec review**
Date: 2026-05-21
Author: jesson-hh (with Claude)

## 1. Goal & scope

Turn a **series of arXiv papers** into an accurate, queryable, plug-and-play **proof graph + grounded knowledge base**, then let a **review agent** decompose each proof and locate suspicious steps / logic gaps — without the agent skimming ("不囫囵吞枣") or hallucinating.

The larger ambition decomposes into four stacked layers:

| Layer | Capability |
|---|---|
| **L0 substrate** | grounded segment-by-segment distillation + context compression + carried memory |
| **L1 control** | stay-on-goal + controlled divergence, no fabrication |
| **L2 proof graph + review** | step-level dependency DAG, plug-and-play; review agent decomposes + checks vs KB |
| **L3 generation** | propose new ideas / extensions / find errors |

**This spec covers the chosen north star: the L0 substrate + the L2 data model & review.** L1 is realized only to the degree the reading pipeline enforces it (strict per-segment schema + a goal contract). **L3 (autonomous idea generation) is explicitly out of scope** — the graph is its future foundation.

### Aligned decisions

1. **North star**: knowledge-asset / proof-graph substrate (review & generation are applications on top).
2. **Verification semantics**: **informal + structured review** — locate suspicious steps / gaps with grounded reasons; **no mathematical guarantee**.
3. **Graph granularity**: **full step-level DAG** (every proof decomposed into assertion-level nodes).
4. **Input / run model**: **explicit paper list → batch deep-dig** (controllable, testable, lowest hallucination risk). Auto-search/expand is future.
5. **Implementation footprint**: **lean in-repo + LLMLingua** (build the rest as focused modules; borrow algorithms, not heavy frameworks).

### Non-goals (this spec)

- Formal/machine-checked proofs (Lean/Coq) — future option; LeanDojo is the entry point if ever pursued.
- Autonomous idea/extension generation (L3).
- Auto-discovery of papers (topic→search→expand) — reuses existing `research` later.
- Heavy external frameworks (GraphRAG / Letta / MemGPT).

## 2. Landscape (why build vs. reuse)

From a survey of four domains:

- **Memory / compression / graph** (mostly reusable tooling): Letta/MemGPT, LLMLingua, GraphRAG. We adopt **LLMLingua only** (prompt compression); the rest we build lean.
- **Proof graph (steps/lemmas DAG)** — *build-it-yourself*; blueprints: **Draft-Sketch-Prove** (informal proof → checkable sketch) and **DAG-Math** (chain-of-thought as DAG: nodes=assertions, edges=inferential deps). No mature math claim-graph library exists.
- **Review / soundness** — *open problem, no turnkey*: **MARG** (distribute paper across agents to beat context limits), **RefGrader / ARES** (sub-step soundness + error-propagation control).
- **Critical caveat**: *"Proof or Bluff?" (USAMO 2025)* — LLM judges are near-chance at proof soundness today. ⇒ the review agent **locates suspicious steps/gaps; it does not certify correctness**.

## 3. Architecture (§1) — fused, zero new CLI

The proof graph is **a deeper output of the distillation that already happens**, not a parallel product.

Current surface (the real one; `docs/ARCHITECTURE.md` is stale at v1.0):
- `paper-distiller-chat` → **AgentLoop** (conversational; LLM picks among **7 tools**) + one-shot subcommands (`distill`/`browse`/`ask`/`resume`/`research`).
- `paper-distiller-arxiv` → local mirror admin.
- Shared async-DAG agent framework (`agents/`); per-vault `proof_store` (SQLite+FTS5) already exists; `distill_by_id` already does `fetch → proof RAG → distill_article → proof_sidecar → ingest`.

Fusion plan:

| Capability | How it fuses | New surface |
|---|---|---|
| **Build graph** | deepen `distill_article`'s `proof_sidecar` from "theorems+techniques" to **step-level DAG (nodes+edges+source spans)**; extend `ingest_sidecar` to write the graph | **0** — `distill_by_id`/`ask`/`research`/`distill` build it automatically; the chosen "list → batch" path *is* `distill_by_id` upgraded |
| **Query graph** | extend the existing `find_proof` tool with graph query types (`dependency_walk`, `by_step`, …) | **0** (same tool, richer) |
| **Review** | the only genuinely new capability | **+1 tool** `review_proof` (7→8); optional `review` one-shot subcommand |
| **Internal plumbing** | new package `proofgraph/` (reader / extractor / linker / reviewer / compress) under existing agents | 0 entry points |

**Keeping the agent from getting lost**: the decision space grows by ~1 tool; the agent never chooses between "distill" and "build graph" — *distilling is building the graph*. The graph is the same vault's deeper artifact, queried by the same `find_proof`. Build depth is a knob — `PD_GRAPH_DEPTH=step|theorem` — so `research`'s large runs aren't forced into the most expensive extraction.

(`docs/ARCHITECTURE.md` will be refreshed as part of this work.)

## 4. Data model (§2) — the spine

Current `proofs/store.py` (SCHEMA_VERSION=1): `theorems` (theorem-level, `techniques_used` JSON + FTS5), `techniques`, `meta`. `ingest_sidecar` is paper-grained idempotent (delete+reinsert per paper).

**Grow it: keep `theorems` as-is (`find_proof` unchanged, zero regression), add three graph tables.** A theorem becomes one kind of node; steps hang under it.

```sql
-- Every assertion in the graph.
CREATE TABLE nodes (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_arxiv_id TEXT NOT NULL,
  paper_slug     TEXT,
  kind           TEXT NOT NULL,   -- theorem|lemma|definition|assumption|proof_step|claim|external_ref
  label          TEXT,            -- "Theorem 4.3" / "Step (a)" / "Def 2.1"
  text           TEXT NOT NULL,   -- normalized assertion
  source_quote   TEXT,            -- verbatim span from paper (written only after the grounding gate) ← anti-hallucination anchor
  loc            TEXT,            -- JSON {"sec":"3.2","char":4120}
  status         TEXT NOT NULL DEFAULT 'extracted', -- extracted|ok|suspicious|gap|unsupported|unstated|hypothesis
  confidence     REAL,            -- reviewer-assigned (null until reviewed)
  parent_id      INTEGER,         -- the theorem/lemma this step belongs to
  ord            INTEGER,         -- step order within parent
  created_at     TEXT NOT NULL
);
CREATE INDEX idx_nodes_paper  ON nodes(paper_arxiv_id);
CREATE INDEX idx_nodes_kind   ON nodes(kind);
CREATE INDEX idx_nodes_parent ON nodes(parent_id);

-- Typed dependency graph (the DAG). Direction: src --rel--> dst means "src depends on / uses dst".
CREATE TABLE edges (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  src_id        INTEGER NOT NULL,
  dst_id        INTEGER NOT NULL,
  rel           TEXT NOT NULL,    -- depends_on|uses_lemma|uses_def|uses_assumption|cites|same_as|specializes|contradicts
  justification TEXT,             -- short grounded "why"
  cross_paper   INTEGER NOT NULL DEFAULT 0,  -- 1 when src.paper != dst.paper ← the "continue/plug-in" substrate
  created_at    TEXT NOT NULL,
  UNIQUE(src_id, dst_id, rel)
);
CREATE INDEX idx_edges_src ON edges(src_id);
CREATE INDEX idx_edges_dst ON edges(dst_id);
CREATE INDEX idx_edges_rel ON edges(rel);

-- node ↔ technique (generalizes today's techniques_used JSON; techniques table reused)
CREATE TABLE node_techniques (
  node_id   INTEGER NOT NULL,
  technique TEXT NOT NULL,
  PRIMARY KEY (node_id, technique)
);

-- FTS5 over node text + verbatim quote → token-efficient targeted retrieval during review
CREATE VIRTUAL TABLE nodes_fts USING fts5(
  label, text, source_quote,
  content='nodes', content_rowid='id',
  tokenize='porter unicode61 remove_diacritics 2'
);
-- + AFTER INSERT / AFTER DELETE triggers mirroring the existing theorems_fts triggers
```

Field → requirement mapping:

| Requirement | Where |
|---|---|
| anti-hallucination / "every claim traceable" | `source_quote` + `loc` (written only after the grounding gate) |
| decompose into executable proof / blockable logic | `parent_id` + `ord` + `depends_on` edges = step-level DAG |
| review annotations | `status` + `confidence` + `contradicts` edges |
| continue / plug-and-play | `same_as` / `uses_lemma` cross-paper edges (`cross_paper=1`) |
| token-efficient precise retrieval | `nodes_fts` + `node_techniques`; review JOINs only the local neighborhood |
| logic gap | a `depends_on` whose dst can't be resolved → status `gap` |

**Migration & consistency**: `SCHEMA_VERSION 1→2`; back the DB up first; backfill existing `theorems` rows into `kind='theorem'` nodes (old vaults get a skeleton; steps added on next re-distill). On migration failure, keep the v1 DB and run in degraded "theorem-only" mode. `ingest_sidecar` stays paper-grained idempotent — the theorem layer and the graph layer are rewritten in the **same transaction per paper**, so they can't drift. Tradeoff: a theorem is stored both as a `theorems` row and a node (mild redundancy) in exchange for zero `find_proof` regression; can be collapsed later.

## 5. Reading / extraction pipeline (§3)

Replaces the single "whole paper → `distill_article`" call inside `_DistillOne`. **[det]** = deterministic code (LLM-free hard gate); **[LLM]** = model call.

```
full_text (pymupdf)
 0. Segment [det]  — split by headings/"Theorem"/"Proof."/"Definition"/paragraphs into segments[];
                     mark proof_blocks ("Proof."…□) → this is the coverage denominator
 ── per-segment loop (one segment at a time) ──
 1. Extract [LLM]  — window = segment text + compressed running memory;
                     fill a STRICT schema: nodes(kind/label/text/source_quote/referenced labels|techniques);
                     proof_block → proof_step nodes + intra-proof depends_on (Draft-Sketch-Prove / DAG-Math style)
 2. Grounding gate [det] — verify each source_quote actually occurs in the segment (normalized + fuzzy for OCR/math noise);
                     miss → one retry → still miss ⇒ drop / mark unsupported.  ★ fabricated nodes cannot enter
 3. Self-check + memory [LLM small] — "any node claiming beyond the text?" → mark suspicious;
                     update STRUCTURED running memory (notation / definitions / established results / open obligations)
 ── loop end ──
 4. Resolve edges [det + small LLM] — turn each step's references into edges:
                     resolves to an in-paper def/lemma/step → uses_def/uses_lemma/depends_on
                     resolves to an external citation        → cites → external_ref node
                     resolves to nothing                     → dangling ⇒ status=gap   ★ logic-gap detection
 5. Write [det]    — paper-grained idempotent write of nodes + node_techniques + intra-paper edges (+ compat write to theorems);
                     emit COVERAGE REPORT: segments total/processed, proof_blocks, nodes by kind,
                     gate rejections, gaps, open-obligations list
```

- **LLMLingua** compresses only (a) the carried running memory and (b) injected prior-theorem context. **Never compresses the segment being read** — that stays verbatim for the gate.
- **Depth knob** `PD_GRAPH_DEPTH=step|theorem`. `step` (batch default) decomposes proof_blocks; `theorem` stops at theorem/lemma level (≈ today + spans) for cheap large runs.
- **One read, two artifacts**: the human-readable 12-section markdown article is composed *from the verified nodes + segments* (not a second read of the paper) — so the prose article is grounded too, and tokens aren't doubled.
- **Path adherence (L1)**: each segment only has the LLM fill fields *about that segment*; an opening lightweight pass identifies the paper's goal + target theorems as a contract injected into each segment. Divergence (searching other papers) is **not** in the reading loop — it is §6's explicit linking step, each import carrying its own source.
- **Honest caveat**: quality rests on Step 0 segmentation; arXiv PDFs are noisy and proof_block detection won't always be clean. Mitigations: fuzzy gate matching; fall back to coarser segments; the coverage report surfaces anomalies (e.g., "only 2 segments found"). No silent skipping.

**Three defenses vs. three failure modes**: fabrication → grounding gate [det]; misinterpretation → self-check [LLM]; skim/skip → coverage report [det]. **Abstain over fabricate**: a step whose justification isn't in the text is marked `gap`/`unstated`, never invented.

## 6. Cross-paper linking (§5)

Runs after the batch is ingested (and incrementally when a paper is added). Creates only **cross-paper edges** (`cross_paper=1`). Reuses the existing proof-RAG philosophy (candidate generation is already built for techniques): **[det] cheap candidates → [LLM] targeted relation classification** (avoids O(N²) all-pairs LLM).

```
A. Candidates [det, cheap] — for each node, find matches in OTHER papers via:
     ① technique overlap (node_techniques intersection)
     ② nodes_fts / BM25 text similarity of statement + source_quote
     ③ explicit citation (a cites edge resolving to a batch paper) → prioritize those node pairs
   keep top-K candidates per node (not all-pairs) ← "token-efficient precise hits"
B. Classify [LLM, small context] — per candidate pair, feed only the two nodes' text + source_quote:
     → same_as | specializes/generalizes | uses_lemma | contradicts | none
     must give justification (which spans justify the link); uncertain ⇒ none (abstain, don't over-link)
C. Write [det] — insert cross_paper=1 edges + justification (idempotent: re-run recomputes batch cross-edges)
```

- **Plug-and-play**: a newly distilled paper is incrementally linked into the existing graph — no rebuild.
- **Continue ("续上")**: traverse `same_as` to see a lemma recur across papers; traverse `uses_lemma`/`depends_on` cross-paper to see how A's result is used/extended in B. This is the substrate L3 will stand on.
- **Scope knob**: link within-batch only vs. against the whole vault graph (the latter accretes a growing knowledge asset).
- **Cost**: candidates are ~free [det]; only top-K pairs get a small [LLM] call → ~N×K small calls, not N².
- **Honest caveat**: cross-paper `same_as` is a soft semantic judgment, made **auditable** by `justification` + both `source_quote`s (you or the reviewer can reject a bad link). Prefer missing a link to inventing one.

## 7. Review agent (§6)

New `review_proof` tool (+ optional `review` subcommand), running over the built graph. Verification semantics: **locate suspicious steps / gaps, no verdict of correctness.**

```
review_proof(target = a theorem / paper / subtree)
  load the target's proof DAG (nodes+edges)
  walk in topological order, per node:
    1. Local context [det, token-efficient] = this node (text+source_quote)
         + its depends_on parents
         + KB hits for its technique (node_techniques index — only the relevant ones, not the whole KB)
         + same_as/uses_lemma cross-paper neighbors (already-ok ones = positive evidence)
    2. Judge [LLM, small] → structured label: ok | suspicious | gap | unsupported | unstated
         + grounded reason (cites this node's source_quote + which KB evidence)
         + confidence (LLM self-confidence is DOWN-WEIGHTED per "Proof or Bluff")
    3. Error propagation [det] — effective trust = own label ∧ min(parent trust);
         a suspicious/gap parent taints its descendants ("downstream of flagged")
  persist nodes.status + confidence; emit a review report
```

- **KB grounding, token-efficient**: a step using `Bernstein` pulls only "prior nodes using Bernstein that are already ok" via the `node_techniques` index — never the whole store; `same_as` neighbors already-ok = positive evidence, `contradicts` = immediate flag. Retrieved exemplars are LLMLingua-compressed. Each node sees a small neighborhood ⇒ cost ≈ #nodes × small-context; review can be scoped to a single subtree.
- **Problem decomposition**: the DAG *is* the decomposition (done at §5 extraction). Additionally, for a `gap` node the reviewer may run a fill sub-routine — propose the missing justification, mark it `status=hypothesis` (**unverified, never silently inserted as fact**), surface it. This is the honest bridge toward L3.
- **Output**: per paper/theorem review report — annotated DAG + a list ranked by "most suspicious / biggest gap", each with `source_quote` (jump-to source) + reason + KB evidence. Written to `surveys/review-<slug>.md`; statuses persisted on nodes. This realizes "review others' papers / find errors": feed a paper, get a grounded, prioritized list of suspicious steps.
- **Scaling (MARG, optional)**: for proofs exceeding context, distribute subtrees across parallel reviewer sub-agents (reuse existing fanout/orchestrator), then a synthesizer merges + runs the propagation pass. Not required for v1.
- **Honest caveat**: the reviewer is "a careful TA's margin notes + priorities", not a judge. Expect false positives/negatives; down-weight confidence. Value = grounded localization + traceability.

## 8. Error handling & testing

### Error handling — degrade gracefully, never crash the batch, surface every degradation

| Failure | Handling |
|---|---|
| Segmentation failure (garbled PDF) | fall back to coarse paragraphs; else mark paper `extraction_failed`, skip graph but still produce the basic article; never abort the batch; report it |
| Grounding-gate rejections | normal (the mechanism working); but an abnormally high per-segment rejection rate (>50%) flags the segment "low-fidelity" rather than silently dropping all |
| LLM failure (extract/self-check/review) | retry once per segment/node → then skip that node / mark segment partial; one paper's failure doesn't kill the batch (reuse PaperProcessor isolation) |
| Unresolved reference / bad cross link | unresolved → `gap` (not an error); linker uncertain → `none`; linker LLM failure → skip the candidate pair, log |
| Store/DB error | never abort distillation (existing pattern); migration 1→2 backs up first, falls back to v1 "theorem-only" mode on failure |
| Budget breach (batch) | check between papers (like research_runner); stop intake, finish in-flight, emit partial graph + report; write resumable progress state |

Idempotency: paper-grained writes (delete+reinsert this paper's nodes/edges) → clean re-distill; re-running the linker recomputes batch cross-edges. All degradations appear in the coverage/review reports — no silent failure.

### Testing — reuse existing patterns (LLM mocked, autouse isolation fixtures, no real API in CI)

- **Deterministic gates get hard unit tests (highest value):** grounding gate (fabricated quote rejected; verbatim/whitespace/OCR-noisy quote accepted; fuzzy threshold boundaries); segmentation (boundaries, proof_block detection, coverage denominator); edge resolution (refs→edges; dangling→gap); error propagation (synthetic DAG with a suspicious parent → descendants tainted); migration 1→2 (v1 db → migrate → nodes backfilled + `find_proof` still works).
- **Graph store CRUD**: node/edge insert+query, idempotent re-ingest (no dupes), `nodes_fts` search, `node_techniques` JOIN.
- **LLM stages with mock LLM**: canned schema JSON → assert correct nodes/edges written, bad-quote retry fires, LLM-failure degradation; reviewer with canned labels → assert topo walk + propagation + report + status persistence.
- **Integration**: a tiny fake-paper fixture → full pipeline (mocked LLM) → assert graph built + report shapes.
- **Isolation**: add a `_isolate_proof_store` autouse fixture (mirrors `_isolate_arxiv_local`) so tests don't leak real DBs.
- **Quality eval (manual, outside CI)**: a small hand-labeled set (papers with known errors/gaps) to track the reviewer's localization precision/recall over time — the only real measure of usefulness given the "Proof or Bluff" caveat.

## 9. Consolidated honest caveats

- **"No hallucination" → enforceable form**: every stored node carries a *verified verbatim source span*; fabricated content cannot enter the graph unflagged; the system abstains/flags rather than invents. True zero-hallucination is not achievable; this is the strongest engineering floor.
- The grounding gate catches **fabrication** (quote absent); **misinterpretation** (quote present, paraphrase distorts) is softer — caught by the self-check pass + reviewer.
- **Correctness is not certified.** The reviewer localizes suspicion/gaps. LLM judges are unreliable on soundness (Proof or Bluff).
- Cross-paper `same_as` is soft but auditable.
- Segmentation quality is the foundation and is best-effort on noisy PDFs.

## 10. Implementation phasing (for the plan)

Each phase independently testable; suggested order:

1. **Graph store + migration 1→2** (`proofs/store.py`): nodes/edges/node_techniques/nodes_fts, CRUD, idempotent ingest, backfill, `find_proof` regression tests.
2. **Segmentation + grounding gate** (`proofgraph/reader.py` stages 0 & 2): pure-function, heavily unit-tested.
3. **Extraction pipeline** (`proofgraph/reader.py` + `extractor.py` stages 1,3,4,5): per-segment loop, structured memory, edge resolution; mock-LLM pipeline tests; `compress.py` (LLMLingua) wrapper.
4. **Wire into distillation**: upgrade `proof_sidecar` schema + `_DistillOne` to call the pipeline; `PD_GRAPH_DEPTH`; one-read-two-artifacts; `distill_by_id` now builds the graph.
5. **Cross-paper linker** (`proofgraph/linker.py`): candidate→classify→write; scope knob.
6. **Review agent** (`proofgraph/reviewer.py`) + `review_proof` tool + `review` subcommand + extend `find_proof` graph queries.
7. **Refresh `docs/ARCHITECTURE.md`** + quality eval harness.

## 11. Success criteria

- Given an explicit list of arXiv IDs, the batch produces a step-level proof graph where **every node has a verified source span**, with a coverage report and no silent skips.
- `find_proof` keeps working unchanged; new graph queries (dependency walk, by-step) work; cross-paper `same_as`/`uses_lemma` edges connect related results.
- `review_proof` returns, for a target, a grounded, priority-ranked list of suspicious steps / gaps with jump-to-source quotes, with early-error taint propagated.
- Full suite green (deterministic gates + mocked-LLM pipeline + integration), no real API in CI; lean footprint (one new dependency: LLMLingua).
