# Architecture

How paper-distiller is laid out and how data flows through it. Read this if you want to extend it, add a new agent, customize the prompts, or understand a failure.

## Two console scripts

paper-distiller ships two console scripts:

| Script | Purpose |
|---|---|
| `paper-distiller-chat` | **Conversational agent** — the main interface. No subcommand = AgentLoop. One-shot subcommands for scripts/CI. |
| `paper-distiller-arxiv` | **Local arXiv mirror admin** — bootstrap, sync, search, stats, doctor. |

### `paper-distiller-chat` — modes

| Mode | Invocation | Description |
|---|---|---|
| **AgentLoop** (default) | `paper-distiller-chat --vault X` | Conversational: LLM picks among 8 tools per user input |
| **distill** | `paper-distiller-chat distill ...` | Single-pass batch: search topic → distill N papers |
| **browse** | `paper-distiller-chat browse ...` | Search + show abstracts, pick which to distill |
| **ask** | `paper-distiller-chat ask ...` | Multi-round QA loop |
| **resume** | `paper-distiller-chat resume ...` | Continue a paused/errored QA session |
| **research** | `paper-distiller-chat research ...` | Long-running autonomous deep-research loop |
| **legacy-repl** | `paper-distiller-chat legacy-repl ...` | Pre-v1.4 slash-command REPL |

### The 8 LLM-callable tools (AgentLoop)

The AgentLoop exposes these tools to the LLM via function-calling (`agent_tools.py`):

| Tool | Description |
|---|---|
| `search` | Search papers — defaults to local arXiv mirror |
| `distill_by_id` | Download PDFs + deep distill + proof sidecar + vault save |
| `show` | Read a vault entry back by slug |
| `ask` | Multi-round QA loop: search → distill → reflect |
| `research` | Long-running 5-phase deep research (default 6h, 40 papers) |
| `ask_user` | Pause and let the user pick between 2–4 options |
| `find_proof` | Query the vault's theorem / proof-graph knowledge base |
| `review_proof` | Structured review of a paper's proof DAG (flags suspicious steps) |

Tool wrappers live in `chat/agent_tools.py`. Each wrapper is synchronous and calls `asyncio.run()` internally; they must be invoked from a synchronous caller. Schemas live in `_*_SCHEMA` constants and are aggregated in `TOOL_SCHEMAS`.

## Package layout

```
src/paper_distiller/
├── __init__.py                __version__ = "1.12.0"
├── config.py                  Config dataclass + load_config / load_config_qa / load_config_research
├── pipeline.py                fetch_with_fallback (PDF + mirror fallback chain)
├── agents/                    Async-DAG agent framework + concrete agents
│   ├── base.py                Agent Protocol, Context, Status
│   ├── dag.py                 DAG — topology validation + topo_levels
│   ├── orchestrator.py        asyncio executor (topological scheduling)
│   ├── fanout.py              FanoutAgent protocol
│   ├── renderer.py            ConsoleRenderer (rich live status table)
│   ├── searchers.py           ArxivSearcher + SemanticScholarSearcher
│   ├── opencli_openalex.py    OpenAlex searcher via CLI
│   ├── curation.py            CandidateMerger + CandidateRanker
│   ├── dedup.py               CandidateDedup (QA-only in-session dedup)
│   ├── processor.py           PaperProcessor (fanout) + _DistillOne
│   ├── writer.py              VaultWriter + SurveyComposer
│   ├── reflector.py           ProgressReflector (QA round-start LLM call)
│   ├── synthesizer.py         AnswerSynthesizer (QA final-answer LLM call)
│   ├── router.py              IntentRouter (legacy-REPL NL → command JSON)
│   ├── theme_clusterer.py     ThemeClusterer (research loop)
│   ├── gap_detector.py        GapDetector (research loop)
│   ├── theorem_extractor.py   TheoremExtractor (research loop)
│   ├── citation_explorer.py   CitationExplorer (research loop)
│   └── prompts/               LLM prompt files for agents
├── chat/                      User-facing CLI + AgentLoop
│   ├── cli.py                 argparse entry + subcommand dispatch
│   ├── agent_loop.py          AgentLoop — conversational brain (8 tools)
│   ├── agent_tools.py         Tool schemas + Python wrappers
│   ├── qa_runner.py           QA-loop driver (multi-round)
│   ├── research_runner.py     Research-loop driver (5-phase)
│   ├── slash_commands.py      Slash-command dispatcher
│   ├── plan_mode.py           Plan-mode preview + confirmation gate
│   ├── permissions.py         Permission mode logic
│   ├── cost_estimator.py      Per-tool cost estimate
│   ├── history.py             Input history (prompt_toolkit)
│   ├── ui.py                  Rich console helpers
│   └── repl/                  Legacy slash-command REPL
│       ├── loop.py            REPL class + input loop
│       ├── commands.py        Slash-command parser + KNOWN_COMMANDS
│       └── helpers.py         Read-only commands (vault/sessions/show/help)
├── llm/openai_compatible.py   LLMClient (OpenAI-compatible HTTP, JSON mode, token accounting)
├── sources/
│   ├── arxiv.py               arxiv API search + PDF download
│   └── semantic_scholar.py    SS API search
├── arxiv_local/               Local arXiv mirror (SQLite + FTS5 + BM25 + OAI-PMH sync)
├── extract/
│   └── pymupdf_extractor.py   PDF → plain text via PyMuPDF
├── distill/
│   ├── article.py             distill_article — LLM 12-section distillation
│   ├── filter.py              rank — LLM candidate ranker
│   └── survey.py              compose — LLM multi-article survey
├── prompts/                   Core LLM prompt .md files (filter / article / survey)
├── vault/
│   ├── schema.py              Category list + Paper / ArticleResult dataclasses
│   ├── store.py               VaultStore — save_entry + find_by_arxiv_id/doi dedup
│   └── crosslink.py           WikiIndex — existing slugs for distill cross-link prompt
├── proofgraph/                Proof-graph subsystem (see section below)
│   ├── reader.py              segment() + verify_quote() grounding gate
│   ├── extractor.py           extract_segment() + self_check()
│   ├── pipeline.py            build_graph_for_paper() + maybe_build_graph() + CoverageReport
│   ├── linker.py              find_candidates() / classify_pair() / link_paper()
│   ├── reviewer.py            review_node() / compute_taint() / review_target()
│   ├── compress.py            Optional LLMLingua wrapper (transparent fallback)
│   ├── extraction_schema.py   JSON schema / dataclasses for extracted nodes
│   ├── memory.py              RunningMemory — cross-segment state during extraction
│   └── prompts/               LLM prompt files for extraction + review
├── proofs/
│   └── store.py               ProofStore — SQLite + FTS5, SCHEMA_VERSION=2
└── qa/
    ├── state.py               SessionState + RoundRecord + disk persistence
    ├── reflection.py          One LLM call: round-start reflection
    ├── answer.py              One LLM call: final answer synthesis
    └── prompts/
        ├── reflect.md
        └── answer.md
```

## The agent framework

### Agent contract

```python
class Agent(Protocol):
    name: str
    deps: list[str]                  # other agent names this depends on

    async def run(self, ctx: Context) -> dict: ...
    # Returns a dict merged into ctx.shared for downstream agents.
```

### Context

```python
@dataclass
class Context:
    cfg: Config
    llm: LLMClient
    vault: VaultStore
    shared: dict                     # mutable inter-agent state
    on_status: Callable              # callback for status events
```

`shared` accumulates as agents run:
- after `arxiv-searcher`: `shared["candidates_arxiv"] = [Paper, ...]`
- after `candidate-merger`: `shared["candidates"] = [Paper, ...]`
- after `candidate-dedup`: `shared["candidates"]` filtered to non-seen papers
- after `candidate-ranker`: `shared["ranked"] = [Paper, ...][:N]`
- after `paper-processor` fanout: `shared["articles"] = [ArticleResult, ...]`
- in QA mode: `shared["qa_state"] = SessionState`
- in QA mode: `shared["reflection"]: dict` (after `progress-reflector`)

### DAG + Orchestrator

```python
class DAG:
    def __init__(self, agents: list[Agent]):
        # validates: no duplicate names, no missing deps, no cycles
    def topo_levels(self) -> list[list[str]]:
        # returns groups: [[level0], [level1], ...] for concurrent scheduling

class Orchestrator:
    async def run(self) -> dict:
        for level in self.dag.topo_levels():
            await asyncio.gather(*(self._run_one(name) for name in level))
        return self.ctx.shared
```

`_run_one` handles regular Agents AND `FanoutAgent`s (agents with `expand()` that produce N sub-agents at runtime). The orchestrator awaits each topological level fully before starting the next.

## `_DistillOne` and the proof graph

`_DistillOne` (in `agents/processor.py`) is the per-paper fanout sub-agent. After distillation it calls:

```python
await asyncio.to_thread(
    maybe_build_graph,
    proof_store, paper_arxiv_id, full_text,
    paper_slug=..., llm=llm,
)
```

`maybe_build_graph` (in `proofgraph/pipeline.py`) checks the `PD_GRAPH_DEPTH` env var:

| Value | Behaviour |
|---|---|
| unset / `off` | Graph not built (default) |
| `theorem` | Extract theorem-level nodes only |
| `step` | Full step-level DAG (every proof decomposed into assertion nodes) |

When active, `build_graph_for_paper` segments the text (`reader.segment`), runs `extract_segment` + `self_check` per segment (with a grounding gate — fabricated nodes are rejected), writes nodes to `ProofStore`, resolves label references into edges, and returns a `CoverageReport`. The operation is idempotent (deletes prior graph data for the paper first).

## `proofgraph/` subsystem

| Module | Role |
|---|---|
| `reader.py` | `segment(text)` — LLM-free splitting into `Segment` objects. `verify_quote(quote, seg)` — grounding gate (SequenceMatcher similarity check) |
| `extractor.py` | `extract_segment(seg, memory, llm, depth)` — LLM call, returns accepted `ExtractedNode`s + rejected count. `self_check(seg, nodes, llm)` — second-pass hallucination filter |
| `pipeline.py` | `build_graph_for_paper` — full pipeline (delete → segment → extract loop → edge resolution → CoverageReport). `maybe_build_graph` — gated entry point for `_DistillOne` |
| `linker.py` | `find_candidates` / `classify_pair` / `link_paper` — cross-paper edge detection (finds `same_as` / `specializes` / `generalizes` / `uses_lemma` / `contradicts` links between nodes in different papers) |
| `reviewer.py` | `review_node(store, node, llm)` — LLM judgment (ok/suspicious/gap/unsupported/unstated) with confidence capped at 0.7. `compute_taint(store, ids, label_by_id)` — propagates problem labels down the `depends_on` DAG. `review_target(store, *, paper_arxiv_id, node_id, llm)` — orchestrates a full paper or subtree review + returns `ReviewReport` |
| `compress.py` | `compress(text, target_ratio)` — optional LLMLingua wrapper; identity passthrough if `llmlingua` is not installed |
| `memory.py` | `RunningMemory` — rolling context of recent nodes, passed between segments so the extractor can reference already-extracted labels |

## ProofStore schema (SCHEMA_VERSION 2)

`proofs/store.py` maintains a per-vault SQLite + FTS5 database at `<vault>/.proof_store/proofs.db`. The schema has two logical layers:

**Theorems layer** (v1, unchanged):

- `theorems` — theorem-level rows (name, statement, proof_sketch, techniques_used JSON)
- `theorems_fts` — FTS5 virtual table (porter tokenizer) over name + statement + proof_sketch
- `techniques` — canonical technique names with first-seen paper
- `meta` — key/value store (holds `schema_version`)

**Graph layer** (v2, added by proof-graph phases):

- `nodes` — every assertion in the graph (kind, label, text, source_quote, status, parent_id, ord, confidence)
- `nodes_fts` — FTS5 over label + text + source_quote
- `edges` — typed dependency edges (src_id, dst_id, rel: `depends_on | uses_lemma | uses_def | uses_assumption | cites | same_as | specializes | contradicts`)
- `node_techniques` — node ↔ technique many-to-many join

Migration is idempotent: opening a v1 database auto-backfills existing `theorems` into `nodes` as `kind='theorem'`.

## Vault format

paper-distiller writes pure Obsidian-flavored markdown:

- YAML frontmatter (`title`, `tags`, `slug`, `arxiv_id`, `doi`, `published`, `depth`)
- Body in markdown with `[[wikilinks]]` for cross-references
- Categories are subdirectory names under `<vault>/`
- Proof store lives at `<vault>/.proof_store/proofs.db` (SQLite, git-ignored by default)
- QA/research sessions at `<vault>/.paper_distiller/qa-sessions/<sid>/state.json`

## LLM client contract

`llm/openai_compatible.py::LLMClient`:

- `complete(messages, temperature, response_format=None)` — one method
- `response_format="json"` enables provider-side JSON mode
- Accumulates `total_tokens_in/out` across calls
- Raises `LLMError` on HTTP/timeout/auth failures

Any OpenAI-compatible endpoint works: Aliyun Bailian (recommended), DeepSeek, OpenRouter, local Ollama.

## Testing

568 tests across `tests/` (run `pytest -q`):

- `tests/agents/` — per-agent unit tests + framework primitives
- `tests/chat/` — AgentLoop, REPL, CLI dispatch, qa_runner, research_runner
- `tests/integration/` — end-to-end distill + ask flows with all subsystems mocked
- `tests/proofgraph/` — reader, extractor, pipeline, linker, reviewer unit tests
- `tests/test_*.py` (root) — primitives: arxiv, SS, distill, config, vault, LLM, proof store, smoke

All LLM calls are mocked; no network calls in CI.
