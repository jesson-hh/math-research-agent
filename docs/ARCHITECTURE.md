# Architecture (v1.0)

How paper-distiller is laid out and how data flows through it. Read this if you want to extend it, add a new agent, customize the prompts, or understand a failure.

## One CLI, three modes + REPL

paper-distiller v1.0 ships a single console script: `paper-distiller-chat`. It has three one-shot subcommands plus an interactive REPL when invoked without a subcommand:

| Mode | Invocation | Use case |
|---|---|---|
| **REPL** | `paper-distiller-chat --vault X` | Interactive — slash commands or natural language |
| **distill** | `paper-distiller-chat distill --topic X --n N` | Single-pass batch: distill N papers on a topic |
| **ask** | `paper-distiller-chat ask --question Y` | Multi-round QA loop: agent plans search across rounds and writes a cited answer |
| **resume** | `paper-distiller-chat resume --session-id <sid>` | Continue a paused/errored QA session |

All four share the same underlying machinery: an async DAG of sub-agents executed by a single orchestrator with a live status table.

## Package layout

```
src/paper_distiller/
├── __init__.py                __version__ = "1.0.0"
├── config.py                  Config dataclass + load_config / load_config_qa
├── agents/                    The v1.0 sub-agent framework + concrete agents
│   ├── base.py                Agent Protocol, Context, Status
│   ├── dag.py                 DAG class — topology validation + topo_levels
│   ├── orchestrator.py        asyncio executor + AgentFailed
│   ├── fanout.py              FanoutAgent protocol
│   ├── renderer.py            ConsoleRenderer (rich live table)
│   ├── searchers.py           ArxivSearcher + SemanticScholarSearcher
│   ├── curation.py            CandidateMerger + CandidateRanker
│   ├── dedup.py               CandidateDedup (QA-only in-session dedup)
│   ├── processor.py           PaperProcessor (fanout) + _DistillOne
│   ├── writer.py              VaultWriter + SurveyComposer
│   ├── reflector.py           ProgressReflector (QA round-start LLM call)
│   ├── synthesizer.py         AnswerSynthesizer (QA final-answer LLM call)
│   ├── router.py              IntentRouter (REPL NL → command JSON)
│   └── prompts/
│       └── route.md           Intent-routing prompt
├── chat/                      The user-facing CLI + REPL
│   ├── cli.py                 argparse entry, subcommand dispatch, REPL launch
│   ├── qa_runner.py           QA-loop driver — orchestrates rounds + state persistence
│   └── repl/
│       ├── commands.py        Slash-command parser + KNOWN_COMMANDS set
│       ├── helpers.py         Read-only commands (vault/sessions/provider/agents/show/help)
│       └── loop.py            REPL class — input loop, dispatch, NL routing
├── llm/openai_compatible.py   LLMClient (OpenAI-compatible HTTP, JSON mode, token accounting)
├── sources/
│   ├── arxiv.py               arxiv API search + PDF download
│   └── semantic_scholar.py    SS API search + openAccessPdf lookup
├── extract/
│   └── pymupdf_extractor.py   PDF → plain text via PyMuPDF
├── distill/
│   ├── article.py             distill_article — LLM paper distillation
│   ├── filter.py              rank — LLM candidate ranker
│   └── survey.py              compose — LLM cluster-survey composition
├── prompts/                   v0.3-era distill prompts (filter / article / survey)
├── vault/
│   ├── schema.py              Category list + Paper / ArticleResult dataclasses
│   ├── store.py               VaultStore — save_entry + find_by_arxiv_id/doi dedup
│   └── crosslink.py           WikiIndex — feeds existing slugs to the distill prompt
└── qa/
    ├── state.py               SessionState + RoundRecord + disk persistence
    ├── reflection.py          One LLM call: round-start reflection
    ├── answer.py              One LLM call: final answer synthesis
    └── prompts/
        ├── reflect.md         Reflection prompt
        └── answer.md          Answer-synthesis prompt
```

The `Paper` dataclass in `vault/schema.py` is the cross-source unification point — every source returns the same shape (`arxiv_id`, `doi`, `ss_paper_id`, etc.), so downstream code never has to special-case the source.

## The agent framework

### Agent contract

```python
class Agent(Protocol):
    name: str
    deps: list[str]                  # other agent names this depends on

    async def run(self, ctx: Context) -> dict: ...
    # Returns a dict that gets merged into ctx.shared for downstream agents.
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
- in QA mode: `shared["qa_state"] = SessionState` (set by qa_runner before each Orchestrator.run)
- in QA mode: `shared["reflection"]: dict` (after `progress-reflector`)
- in QA mode: `shared["answer_survey_slug"]` (after `answer-synthesizer`)

### DAG + Orchestrator

```python
class DAG:
    def __init__(self, agents: list[Agent]):
        # validates: no duplicate names, no missing deps, no cycles
    def topo_levels(self) -> list[list[str]]:
        # returns groups: [[level0], [level1], ...] where each group runs concurrently

class Orchestrator:
    async def run(self) -> dict:
        for name in self.dag.agents:
            self.ctx.on_status(name, Status.QUEUED)
        for level in self.dag.topo_levels():
            await asyncio.gather(*(self._run_one(name) for name in level))
        return self.ctx.shared
```

`_run_one` handles regular agents AND `FanoutAgent`s (agents with `expand()` instead of `run()` that produce N sub-agents at runtime). The orchestrator awaits each topological level fully before starting the next.

### ConsoleRenderer

`ConsoleRenderer.on_status` is the `Context.on_status` callback. It accumulates row state (queued / running / done / failed with elapsed times). `ConsoleRenderer.build_table()` returns a fresh `rich.table.Table` that the chat CLI wraps with `rich.live.Live` for ~10 Hz auto-refresh during execution.

## Three DAG shapes

paper-distiller composes the 11 agents into three DAG shapes, one per mode.

### Single-pass DAG (`distill`)

```
arxiv-searcher  ss-searcher      ← Phase 1 (parallel, deps=[])
       └────┬────┘
       candidate-merger
              │
       candidate-dedup            ← QA-only filter; passthrough in single-pass
              │
       candidate-ranker (LLM)
              │
       paper-processor (fanout)   ← Phase 3 (×N parallel sub-agents)
              │
       vault-writer
              │
       survey-composer (LLM)      ← optional, only if N >= min_papers_for_survey
```

Each `paper-processor` fan-out instance does `fetch_with_fallback` + PyMuPDF extract + `distill_article` LLM call. Per-paper LLMError is swallowed (the failed paper is dropped; other papers continue).

### Reflection DAG (`ask`, per round)

```
progress-reflector (LLM)
```

A single-node DAG. Reads `qa_state.articles_distilled` + `qa_state.history` (prior queries) and asks the LLM:
- is the answer complete enough to stop? (`is_done`, `confidence`)
- if not, what should we search next? (`next_query`, `next_query_rationale`)
- diminishing returns flag? (`suggest_stop`)

Returns this JSON in `ctx.shared["reflection"]`. The `qa_runner.py` driver inspects it and decides whether to continue.

### Distillation DAG (`ask`, per round, if continuing)

Same shape as the single-pass DAG, with `next_query` set in `ctx.shared` and `qa_state` available for `candidate-dedup` to filter out already-seen papers. No `survey-composer` (the final synthesis is `answer-synthesizer` after the loop terminates).

### Synthesis DAG (`ask`, once after loop ends)

```
answer-synthesizer (LLM)
```

Wraps `qa.answer.synthesize` — composes the final cited answer + appends an audit trail markdown table summarizing per-round queries, confidence, cost. Writes the result to `<vault>/surveys/qa-<slug>-<date>.md`.

## QA loop driver (`chat/qa_runner.py`)

The Orchestrator runs one DAG once — it has no native loop concept. So the multi-round QA flow is driven by `chat/qa_runner.py::run_qa_loop`, which calls `Orchestrator.run()` once per phase:

```python
async def _arun_qa_loop(cfg):
    state = SessionState(...) or read_state(...)
    with Live(renderer.build_table(), ...):
        while True:
            # 1. Reflection DAG
            ctx = Context(cfg, llm, vault, {"qa_state": state}, renderer.on_status)
            await Orchestrator(reflection_dag, ctx).run()
            reflection = ctx.shared["reflection"]
            state.last_reflection = reflection

            # 2. Check stop conditions
            if state.rounds_completed >= cfg.qa_max_rounds: break  # max_rounds
            if reflection["is_done"] and confidence >= threshold: break  # llm_done
            if reflection["suggest_stop"]: break  # llm_brake
            if not reflection["next_query"]: break  # no_candidates

            # 3. Distillation DAG
            ctx = fresh_ctx_with_shared({"qa_state": state, "next_query": ...})
            await Orchestrator(distillation_dag, ctx).run()

            # 4. Process round's results, update state, persist
            new_articles = [a for a in ctx.shared["articles"] if a.slug not in seen]
            if not new_articles and not ctx.shared["candidates"]: break  # no_candidates
            state.articles_distilled.extend(new_articles)
            # populate seen_ids from successful articles' refs
            state.history.append(RoundRecord(...))
            state.rounds_completed += 1
            write_state(cfg.vault_path, state)

            # 5. Budget caps
            if len(state.articles_distilled) >= cfg.qa_max_articles: break  # max_articles
            if state.cost_cny >= cfg.qa_max_cost_cny: break  # max_cost

    # 6. Synthesis (if any articles)
    if state.articles_distilled:
        await Orchestrator(synthesis_dag, fresh_ctx).run()

    # 7. is_done semantics
    state.is_done = state.stop_reason not in ("user_quit",) and not state.stop_reason.startswith("error:")
    write_state(cfg.vault_path, state)
    return state
```

### Seven stop reasons

| Reason | Trigger | `is_done`? |
|---|---|---|
| `max_rounds` | `state.rounds_completed >= cfg.qa_max_rounds` | yes |
| `llm_done` | `reflection.is_done == True` AND `confidence >= threshold` | yes |
| `llm_brake` | `reflection.suggest_stop == True` (diminishing-returns judgement) | yes |
| `no_candidates` | Search + dedup yielded nothing new, OR `next_query` was empty | yes |
| `max_articles` | Total articles distilled hit `cfg.qa_max_articles` | yes |
| `max_cost` | Accumulated `state.cost_cny >= cfg.qa_max_cost_cny` | yes |
| `user_quit` | Ctrl+C OR REPL interactive `n` | **no** (resumable) |
| `error: <details>` | Uncaught exception in reflection or distillation DAG | **no** (resumable) |

`is_done = False` makes the session resumable via `paper-distiller-chat resume <sid>`. Done sessions cannot be resumed (raises ValueError).

## REPL (`chat/repl/loop.py`)

When the user types `paper-distiller-chat --vault X` (no subcommand), the chat CLI launches the REPL.

```python
class REPL:
    def dispatch_one(self, line) -> str | None:
        if line.startswith("/"):
            return self._dispatch_slash(line)
        return self._dispatch_natural_language(line)
```

**Slash commands** dispatch directly:
- Read-only commands (`/vault`, `/sessions`, `/provider`, `/agents`, `/show`, `/help`) call into `chat/repl/helpers.py` — pure utility, no LLM.
- Action commands (`/distill`, `/ask`, `/resume`) build synthetic argv and call `chat/cli.py::main` — reuses the one-shot subcommand handlers.
- `/quit` returns a "QUIT" sentinel that the input loop respects.

**Natural-language input** goes through `IntentRouter`:
1. One LLM call (`agents/prompts/route.md`) classifies into one of 4 commands (`distill` / `ask` / `resume` / `show`) with extracted params + missing-params list + confidence.
2. REPL prints the proposal:
   ```
   [intent-router] Intent: ask  | confidence 9
     question: 扩散在金融时序的最新进展
   Missing: max_rounds, per_round, max_cost_cny
   Apply defaults (max_rounds=3, per_round=2, max_cost_cny=5.0) and run? [Y/n]
   ```
3. On `Y`, REPL applies `_AGENT_DEFAULTS` for missing params, builds argv, dispatches.
4. `show` is special-cased — routes directly to `handle_show(vault_path, slug)` because `cli.main` has no `show` subcommand.

The input loop uses `prompt_toolkit.PromptSession` with `WordCompleter` for tab-completion of slash commands and arrow-key history.

## Cost accounting

`LLMClient` maintains `total_tokens_in` and `total_tokens_out` accumulators. After every round, `qa_runner._update_cost(state, llm)` rolls them into `state.cost_cny` using qwen-plus pricing:

```python
_PRICE_IN_CNY_PER_M = 2.1
_PRICE_OUT_CNY_PER_M = 12.7
state.cost_cny = (tokens_in * 2.1 + tokens_out * 12.7) / 1_000_000
```

This is for the cost circuit breaker (`--max-cost-cny`); it is **not** billing-accurate and does not account for provider-specific overhead.

## State persistence

QA sessions persist their `SessionState` after every round:

```
<vault>/.paper_distiller/qa-sessions/<session_id>/state.json
```

`state.json` is a JSON-serialized `SessionState` with all rounds, articles distilled inline (so resume doesn't have to refetch), seen IDs, cost, and stop reason. The `articles_seen_ids` set is stored as a sorted list and restored as a Python set on read.

`paper-distiller-chat resume --session-id <sid>` reads this file, sets `cfg.qa_resume_session_id`, and re-enters the loop at the next round. Already-distilled articles are not re-fetched.

## LLM client contract

`llm/openai_compatible.py::LLMClient` is a minimal HTTP wrapper:

- `complete(messages, temperature, response_format=None)` — one method
- `response_format="json"` enables strict-JSON mode (provider-side `response_format: {type: json_object}`)
- Accumulates `total_tokens_in/out` across calls
- Raises `LLMError` on HTTP/timeout/auth failures (caught upstream by `PaperProcessor`)

Any OpenAI-compatible endpoint works: Aliyun Bailian (recommended), DeepSeek, OpenRouter, local Ollama.

## Prompts as plain markdown

All 6 LLM prompts live as plain `.md` files:

- `src/paper_distiller/prompts/filter.md` (rank candidates)
- `src/paper_distiller/prompts/article.md` (distill one paper)
- `src/paper_distiller/prompts/survey.md` (compose multi-article survey)
- `src/paper_distiller/qa/prompts/reflect.md` (judge QA loop progress)
- `src/paper_distiller/qa/prompts/answer.md` (synthesize final answer)
- `src/paper_distiller/agents/prompts/route.md` (REPL intent classification)

Edit them directly to change tone, structure, or output language. No Python changes needed — they use Python `str.format()` interpolation with named parameters like `{question}`, `{user_input}`, etc.

## Vault format

paper-distiller writes pure Obsidian-flavored markdown — no custom format:

- YAML frontmatter at the top (`title`, `tags`, `slug`, `arxiv_id`, `doi`, `published`, `depth`)
- Body in markdown
- Cross-references via `[[wikilink]]` or `[[wikilink|Display]]`
- Categories are subdirectory names

The chat REPL's `/sessions` command and the resume flow both rely on `.paper_distiller/qa-sessions/<sid>/state.json` — a hidden directory Obsidian ignores by default.

## Testing

168 tests across `tests/` (run `pytest -q`):

- `tests/agents/` — per-agent unit tests (each agent + framework primitives)
- `tests/chat/` — REPL parsing, helpers, CLI dispatch, qa_runner
- `tests/integration/` — end-to-end distill + ask flows with all subsystems mocked
- `tests/test_*.py` (root) — primitives: arxiv source, SS source, distill, config, vault, LLM, smoke

All LLM calls are mocked (`unittest.mock` / `pytest-mock`); no real API calls in CI. Real-API smoke tests are documented in CHANGELOG entries; not part of CI.

CI matrix: Python 3.10 / 3.11 / 3.12 on Ubuntu, triggered on push to main and on every PR.
