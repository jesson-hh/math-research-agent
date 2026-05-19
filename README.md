# paper-distiller

> Chat-first paper distillation. Turn arXiv papers into an Obsidian-ready knowledge base — via REPL, one-shot commands, or natural language.

[![CI](https://github.com/jesson-hh/paper-distiller/actions/workflows/ci.yml/badge.svg)](https://github.com/jesson-hh/paper-distiller/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/paper-distiller.svg)](https://pypi.org/project/paper-distiller/)
[![Python versions](https://img.shields.io/pypi/pyversions/paper-distiller.svg)](https://pypi.org/project/paper-distiller/)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

paper-distiller is a command-line tool that searches academic paper sources (arXiv + Semantic Scholar), downloads PDFs, has an LLM distill each one into a structured markdown note, and writes everything to a folder that opens directly in [Obsidian](https://obsidian.md).

v1.0 ships a single `paper-distiller-chat` command with three modes:

| Mode | When to use |
|---|---|
| **`paper-distiller-chat`** (no args) | Interactive REPL — slash commands + natural-language input |
| **`paper-distiller-chat distill`** | One-shot: search a topic, distill N papers |
| **`paper-distiller-chat ask`** | One-shot: ask a research question, multi-round QA loop |
| **`paper-distiller-chat resume`** | One-shot: continue a paused/errored QA session |

Output is plain markdown with YAML frontmatter and `[[wikilink]]` cross-references — no proprietary format, no lock-in. Graph view, Dataview, tags, and full-text search all work out of the box.

---

## Install

```bash
pip install paper-distiller
```

Requires Python 3.10+. From source:

```bash
git clone https://github.com/jesson-hh/paper-distiller
cd paper-distiller
pip install -e ".[dev]"
```

---

## Configure

paper-distiller needs an OpenAI-compatible LLM endpoint. Cheapest reliable option: Aliyun Bailian's `qwen-plus` (~¥0.02 per paper).

```bash
cp examples/example.env .env
# Edit .env — set PD_API_KEY, PD_BASE_URL, PD_MODEL
```

| Env var | Required | Default | Purpose |
|---|---|---|---|
| `PD_API_KEY` | ✓ | — | Any OpenAI-compatible API key |
| `PD_BASE_URL` | ✓ | — | API endpoint base URL |
| `PD_MODEL` | ✓ | — | Model identifier |
| `PD_PROVIDER_NAME` |   | `unspecified` | Logging tag only |
| `PD_PDF_TIMEOUT` |   | `60` | PDF download timeout (seconds) |
| `PD_MIN_SURVEY` |   | `2` | Min articles before composing a session survey |
| `PD_SS_API_KEY` |   | (none) | Optional — higher Semantic Scholar rate limit |

### Provider quick reference

| Provider | `PD_BASE_URL` | `PD_MODEL` |
|---|---|---|
| **Aliyun Bailian** (recommended) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| Aliyun Bailian (coding plan) | `https://coding.dashscope.aliyuncs.com/v1` | `qwen3.5-plus` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenRouter | `https://openrouter.ai/api/v1` | `qwen/qwen3.5-plus` |
| Local Ollama | `http://localhost:11434/v1` | `qwen2.5` |

---

## Use it

### Interactive REPL (recommended)

```bash
paper-distiller-chat --vault /path/to/your/vault
```

You see a welcome banner with provider + vault info, then a prompt. Type slash commands or natural language:

```
> /help
[command list]

> /vault
Vault: /path/to/your/vault
  articles: 47
  surveys: 6
  ...

> /distill diffusion models --n 3
[live status table during execution]

> 帮我研究下扩散模型在长周期金融时序生成上的最新进展
[intent-router] Intent: ask  | confidence 9
  question: 扩散模型在长周期金融时序生成上的最新进展
Missing: max_rounds, per_round, max_cost_cny
Apply defaults (max_rounds=3, per_round=2, max_cost_cny=5.0) and run? [Y/n]
> Y
[live status table for 3-round QA loop]

> /quit
  (bye)
```

10 slash commands available: `/distill`, `/ask`, `/resume`, `/sessions`, `/vault`, `/provider`, `/agents`, `/show`, `/help`, `/quit`.

Natural-language input goes through an LLM intent-router that classifies into one of `distill`/`ask`/`resume`/`show` and proposes defaults for any missing parameters. You confirm before any expensive operation runs.

### One-shot mode (good for scripts / cron)

**Distill N papers on a topic:**

```bash
paper-distiller-chat distill --vault /path/to/your/vault \
    --topic "diffusion models for finance" --n 5
```

**Answer a question across multiple rounds:**

```bash
paper-distiller-chat ask --vault /path/to/your/vault \
    --question "What are recent advances in long-horizon time-series diffusion?" \
    --max-rounds 3 --per-round 2 --max-cost-cny 5
```

**Resume a paused / errored session:**

```bash
paper-distiller-chat resume --vault /path/to/your/vault \
    --session-id 20260519-1635-a3f7
```

Use `--dry-run` on any subcommand to validate config without spending API budget.

### Helpful flags

```
paper-distiller-chat [--vault PATH]
                     {distill | ask | resume}
                     [subcommand-specific flags]
```

`paper-distiller-chat distill --help` etc. show every flag for that subcommand.

---

## What you get — a sample distilled article

````markdown
---
title: "Convergence Rates of Conditional Flow Matching..."
category: articles
slug: cnf-convergence
tags: [generative-models, theory, distribution-estimation, arxiv-2024]
refs: [arxiv:2410.12345]
depth: full-pdf
---

# CFM 的样本复杂度上界

> **场合**: arxiv preprint, 2024 Oct
> **主题**: 给 CFM 训练给出第一个匹配 nonparametric minimax rate 的有限样本界
> **领域**: 统计 / 生成模型理论

## 一句话
作者证明 CFM 训练在 $\beta$-平滑目标密度下达到 $n^{-\beta/(2\beta+d)}$ 的 $W_2$ 收敛速度…

## 方法
核心是把 vector-field 估计误差 decompose 成 (1) approximation error 由 $\beta$-Hölder ball
覆盖控制 (2) statistical error 用 local Rademacher 处理 (3) discretization error 显式给…

## 与已有 wiki 的关联
对 [[cnf-convergence-distribution-learning]] 的分析路线是个自然的强化…

## 我的 take
最有意思的是 time-singularity 在 CFM 训练里其实从未出现…
````

Open the vault in Obsidian and this article cross-links automatically with everything else you've distilled.

---

## Vault layout

paper-distiller writes into a vault with these subdirectories (auto-created on first run):

| Directory | Auto-written by tool | Description |
|---|---|---|
| `articles/` | ✓ | One file per paper |
| `surveys/` | ✓ | Multi-article surveys + `qa-…` final answer docs |
| `techniques/`, `directions/`, `open-problems/`, `authors/` | — | Reserved for human-curated notes |

QA sessions persist resume state at `<vault>/.paper_distiller/qa-sessions/<sid>/state.json`.

---

## How it works

paper-distiller v1.0 is built around an async DAG of sub-agents:

```
Single-pass (distill):
  arxiv-searcher  ss-searcher          (parallel)
        └────┬────┘
        candidate-merger
              │
        candidate-ranker (LLM)
              │
        paper-processor × N            (parallel: fetch PDF → extract → distill LLM)
              │
        vault-writer
              │
        survey-composer (LLM, optional)

Multi-round (ask):
  ┌──────────────────────────────────────────────────────┐
  │  progress-reflector (LLM)                             │
  │      ↓                                                │
  │  [stop check: max_rounds / llm_done / llm_brake / ...] │
  │      ↓                                                │
  │  search → dedup → rank → distill × N → write          │
  └────────────────────────────────────────────────────────┘
                          ↓
                  answer-synthesizer (LLM) → surveys/qa-<slug>-<date>.md
```

11 agents, 4 stop reasons in QA mode, all wired together by a topological-level scheduler. For module structure, full data flow, and internal contracts, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Cost

Aliyun Bailian `qwen-plus` pricing — roughly ¥2.1/M input tokens, ¥12.7/M output tokens.

| Operation | Typical cost |
|---|---|
| 1 paper distilled | ~¥0.02 (~$0.003) |
| 5-paper single-pass + survey | ~¥0.7 (~$0.10) |
| 3-round QA session @ 2 papers/round | ~¥1.5–3 |
| 5-round QA session @ 3 papers/round | ~¥4–8 |

`paper-distiller-chat ask` enforces `--max-cost-cny` (default ¥20). The cost number is for the circuit breaker — not billing-accurate.

---

## Customize the output

All 6 LLM prompts are plain markdown — edit them to change tone, structure, or output language. No Python changes needed.

- `src/paper_distiller/prompts/{filter,article,survey}.md` — distill mode
- `src/paper_distiller/agents/prompts/route.md` — intent router
- `src/paper_distiller/qa/prompts/{reflect,answer}.md` — QA mode

Defaults produce **Chinese-primary** notes with this 5-section structure: 一句话 / 问题动因 / 方法 / 关键结果 / 我的 take.

---

## Optional companion: semantic search via vault-mcp

paper-distiller does NOT ship its own semantic-search engine for your vault. To search by meaning (not keywords) from Claude Code, Cursor, or any MCP-aware agent, pair it with [**vault-mcp**](https://github.com/robbiemu/vault-mcp).

See [docs/vault-mcp-recommendation.md](docs/vault-mcp-recommendation.md) for setup and rationale.

---

## Status & roadmap

**v1.0.0 — beta.** Chat-first architecture stable; 168 tests passing on Python 3.10 / 3.11 / 3.12.

### Migration from v0.5

| v0.5.x | v1.0 |
|---|---|
| `paper-distiller --topic X --n N` | `paper-distiller-chat distill --topic X --n N` |
| `paper-distiller-qa --question Y --max-rounds R` | `paper-distiller-chat ask --question Y --max-rounds R` |
| (no resume command) | `paper-distiller-chat resume --session-id <sid>` |
| (no interactive mode) | `paper-distiller-chat` (no subcommand) |

Flag names and defaults are otherwise preserved. See [CHANGELOG](CHANGELOG.md) for full details.

### Coming

- **v1.1** — citation-graph traversal: given a seed article, follow references / cited-by edges and rank them for inclusion.
- **v1.2** — broaden sources beyond arxiv + SS: integrate browser-session scraping for ACM, IEEE, 知乎 etc.
- **Later** — per-vault `paper-distiller.toml` for custom category schemas; LEANN in-pipeline crosslink retrieval for vaults > 500 entries.

### Known limitations

- arxiv.org and Semantic Scholar occasionally rate-limit (HTTP 429); QA sessions exit with `error: search failed` (resumable via `paper-distiller-chat resume <sid>`).
- Scanned-only PDFs fall through to abstract-only mode (PyMuPDF doesn't OCR — by design we flag rather than silently distill wrong text).

---

## Contributing

Issues and PRs welcome.

```bash
git clone https://github.com/jesson-hh/paper-distiller
cd paper-distiller
pip install -e ".[dev]"
pytest -v
```

CI runs the same matrix on every PR. For a tour of the codebase, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## License

MIT — see [LICENSE](LICENSE).
