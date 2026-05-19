# paper-distiller

> Turn arXiv papers into an Obsidian-ready knowledge base — with two modes, one vault.

[![CI](https://github.com/jesson-hh/paper-distiller/actions/workflows/ci.yml/badge.svg)](https://github.com/jesson-hh/paper-distiller/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/paper-distiller.svg)](https://pypi.org/project/paper-distiller/)
[![Python versions](https://img.shields.io/pypi/pyversions/paper-distiller.svg)](https://pypi.org/project/paper-distiller/)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

paper-distiller is a command-line tool that searches academic paper sources (arXiv + Semantic Scholar), downloads PDFs, has an LLM distill each one into a structured markdown note, and writes everything to a folder that opens directly in [Obsidian](https://obsidian.md).

Two modes:

| Mode | When to reach for it |
|---|---|
| **`paper-distiller`** | You know the topic; you want N papers added to the vault in one shot. |
| **`paper-distiller-qa`** | You have a research *question*; you want the tool to plan multiple search rounds itself and write you a cited answer. |

Output is plain markdown with YAML frontmatter and `[[wikilink]]` cross-references — no proprietary format, no lock-in. Graph view, Dataview, tags, and full-text search all work out of the box.

---

## Why use this?

| Alternative | What paper-distiller does differently |
|---|---|
| **Asking ChatGPT directly** | paper-distiller cites the actual PDFs you can verify, persists notes locally, dedups across runs |
| **Zotero / Mendeley** | Those are reference managers; this *summarizes* each paper into a structured note you can read |
| **Manual notes** | Automates the "fingertip understanding" pass — skim 5× faster, then deep-dive selectively |
| **Cloud "AI research assistants"** | Writes to YOUR local files in a standard format. No cloud lock-in, no proprietary database. |

---

## What you get

A distilled article in your vault looks like this (auto-generated; the structure is fixed, the content reflects the paper):

````markdown
---
title: "Conditional Flow Matching with Sample-Complexity Bounds"
category: articles
slug: cfm-sample-complexity-bounds
tags: [generative-models, flow-matching, theory, sample-complexity, arxiv-2024]
refs: [arxiv:2410.12345]
depth: full-pdf
---

# Conditional Flow Matching 的 Sample Complexity 上界

> **场合**: arxiv preprint, 2024 Oct
> **主题**: 给 CFM 训练给出第一个匹配 nonparametric minimax rate 的有限样本界
> **领域**: 统计 / 生成模型理论

## 一句话
作者证明 CFM 训练在 $\beta$-平滑目标密度下达到 $n^{-\beta/(2\beta+d)}$ 的 $W_2$ 收敛速度,
不需要 score-based 方法那个 time-singularity log 因子。

## 问题动因
之前的 score-based 收敛分析普遍要求 $t \to 0$ 处加 ε-正则化, 否则要付额外 $\log(1/\varepsilon)$ 项 …

## 方法
核心是把 vector-field 估计误差 decompose 成 (1) approximation error 由 $\beta$-Hölder ball
覆盖控制 (2) statistical error 用 local Rademacher 处理 (3) discretization error 显式给 …

## 关键结果
| 估计量 | 收敛速度 |
|---|---|
| Score matching (prior) | $\tilde O(n^{-\beta/(2\beta+d+5)})$ |
| CFM (this work) | $\tilde O(n^{-\beta/(2\beta+d)})$ |

## 与已有 wiki 的关联
对 [[cnf-convergence-distribution-learning]] 的分析路线是个自然的强化 …

## 我的 take
最有意思的是 time-singularity 在 CFM 训练里其实从未出现 — 不需要回避它而是根本就不会进入估计误差。
仍待研究的是 …
````

Open the vault in Obsidian and this note is automatically cross-linked with everything else you've distilled.

---

## Install

**From PyPI**:

```bash
pip install paper-distiller
```

**From source** (for development):

```bash
git clone https://github.com/jesson-hh/paper-distiller
cd paper-distiller
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -e .
```

Requires Python 3.10+.

---

## Configure

paper-distiller needs an OpenAI-compatible LLM endpoint. Cheapest reliable choice: Aliyun Bailian's `qwen-plus` (~¥0.02 per paper).

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

CLI flags `--model` and `--provider` override env vars where set.

### Provider quick reference

| Provider | `PD_BASE_URL` | `PD_MODEL` |
|---|---|---|
| **Aliyun Bailian** (cheapest, recommended) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| Aliyun Bailian (coding plan) | `https://coding.dashscope.aliyuncs.com/v1` | `qwen3.5-plus` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenRouter | `https://openrouter.ai/api/v1` | `qwen/qwen3.5-plus` |
| Local Ollama | `http://localhost:11434/v1` | `qwen2.5` |

---

## Use it

### Single-pass — distill N papers on a topic

```bash
paper-distiller --vault /path/to/your/vault \
                --topic "diffusion models for finance" --n 5
```

paper-distiller searches both arxiv and Semantic Scholar, dedups, has the LLM rank the top 5, and distills each. Open the vault in Obsidian — new articles appear under `articles/`, an optional session survey under `surveys/`. Cost: ~¥0.7 (~$0.10).

### Question-driven — let the agent plan multiple rounds

```bash
paper-distiller-qa --vault /path/to/your/vault \
                   --question "What are recent advances in diffusion models for long-horizon time-series forecasting?" \
                   --max-rounds 3 --per-round 2 --max-cost-cny 5
```

Each round the agent:

1. **Reflects** on what's known so far vs. what's missing
2. **Plans the next search query** based on that reflection
3. **Distills** the top N papers for that query
4. **Repeats** until the LLM is confident OR a budget cap fires

Then it **synthesizes a cited answer survey** written to `surveys/qa-<slug>-<date>.md`, with an audit trail of every round.

Pause anytime with `Ctrl+C`; resume later:

```bash
paper-distiller-qa --vault ... --resume 20260519-0935-c6e43
```

The seven stop reasons that can end a QA session:

| Reason | What it means |
|---|---|
| `llm_done` | LLM judged it's done (confidence ≥ threshold) |
| `llm_brake` | LLM flagged diminishing returns (`suggest_stop=True`) |
| `max_rounds` | Hit `--max-rounds` |
| `max_articles` | Hit `--max-articles` |
| `max_cost` | Hit `--max-cost-cny` |
| `no_candidates` | All search hits already in the vault (full dedup) |
| `user_quit` | Ctrl+C or interactive `n`/`q` |

`user_quit` and transient `error:*` stops leave the session resumable; the others terminate it.

---

## CLI reference

```
paper-distiller --vault <path> {--topic <str> | --author <str>}
                [--n 5] [--pool 30] [--source {arxiv,ss,both}]
                [--force] [--dry-run] [--verbose] [--model <name>] [--provider <name>]

paper-distiller-qa --vault <path> --question <str>
                   [--max-rounds 5] [--max-articles 15] [--max-cost-cny 20.0]
                   [--confidence-threshold 8] [--per-round 2]
                   [--source {arxiv,ss,both}] [--interactive] [--resume <session-id>]
                   [--dry-run] [--verbose] [--model <name>] [--provider <name>]
```

Run either with `--help` for the full flag list. `--dry-run` skips all LLM calls and vault writes — useful for verifying config before spending API budget.

---

## Vault layout

paper-distiller writes into a vault with these subdirectories (created on first run):

| Directory | What goes there |
|---|---|
| `articles/` | One file per paper |
| `surveys/` | Multi-article surveys (single-pass session summary, or QA-mode `qa-…` answer doc) |
| `techniques/`, `directions/`, `open-problems/`, `authors/` | Reserved for human-curated content — paper-distiller does not touch these |

QA-mode also persists state under `<vault>/.paper_distiller/qa-sessions/<sid>/state.json` — a hidden directory Obsidian ignores by default. This is what `--resume` reads.

---

## How it works (in two diagrams)

**Single-pass:**

```
search arxiv + SS  →  LLM filter (top N)  →  fetch PDF (with SS openAccessPdf fallback)
                  →  PyMuPDF extract  →  LLM distill  →  vault.save_entry (dedups by arxiv-id/DOI)
                  →  (if N >= PD_MIN_SURVEY)  LLM compose survey
```

**Multi-round:**

```
loop {
   LLM reflect (judge progress, propose next query)  ─┐
   → break if budget hit / LLM confident / no new     │
   → single-pass for this round's query               │
}                                                     │
                                                      ↓
LLM synthesize cited answer  →  surveys/qa-<slug>-<date>.md
                                  (with audit trail of every round)
```

For module structure, the data flow internals, prompt locations, state persistence format, and cost-accounting math, see **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Cost

Aliyun Bailian `qwen-plus` pricing — roughly ¥2.1/M input tokens, ¥12.7/M output tokens.

| Run | Typical cost |
|---|---|
| 1 paper distilled | ~¥0.02 (~$0.003) |
| 5-paper single-pass run + survey | ~¥0.7 (~$0.10) |
| 3-round QA session @ 2 papers/round | ~¥1.5–3 |
| 5-round QA session @ 3 papers/round | ~¥4–8 |

`paper-distiller-qa` enforces these via `--max-cost-cny` (default ¥20). The cost number is for the circuit breaker — it is **not** billing-accurate.

---

## Customize the output

All 5 LLM prompts are plain markdown — edit them to change tone, structure, or output language. No Python changes needed.

- `src/paper_distiller/prompts/{filter,article,survey}.md` — single-pass mode
- `src/paper_distiller/qa/prompts/{reflect,answer}.md` — question-driven mode

The defaults produce **Chinese-primary** notes with this 5-section structure: 一句话 / 问题动因 / 方法 / 关键结果 / 我的 take. To switch the output language, edit `article.md` and `answer.md`.

---

## Optional companion: semantic search via vault-mcp

paper-distiller does NOT ship its own semantic-search engine for your vault. To search by meaning (not keywords) — from Claude Code, Cursor, or any MCP-aware agent — pair it with [**vault-mcp**](https://github.com/robbiemu/vault-mcp), a standalone MCP server purpose-built for markdown vaults.

See [docs/vault-mcp-recommendation.md](docs/vault-mcp-recommendation.md) for setup and rationale.

---

## Status & roadmap

**v0.5.1 — alpha.** Both CLIs work end-to-end; 78 tests passing on Python 3.10 / 3.11 / 3.12 via GitHub Actions.

### Shipped

- **v0.1** — Single-pass against arxiv; LLM filter + ranker; PyMuPDF extraction; markdown output.
- **v0.2** — arxiv-id-based dedup (prevents sibling notes for the same paper).
- **v0.3** — Semantic Scholar as second source (`--source {arxiv,ss,both}`); PDF fallback chain (try SS `openAccessPdf` when arxiv PDF 4xx's); DOI dedup.
- **v0.5** — `paper-distiller-qa` question-driven multi-round loop. State-machine with 7 stop reasons, `--interactive` and `--resume`.
- **v0.5.1** — First PyPI release.

### Coming

- **v0.6** — Citation-graph traversal: given a seed article, follow its `references` / `cited-by` edges and rank them for inclusion.
- **v0.7** — Sources beyond arxiv + SS: likely [OpenCLI](https://github.com/jackwener/OpenCLI) integration for logged-in browser sessions (ACM Digital Library, IEEE Xplore, 知乎, 等).
- **Later** — Per-vault `paper-distiller.toml` for custom category schemas; LEANN-backed in-pipeline crosslink retrieval (useful only when vault > 500 entries).

### Known limitations

- arxiv.org and Semantic Scholar occasionally rate-limit (`HTTP 429`); paper-distiller exits gracefully with a `error: search failed` stop reason that's resumable via `--resume`.
- Scanned-only PDFs fall through to abstract-only mode (PyMuPDF doesn't OCR — by design we'd rather flag it than silently distill from a wrong text source).

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
