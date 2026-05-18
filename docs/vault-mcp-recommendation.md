# Recommended companion: vault-mcp for semantic search

paper-distiller writes into your Obsidian vault. To **search** that vault by meaning (not keywords) from Claude Code or any MCP-aware AI, we recommend pairing it with [**vault-mcp**](https://github.com/robbiemu/vault-mcp) — a standalone MCP server purpose-built for markdown vaults.

paper-distiller does NOT ship its own semantic-search engine. This document explains why, and how to wire vault-mcp up.

## Why vault-mcp specifically

| Feature | vault-mcp |
|---|---|
| MCP-native | Yes — installs as a Claude Code MCP server |
| Vault formats | Obsidian, Joplin, plain markdown directories |
| Live sync | Yes — watchdog re-indexes on file change |
| Engine | ChromaDB + LlamaIndex + LiteLLM |
| Embedding | Sentence Transformers (local) OR any LiteLLM-supported API (OpenAI, Aliyun, Anthropic, etc.) |
| Chunking | Quality-based, structure-aware markdown parsing |
| Obsidian must be running? | No |
| Activates without modifying paper-distiller | Yes |

This is exactly the integration shape paper-distiller wants its users to have — a separate, well-engineered server that watches the same `wiki/` directory paper-distiller writes into.

## Setup (with paper-distiller's vault)

From your home directory or anywhere outside paper-distiller:

```bash
git clone https://github.com/robbiemu/vault-mcp
cd vault-mcp
uv venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
uv sync
```

Configure it to index your paper-distiller vault (consult vault-mcp's README for the latest config format). Then register the resulting MCP server in Claude Code's `.mcp.json` either at the paper-distiller repo root (project-scoped) or in `~/.claude/settings.json` (user-scoped).

After restarting Claude Code, you'll have semantic search tools like `mcp__vault-mcp__search` available in chat:

> "Search my vault for entries about diffusion models in finance"

## Why paper-distiller doesn't ship its own MCP search

We evaluated rolling our own ([LEANN](https://github.com/yichuan-w/LEANN)-backed) and concluded it's the wrong scope for this project. Documenting the decision so contributors don't re-propose:

1. **Scope discipline.** paper-distiller's value is the **distillation pipeline** (arxiv → LLM → markdown). MCP search infrastructure is a different problem domain; reinventing it within paper-distiller would split focus and lose to specialized tools like vault-mcp.
2. **vault-mcp's stack is more mature.** ChromaDB + LlamaIndex + LiteLLM are battle-tested; LEANN is a smaller alpha-stage research project.
3. **Live sync matters.** vault-mcp has it; we'd have had to build it.
4. **LiteLLM gives multi-provider for free.** Aliyun / OpenAI / Anthropic / local Ollama all just work; with LEANN we'd be wiring each one.
5. **CPU embedding is slow.** Our smoke test with LEANN + Qwen3-Embedding-0.6B took 30+ minutes per indexing run on CPU — unusable. vault-mcp's defaults are more sensible (smaller models like `all-MiniLM-L6-v2` work fine), and switching to an API embedding via LiteLLM is one config line.

## Alternative: when LEANN becomes interesting later

LEANN's value proposition is **97% storage savings via selective recomputation** — irrelevant at 50 entries, but compelling at 100K+. If your vault grows to that scale, revisit. Otherwise, vault-mcp's standard ANN approach is fine.

## Future paper-distiller integrations

A future paper-distiller release may directly call vault-mcp's REST API (it exposes one alongside the MCP interface) to enrich the in-pipeline crosslink suggestions. This is on the v0.5+ roadmap, not the current release.
