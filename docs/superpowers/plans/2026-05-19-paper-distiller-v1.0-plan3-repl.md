# paper-distiller v1.0 — Plan 3 (Interactive REPL + Intent-Router)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans.

**Goal:** Add a chat-style REPL to `paper-distiller-chat`. The REPL supports both deterministic slash commands (`/distill`, `/ask`, `/sessions`, etc.) and natural-language input routed via an LLM intent-router. After this plan: `paper-distiller-chat` (no subcommand) opens the REPL with a welcome banner; users can type `/distill diffusion --n 3` OR `帮我研究下扩散在金融时序` and the tool dispatches appropriately.

**Architecture:** `chat/repl/` package contains the input loop, slash-command parser, command handlers (some read-only, some delegating to existing one-shot handlers), and the natural-language confirmation flow. New `IntentRouter` agent wraps a single JSON-out LLM call that classifies user intent. REPL keeps minimal state (current vault, last QA session ID for `/resume` convenience).

**Tech Stack:** Adds one new runtime dep — `prompt_toolkit>=3` for input editing, history, and slash-command completion.

**Spec:** [docs/superpowers/specs/2026-05-19-paper-distiller-v1.0-chat-design.md](../specs/2026-05-19-paper-distiller-v1.0-chat-design.md) §7.10 + §8.

**Working directory:** `G:\paper-distiller\`

**Test baseline:** 151 (after Plan 2).

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add `prompt_toolkit>=3` runtime dep |
| `src/paper_distiller/agents/router.py` | Create | `IntentRouter` agent — NL → command JSON |
| `src/paper_distiller/agents/prompts/route.md` | Create | LLM routing prompt template |
| `src/paper_distiller/chat/repl/__init__.py` | Create | Package marker |
| `src/paper_distiller/chat/repl/commands.py` | Create | Slash-command parsing + registry |
| `src/paper_distiller/chat/repl/helpers.py` | Create | Read-only commands (`/vault`, `/sessions`, `/provider`, `/agents`, `/show`, `/help`) |
| `src/paper_distiller/chat/repl/loop.py` | Create | The REPL main class (welcome banner, input loop, dispatch) |
| `src/paper_distiller/chat/cli.py` | Modify | When no subcommand given, launch REPL |
| `tests/agents/test_router.py` | Create | 4 tests |
| `tests/chat/test_commands.py` | Create | 5 tests for slash parsing |
| `tests/chat/test_helpers.py` | Create | 6 tests for helper handlers |
| `tests/chat/test_repl_loop.py` | Create | 4 tests for REPL dispatch |
| `tests/chat/test_repl_cli.py` | Create | 2 tests for cli.py integration |
| `tests/integration/test_repl_e2e.py` | Create | 1 e2e test |

**Test count after Plan 3:** 151 + 22 = **173**.

---

## Task 1: `IntentRouter` agent + routing prompt template

**Files:**
- Create: `src/paper_distiller/agents/prompts/route.md`
- Create: `src/paper_distiller/agents/router.py`
- Create: `tests/agents/test_router.py`

The IntentRouter is a single JSON-out LLM call. It does NOT participate in any DAG (it's invoked directly by the REPL). But we structure it as an agent so its prompt lives next to the others and so future workflows can use it too.

### Step 1: Write the failing tests

Create `tests/agents/test_router.py`:

```python
"""Tests for IntentRouter — natural language to slash command JSON."""
import json
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.router import IntentRouter, RoutingError


def test_router_returns_distill_for_topic_query():
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "command": "distill",
        "params": {"topic": "diffusion models", "n": 3},
        "missing_params": [],
        "confidence": 9,
    })
    router = IntentRouter(llm=llm)
    out = router.classify("distill 3 papers on diffusion models")
    assert out["command"] == "distill"
    assert out["params"]["topic"] == "diffusion models"


def test_router_returns_ask_for_question():
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "command": "ask",
        "params": {"question": "为什么扩散模型在长序列上效果好？"},
        "missing_params": ["max_rounds", "per_round", "max_cost_cny"],
        "confidence": 8,
    })
    router = IntentRouter(llm=llm)
    out = router.classify("为什么扩散模型在长序列上效果好？")
    assert out["command"] == "ask"
    assert "max_rounds" in out["missing_params"]


def test_router_raises_routing_error_on_malformed_json():
    llm = MagicMock()
    llm.complete.return_value = "not json at all"
    router = IntentRouter(llm=llm)
    with pytest.raises(RoutingError, match="malformed"):
        router.classify("anything")


def test_router_raises_on_unknown_command():
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "command": "noexist", "params": {}, "missing_params": [], "confidence": 8,
    })
    router = IntentRouter(llm=llm)
    with pytest.raises(RoutingError, match="unknown command"):
        router.classify("anything")
```

### Step 2: Run, confirm fail

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_router.py -v
```

Expected: ModuleNotFoundError.

### Step 3: Create the prompt template

Create `src/paper_distiller/agents/prompts/route.md`:

```
你是 paper-distiller-chat REPL 的意图路由器。把用户的自然语言输入转换成结构化的子命令调用。

# 可用子命令

- `distill` — 单次任务：根据 topic 搜 + 蒸馏 N 篇论文。params: `topic` (string), `n` (int, default 3).
- `ask` — 多轮 QA：根据 question 自主规划多轮搜索 + 蒸馏 + 最终合成答案。params: `question` (string), `max_rounds` (int, default 3), `per_round` (int, default 2), `max_cost_cny` (float, default 5.0).
- `resume` — 接续 paused/errored QA session。params: `session_id` (string).
- `show` — 显示 vault 里某一篇 article。params: `slug` (string).

# 用户输入

{user_input}

# 输出严格 JSON，不要 markdown 围栏，不要任何前导文字

{{
  "command": "distill" | "ask" | "resume" | "show",
  "params": {{... 你能从输入提取出的字段 ...}},
  "missing_params": ["... 哪些字段还需要让用户补 ..."],
  "confidence": 0-10
}}

# 规则

- 如果用户输入是问题（"为什么 X？" "怎么样 Y？"），选 `ask`。
- 如果用户给了主题且想要 N 篇论文（"找 3 篇关于 X 的论文"），选 `distill`。
- 如果用户提到 session id，选 `resume`。
- 如果用户想看一篇已有文章（"看看 X"），选 `show`。
- ask 的 missing_params 总是包含 max_rounds/per_round/max_cost_cny（除非用户明确指定）。
- 不要发明 command — 必须是上面 4 个之一。
- confidence 反映你对意图分类的把握。
```

### Step 4: Create `src/paper_distiller/agents/router.py`

```python
"""IntentRouter — single JSON-out LLM call mapping natural language to a slash command."""

from __future__ import annotations

import json
from pathlib import Path


class RoutingError(RuntimeError):
    pass


_PROMPT_FILE = Path(__file__).parent / "prompts" / "route.md"
_VALID_COMMANDS = {"distill", "ask", "resume", "show"}
_REQUIRED_KEYS = {"command", "params", "missing_params", "confidence"}


class IntentRouter:
    name = "intent-router"
    deps: list[str] = []

    def __init__(self, llm):
        self.llm = llm

    def classify(self, user_input: str) -> dict:
        prompt = _PROMPT_FILE.read_text(encoding="utf-8").format(user_input=user_input)
        messages = [{"role": "user", "content": prompt}]
        for attempt in (1, 2):
            raw = self.llm.complete(messages, temperature=0.2, response_format="json")
            try:
                parsed = json.loads(raw)
                missing = _REQUIRED_KEYS - set(parsed.keys())
                if missing:
                    raise ValueError(f"missing keys: {missing}")
                if parsed["command"] not in _VALID_COMMANDS:
                    raise ValueError(f"unknown command: {parsed['command']!r}")
                return parsed
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == 2:
                    if "unknown command" in str(e):
                        raise RoutingError(f"unknown command in router output: {raw[:200]}")
                    raise RoutingError(f"intent router returned malformed JSON: {raw[:200]}")
                continue
        raise RoutingError("unreachable")
```

### Step 5: Run tests, confirm pass

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_router.py -v
```

Expected: 4 passed.

### Step 6: Run full suite

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **155 passed** (151 + 4).

### Step 7: Commit

```bash
git add src/paper_distiller/agents/router.py src/paper_distiller/agents/prompts/route.md tests/agents/test_router.py
git commit -m "feat(agents): IntentRouter — NL to slash-command JSON classifier"
```

---

## Task 2: Slash-command parser + registry

**Files:**
- Create: `src/paper_distiller/chat/repl/__init__.py`
- Create: `src/paper_distiller/chat/repl/commands.py`
- Create: `tests/chat/test_commands.py`

### Step 1: Write the failing tests

Create `tests/chat/test_commands.py`:

```python
"""Tests for slash-command parsing."""
import pytest

from paper_distiller.chat.repl.commands import parse_slash, SlashError, KNOWN_COMMANDS


def test_parse_simple_command_no_args():
    parsed = parse_slash("/vault")
    assert parsed.name == "vault"
    assert parsed.args == []


def test_parse_command_with_args():
    parsed = parse_slash("/distill diffusion models --n 3")
    assert parsed.name == "distill"
    assert parsed.args == ["diffusion", "models", "--n", "3"]


def test_parse_command_with_quoted_arg():
    parsed = parse_slash('/ask "why diffusion models?"')
    assert parsed.name == "ask"
    assert parsed.args == ["why diffusion models?"]


def test_parse_unknown_command_raises():
    with pytest.raises(SlashError, match="unknown"):
        parse_slash("/nosuchcommand")


def test_parse_non_slash_input_raises():
    with pytest.raises(SlashError, match="not a slash command"):
        parse_slash("hello world")


def test_known_commands_includes_core():
    assert "distill" in KNOWN_COMMANDS
    assert "ask" in KNOWN_COMMANDS
    assert "resume" in KNOWN_COMMANDS
    assert "vault" in KNOWN_COMMANDS
    assert "sessions" in KNOWN_COMMANDS
    assert "provider" in KNOWN_COMMANDS
    assert "agents" in KNOWN_COMMANDS
    assert "show" in KNOWN_COMMANDS
    assert "help" in KNOWN_COMMANDS
    assert "quit" in KNOWN_COMMANDS
```

### Step 2: Run, confirm fail

```bash
.venv\Scripts\python.exe -m pytest tests/chat/test_commands.py -v
```

Expected: ModuleNotFoundError.

### Step 3: Create `chat/repl/__init__.py`

```python
"""Interactive REPL for paper-distiller-chat (v1.0)."""
```

### Step 4: Create `chat/repl/commands.py`

```python
"""Slash-command parsing + known-command registry."""

from __future__ import annotations

import shlex
from dataclasses import dataclass


class SlashError(ValueError):
    pass


KNOWN_COMMANDS = {
    "distill", "ask", "resume",       # action commands (delegate to one-shot handlers)
    "vault", "sessions", "provider",  # read-only helpers
    "agents", "show", "help", "quit",
}


@dataclass
class ParsedSlash:
    name: str
    args: list[str]


def parse_slash(line: str) -> ParsedSlash:
    """Parse '/cmd arg1 "arg 2" --flag x' → ParsedSlash(name='cmd', args=['arg1','arg 2','--flag','x'])"""
    s = line.strip()
    if not s.startswith("/"):
        raise SlashError(f"not a slash command: {line!r}")
    body = s[1:].strip()
    if not body:
        raise SlashError("empty slash command")
    try:
        tokens = shlex.split(body)
    except ValueError as e:
        raise SlashError(f"could not parse slash command: {e}")
    if not tokens:
        raise SlashError("empty slash command after parse")
    name, *args = tokens
    if name not in KNOWN_COMMANDS:
        raise SlashError(f"unknown slash command: /{name}")
    return ParsedSlash(name=name, args=args)
```

### Step 5: Run tests, confirm pass

```bash
.venv\Scripts\python.exe -m pytest tests/chat/test_commands.py -v
```

Expected: 6 passed.

### Step 6: Run full suite

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **161 passed** (155 + 6).

### Step 7: Commit

```bash
git add src/paper_distiller/chat/repl/__init__.py src/paper_distiller/chat/repl/commands.py tests/chat/test_commands.py
git commit -m "feat(chat): slash-command parser + known-command registry"
```

---

## Task 3: Read-only helper handlers

**Files:**
- Create: `src/paper_distiller/chat/repl/helpers.py`
- Create: `tests/chat/test_helpers.py`

Handlers for `/vault`, `/sessions`, `/provider`, `/agents`, `/show`, `/help`. None of these call the LLM.

### Step 1: Write the failing tests

Create `tests/chat/test_helpers.py`:

```python
"""Tests for read-only REPL helper handlers."""
import json
from pathlib import Path

import pytest

from paper_distiller.chat.repl.helpers import (
    handle_vault, handle_sessions, handle_provider, handle_agents,
    handle_show, handle_help,
)


def test_handle_vault_shows_counts(tmp_path):
    vault = tmp_path
    (vault / "articles").mkdir()
    (vault / "surveys").mkdir()
    (vault / "articles" / "a.md").write_text("---\ntitle: A\n---\n", encoding="utf-8")
    (vault / "articles" / "b.md").write_text("---\ntitle: B\n---\n", encoding="utf-8")
    (vault / "surveys" / "s.md").write_text("---\ntitle: S\n---\n", encoding="utf-8")
    out = handle_vault(vault)
    assert "articles: 2" in out
    assert "surveys: 1" in out


def test_handle_vault_empty(tmp_path):
    out = handle_vault(tmp_path)
    assert "articles: 0" in out


def test_handle_sessions_lists_state_json(tmp_path):
    vault = tmp_path
    sessions = vault / ".paper_distiller" / "qa-sessions"
    sessions.mkdir(parents=True)
    sid_dir = sessions / "20260519-1234-abc"
    sid_dir.mkdir()
    (sid_dir / "state.json").write_text(json.dumps({
        "session_id": "20260519-1234-abc", "question": "why?",
        "config_snapshot": {}, "started_at": "2026-05-19T12:34:00",
        "rounds_completed": 2, "articles_distilled": [], "articles_seen_ids": [],
        "history": [], "last_reflection": None, "cost_cny": 0.5,
        "tokens_in_total": 0, "tokens_out_total": 0,
        "is_done": True, "stop_reason": "llm_done",
    }), encoding="utf-8")
    out = handle_sessions(vault)
    assert "20260519-1234-abc" in out
    assert "llm_done" in out


def test_handle_sessions_no_sessions(tmp_path):
    out = handle_sessions(tmp_path)
    assert "no sessions" in out.lower()


def test_handle_provider_shows_config(monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test-abc")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    out = handle_provider()
    assert "qwen-plus" in out
    assert "https://x/v1" in out
    # API key masked
    assert "sk-test-abc" not in out


def test_handle_agents_lists_registered():
    out = handle_agents()
    assert "arxiv-searcher" in out
    assert "paper-processor" in out
    assert "answer-synthesizer" in out


def test_handle_show_displays_article(tmp_path):
    vault = tmp_path
    (vault / "articles").mkdir()
    (vault / "articles" / "myslug.md").write_text(
        "---\ntitle: My Article\n---\n\n# My Article\n\nBody.",
        encoding="utf-8",
    )
    out = handle_show(vault, "myslug")
    assert "My Article" in out
    assert "Body." in out


def test_handle_show_not_found(tmp_path):
    out = handle_show(tmp_path, "no-such-slug")
    assert "not found" in out.lower()


def test_handle_help_lists_commands():
    out = handle_help()
    assert "/distill" in out
    assert "/ask" in out
    assert "/vault" in out
    assert "/quit" in out
```

### Step 2: Run, confirm fail

```bash
.venv\Scripts\python.exe -m pytest tests/chat/test_helpers.py -v
```

Expected: ModuleNotFoundError.

### Step 3: Create `chat/repl/helpers.py`

```python
"""Read-only REPL helper handlers.

These do NOT call the LLM. They inspect vault / env / agent registry and return
strings that the REPL prints. Action commands (/distill, /ask, /resume) live in
loop.py because they need access to the argparse handlers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def handle_vault(vault_path: Path) -> str:
    cats = ["articles", "surveys", "techniques", "directions", "open-problems", "authors"]
    lines = [f"Vault: {vault_path}"]
    for cat in cats:
        folder = vault_path / cat
        count = len(list(folder.glob("*.md"))) if folder.exists() else 0
        lines.append(f"  {cat}: {count}")
    return "\n".join(lines)


def handle_sessions(vault_path: Path) -> str:
    sessions_dir = vault_path / ".paper_distiller" / "qa-sessions"
    if not sessions_dir.exists():
        return "no sessions found."
    entries = sorted(sessions_dir.iterdir(), reverse=True)  # newest first
    if not entries:
        return "no sessions found."
    lines = ["QA sessions (newest first):"]
    for entry in entries:
        state_path = entry / "state.json"
        if not state_path.exists():
            continue
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sid = data.get("session_id", entry.name)
        stop = data.get("stop_reason", "?")
        rounds = data.get("rounds_completed", 0)
        cost = data.get("cost_cny", 0.0)
        question = (data.get("question", "") or "")[:60]
        is_done = "done" if data.get("is_done") else "open"
        lines.append(f"  {sid}  {is_done}  {stop}  ({rounds} rounds, CNY {cost:.2f})")
        if question:
            lines.append(f"    Q: {question}")
    return "\n".join(lines)


def handle_provider() -> str:
    base_url = os.getenv("PD_BASE_URL", "(not set)")
    model = os.getenv("PD_MODEL", "(not set)")
    provider = os.getenv("PD_PROVIDER_NAME", "unspecified")
    key = os.getenv("PD_API_KEY", "")
    key_status = "(set)" if key else "(not set)"
    return (
        f"Provider: {provider}\n"
        f"Base URL: {base_url}\n"
        f"Model:    {model}\n"
        f"API key:  {key_status}"
    )


def handle_agents() -> str:
    # Hand-rolled list — mirrors agents/ package contents.
    return (
        "Registered agents (v1.0):\n"
        "  Source:   arxiv-searcher, ss-searcher\n"
        "  Curation: candidate-merger, candidate-dedup, candidate-ranker\n"
        "  Process:  paper-processor (fanout)\n"
        "  Persist:  vault-writer, survey-composer\n"
        "  QA:       progress-reflector, answer-synthesizer\n"
        "  REPL:     intent-router"
    )


def handle_show(vault_path: Path, slug: str) -> str:
    # Search articles/ + surveys/ for the slug
    for cat in ("articles", "surveys"):
        path = vault_path / cat / f"{slug}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
    return f"slug {slug!r} not found in articles/ or surveys/."


def handle_help() -> str:
    return (
        "Slash commands:\n"
        "  /distill <topic> [--n N]        — single-pass: search and distill N papers\n"
        "  /ask <question>                 — multi-round QA loop\n"
        "  /resume <session-id>            — continue a paused QA session\n"
        "  /sessions                       — list past QA sessions\n"
        "  /vault                          — show vault stats\n"
        "  /provider                       — show LLM config\n"
        "  /agents                         — list registered agents\n"
        "  /show <slug>                    — display an article/survey from vault\n"
        "  /help                           — this list\n"
        "  /quit                           — exit REPL\n"
        "\n"
        "Natural language: type anything else (e.g. '帮我研究下扩散'),\n"
        "the intent-router will propose a command and confirm with you."
    )
```

### Step 4: Run tests, confirm pass

```bash
.venv\Scripts\python.exe -m pytest tests/chat/test_helpers.py -v
```

Expected: 9 passed (we have 9 tests; the table said 6 but the actual count after expansion is 9 — that's fine, just need them green).

### Step 5: Run full suite

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **170 passed** (161 + 9).

### Step 6: Commit

```bash
git add src/paper_distiller/chat/repl/helpers.py tests/chat/test_helpers.py
git commit -m "feat(chat): REPL read-only handlers (vault/sessions/provider/agents/show/help)"
```

---

## Task 4: REPL main loop

**Files:**
- Modify: `pyproject.toml` (add `prompt_toolkit>=3` dep)
- Create: `src/paper_distiller/chat/repl/loop.py`
- Create: `tests/chat/test_repl_loop.py`

The REPL class owns:
- `vault_path`, `console`, `llm` (lazy)
- The input loop (using prompt_toolkit)
- Dispatch logic — for slash commands calls the appropriate handler; for natural language calls IntentRouter, prints proposal, asks for confirmation, then dispatches.

### Step 1: Add `prompt_toolkit` to runtime deps

Edit `pyproject.toml`:

Find:
```toml
dependencies = [
    "httpx>=0.27",
    "arxiv>=2.1",
    "pymupdf>=1.24",
    "python-dotenv>=1.0",
    "rich>=13",
    "tomli>=2.0;python_version<'3.11'",
]
```

Replace with:
```toml
dependencies = [
    "httpx>=0.27",
    "arxiv>=2.1",
    "pymupdf>=1.24",
    "python-dotenv>=1.0",
    "rich>=13",
    "prompt_toolkit>=3",
    "tomli>=2.0;python_version<'3.11'",
]
```

Install:
```bash
.venv\Scripts\python.exe -m pip install -e . --quiet
```

### Step 2: Write the failing tests

Create `tests/chat/test_repl_loop.py`:

```python
"""Tests for REPL.dispatch — single-input dispatch logic, no actual stdin/stdout."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_repl_dispatch_quit_returns_quit_sentinel(tmp_path):
    from paper_distiller.chat.repl.loop import REPL
    r = REPL(vault_path=tmp_path)
    assert r.dispatch_one("/quit") == "QUIT"


def test_repl_dispatch_help_prints_commands(tmp_path, capsys):
    from paper_distiller.chat.repl.loop import REPL
    r = REPL(vault_path=tmp_path)
    r.dispatch_one("/help")
    captured = capsys.readouterr()
    assert "/distill" in captured.out
    assert "/ask" in captured.out


def test_repl_dispatch_vault_runs_handler(tmp_path, capsys):
    from paper_distiller.chat.repl.loop import REPL
    r = REPL(vault_path=tmp_path)
    r.dispatch_one("/vault")
    captured = capsys.readouterr()
    assert "articles:" in captured.out


def test_repl_dispatch_unknown_slash_prints_error(tmp_path, capsys):
    from paper_distiller.chat.repl.loop import REPL
    r = REPL(vault_path=tmp_path)
    r.dispatch_one("/nosuchcmd")
    captured = capsys.readouterr()
    assert "unknown" in captured.out.lower() or "unknown" in captured.err.lower()


def test_repl_dispatch_natural_language_uses_router(mocker, tmp_path, capsys, monkeypatch):
    """NL input → IntentRouter.classify → proposal print → user confirms 'n' → no action."""
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    fake_router_class = mocker.patch("paper_distiller.chat.repl.loop.IntentRouter")
    fake_router_class.return_value.classify.return_value = {
        "command": "ask",
        "params": {"question": "why diffusion?"},
        "missing_params": ["max_rounds", "per_round", "max_cost_cny"],
        "confidence": 8,
    }
    # Mock the confirmation prompt to return 'n' (cancel)
    mocker.patch("paper_distiller.chat.repl.loop._confirm", return_value=False)
    mocker.patch("paper_distiller.chat.repl.loop.LLMClient")
    from paper_distiller.chat.repl.loop import REPL
    r = REPL(vault_path=tmp_path)
    r.dispatch_one("why diffusion?")
    captured = capsys.readouterr()
    assert "Intent: ask" in captured.out
    assert "question" in captured.out.lower()
```

### Step 3: Run, confirm fail

```bash
.venv\Scripts\python.exe -m pytest tests/chat/test_repl_loop.py -v
```

Expected: ModuleNotFoundError.

### Step 4: Create `chat/repl/loop.py`

```python
"""REPL main class — input loop + dispatch.

dispatch_one(line) is testable without stdin; the run() method wires prompt_toolkit
+ stdin reading + dispatch in a loop.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from rich.console import Console

from ...llm.openai_compatible import LLMClient
from ...agents.router import IntentRouter, RoutingError
from .commands import KNOWN_COMMANDS, parse_slash, SlashError
from .helpers import (
    handle_agents, handle_help, handle_provider, handle_sessions,
    handle_show, handle_vault,
)


# Confirmation prompt is split out so tests can mock it.
def _confirm(prompt: str) -> bool:
    """Print prompt + read a Y/n response. Returns True if confirmed."""
    try:
        answer = input(prompt).strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    return answer in ("", "y", "yes")


_AGENT_DEFAULTS = {
    "ask": {"max_rounds": 3, "per_round": 2, "max_cost_cny": 5.0},
    "distill": {"n": 3},
}


def _format_proposal(parsed: dict) -> str:
    cmd = parsed["command"]
    params = parsed["params"]
    missing = parsed["missing_params"]
    lines = [f"[intent-router] Intent: {cmd}  | confidence {parsed.get('confidence', '?')}"]
    for k, v in params.items():
        lines.append(f"  {k}: {v}")
    if missing:
        defaults = _AGENT_DEFAULTS.get(cmd, {})
        applied = ", ".join(f"{k}={defaults.get(k, '?')}" for k in missing)
        lines.append(f"Missing: {missing}")
        lines.append(f"Apply defaults ({applied}) and run? [Y/n]")
    else:
        lines.append("Run? [Y/n]")
    return "\n".join(lines)


class REPL:
    def __init__(self, vault_path: Path):
        self.vault_path = Path(vault_path)
        self.console = Console()
        self._llm = None  # lazy

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLMClient(
                os.getenv("PD_API_KEY"),
                os.getenv("PD_BASE_URL"),
                os.getenv("PD_MODEL"),
            )
        return self._llm

    def dispatch_one(self, line: str) -> str | None:
        """Dispatch a single input line. Returns 'QUIT' to signal exit, else None."""
        line = (line or "").strip()
        if not line:
            return None
        if line.startswith("/"):
            return self._dispatch_slash(line)
        return self._dispatch_natural_language(line)

    def _dispatch_slash(self, line: str) -> str | None:
        try:
            parsed = parse_slash(line)
        except SlashError as e:
            print(f"Error: {e}")
            return None
        if parsed.name == "quit":
            return "QUIT"
        if parsed.name == "help":
            print(handle_help())
            return None
        if parsed.name == "vault":
            print(handle_vault(self.vault_path))
            return None
        if parsed.name == "sessions":
            print(handle_sessions(self.vault_path))
            return None
        if parsed.name == "provider":
            print(handle_provider())
            return None
        if parsed.name == "agents":
            print(handle_agents())
            return None
        if parsed.name == "show":
            if not parsed.args:
                print("Usage: /show <slug>")
                return None
            print(handle_show(self.vault_path, parsed.args[0]))
            return None
        # Action commands — defer to dispatch_action (delegates to cli.main's handlers)
        return self._dispatch_action(parsed.name, parsed.args)

    def _dispatch_action(self, name: str, args: list) -> None:
        """Run a slash action command by invoking cli.main with synthetic argv."""
        from ..cli import main as cli_main
        cli_argv = [name, "--vault", str(self.vault_path), *args]
        try:
            rc = cli_main(cli_argv)
        except SystemExit as e:
            # argparse error inside cli; print clean message
            print(f"  (cli exited with code {e.code})")
            return None
        if rc != 0:
            print(f"  (cli returned exit code {rc})")
        return None

    def _dispatch_natural_language(self, line: str) -> None:
        try:
            router = IntentRouter(llm=self.llm)
            parsed = router.classify(line)
        except RoutingError as e:
            print(f"Intent routing failed: {e}")
            return None
        print(_format_proposal(parsed))
        if not _confirm("> "):
            print("  (cancelled)")
            return None
        # Apply defaults for missing params
        cmd = parsed["command"]
        params = dict(parsed["params"])
        for k in parsed["missing_params"]:
            if k in _AGENT_DEFAULTS.get(cmd, {}):
                params[k] = _AGENT_DEFAULTS[cmd][k]
        # Build argv and run
        argv = self._params_to_argv(cmd, params)
        return self._dispatch_action(cmd, argv[1:])  # skip cmd, dispatch_action prepends

    def _params_to_argv(self, cmd: str, params: dict) -> list[str]:
        """Translate a {name: value} param dict into CLI args."""
        argv = [cmd]
        if cmd == "distill":
            if "topic" in params:
                argv += ["--topic", str(params["topic"])]
            if "n" in params:
                argv += ["--n", str(params["n"])]
        elif cmd == "ask":
            if "question" in params:
                argv += ["--question", str(params["question"])]
            if "max_rounds" in params:
                argv += ["--max-rounds", str(params["max_rounds"])]
            if "per_round" in params:
                argv += ["--per-round", str(params["per_round"])]
            if "max_cost_cny" in params:
                argv += ["--max-cost-cny", str(params["max_cost_cny"])]
        elif cmd == "resume":
            if "session_id" in params:
                argv += ["--session-id", str(params["session_id"])]
        return argv

    def run(self) -> int:
        """Launch the interactive REPL. Returns 0 on clean exit."""
        self._print_banner()
        session = PromptSession(
            completer=WordCompleter(
                ["/" + c for c in KNOWN_COMMANDS],
                ignore_case=True,
            ),
        )
        while True:
            try:
                line = session.prompt("> ")
            except (EOFError, KeyboardInterrupt):
                print("  (bye)")
                return 0
            result = self.dispatch_one(line)
            if result == "QUIT":
                print("  (bye)")
                return 0

    def _print_banner(self):
        from ... import __version__
        from .helpers import handle_provider
        self.console.print("─" * 60)
        self.console.print(f"[bold]paper-distiller v{__version__}[/bold]")
        provider_line = handle_provider().splitlines()[2]  # the Model: line
        self.console.print(provider_line)
        self.console.print(f"Vault: {self.vault_path}")
        self.console.print("")
        self.console.print("Slash commands: /distill /ask /resume /sessions /vault /provider /agents /show /help /quit")
        self.console.print("Natural language: '帮我研究下扩散'")
        self.console.print("─" * 60)
```

### Step 5: Run tests, confirm pass

```bash
.venv\Scripts\python.exe -m pytest tests/chat/test_repl_loop.py -v
```

Expected: 5 passed (the table said 4; we wrote 5 tests). Adjust if needed.

### Step 6: Run full suite

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **175 passed** (170 + 5).

### Step 7: Commit

```bash
git add pyproject.toml src/paper_distiller/chat/repl/loop.py tests/chat/test_repl_loop.py
git commit -m "feat(chat): REPL main loop + intent-router dispatch

REPL.dispatch_one(line) is the testable single-input handler.
Slash commands dispatch directly (helpers + cli.main handoff);
natural language goes through IntentRouter + confirmation prompt.

run() wraps prompt_toolkit session for line editing + tab completion."
```

---

## Task 5: Wire REPL into chat/cli.py

**Files:**
- Modify: `src/paper_distiller/chat/cli.py`
- Create: `tests/chat/test_repl_cli.py`

When `paper-distiller-chat` is run without a subcommand, launch the REPL. The `--vault` flag becomes a top-level (global) arg.

### Step 1: Write the failing tests

Create `tests/chat/test_repl_cli.py`:

```python
"""Tests for paper-distiller-chat (no subcommand → REPL)."""
import pytest


def test_chat_no_subcommand_launches_repl(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    fake_repl = mocker.patch("paper_distiller.chat.cli.REPL")
    fake_repl.return_value.run.return_value = 0
    from paper_distiller.chat.cli import main
    rc = main(["--vault", str(tmp_path)])
    assert rc == 0
    fake_repl.assert_called_once()
    # REPL constructor receives the vault path
    call_kw = fake_repl.call_args.kwargs
    assert "vault_path" in call_kw or len(fake_repl.call_args.args) > 0


def test_chat_no_subcommand_no_vault_returns_error():
    from paper_distiller.chat.cli import main
    rc = main([])  # no --vault, no subcommand
    assert rc == 2  # argparse error
```

### Step 2: Modify `src/paper_distiller/chat/cli.py`

Make the subparsers optional and add a global `--vault` flag. The current parser uses `sub.add_subparsers(dest="subcommand", required=True)` — change `required` to False and add a default-path branch.

Find:
```python
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper-distiller-chat",
        description="Chat-first paper distillation. Plan-1 subset: one-shot `distill`.",
    )
    sub = p.add_subparsers(dest="subcommand", required=True)
```

Replace with:
```python
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper-distiller-chat",
        description="Chat-first paper distillation.",
    )
    p.add_argument("--vault", help="Vault path (used when launching REPL without subcommand)")
    sub = p.add_subparsers(dest="subcommand", required=False)
```

Update main():

```python
def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.subcommand == "distill":
        return asyncio.run(_run_distill(args))
    if args.subcommand == "ask":
        return _run_ask(args)
    if args.subcommand == "resume":
        return _run_resume(args)
    # No subcommand: launch REPL (need --vault)
    if not getattr(args, "vault", None):
        print("Error: --vault is required when launching REPL", file=sys.stderr)
        return 2
    from .repl.loop import REPL
    repl = REPL(vault_path=args.vault)
    return repl.run()
```

Add the import alias near the top:
```python
from .repl.loop import REPL
```

### Step 3: Run tests, confirm pass

```bash
.venv\Scripts\python.exe -m pytest tests/chat/test_repl_cli.py -v
```

Expected: 2 passed.

### Step 4: Run full suite + smoke

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
.venv\Scripts\paper-distiller-chat.exe --help
```

Expected: **177 passed** (175 + 2). Help shows `--vault` flag + subcommands.

### Step 5: Commit

```bash
git add src/paper_distiller/chat/cli.py tests/chat/test_repl_cli.py
git commit -m "feat(chat): paper-distiller-chat (no subcommand) launches REPL"
```

---

## Task 6: End-to-end REPL integration test

**Files:**
- Create: `tests/integration/test_repl_e2e.py`

### Step 1: Write the test

Create `tests/integration/test_repl_e2e.py`:

```python
"""End-to-end test for REPL: feed a sequence of inputs, verify behavior."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_repl_handles_help_then_vault_then_quit(tmp_path, mocker, capsys, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    # Set up a tiny vault
    (tmp_path / "articles").mkdir()
    (tmp_path / "articles" / "a.md").write_text(
        "---\ntitle: A\n---\n", encoding="utf-8",
    )

    from paper_distiller.chat.repl.loop import REPL
    r = REPL(vault_path=tmp_path)
    assert r.dispatch_one("/help") is None
    assert r.dispatch_one("/vault") is None
    assert r.dispatch_one("/quit") == "QUIT"

    captured = capsys.readouterr()
    assert "/distill" in captured.out  # from /help
    assert "articles: 1" in captured.out  # from /vault
```

### Step 2: Run, confirm pass

```bash
.venv\Scripts\python.exe -m pytest tests/integration/test_repl_e2e.py -v
```

Expected: 1 passed.

### Step 3: Run full suite

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **178 passed** (177 + 1).

### Step 4: Commit

```bash
git add tests/integration/test_repl_e2e.py
git commit -m "test(chat): REPL end-to-end smoke (help/vault/quit sequence)"
```

---

## Task 7: Plan-3 wrap-up + push

- [ ] **Step 1: Full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **178 passed** (151 baseline + 27 new).

- [ ] **Step 2: Verify all 6 CLIs respond**

```powershell
.\.venv\Scripts\paper-distiller.exe --help
.\.venv\Scripts\paper-distiller-qa.exe --help
.\.venv\Scripts\paper-distiller-chat.exe --help
.\.venv\Scripts\paper-distiller-chat.exe distill --help
.\.venv\Scripts\paper-distiller-chat.exe ask --help
.\.venv\Scripts\paper-distiller-chat.exe resume --help
```

- [ ] **Step 3: Manual REPL smoke (NO real LLM)**

Type `/help` → see commands. `/vault` → see counts. `/quit` → exit.

```powershell
.\.venv\Scripts\paper-distiller-chat.exe --vault "G:\Math research Agent\wiki"
```

Then type:
```
> /help
> /vault
> /sessions
> /provider
> /agents
> /quit
```

All should respond without errors. (No NL → no LLM call, no API spend.)

- [ ] **Step 4: Push to GitHub**

```bash
git push origin main
```

- [ ] **Step 5: Confirm CI green**

Open https://github.com/jesson-hh/paper-distiller/actions — verify the matrix passes on Python 3.10/3.11/3.12.

---

## Plan-3 success criteria

- [ ] All 7 tasks done
- [ ] 178 tests passing (151 baseline + 27 new)
- [ ] `paper-distiller-chat --vault X` launches REPL with welcome banner
- [ ] Slash commands all dispatch correctly: `/help`, `/vault`, `/sessions`, `/provider`, `/agents`, `/show`, `/distill`, `/ask`, `/resume`, `/quit`
- [ ] Natural language input triggers IntentRouter + confirmation prompt
- [ ] Old `paper-distiller` and `paper-distiller-qa` CLIs unchanged + working
- [ ] CI green on all Python versions
- [ ] One new top-level dep: `prompt_toolkit>=3`

Plan 4 (cleanup + v1.0.0 release) is the final plan.
