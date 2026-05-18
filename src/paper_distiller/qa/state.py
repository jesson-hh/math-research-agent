"""SessionState dataclass + on-disk persistence for the QA loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class RoundRecord:
    round: int
    query: str
    rationale: str
    candidates_found: int
    new_articles: int
    article_slugs: list
    what_we_know: str
    what_is_missing: str
    confidence: int
    timestamp: str


@dataclass
class SessionState:
    session_id: str
    question: str
    config_snapshot: dict
    started_at: str

    rounds_completed: int = 0
    articles_distilled: list = field(default_factory=list)  # list of dicts (article kwargs)
    articles_seen_ids: set = field(default_factory=set)     # arxiv_id ∪ doi
    history: list = field(default_factory=list)             # list[RoundRecord]
    last_reflection: dict | None = None

    cost_cny: float = 0.0
    tokens_in_total: int = 0
    tokens_out_total: int = 0

    is_done: bool = False
    stop_reason: str = ""


def _session_dir(vault_path: Path, session_id: str) -> Path:
    return vault_path / ".paper_distiller" / "qa-sessions" / session_id


def write_state(vault_path: Path, state: SessionState) -> None:
    """Persist the latest SessionState snapshot to <vault>/.paper_distiller/qa-sessions/<sid>/state.json."""
    session_dir = _session_dir(vault_path, state.session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    payload = asdict(state)
    payload["articles_seen_ids"] = sorted(state.articles_seen_ids)

    (session_dir / "state.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_state(vault_path: Path, session_id: str) -> SessionState | None:
    """Read a previously persisted SessionState. Returns None if not found."""
    state_path = _session_dir(vault_path, session_id) / "state.json"
    if not state_path.exists():
        return None
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    raw["articles_seen_ids"] = set(raw.get("articles_seen_ids") or [])
    history_raw = raw.get("history") or []
    raw["history"] = [RoundRecord(**r) for r in history_raw]
    return SessionState(**raw)
