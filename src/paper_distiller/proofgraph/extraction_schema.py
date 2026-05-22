"""Extraction JSON schema dataclasses + tolerant parser.

The LLM is asked to return JSON matching the contract below.  This module
defines the in-memory dataclasses and a ``parse_extraction`` function that
converts the raw LLM string (or already-decoded dict) to a list of
``ExtractedNode`` objects, dropping any node that is missing required fields
and coercing unexpected types gracefully.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ExtractedRef:
    """A reference from one node to another (labelled, not yet resolved to an ID)."""
    rel: str    # "depends_on" | "uses_lemma" | "uses_def" | "uses_assumption"
    target: str  # label string e.g. "Lemma 3.1"


@dataclass
class ExtractedNode:
    """One node extracted by the LLM from a segment.

    ``status`` is mutable: the grounding gate sets it to ``"unsupported"`` for
    nodes whose ``source_quote`` cannot be verified; the self-check pass sets it
    to ``"suspicious"`` for nodes that claim beyond the segment text.

    ``key`` is a short local identifier (e.g. ``"n1"``) assigned by the LLM
    within a single extraction response, used to resolve intra-proof step refs
    without requiring a formal label.
    """
    kind: str
    text: str
    source_quote: str
    label: str | None = None
    key: str | None = None
    techniques: list[str] = field(default_factory=list)
    refs: list[ExtractedRef] = field(default_factory=list)
    status: str = "extracted"


_VALID_KINDS = {
    "theorem", "lemma", "definition", "assumption", "proof_step", "claim",
}


def _parse_node(raw: dict) -> ExtractedNode | None:
    """Parse one raw dict into an ExtractedNode, or return None if invalid."""
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    text = raw.get("text")
    source_quote = raw.get("source_quote")
    # Required fields
    if not (kind and isinstance(kind, str) and
            text and isinstance(text, str) and
            source_quote and isinstance(source_quote, str)):
        return None

    label = raw.get("label")
    if not isinstance(label, str) or not label.strip():
        label = None

    key = raw.get("key")
    if not isinstance(key, str) or not key.strip():
        key = None

    # techniques: must be list of strings; coerce otherwise
    raw_techs = raw.get("techniques")
    if isinstance(raw_techs, list):
        techniques = [t for t in raw_techs if isinstance(t, str)]
    else:
        techniques = []

    # refs: must be list of dicts with rel + target; coerce otherwise
    raw_refs = raw.get("refs")
    refs: list[ExtractedRef] = []
    if isinstance(raw_refs, list):
        for r in raw_refs:
            if not isinstance(r, dict):
                continue
            rel = r.get("rel")
            target = r.get("target")
            if isinstance(rel, str) and isinstance(target, str) and rel and target:
                refs.append(ExtractedRef(rel=rel, target=target))

    return ExtractedNode(
        kind=kind,
        text=text,
        source_quote=source_quote,
        label=label,
        key=key,
        techniques=techniques,
        refs=refs,
        status="extracted",
    )


def parse_extraction(raw: str | dict) -> list[ExtractedNode]:
    """Parse the LLM extraction output into a list of ``ExtractedNode``.

    Accepts:
    - A JSON string (will be decoded).
    - An already-decoded dict.

    Returns an empty list on any parse error or if the structure is unexpected.
    Individual invalid nodes are silently skipped.
    """
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
    elif isinstance(raw, dict):
        data = raw
    else:
        return []

    if not isinstance(data, dict):
        return []

    nodes_raw = data.get("nodes")
    if not isinstance(nodes_raw, list):
        return []

    result: list[ExtractedNode] = []
    for item in nodes_raw:
        node = _parse_node(item)
        if node is not None:
            result.append(node)
    return result
