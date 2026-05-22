"""Structured running memory carried across segments during extraction.

``RunningMemory`` collects what has been established so far (definitions,
theorems/lemmas, notation) and what is still unresolved (obligations).
The ``render()`` method produces a compact text block injected into the
per-segment extraction prompt so the LLM has local context without seeing
the entire paper again.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Cap on each list: keeps ``render()`` output bounded and the injected prompt
# token-efficient.  Only the *most recent* items are kept.
_CAP = 20


@dataclass
class RunningMemory:
    """Structured carry-forward state across segment extractions.

    Fields:
    - notation: symbol → meaning mapping accumulated from definitions.
    - definitions: compact dicts {"label":..., "text":...} (last _CAP).
    - established: theorems/lemmas seen so far (last _CAP).
    - obligations: labels referenced but not yet defined/established (deduped).
    """
    notation: dict[str, str] = field(default_factory=dict)
    definitions: list[dict] = field(default_factory=list)
    established: list[dict] = field(default_factory=list)
    obligations: list[str] = field(default_factory=list)

    def update(
        self,
        nodes: list,  # list[ExtractedNode] (avoid circular import at type level)
        resolved_labels: set[str],
    ) -> None:
        """Incorporate newly extracted nodes into running memory.

        - definition nodes → ``definitions``
        - theorem/lemma/proposition/corollary nodes → ``established``
        - any ref target not in ``resolved_labels`` → ``obligations`` (deduped)

        Lists are capped at ``_CAP`` (most-recent retained).
        """
        for node in nodes:
            kind = (node.kind or "").lower()
            label = node.label
            text = node.text

            entry = {"label": label, "text": text[:200] if text else ""}

            if kind == "definition":
                self.definitions.append(entry)
                if len(self.definitions) > _CAP:
                    self.definitions = self.definitions[-_CAP:]

            elif kind in {"theorem", "lemma", "proposition", "corollary", "claim"}:
                self.established.append(entry)
                if len(self.established) > _CAP:
                    self.established = self.established[-_CAP:]

            # Collect unresolved references as obligations
            for ref in (node.refs or []):
                target = ref.target
                if target and target not in resolved_labels:
                    if target not in self.obligations:
                        self.obligations.append(target)

    def render(self) -> str:
        """Return a compact, human-readable text block for prompt injection.

        Caps each section to keep token cost bounded.
        """
        lines: list[str] = []

        if self.established:
            lines.append("## Established results")
            for entry in self.established[-_CAP:]:
                lbl = entry.get("label") or "(unlabelled)"
                txt = entry.get("text") or ""
                lines.append(f"- {lbl}: {txt[:150]}")

        if self.definitions:
            lines.append("## Definitions")
            for entry in self.definitions[-_CAP:]:
                lbl = entry.get("label") or "(unlabelled)"
                txt = entry.get("text") or ""
                lines.append(f"- {lbl}: {txt[:150]}")

        if self.obligations:
            lines.append("## Open obligations (referenced but unresolved)")
            for ob in self.obligations[-_CAP:]:
                lines.append(f"- {ob}")

        return "\n".join(lines)
