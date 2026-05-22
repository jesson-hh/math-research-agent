"""Deterministic, LLM-free reading primitives for the proof-graph pipeline.

- segment(text): split a paper's plain text into ordered Segments, flagging
  theorem-statement and proof-block regions. This is the coverage denominator
  for "don't skim" — every segment must later be visited.
- verify_quote(quote, segment_text): the grounding gate (Task 9).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class Segment:
    id: int
    kind_hint: str          # "prose" | "theorem" | "proof" | "definition" | "heading"
    section: str | None     # nearest preceding heading label, e.g. "2 Main Result"
    text: str
    char_start: int
    char_end: int
    is_proof_block: bool


# A heading line: "1 Introduction", "2.1 Setup", "Appendix A", etc.
_HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*\s+\S.*|Appendix\s+\S.*)$")
# Start of a theorem-like statement.
_THEOREM_RE = re.compile(
    r"^\s*(Theorem|Lemma|Proposition|Corollary|Claim|Definition)\b", re.IGNORECASE)
_PROOF_START_RE = re.compile(r"^\s*Proof\b", re.IGNORECASE)


def _classify(block: str) -> str:
    head = block.lstrip()
    if _PROOF_START_RE.match(head):
        return "proof"
    if _THEOREM_RE.match(head):
        first = head.split(None, 1)[0].lower()
        return "definition" if first.startswith("defin") else "theorem"
    if _HEADING_RE.match(head.splitlines()[0] if head else ""):
        return "heading"
    return "prose"


def segment(text: str) -> list[Segment]:
    """Split into structural blocks — a new block starts at each heading /
    Theorem-like / Proof line, and at blank-line paragraph breaks. Classify
    each block and record char offsets so downstream code can reconstruct and
    ground to source. This list is the coverage denominator for "don't skim"."""
    if not text or not text.strip():
        return []
    lines = text.splitlines(keepends=True)
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln)

    segments: list[Segment] = []
    cur: list[int] = []
    state = {"sid": 0, "section": None}

    def _is_boundary(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        return bool(_HEADING_RE.match(s) or _THEOREM_RE.match(s)
                    or _PROOF_START_RE.match(s))

    def _flush() -> None:
        if not cur:
            return
        start = offsets[cur[0]]
        end = offsets[cur[-1]] + len(lines[cur[-1]])
        block = text[start:end]
        cur.clear()
        if not block.strip():
            return
        kind = _classify(block)
        if kind == "heading":
            state["section"] = block.strip().splitlines()[0].strip()
        segments.append(Segment(
            id=state["sid"], kind_hint=kind, section=state["section"],
            text=block, char_start=start, char_end=end,
            is_proof_block=bool(_PROOF_START_RE.match(block.lstrip())),
        ))
        state["sid"] += 1

    for i, ln in enumerate(lines):
        if not ln.strip():
            _flush()
            continue
        if cur and _is_boundary(ln):
            _flush()
        cur.append(i)
    _flush()
    return segments


@dataclass
class GateResult:
    ok: bool
    score: float            # 1.0 = exact (after whitespace norm); else best fuzzy ratio
    matched_span: str | None  # the source substring that best matches, if any


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def verify_quote(
    quote: str, segment_text: str, threshold: float = 0.85,
) -> GateResult:
    """The grounding gate. Returns ok=True iff `quote` is found in
    `segment_text` exactly (after whitespace normalization) or with a best
    fuzzy ratio >= threshold over a sliding window. Fabricated quotes that
    aren't really in the source score low and are rejected — this is what
    structurally keeps hallucinated nodes out of the graph.
    """
    q = _norm_ws(quote)
    if not q:
        return GateResult(ok=False, score=0.0, matched_span=None)
    hay = _norm_ws(segment_text)
    if q in hay:
        return GateResult(ok=True, score=1.0, matched_span=q)
    # Fuzzy: slide fixed-size windows (by word boundaries) across the haystack.
    # We use two target character lengths — qlen and 1.3*qlen — so we don't
    # miss the best alignment when the quote has slightly different whitespace
    # than the source.  ONE SequenceMatcher is reused across all windows
    # (set_seq1 is called once; set_seq2 is called per window), avoiding the
    # O(words²) fresh-construction cost of the original loop.
    words = hay.split(" ")
    qlen = len(q)
    target_lens = (qlen, int(1.3 * qlen))
    best = 0.0
    best_span: str | None = None

    # Pre-build the word-start character offsets so we can slice windows
    # directly from `hay` rather than re-joining word lists each iteration.
    word_starts: list[int] = []
    pos = 0
    for w in words:
        word_starts.append(pos)
        pos += len(w) + 1  # +1 for the space separator

    sm = SequenceMatcher(autojunk=False)
    sm.set_seq1(q)
    # Length-ratio bound: ratio() <= 2*min(|a|,|b|)/(|a|+|b|), so if
    # len(window)/qlen is too far from 1 the score cannot reach `threshold`.
    # We derive the allowed window-length range from this inequality.
    lo_len = int(qlen * threshold / (2.0 - threshold)) + 1
    hi_len = int(qlen * (2.0 - threshold) / threshold)

    nw = len(words)
    for target in target_lens:
        # For each word-start position i, find the ending word index j such
        # that the window [word_starts[i], word_starts[j]+len(words[j])]
        # has character length >= target.  Keep a running right pointer to
        # avoid O(words²) inner loops.
        j = 0
        for i in range(nw):
            if j < i:
                j = i
            # Advance j until window length >= target or we exhaust words.
            while j < nw - 1 and (word_starts[j] + len(words[j]) - word_starts[i]) < target:
                j += 1
            win_end = word_starts[j] + len(words[j])
            win_len = win_end - word_starts[i]
            # Length-ratio pre-filter: skip windows whose length cannot
            # possibly produce a ratio >= max(best, threshold).
            if win_len < lo_len or win_len > hi_len:
                continue
            window = hay[word_starts[i]:win_end]
            if not window:
                continue
            sm.set_seq2(window)
            if sm.quick_ratio() < max(best, threshold):
                continue
            ratio = sm.ratio()
            if ratio > best:
                best, best_span = ratio, window
                if best == 1.0:
                    break
        if best == 1.0:
            break

    return GateResult(ok=best >= threshold, score=round(best, 3),
                      matched_span=best_span if best >= threshold else None)
