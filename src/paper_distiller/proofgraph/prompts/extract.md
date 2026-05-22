You are a mathematical knowledge extractor. Your task is to extract structured nodes from the following paper segment.

## Running memory (what has been established so far)
{memory}

## Current segment
Kind hint: {kind_hint}
Section: {section}

```
{segment_text}
```

## Extraction depth
Depth: {depth}

## Instructions

Extract all mathematically significant assertions from THIS segment only. Do NOT reference content from other segments or papers.

Return ONLY a JSON object with this exact structure — no prose, no markdown fences:

{{"nodes": [
  {{
    "kind": "<theorem|lemma|definition|assumption|proof_step|claim>",
    "key": "<short unique id within THIS response, e.g. 'n1', 'n2', ...>",
    "label": "<label if stated, e.g. 'Theorem 4.3', or null>",
    "text": "<normalized assertion in your own words>",
    "source_quote": "<VERBATIM text copied character-for-character from the segment above>",
    "techniques": ["<technique name>", ...],
    "refs": [
      {{"rel": "<depends_on|uses_lemma|uses_def|uses_assumption>", "target": "<key or label>"}}
    ]
  }}
]}}

## Critical rules (violation = discarded node)

1. `source_quote` MUST be a verbatim substring copied from the segment text above. Do NOT paraphrase, summarize, or invent. Copy the exact characters.
2. If you cannot find a verbatim quote for a claim, OMIT that node entirely. Abstain rather than fabricate.
3. Each node MUST have a short unique `key` (e.g. `"n1"`, `"n2"`, …) that is unique within this single response. This allows other nodes in the same response to reference it.
4. For `refs`, set `target` to EITHER:
   - the `key` of another node in THIS SAME response (use this for steps depending on earlier steps in the same proof), OR
   - the label of a previously-established result visible in the running memory (e.g. `"Theorem 1"`, `"Lemma 2.3"`) for cross-segment dependencies.
   Do not invent labels or keys that do not exist.
5. For `depth=theorem`: extract only theorem/lemma/definition/claim nodes; skip proof_step decomposition.
6. For `depth=step`: fully decompose proof blocks into individual proof_step nodes, each with its own verbatim quote.
7. Return an empty nodes list `{{"nodes": []}}` if there is nothing to extract from this segment.
8. Do not include commentary, explanation, or any text outside the JSON object.
