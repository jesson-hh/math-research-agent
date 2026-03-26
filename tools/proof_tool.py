import json
import re

from config import PROOF_MODEL, PROOF_MAX_TOKENS

PROOF_SYSTEM_PROMPT = """You are a rigorous mathematical proof assistant with expertise across all areas of mathematics.

Given a theorem and context, you will:
1. Identify the most appropriate proof strategy
2. Decompose the proof into clear, numbered steps
3. For each step, provide: the claim, the mathematical justification, and any definitions used
4. Identify any gaps, assumptions, or sub-lemmas that need separate proofs
5. Assess your confidence in the proof's correctness

Return your response as a JSON object with this exact structure:
{
  "theorem": "<restate the theorem precisely>",
  "strategy": "<chosen proof strategy and why>",
  "key_ideas": ["<main insight 1>", "<main insight 2>"],
  "steps": [
    {
      "step_num": 1,
      "claim": "<what we show in this step>",
      "justification": "<mathematical reasoning>",
      "definitions_used": ["<def 1>", "<def 2>"]
    }
  ],
  "gaps": ["<any unproven assumptions or gaps>"],
  "needed_lemmas": ["<lemma 1 that needs separate proof>"],
  "confidence": "high|medium|low",
  "notes": "<any additional mathematical insights>"
}

Be rigorous. Clearly distinguish between what is proven and what is assumed."""


def proof_assist(
    theorem: str,
    context: str = "",
    strategy: str = "auto",
    mode: str = "detailed",
) -> dict:
    from llm import get_client

    client = get_client(model_override=PROOF_MODEL)

    mode_instruction = {
        "outline": "Provide a high-level outline with 3-6 key steps only. Keep steps brief.",
        "detailed": "Provide a complete, detailed proof with full justifications for each step.",
        "lemmas": "Focus on identifying all lemmas and sub-results needed. List them with brief descriptions.",
    }.get(mode, "Provide a complete detailed proof.")

    strategy_instruction = (
        f"Use the '{strategy}' proof strategy."
        if strategy != "auto"
        else "Choose the most elegant and appropriate proof strategy."
    )

    user_content = f"""Theorem: {theorem}

Mathematical context/known results:
{context or "Standard mathematical axioms and well-known theorems may be used freely."}

Instructions:
- {strategy_instruction}
- {mode_instruction}
- Return your response as valid JSON matching the specified structure."""

    result = client.chat(
        system=PROOF_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        max_tokens=PROOF_MAX_TOKENS,
    )

    # Extract text from content blocks
    text = ""
    for block in result["content_blocks"]:
        if block["type"] == "text":
            text += block["text"]

    # Try to parse JSON from response
    proof_data = _extract_json(text)
    if proof_data is None:
        proof_data = {
            "theorem": theorem,
            "strategy": strategy,
            "raw_proof": text,
            "confidence": "unknown",
        }

    return proof_data


def _extract_json(text: str) -> dict | None:
    """Extract the first complete JSON object from text, handling nested braces."""
    # Try fenced code block first
    fence_match = re.search(r"```(?:json)?\s*", text)
    if fence_match:
        result = _find_balanced_json(text, fence_match.end())
        if result is not None:
            return result

    # Try raw JSON
    brace_idx = text.find("{")
    if brace_idx >= 0:
        result = _find_balanced_json(text, brace_idx)
        if result is not None:
            return result

    # Last resort: try parsing entire text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _find_balanced_json(text: str, start: int) -> dict | None:
    """Find a balanced JSON object starting from position start and parse it."""
    idx = text.find("{", start)
    if idx < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(idx, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[idx:i + 1])
                except json.JSONDecodeError:
                    return None
    return None
