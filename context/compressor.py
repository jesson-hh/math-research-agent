import json

from config import CONTEXT_THRESHOLD


def estimate_tokens(messages: list) -> int:
    """Rough token estimate: ~4 chars per token."""
    total = sum(len(json.dumps(m, default=str)) for m in messages)
    return total // 4


def maybe_compress(messages: list, threshold: int = CONTEXT_THRESHOLD) -> list:
    """
    If estimated tokens exceed threshold, truncate the content of old tool
    results (keeping the most recent 4 messages intact).
    """
    if estimate_tokens(messages) <= threshold:
        return messages

    keep_recent = 4  # Always preserve the last N messages in full
    cutoff = max(0, len(messages) - keep_recent)

    compressed = []
    for i, msg in enumerate(messages):
        if i >= cutoff:
            # Keep recent messages untouched
            compressed.append(msg)
            continue

        if msg["role"] == "user" and isinstance(msg["content"], list):
            new_content = []
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content_str = str(block.get("content", ""))
                    if len(content_str) > 400:
                        new_content.append({
                            **block,
                            "content": _smart_truncate(content_str),
                        })
                    else:
                        new_content.append(block)
                else:
                    new_content.append(block)
            compressed.append({**msg, "content": new_content})
        else:
            compressed.append(msg)

    return compressed


def _smart_truncate(content_str: str, max_len: int = 400) -> str:
    """Intelligently truncate tool result content, preserving JSON structure."""
    if len(content_str) <= max_len:
        return content_str

    # Try to parse as JSON and keep structure with shortened values
    try:
        data = json.loads(content_str)
        if isinstance(data, dict):
            summary = {}
            for k, v in data.items():
                if isinstance(v, list) and k in ("papers", "steps", "results"):
                    summary[k] = f"[{len(v)} items]"
                elif isinstance(v, str) and len(v) > 100:
                    summary[k] = v[:100] + "..."
                else:
                    summary[k] = v
            result = json.dumps(summary, ensure_ascii=False, default=str)
            if len(result) <= max_len:
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    return content_str[:max_len - 30] + "\n...[truncated for context management]"
