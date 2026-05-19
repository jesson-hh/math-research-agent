"""Per-tool cost estimates used by plan_mode.should_show_plan."""

from __future__ import annotations


def estimate_tool_cost_cny(name: str, arguments: dict) -> float:
    """Conservative pre-execution cost estimate for a tool call."""
    args = arguments or {}
    if name == "research":
        return float(args.get("max_cost_cny", 15.0))
    if name == "ask":
        return float(args.get("max_cost_cny", 5.0))
    if name == "distill_by_id":
        ids = args.get("ids", []) or []
        return 0.2 * len(ids)
    if name == "search":
        return 0.05
    return 0.0
