"""Plan-mode preview cards for expensive tool calls.

When a tool's estimated cost exceeds `PD_PLAN_THRESHOLD_CNY` (default ¥10),
the AgentLoop pauses before execution, prints a structured plan card listing
the tool name + arguments + cost estimate, and waits for the user to press
Enter (proceed) or 'q' (cancel). After `countdown_sec` of inactivity, the
plan auto-proceeds.

Override threshold via env var `PD_PLAN_THRESHOLD_CNY`. Auto-mode (`/auto`
slash command) bypasses plan-mode entirely.
"""

from __future__ import annotations

import os
import sys
import threading

from .cost_estimator import estimate_tool_cost_cny


def should_show_plan(name: str, arguments: dict) -> bool:
    """True if this tool call's estimated cost exceeds the plan-mode threshold."""
    threshold = float(os.getenv("PD_PLAN_THRESHOLD_CNY", "10.0"))
    return estimate_tool_cost_cny(name, arguments) >= threshold


def render_plan_card(
    name: str, arguments: dict, estimated_cost_cny: float
) -> str:
    """Build the plan-card display string. Pure function; no I/O."""
    lines = [f"plan: {name}  ·  estimated ¥{estimated_cost_cny:.2f} budget"]
    for k, v in (arguments or {}).items():
        val = v if isinstance(v, str) else repr(v)
        if len(val) > 60:
            val = val[:57] + "..."
        lines.append(f"  {k}: {val}")
    return "\n".join(lines)


def confirm_plan(
    name: str,
    arguments: dict,
    estimated_cost_cny: float,
    countdown_sec: int = 5,
) -> bool:
    """Display plan card + wait for user input. Return True to proceed.

    Behavior:
    - Enter / empty input → proceed (True)
    - 'q' / 'quit' / 'cancel' / 'n' / 'no' → cancel (False)
    - countdown elapses → proceed (True)
    - Ctrl-C / EOF → cancel (False)
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    card_body = render_plan_card(name, arguments, estimated_cost_cny)
    if countdown_sec > 0:
        prompt_hint = (
            f"Enter to proceed (auto-proceed in {countdown_sec}s) · "
            "q to cancel · /auto to disable previews"
        )
    else:
        prompt_hint = "Enter to proceed · q to cancel"
    console.print(Panel(
        card_body + "\n\n" + prompt_hint,
        title="[bold yellow]PLAN MODE[/bold yellow]",
        border_style="yellow",
    ))

    if countdown_sec <= 0:
        try:
            raw = input("  proceed? [Y/n/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return raw not in ("q", "quit", "cancel", "n", "no")

    answer: dict = {"value": None}

    def _read():
        try:
            answer["value"] = input(f"  proceed? [Y/n/q] ({countdown_sec}s): ")
        except (EOFError, KeyboardInterrupt):
            answer["value"] = "__cancel__"

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout=countdown_sec)
    if t.is_alive():
        print(file=sys.stderr)
        return True
    raw = (answer["value"] or "").strip().lower()
    if raw == "__cancel__":
        return False
    return raw not in ("q", "quit", "cancel", "n", "no")
