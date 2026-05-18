"""paper-distiller-qa command-line entry point."""

from __future__ import annotations

import argparse
import sys

from ..config import load_config_qa
from .loop import run as loop_run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper-distiller-qa",
        description="Multi-round question-driven research loop over arxiv + "
                    "Semantic Scholar, writing a synthesized answer survey doc.",
    )
    p.add_argument("--vault", required=True, help="Path to your Obsidian vault.")
    p.add_argument("--question", required=True, help="Research question to answer.")
    p.add_argument("--max-rounds", type=int, default=5,
                   help="Hard upper bound on loop rounds (default 5).")
    p.add_argument("--max-articles", type=int, default=15,
                   help="Hard upper bound on total articles distilled (default 15).")
    p.add_argument("--max-cost-cny", type=float, default=20.0,
                   help="Cost circuit breaker in CNY (default 20.0).")
    p.add_argument("--confidence-threshold", type=int, default=8,
                   help="LLM is_done confidence required to stop (0-10, default 8).")
    p.add_argument("--per-round", type=int, default=2,
                   help="Articles to distill each round (default 2).")
    p.add_argument("--source", choices=["arxiv", "ss", "both"], default="both",
                   help="Paper source(s) to search (default both).")
    p.add_argument("--interactive", action="store_true",
                   help="Pause after each round and prompt to continue (Y/n/q).")
    p.add_argument("--resume", help="Resume a paused session by its session_id.")
    p.add_argument("--verbose", "-v", action="store_true", help="Detailed logging.")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan only; no LLM, no vault writes.")
    p.add_argument("--model", help="Override PD_MODEL env var.")
    p.add_argument("--provider", help="Override PD_PROVIDER_NAME label.")
    return p


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config_qa(
            vault_path=args.vault,
            question=args.question,
            max_rounds=args.max_rounds,
            max_articles=args.max_articles,
            max_cost_cny=args.max_cost_cny,
            confidence_threshold=args.confidence_threshold,
            per_round=args.per_round,
            source=args.source,
            interactive=args.interactive,
            resume_session_id=args.resume,
            verbose=args.verbose,
            dry_run=args.dry_run,
            model_override=args.model,
            provider_override=args.provider,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    try:
        summary = loop_run(cfg)
    except Exception as e:
        print(f"\nError during QA loop: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        else:
            print("(run with --verbose for full traceback)", file=sys.stderr)
        return 3

    print()
    print(f"  Session:        {summary['session_id']}")
    print(f"  Stop reason:    {summary['stop_reason']}")
    print(f"  Rounds:         {summary['rounds_completed']}")
    print(f"  Articles:       {summary['articles_distilled_count']}")
    print(f"  Survey slug:    {summary.get('survey_slug') or '(none -- no articles)'}")
    print(f"  Cost:           CNY {summary['cost_cny']:.2f}")
    print(f"  Tokens in/out:  {summary['tokens_in_total']} / {summary['tokens_out_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
