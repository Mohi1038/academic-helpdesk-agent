"""Command-line chat with the helpdesk.

Usage:
  uv run python -m helpdesk.cli "What is the slot for Autonomous Systems?"
  uv run python -m helpdesk.cli --mode multi --model llama3.1:8b      # interactive
"""

import argparse
import asyncio
import json

from helpdesk.runner import Helpdesk, Turn


def show(turn: Turn, trace: bool) -> None:
    if trace:
        for call in turn.tool_calls:
            print(f"  [{call.agent}] {call.name}({json.dumps(call.args, ensure_ascii=False)})")
    print(f"\n{turn.answer or '(no answer)'}")
    for draft in turn.drafts:
        print(f"\n----- draft (not sent) -----\n{draft}----------------------------")
    print(f"\n  ({' -> '.join(turn.agents)}, {turn.latency_s}s{', ERROR: ' + turn.error if turn.error else ''})\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-trace", action="store_true", help="hide tool calls")
    args = parser.parse_args()

    helpdesk = Helpdesk(args.mode, args.model)
    session = await helpdesk.new_session()
    if args.question:
        show(await helpdesk.ask(args.question, session), not args.no_trace)
        return
    print(f"IITJ Academic Helpdesk ({args.mode}, {helpdesk.model}). Empty line to quit.")
    while question := input("> ").strip():
        show(await helpdesk.ask(question, session), not args.no_trace)


if __name__ == "__main__":
    asyncio.run(main())
