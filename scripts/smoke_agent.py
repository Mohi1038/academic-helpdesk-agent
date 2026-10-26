"""Run the assignment's sample questions through an agent and print answers, tool calls and timing.

Usage:  uv run python scripts/smoke_agent.py [--mode single|multi] [--model qwen3:8b]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpdesk.runner import Helpdesk  # noqa: E402

QUESTIONS = [
    "What is the attendance requirement to sit in the Major exam?",
    "What is the slot for the course Autonomous Systems?",
    "How many 400 level electives have been floated by the Humanities department in this semester?",
    "How many Open Elective credits do I need to earn in order to get a B.Tech. Degree?",
    "Write an email to the course instructor of CSL7570 that I need leave for 2 days",
    "What are the available dining options on the campus?",
    "What is the best way to reach the campus from Jodhpur city center?",
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    helpdesk = Helpdesk(args.mode, args.model)
    for q in QUESTIONS:
        turn = await helpdesk.ask(q)
        print(f"\n### {q}")
        for c in turn.tool_calls:
            print(f"  [{c.agent}] {c.name}({json.dumps(c.args, ensure_ascii=False)})")
        print(turn.answer or f"(no answer) {turn.error or ''}")
        print(f"  -- {' -> '.join(turn.agents)} | {turn.latency_s}s")


if __name__ == "__main__":
    asyncio.run(main())
