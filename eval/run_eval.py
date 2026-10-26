"""Run the evaluation questions through every (mode, model, run) combination and record each turn.

Each combination writes eval/results/runs/<mode>__<model>__r<run>.jsonl; reruns resume and retry
turns that errored. Every combination gets its own unanswered-questions log, so "was the refusal
logged?" can be checked per answer.

Usage:
  uv run python eval/run_eval.py                                  # full grid: 2 modes x 3 models x 3 runs
  uv run python eval/run_eval.py --models qwen3:8b --runs 1 --ids P01 S02
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helpdesk import config  # noqa: E402
from helpdesk.runner import Helpdesk  # noqa: E402

RESULTS = ROOT / "eval" / "results"
QUESTIONS = ROOT / "eval" / "questions.jsonl"
DEFAULT_MODELS = ["qwen3:8b", "llama3.1:8b", "granite3.3:8b"]


def run_name(mode: str, model: str, run: int) -> str:
    return f"{mode}__{model.replace(':', '-')}__r{run}"


def load_questions(ids: list[str] | None) -> list[dict]:
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if not ids or r["id"] in ids]


def logged_questions(log: Path) -> set[str]:
    if not log.exists():
        return set()
    return {json.loads(line)["question"].strip() for line in log.read_text(encoding="utf-8").splitlines()}


async def run_combination(desk: Helpdesk, run: int, questions: list[dict], concurrency: int) -> None:
    name = run_name(desk.mode, desk.model, run)
    out = RESULTS / "runs" / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row["error"]:
                done[row["id"]] = row
    todo = [q for q in questions if q["id"] not in done]
    if not todo:
        print(f"{name}: complete")
        return

    # Per-combination side effects (tools read these paths at call time)
    config.UNANSWERED_LOG = RESULTS / "logs" / f"{name}.jsonl"
    config.DRAFTS_DIR = RESULTS / "drafts" / name

    gate = asyncio.Semaphore(concurrency)

    async def one(q: dict) -> dict:
        async with gate:
            turn = await desk.ask(q["question"])  # fresh session per question
        print(f"  {name} {q['id']} {turn.latency_s:>5}s {turn.route or '-':<15} {turn.error or ''}")
        return {
            "id": q["id"], "mode": desk.mode, "model": desk.model, "run": run, "question": q["question"],
            "answer": turn.answer, "drafts": turn.drafts, "agents": turn.agents, "route": turn.route,
            "tool_calls": [asdict(c) for c in turn.tool_calls], "latency_s": turn.latency_s, "error": turn.error,
        }

    rows = await asyncio.gather(*(one(q) for q in todo))
    logged = logged_questions(config.UNANSWERED_LOG)
    for row in rows:
        row["logged_unanswered"] = row["question"].strip() in logged
        done[row["id"]] = row
    order = [q["id"] for q in load_questions(None)]
    out.write_text("".join(json.dumps(done[i], ensure_ascii=False) + "\n" for i in order if i in done),
                   encoding="utf-8")
    errors = sum(1 for r in rows if r["error"])
    print(f"{name}: {len(rows)} answered, {errors} errors")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", default=["single", "multi"], choices=["single", "multi", "multi_nothink"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--ids", nargs="*", help="only these question ids")
    parser.add_argument("--concurrency", type=int, default=4, help="match OLLAMA_NUM_PARALLEL on the server")
    args = parser.parse_args()

    questions = load_questions(args.ids)
    for model in args.models:  # one model at a time, so Ollama is not swapping models in and out
        for mode in args.modes:
            desk = Helpdesk(mode, model)
            for run in range(1, args.runs + 1):
                await run_combination(desk, run, questions, args.concurrency)


if __name__ == "__main__":
    asyncio.run(main())
