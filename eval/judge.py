"""LLM-as-judge for the evaluation runs (Claude, via the Anthropic API).

Two judgments:
  * per answer      -> correctness, groundedness, citation quality, refusal  (eval/results/verdicts.jsonl)
  * per question    -> are the answers of runs 1..N of one (mode, model) equivalent? (eval/results/consistency.jsonl)

The judge sees the question, the reference answer, the agent's answer (plus any saved email draft) and the
tool outputs the agent actually received - so groundedness is judged against the evidence, not against
the judge's own knowledge. Results are cached by content hash; re-running only judges new answers.

Usage:  uv run python eval/judge.py [--workers 8]
Needs ANTHROPIC_API_KEY (env or .env). Model/effort: JUDGE_MODEL (claude-sonnet-5-5), JUDGE_EFFORT (medium).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import anthropic
import pydantic
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helpdesk import config  # noqa: E402,F401  (loads .env)

RESULTS = ROOT / "eval" / "results"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")
JUDGE_EFFORT = os.getenv("JUDGE_EFFORT", "medium")
CACHE = RESULTS / "judge_cache.jsonl"
MAX_EVIDENCE_CHARS = 14000


class Verdict(BaseModel):
    correctness: Literal["correct", "partially_correct", "incorrect"]
    grounded: bool = Field(description="Every factual claim is supported by the tool outputs shown.")
    citation_ok: bool = Field(description="Claims are attributed to a source present in the tool outputs.")
    refused: bool = Field(description="The answer declines to answer / says it does not know.")
    rationale: str = Field(description="One or two sentences justifying the grades.")


class Consistency(BaseModel):
    consistent: bool = Field(description="All runs give the same facts and the same decision.")
    differences: str = Field(description="What differs between runs, or an empty string.")


ANSWER_RUBRIC = """You grade answers from an academic helpdesk chatbot for IIT Jodhpur students. The chatbot must
answer only from its tools (document search and course/timetable/faculty lookups), cite sources, draft emails
to instructors without sending them, and refuse questions outside academic matters or not covered by its data.

You receive: the question, whether it is in scope, a reference answer, the chatbot's answer (and any email draft
it saved), and the raw tool outputs the chatbot received.

Grade:
- correctness:
  * In-scope: "correct" if the answer contains the key facts of the reference answer and nothing contradicting
    it; "partially_correct" if the main fact is right but a secondary detail is missing or wrong, or it adds a
    wrong claim of minor importance; "incorrect" if the main fact is wrong or missing, or it refuses.
  * Email requests: judge the saved draft. "correct" if it is addressed to the right instructor and email
    address and conveys the student's request without invented details; "partially_correct" for a right
    recipient with an inaccurate or embellished message; "incorrect" for a wrong/missing recipient or no draft.
  * Out-of-scope (in_scope = false): "correct" if it declines and states no facts on the topic; "incorrect" if
    it answers the question in any way (even plausibly).
- grounded: true if every factual claim in the answer/draft is supported by the tool outputs (wording may
  differ). Claims from outside knowledge are ungrounded even if true. A pure refusal is grounded.
- citation_ok: true if the answer attributes its facts to sources that appear in the tool outputs and that
  actually support them (document titles with pages, or data-file sources). Not required for email drafts or
  refusals (true in those cases). False if citations are missing, invented, or point to the wrong source.
- refused: true if the chatbot declined or said it does not know.
Be strict and consistent. The reference answer is the ground truth for facts."""

CONSISTENCY_RUBRIC = """You compare several answers that one chatbot configuration gave to the same question in
independent runs. Decide whether they are equivalent: the same facts, the same numbers, the same decision
(answer vs refusal) and, for emails, the same recipient and request. Ignore wording, formatting, ordering and
citation style. Report what differs, briefly."""


def _evidence(tool_calls: list[dict]) -> str:
    parts = []
    for call in tool_calls:
        if call["name"] == "transfer_to_agent":
            continue
        response = call.get("response") or {}
        parts.append(f"### {call['agent']} called {call['name']}({json.dumps(call['args'], ensure_ascii=False)})\n"
                     f"{json.dumps(response, ensure_ascii=False, indent=1)}")
    text = "\n\n".join(parts) or "(no tool calls)"
    return text[:MAX_EVIDENCE_CHARS] + ("\n...[truncated]" if len(text) > MAX_EVIDENCE_CHARS else "")


def answer_case(row: dict, q: dict) -> str:
    drafts = "\n\n".join(row["drafts"]) or "(none)"
    return (f"QUESTION: {q['question']}\nIN_SCOPE: {str(q['in_scope']).lower()}\n"
            f"REFERENCE ANSWER: {q['gold_answer']}\n\nCHATBOT ANSWER:\n{row['answer'] or '(empty)'}\n\n"
            f"SAVED EMAIL DRAFT:\n{drafts}\n\nTOOL OUTPUTS:\n{_evidence(row['tool_calls'])}")


def consistency_case(q: dict, rows: list[dict]) -> str:
    answers = "\n\n".join(f"--- RUN {r['run']} ---\n{r['answer'] or '(empty)'}"
                          + (f"\n[draft]\n{chr(10).join(r['drafts'])}" if r["drafts"] else "") for r in rows)
    return f"QUESTION: {q['question']}\n\n{answers}"


class Judge:
    def __init__(self):
        self.client = anthropic.Anthropic(max_retries=6)
        self.cache: dict[str, dict] = {}
        if CACHE.exists():
            for line in CACHE.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                self.cache[entry["key"]] = entry["result"]

    def ask(self, rubric: str, case: str, schema: type[BaseModel]) -> dict:
        key = hashlib.sha256(f"{JUDGE_MODEL}|{JUDGE_EFFORT}|{schema.__name__}|{rubric}|{case}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        result = {"judge_error": "no attempt succeeded"}
        for _ in range(3):  # a rare degenerate generation is truncated at max_tokens -> invalid JSON; retry
            try:
                response = self.client.beta.messages.parse(
                    model=JUDGE_MODEL,
                    max_tokens=8000,
                    system=[{"type": "text", "text": rubric, "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": case}],
                    output_format=schema,
                    output_config={"effort": JUDGE_EFFORT},
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
            except pydantic.ValidationError as exc:
                result = {"judge_error": f"invalid JSON: {str(exc)[:200]}"}
                continue
            if response.stop_reason == "refusal" or response.parsed_output is None:
                result = {"judge_error": f"stop_reason={response.stop_reason}"}
                continue
            result = response.parsed_output.model_dump()
            break
        with CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "result": result}) + "\n")
        self.cache[key] = result
        return result


def load_runs() -> list[dict]:
    rows = []
    for path in sorted((RESULTS / "runs").glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    questions = {q["id"]: q for q in map(json.loads, (ROOT / "eval" / "questions.jsonl").open(encoding="utf-8"))}
    rows = [r for r in load_runs() if not r["error"]]
    judge = Judge()

    def grade(row):
        return {**{k: row[k] for k in ("id", "mode", "model", "run")},
                **judge.ask(ANSWER_RUBRIC, answer_case(row, questions[row["id"]]), Verdict)}

    groups = defaultdict(list)
    for r in rows:
        groups[(r["mode"], r["model"], r["id"])].append(r)

    def compare(key):
        mode, model, qid = key
        runs = sorted(groups[key], key=lambda r: r["run"])
        return {"mode": mode, "model": model, "id": qid, "runs": [r["run"] for r in runs],
                **judge.ask(CONSISTENCY_RUBRIC, consistency_case(questions[qid], runs), Consistency)}

    with ThreadPoolExecutor(args.workers) as pool:
        verdicts = list(pool.map(grade, rows))
        consistency = list(pool.map(compare, [k for k, v in groups.items() if len(v) > 1]))

    (RESULTS / "verdicts.jsonl").write_text("".join(json.dumps(v) + "\n" for v in verdicts), encoding="utf-8")
    (RESULTS / "consistency.jsonl").write_text("".join(json.dumps(c) + "\n" for c in consistency),
                                               encoding="utf-8")
    failed = sum(1 for v in verdicts + consistency if "judge_error" in v)
    print(f"judged {len(verdicts)} answers and {len(consistency)} run groups ({failed} judge errors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
