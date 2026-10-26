"""Aggregate evaluation runs (+ judge verdicts, if present) into metrics tables.

Deterministic metrics (no judge needed): refusal behaviour, unanswered-question logging, routing accuracy,
tool selection, whether the answer cites a source that the tools actually returned, latency, run-to-run
changes in the refusal decision. Judge metrics: accuracy, groundedness, citation quality, consistency.

Usage:  uv run python eval/score.py       -> eval/results/summary.md, metrics.csv, per_answer.csv
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helpdesk.agents.callbacks import REFUSAL_PATTERN  # noqa: E402

RESULTS = ROOT / "eval" / "results"
SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}
STRUCTURED_FILES = ["courses.csv", "timetable.csv", "faculty.csv", "student_profile.json"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def doc_titles() -> dict[str, str]:
    with (ROOT / "data" / "manifest.csv").open(newline="", encoding="utf-8") as f:
        return {r["title"]: r["doc_id"] for r in csv.DictReader(f)}


TITLES = doc_titles()


def sources_in(text: str) -> set[str]:
    """doc_ids and structured-data files mentioned in a piece of text."""
    found = {doc_id for title, doc_id in TITLES.items() if title in text}
    return found | {f for f in STRUCTURED_FILES if f in text}


def evidence_sources(row: dict) -> set[str]:
    found = set()
    for call in row["tool_calls"]:
        resp = call.get("response") or {}
        for r in resp.get("results", []):
            found |= sources_in(r.get("citation", ""))
        found |= sources_in(resp.get("source", ""))
    return found


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def pct(values) -> str:
    m = mean(values)
    return "-" if m is None else f"{100 * m:.0f}%"


def p90(values) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(0.9 * len(values)))] if values else 0.0


def main() -> int:
    questions = {q["id"]: q for q in read_jsonl(ROOT / "eval" / "questions.jsonl")}
    rows = [r for p in sorted((RESULTS / "runs").glob("*.jsonl")) for r in read_jsonl(p)]
    verdicts = {(v["mode"], v["model"], v["run"], v["id"]): v for v in read_jsonl(RESULTS / "verdicts.jsonl")}
    consistency = {(c["mode"], c["model"], c["id"]): c for c in read_jsonl(RESULTS / "consistency.jsonl")}

    per_answer = []
    for r in rows:
        q = questions[r["id"]]
        v = verdicts.get((r["mode"], r["model"], r["run"], r["id"]), {})
        called = [c["name"] for c in r["tool_calls"]]
        text = r["answer"] or ""
        refused = bool(REFUSAL_PATTERN.search(text)) or (not text.strip() and not r["drafts"])
        cited = sources_in(text)
        evidence = evidence_sources(r)
        per_answer.append({
            "mode": r["mode"], "model": r["model"], "run": r["run"], "id": r["id"], "category": q["category"],
            "in_scope": q["in_scope"], "error": bool(r["error"]), "empty": not text.strip(),
            "refused": refused, "logged": r.get("logged_unanswered", False),
            "route": r["route"], "route_ok": r["route"] == q["expected_agent"] if r["mode"].startswith("multi") else None,
            "tools_ok": set(q["expected_tools"]) <= set(called) if q["in_scope"] else None,
            "domain_tools_on_oos": (not q["in_scope"]) and any(n not in {"log_unanswered", "transfer_to_agent"}
                                                                for n in called),
            "cites_evidence": bool(cited & evidence) if q["in_scope"] and q["category"] != "email" else None,
            "evidence_has_gold": bool(set(q["gold_sources"]) & evidence) if q["in_scope"] else None,
            "latency_s": r["latency_s"],
            "correctness": SCORE.get(v.get("correctness")),
            "grounded": v.get("grounded"), "citation_ok": v.get("citation_ok"),
            "judge_rationale": v.get("rationale", v.get("judge_error", "")),
            "answer": text,
        })

    configs = sorted({(a["mode"], a["model"]) for a in per_answer}, key=lambda k: (k[1], k[0]))
    metrics = []
    for mode, model in configs:
        a = [x for x in per_answer if (x["mode"], x["model"]) == (mode, model)]
        ins = [x for x in a if x["in_scope"]]
        oos = [x for x in a if not x["in_scope"]]
        answered_ins = [x for x in ins if not x["refused"]]
        groups = [c for (m, mo, _), c in consistency.items() if (m, mo) == (mode, model)]
        metrics.append({
            "mode": mode, "model": model, "answers": len(a),
            "accuracy_in_scope": mean(x["correctness"] for x in ins),
            "accuracy_overall": mean(x["correctness"] for x in a),
            "grounded": mean(x["grounded"] for x in answered_ins),
            "citation_ok": mean(x["citation_ok"] for x in answered_ins if x["category"] != "email"),
            "cites_evidence": mean(x["cites_evidence"] for x in answered_ins),
            "evidence_has_gold": mean(x["evidence_has_gold"] for x in ins),
            "oos_refused": mean(x["refused"] for x in oos),
            "oos_logged": mean(x["logged"] for x in oos),
            "false_refusal": mean(x["refused"] for x in ins),
            "routing_acc": mean(x["route_ok"] for x in a) if mode.startswith("multi") else None,
            "tool_selection": mean(x["tools_ok"] for x in ins),
            "empty_answers": sum(x["empty"] for x in a),
            "errors": sum(x["error"] for x in a),
            "consistent_questions": mean(c.get("consistent") for c in groups),
            "latency_mean_s": round(statistics.mean(x["latency_s"] for x in a), 1),
            "latency_p90_s": round(p90([x["latency_s"] for x in a]), 1),
        })

    # Questions whose answers changed between runs
    changed = []
    by_group = defaultdict(list)
    for x in per_answer:
        by_group[(x["mode"], x["model"], x["id"])].append(x)
    for key, xs in sorted(by_group.items()):
        c = consistency.get(key, {})
        scores = {x["correctness"] for x in xs}
        refusals = {x["refused"] for x in xs}
        if c.get("consistent") is False or len(scores) > 1 or len(refusals) > 1:
            changed.append({"mode": key[0], "model": key[1], "id": key[2],
                            "scores": [x["correctness"] for x in sorted(xs, key=lambda x: x["run"])],
                            "differences": c.get("differences", "refusal decision changed" if len(refusals) > 1 else "")})

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "per_answer.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_answer[0]))
        w.writeheader()
        w.writerows(per_answer)
    with (RESULTS / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0]))
        w.writeheader()
        w.writerows(metrics)

    lines = ["# Evaluation summary", "",
             f"{len(questions)} questions; {len(per_answer)} answers across {len(configs)} configurations.", "",
             "## Main metrics", "",
             "| Model | Mode | Accuracy (in-scope) | Grounded | Citation OK | Correct refusals (OOS) | "
             "OOS logged | False refusals | Routing | Tool selection | Consistent Qs | Latency mean / p90 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in metrics:
        lines.append(f"| {m['model']} | {m['mode']} | {pct([m['accuracy_in_scope']])} | {pct([m['grounded']])} | "
                     f"{pct([m['citation_ok']])} | {pct([m['oos_refused']])} | {pct([m['oos_logged']])} | "
                     f"{pct([m['false_refusal']])} | {pct([m['routing_acc']])} | {pct([m['tool_selection']])} | "
                     f"{pct([m['consistent_questions']])} | {m['latency_mean_s']}s / {m['latency_p90_s']}s |")
    lines += ["", "## Accuracy by category", "", "| Model | Mode | " + " | ".join(
        c for c in ["policy", "schedule", "email", "out_of_scope"]) + " |", "|---|---|---|---|---|---|"]
    for mode, model in configs:
        cells = []
        for cat in ["policy", "schedule", "email", "out_of_scope"]:
            cells.append(pct([x["correctness"] for x in per_answer
                              if (x["mode"], x["model"], x["category"]) == (mode, model, cat)]))
        lines.append(f"| {model} | {mode} | " + " | ".join(cells) + " |")
    lines += ["", "## Deterministic checks", "",
              "| Model | Mode | Cites a returned source | Gold source retrieved | Empty answers | Errors |",
              "|---|---|---|---|---|---|"]
    for m in metrics:
        lines.append(f"| {m['model']} | {m['mode']} | {pct([m['cites_evidence']])} | "
                     f"{pct([m['evidence_has_gold']])} | {m['empty_answers']} | {m['errors']} |")
    lines += ["", f"## Answers that changed across runs ({len(changed)})", "",
              "| Model | Mode | Question | Scores per run | What changed |", "|---|---|---|---|---|"]
    for c in changed:
        diff = c["differences"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {c['model']} | {c['mode']} | {c['id']} | {c['scores']} | {diff} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4 + 2 + len(metrics) + 2]))
    print(f"\n-> {RESULTS / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
