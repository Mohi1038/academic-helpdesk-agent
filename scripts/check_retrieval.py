"""Measure retrieval quality before any agent exists.

For in-scope questions: hit@k (a gold document appears in the top k) and MRR.
For all questions: the best dense similarity, used to calibrate MIN_SIMILARITY (the refusal threshold).

Usage:  uv run python scripts/check_retrieval.py [--k 5]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helpdesk import config  # noqa: E402
from helpdesk.rag import retriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    rows = [json.loads(line) for line in (ROOT / "eval" / "retrieval_set.jsonl").open(encoding="utf-8")]
    hits, rr, in_sims, out_sims = 0, 0.0, [], []
    for row in rows:
        results = retriever.search(row["q"], k=args.k)
        best = retriever.best_similarity(results)
        if not row["gold"]:
            out_sims.append(best)
            print(f"  OOS  sim={best:.3f}  {row['q']}")
            continue
        in_sims.append(best)
        ranks = [i for i, h in enumerate(results, 1) if h.doc_id in row["gold"]]
        hits += bool(ranks)
        rr += 1 / ranks[0] if ranks else 0
        mark = f"@{ranks[0]}" if ranks else "MISS"
        print(f"{mark:>5}  sim={best:.3f}  {row['q']}  -> {[h.doc_id for h in results]}")

    n = len(in_sims)
    print(f"\nhit@{args.k} = {hits}/{n} = {hits / n:.0%}   MRR = {rr / n:.3f}")
    print(f"best-similarity  in-scope: min {min(in_sims):.3f} / mean {sum(in_sims) / n:.3f}"
          f"   out-of-scope: max {max(out_sims):.3f} / mean {sum(out_sims) / len(out_sims):.3f}"
          f"   (MIN_SIMILARITY = {config.MIN_SIMILARITY})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
