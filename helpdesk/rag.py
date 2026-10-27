"""Hybrid retrieval over the document corpus: dense (ChromaDB) + BM25, fused with reciprocal rank fusion.

Exact tokens such as course codes ("CSL7570") and clause numbers are where BM25 beats embeddings, while
paraphrased questions are where embeddings win; fusing both covers each one's blind spot.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property

import chromadb
from rank_bm25 import BM25Okapi

from helpdesk import config
from helpdesk.embeddings import embed

RRF_K = 60


def tokenize(text: str) -> list[str]:
    # Strip a plural "s" so "matters"/"matter" and "academics"/"academic" match (not "ss" as in "class").
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t
            for t in re.findall(r"[a-z0-9]+", text.lower())]


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    source_url: str
    page_start: int
    page_end: int
    section: str
    text: str
    similarity: float  # cosine similarity from the dense retriever (0 if only BM25 found it)
    score: float  # fused RRF score

    @property
    def citation(self) -> str:
        if not self.page_start:
            return self.title
        if self.page_start == self.page_end:
            return f"{self.title}, p. {self.page_start}"
        return f"{self.title}, pp. {self.page_start}-{self.page_end}"


class Retriever:
    @cached_property
    def _collection(self):
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        return client.get_collection(config.COLLECTION, embedding_function=None)

    @cached_property
    def _chunks(self) -> list[dict]:
        with config.CHUNKS_FILE.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @cached_property
    def _bm25(self) -> BM25Okapi:
        return BM25Okapi([tokenize(f"{c['title']} {c['section']} {c['text']}") for c in self._chunks])

    def search(self, query: str, k: int = config.TOP_K) -> list[Hit]:
        n = config.CANDIDATES
        dense = self._collection.query(
            query_embeddings=embed([config.QUERY_PREFIX + query]), n_results=n, include=["distances"],
        )
        dense_ids = dense["ids"][0]
        similarity = {cid: 1 - dist for cid, dist in zip(dense_ids, dense["distances"][0])}

        scores = self._bm25.get_scores(tokenize(query))
        bm25_ids = [self._chunks[i]["chunk_id"]
                    for i in sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:n] if scores[i] > 0]

        fused: dict[str, float] = {}
        for ranking in (dense_ids, bm25_ids):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1 / (RRF_K + rank + 1)

        by_id = {c["chunk_id"]: c for c in self._chunks}
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(**{key: by_id[cid][key] for key in
                       ("chunk_id", "doc_id", "title", "source_url", "page_start", "page_end", "section", "text")},
                    similarity=round(similarity.get(cid, 0.0), 4), score=round(fused[cid], 5))
                for cid in top]

    def best_similarity(self, hits: list[Hit]) -> float:
        return max((h.similarity for h in hits), default=0.0)


retriever = Retriever()
