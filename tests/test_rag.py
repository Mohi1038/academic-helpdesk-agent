"""Chunking and hybrid-retrieval tests. A hashing embedder stands in for Ollama, so no server is needed."""

import hashlib
import json
from dataclasses import asdict

import chromadb
import pytest

from helpdesk import config, ingest, rag


def fake_embed(texts, batch_size=32):
    """Bag-of-words hashed into 64 dims: texts sharing words get similar vectors."""
    out = []
    for text in texts:
        vec = [0.0] * 64
        for tok in rag.tokenize(text):
            vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 64] += 1.0
        out.append(vec)
    return out


def test_boilerplate_and_page_numbers_removed():
    header = "Indian Institute of Technology Jodhpur Office of Academic Affairs"
    pages = [(i, [header, str(i), f"body text of page {i}"]) for i in range(1, 5)]
    paras = ingest.strip_boilerplate(pages)
    assert [p.text for p in paras] == [f"body text of page {i}" for i in range(1, 5)]


@pytest.mark.parametrize("text,expected", [
    ("15. LEAVE RULES", True),
    ("4.1.9 Attendance Requirements", True),
    ("# Academic Calendar", True),
    ("A student should have full attendance in each course.", False),
])
def test_heading_detection(text, expected):
    assert ingest.is_heading(text) is expected


def test_chunks_respect_size_overlap_and_pages(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_WORDS", 50)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_WORDS", 10)
    paras = [ingest.Paragraph("1. INTRODUCTION", 1)] + [
        ingest.Paragraph(" ".join(f"w{p}_{i}" for i in range(8)), page=1 + p // 5) for p in range(20)]
    meta = {"doc_id": "d", "title": "Doc", "category": "x", "source_url": "u"}
    chunks = ingest.chunk_document(paras, meta)

    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 50 + 8 for c in chunks)
    assert chunks[0].section == "1. INTRODUCTION"
    # consecutive chunks share their boundary paragraph
    assert chunks[0].text.splitlines()[-1] == chunks[1].text.splitlines()[0]
    # every paragraph survives
    joined = "\n".join(c.text for c in chunks)
    assert all(p.text in joined for p in paras)
    assert chunks[-1].page_end == 4


@pytest.fixture
def tiny_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path)
    monkeypatch.setattr(config, "CHUNKS_FILE", tmp_path / "chunks.jsonl")
    monkeypatch.setattr(rag, "embed", fake_embed)
    docs = [
        ("reg::000", "Regulations", 7, "Attendance requirements: a student should have full attendance in each course."),
        ("cal::000", "Calendar", 2, "Major examination 19-26 November 2026 for semester I."),
        ("cse::000", "CSE Curriculum", 11, "Elective CSL7570 Natural Language Understanding is a 700 level course."),
    ]
    chunks = [ingest.Chunk(cid, cid.split("::")[0], title, "c", "url", page, page, "", text)
              for cid, title, page, text in docs]
    client = chromadb.PersistentClient(path=str(tmp_path))
    col = client.create_collection(config.COLLECTION, embedding_function=None,
                                   configuration={"hnsw": {"space": "cosine"}})
    col.add(ids=[c.chunk_id for c in chunks], embeddings=fake_embed([c.text for c in chunks]),
            documents=[c.text for c in chunks], metadatas=[{"doc_id": c.doc_id} for c in chunks])
    (tmp_path / "chunks.jsonl").write_text("\n".join(json.dumps(asdict(c)) for c in chunks))
    return rag.Retriever()


def test_hybrid_search_finds_course_code_and_formats_citation(tiny_store):
    hits = tiny_store.search("Who teaches CSL7570?", k=2)
    assert hits[0].doc_id == "cse"
    assert hits[0].citation == "CSE Curriculum, p. 11"


def test_search_ranks_paraphrase(tiny_store):
    hits = tiny_store.search("attendance needed in each course", k=1)
    assert hits[0].doc_id == "reg"
    assert 0 < hits[0].similarity <= 1
