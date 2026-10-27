"""Build the vector store: parse documents -> section-aware chunks -> Ollama embeddings -> ChromaDB.

Usage:  uv run python -m helpdesk.ingest
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import chromadb
import pymupdf

from helpdesk import config
from helpdesk.embeddings import embed

PAGE_MARKER = re.compile(r"<!-- page (\d+) -->")
NUMBERED_HEADING = re.compile(r"^\d{1,2}(\.\d{1,2})*\.?\s+[A-Z][^.:;]{2,90}$")


@dataclass
class Paragraph:
    text: str
    page: int  # 0 for documents without pages (web pages)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    category: str
    source_url: str
    page_start: int
    page_end: int
    section: str
    text: str


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_pages(path: Path) -> list[tuple[int, list[str]]]:
    """Return (page number, paragraphs) for a document, preferring OCR text for scanned PDFs."""
    ocr = path.with_suffix(".ocr.md")
    if path.suffix == ".pdf" and ocr.exists():
        parts = PAGE_MARKER.split(ocr.read_text(encoding="utf-8"))[1:]
        return [(int(num), [_clean(p) for p in body.split("\n\n")]) for num, body in zip(parts[::2], parts[1::2])]
    if path.suffix == ".pdf":
        with pymupdf.open(path) as doc:
            return [(i, [_clean(b[4]) for b in page.get_text("blocks") if b[6] == 0])
                    for i, page in enumerate(doc, start=1)]
    # Markdown web page: one paragraph per line, no page numbers
    return [(0, [_clean(line) for line in path.read_text(encoding="utf-8").splitlines()])]


def strip_boilerplate(pages: list[tuple[int, list[str]]]) -> list[Paragraph]:
    """Drop empty lines, bare page numbers, and short lines repeated on most pages (headers/footers)."""
    repeated: set[str] = set()
    if len(pages) >= 3:
        counts = Counter(p for _, paras in pages for p in set(paras) if len(p) < 150)
        repeated = {p for p, n in counts.items() if n >= len(pages) / 2}
    return [Paragraph(p, num) for num, paras in pages for p in paras
            if p and not p.isdigit() and p not in repeated]


def is_heading(text: str) -> bool:
    words = text.split()
    if text.startswith("#"):
        return True
    if len(words) > 12:
        return False
    if NUMBERED_HEADING.match(text):
        return True
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 4 and all(c.isupper() for c in letters)


def chunk_document(paragraphs: list[Paragraph], meta: dict) -> list[Chunk]:
    """Greedily pack paragraphs into ~CHUNK_WORDS windows with a paragraph-level overlap."""
    # Split very long paragraphs (e.g. OCR blobs) into window-sized units first.
    units: list[tuple[Paragraph, str]] = []  # (paragraph, section in force)
    section = ""
    for para in paragraphs:
        if is_heading(para.text):
            section = para.text.lstrip("# ").strip()
        words = para.text.split()
        for i in range(0, len(words), config.CHUNK_WORDS):
            units.append((Paragraph(" ".join(words[i:i + config.CHUNK_WORDS]), para.page), section))

    chunks: list[Chunk] = []
    window: list[tuple[Paragraph, str]] = []

    def emit():
        text = "\n".join(p.text for p, _ in window)
        pages = [p.page for p, _ in window]
        chunks.append(Chunk(
            chunk_id=f"{meta['doc_id']}::{len(chunks):03d}",
            doc_id=meta["doc_id"], title=meta["title"], category=meta["category"],
            source_url=meta["source_url"], page_start=min(pages), page_end=max(pages),
            section=window[0][1], text=text,
        ))

    has_new = False  # does the window hold anything not already emitted?
    for unit in units:
        window.append(unit)
        has_new = True
        if sum(len(p.text.split()) for p, _ in window) >= config.CHUNK_WORDS:
            emit()
            # keep trailing paragraphs that fit in the overlap budget
            tail, budget = [], config.CHUNK_OVERLAP_WORDS
            for p, s in reversed(window):
                budget -= len(p.text.split())
                if budget < 0:
                    break
                tail.insert(0, (p, s))
            window, has_new = tail, False
    if has_new:
        emit()
    return chunks


def embedding_text(chunk: Chunk) -> str:
    """Prefix each chunk with its document title and section so the vector carries that context."""
    header = chunk.title + (f" | {chunk.section}" if chunk.section else "")
    return f"{config.DOC_PREFIX}{header}\n{chunk.text}"


def build_chunks() -> list[Chunk]:
    with config.MANIFEST.open(newline="", encoding="utf-8") as f:
        docs = list(csv.DictReader(f))
    chunks: list[Chunk] = []
    for meta in docs:
        paragraphs = strip_boilerplate(read_pages(config.DATA_DIR.parent / meta["file"]))
        chunks.extend(chunk_document(paragraphs, meta))
    return chunks


def main() -> int:
    chunks = build_chunks()
    print(f"{len(chunks)} chunks from {len({c.doc_id for c in chunks})} documents; embedding with {config.EMBED_MODEL} ...")
    vectors = embed([embedding_text(c) for c in chunks])

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if config.COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(config.COLLECTION)
    collection = client.create_collection(
        config.COLLECTION, embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embed_model": config.EMBED_MODEL},
    )
    for i in range(0, len(chunks), 500):
        batch = chunks[i:i + 500]
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors[i:i + 500],
            documents=[c.text for c in batch],
            metadatas=[{k: v for k, v in asdict(c).items() if k not in {"chunk_id", "text"}} for c in batch],
        )
    with config.CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"stored {collection.count()} chunks in {config.CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
