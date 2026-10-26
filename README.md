# academic-helpdesk-agent

Multi-agent academic helpdesk for IIT Jodhpur, built with Google ADK, Ollama-hosted 8B models and a
ChromaDB RAG pipeline. It answers questions on programmes, curricula, rules and regulations, and the timetable.
It also drafts emails to course instructors, and refuses (and logs) questions it cannot answer.

> Work in progress. Phase 1 (data corpus) is complete.

## Setup

```bash
uv sync --python 3.12
```

## Data corpus

```bash
uv run python scripts/fetch_docs.py   # download 29 public IITJ documents -> data/raw, data/manifest.csv
uv run python scripts/ocr_scans.py    # OCR scanned PDFs (needs tesseract) -> data/raw/*.ocr.md
uv run pytest -q                      # integrity checks on the manifest and the structured CSVs
```

- `data/raw/`: 24 PDFs and 5 web pages saved as Markdown. They cover academic regulations, attendance and examination circulars, the 2026-27 academic calendar, holidays, fees, scholarships, forms, and the B.Tech./M.Tech. curricula, including humanities courses.
- `data/manifest.csv`: provenance for each document (title, category, source URL, fetch date, page count, hash).
- `data/structured/`: course offerings, the official 2026-27 Semester I slot system, and the real faculty directory. Course codes are partly synthetic and course-to-instructor assignments are illustrative. See [data/structured/README.md](data/structured/README.md).
