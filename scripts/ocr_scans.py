"""OCR the scanned PDFs in data/raw that have no text layer.

For every PDF whose extractable text is (almost) empty, each page is rendered at
300 dpi and passed through the tesseract CLI. The result is written next to the
PDF as <doc_id>.ocr.md with one "<!-- page N -->" marker per page, which the
ingestion step uses instead of the empty text layer.

Usage:  uv run python scripts/ocr_scans.py [--force]
Requires: tesseract on PATH (brew install tesseract / apt install tesseract-ocr)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MIN_CHARS_PER_PAGE = 50  # below this a page is treated as scanned


def needs_ocr(pdf: Path) -> bool:
    with pymupdf.open(pdf) as doc:
        chars = sum(len(page.get_text().strip()) for page in doc)
        return chars < MIN_CHARS_PER_PAGE * doc.page_count


def ocr_pdf(pdf: Path) -> str:
    parts = []
    with pymupdf.open(pdf) as doc, tempfile.TemporaryDirectory() as tmp:
        for i, page in enumerate(doc, start=1):
            png = Path(tmp) / f"p{i}.png"
            page.get_pixmap(dpi=300).save(png)
            text = subprocess.run(
                ["tesseract", str(png), "-", "-l", "eng", "--psm", "6"],
                capture_output=True, text=True, check=True,
            ).stdout
            parts.append(f"<!-- page {i} -->\n{text.strip()}\n")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="redo OCR even if output exists")
    args = parser.parse_args()

    if not shutil.which("tesseract"):
        print("tesseract not found on PATH", file=sys.stderr)
        return 1

    for pdf in sorted(RAW_DIR.glob("*.pdf")):
        out = pdf.with_suffix(".ocr.md")
        if not needs_ocr(pdf) or (out.exists() and not args.force):
            continue
        out.write_text(ocr_pdf(pdf), encoding="utf-8")
        print(f"ocr  {pdf.name} -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
