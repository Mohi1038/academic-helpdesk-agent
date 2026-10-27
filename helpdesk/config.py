"""Central settings. Every value can be overridden through the environment or a .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Ollama (local, or a remote server reached through an SSH tunnel)
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:8b")
NUM_CTX = int(os.getenv("NUM_CTX", 8192))  # Ollama's default context window silently truncates prompts
TEMPERATURE = float(os.getenv("TEMPERATURE")) if os.getenv("TEMPERATURE") else None  # None: model default

# Data locations
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STRUCTURED_DIR = DATA_DIR / "structured"
MANIFEST = DATA_DIR / "manifest.csv"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", DATA_DIR / "chroma"))
CHUNKS_FILE = CHROMA_DIR / "chunks.jsonl"
COLLECTION = "iitj_docs"

# Chunking (sizes in words; ~1.3 tokens per word for this corpus)
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", 350))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", 60))

# Retrieval
TOP_K = int(os.getenv("TOP_K", 5))
CANDIDATES = int(os.getenv("CANDIDATES", 20))  # per retriever, before rank fusion
# Cosine similarity of the best dense hit below which retrieval is treated as "nothing relevant found".
# Calibrated with scripts/check_retrieval.py: in-scope questions bottom out at ~0.71 and out-of-scope
# ones top out at ~0.66. The set is small, so the cut sits on the safe side of the gap; it screens clearly
# off-topic queries, and the agent prompt still judges whether the passages actually answer the question.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", 0.65))

# nomic-embed-text is trained with task prefixes; other embedding models take none.
_NOMIC = EMBED_MODEL.startswith("nomic-embed-text")
DOC_PREFIX = "search_document: " if _NOMIC else ""
QUERY_PREFIX = "search_query: " if _NOMIC else ""

CURRENT_SEM = os.getenv("CURRENT_SEM", "2026-27-I")

# Runtime outputs
DRAFTS_DIR = Path(os.getenv("DRAFTS_DIR", ROOT / "drafts"))
UNANSWERED_LOG = Path(os.getenv("UNANSWERED_LOG", ROOT / "logs" / "unanswered.jsonl"))
# Characters of each retrieved chunk passed to the model (keeps 5 hits well inside an 8k context)
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", 2000))
