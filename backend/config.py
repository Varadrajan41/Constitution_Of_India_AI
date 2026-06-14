"""Central runtime configuration for the Constitution RAG lab.

All values can be overridden via environment variables (see .env.example) so the
same code runs unchanged on different machines.
"""
import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("CONSTITUTION_DATA_DIR", BASE_DIR / "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DB_DIR", BASE_DIR / "chroma_db"))

# Source document to ingest (.pdf or .txt). Drop the file into data/.
SOURCE_FILE = Path(
    os.getenv("CONSTITUTION_SOURCE", DATA_DIR / "Constitution of India.pdf")
)

# Cached plain-text extraction (built once from the PDF, reused by both chunkers).
TEXT_CACHE = Path(os.getenv("CONSTITUTION_TEXT_CACHE", DATA_DIR / "constitution.txt"))

# --- Vector store ----------------------------------------------------------
# Two collections so the two chunking strategies can be compared side by side.
COLLECTION_NAIVE = os.getenv("CHROMA_COLLECTION_NAIVE", "constitution_naive")
COLLECTION_STRUCTURED = os.getenv(
    "CHROMA_COLLECTION_STRUCTURED", "constitution_structured"
)

# --- Chunking --------------------------------------------------------------
# Naive (char-window) strategy.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Structured strategy: Articles longer than this are sub-split into windows.
MAX_ARTICLE_CHARS = int(os.getenv("MAX_ARTICLE_CHARS", "2500"))

# --- Embeddings ------------------------------------------------------------
# Same model embeds documents (ingest) and queries (retrieval). BGE v1.5 gains
# from a query-side instruction prefix; documents are embedded plain.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "auto")  # auto | cuda | cpu
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))
BGE_QUERY_INSTRUCTION = os.getenv(
    "BGE_QUERY_INSTRUCTION",
    "Represent this sentence for searching relevant passages: ",
)

# --- Reranker (cross-encoder) ---------------------------------------------
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") not in ("0", "false", "False")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
# Passages scoring below this absolute rerank score are dropped from the context
# (the top passage is always kept). bge-reranker-v2-m3 scores are ~0..1.
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.1"))
# Also drop passages weaker than this fraction of the top passage's score.
RERANK_KEEP_RATIO = float(os.getenv("RERANK_KEEP_RATIO", "0.3"))

# --- Retrieval / LLM -------------------------------------------------------
# Which collection a single-strategy query uses: "naive" or "structured".
DEFAULT_STRATEGY = os.getenv("RAG_STRATEGY", "structured")
# Hybrid pulls from BOTH collections, then the reranker picks the best.
HYBRID = os.getenv("RAG_HYBRID", "1") not in ("0", "false", "False")
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "20"))   # candidates per collection
TOP_N = int(os.getenv("TOP_N", "6"))              # passages kept after rerank
N_RESULTS = int(os.getenv("N_RESULTS", "4"))      # used by single-strategy path

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "saul-7b-instruct")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")


def collection_for(strategy: str) -> str:
    strategy = (strategy or DEFAULT_STRATEGY).lower()
    if strategy == "naive":
        return COLLECTION_NAIVE
    if strategy == "structured":
        return COLLECTION_STRUCTURED
    raise ValueError(f"Unknown strategy '{strategy}'. Use 'naive' or 'structured'.")


def resolve_device(preferred: str = None) -> str:
    """Resolve 'auto' to cuda/cpu without importing torch at module load."""
    dev = (preferred or EMBED_DEVICE or "auto").lower()
    if dev != "auto":
        return dev
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
