"""Build ChromaDB collections from the Constitution PDF.

Runs one or both chunking strategies into separate collections so they can be
compared and queried independently:

    python -m backend.ingest                # both strategies (default)
    python -m backend.ingest --mode naive
    python -m backend.ingest --mode structured
    python -m backend.ingest --force        # re-extract text from the PDF

Metadata values are coerced to Chroma-safe scalars (str/int/float/bool).
"""
import argparse

import chromadb

from backend import config, chunkers
from backend.embeddings import BGEEmbeddingFunction
from backend.extract import extract


def _safe_meta(meta: dict) -> dict:
    out = {}
    for k, v in meta.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def ingest_strategy(strategy: str, text: str, client, ef) -> int:
    collection_name = config.collection_for(strategy)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    chunks = chunkers.build(strategy, text)
    print(f"[{strategy}] {len(chunks)} chunks -> '{collection_name}'")

    BATCH = 200
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        collection.add(
            documents=[c.text for c in batch],
            ids=[f"{strategy}_{i + j}" for j in range(len(batch))],
            metadatas=[
                _safe_meta(dict(c.metadata, char_start=c.start, char_end=c.end))
                for c in batch
            ],
        )
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest the Constitution into ChromaDB.")
    parser.add_argument(
        "--mode",
        choices=["naive", "structured", "both"],
        default="both",
        help="Which chunking strategy/strategies to build (default: both).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-extract text from the PDF (ignore cache)."
    )
    args = parser.parse_args()

    text = extract(force=args.force)

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    ef = BGEEmbeddingFunction()

    strategies = ["naive", "structured"] if args.mode == "both" else [args.mode]
    print(f"--- Ingesting strategies: {', '.join(strategies)} ---")
    for strategy in strategies:
        ingest_strategy(strategy, text, client, ef)
    print(f"--- Done. Vector store at {config.CHROMA_DIR} ---")
    print("    Compare them with: python -m backend.compare")


if __name__ == "__main__":
    main()
