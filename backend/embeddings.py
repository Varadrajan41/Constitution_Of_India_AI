"""Embedding backend: BAAI/bge-base-en-v1.5 via sentence-transformers.

The same model embeds documents (at ingest) and queries (at retrieval), which is
required for vectors to be comparable. BGE v1.5 retrieval improves when the query
carries an instruction prefix, so:

- documents  -> embedded plain (passage mode)
- queries    -> embedded with config.BGE_QUERY_INSTRUCTION prepended

All vectors are L2-normalized and collections use cosine space.

Heavy imports (torch / sentence_transformers) are deferred so that modules which
only need chunking (compare, dump, extract) keep working without them installed.
"""
from __future__ import annotations

from backend import config

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        device = config.resolve_device()
        print(f"[embeddings] loading {config.EMBEDDING_MODEL} on {device}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    return _model


def embed_documents(texts) -> list[list[float]]:
    model = _get_model()
    vecs = model.encode(
        list(texts),
        normalize_embeddings=True,
        batch_size=config.EMBED_BATCH,
        show_progress_bar=False,
    )
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    model = _get_model()
    vec = model.encode(
        [config.BGE_QUERY_INSTRUCTION + text],
        normalize_embeddings=True,
    )
    return vec.tolist()[0]


class BGEEmbeddingFunction:
    """ChromaDB-compatible embedding function (passage/document mode).

    Queries should be embedded via `embed_query` and passed as
    `query_embeddings`, so this function is only invoked for documents.
    """

    def __call__(self, input):  # noqa: A002 - Chroma requires the name `input`
        return embed_documents(input)

    @staticmethod
    def name() -> str:
        return "bge-base-en-v1.5"
