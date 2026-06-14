"""Cross-encoder reranking with BAAI/bge-reranker-v2-m3.

A bi-encoder (the embedding model) is fast but coarse; the cross-encoder reads
the (query, passage) pair jointly and scores true relevance much better. We use
it to re-order the union of hybrid candidates and keep the best few.

The model is loaded lazily and cached as a module-level singleton.
"""
from __future__ import annotations

from backend import config

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        device = config.resolve_device()
        print(f"[rerank] loading {config.RERANKER_MODEL} on {device}")
        _reranker = CrossEncoder(config.RERANKER_MODEL, max_length=512, device=device)
    return _reranker


def rerank(query: str, candidates: list[tuple[str, dict]]):
    """Score and sort candidates.

    candidates: list of (document_text, metadata).
    Returns: list of (document_text, metadata, score) sorted high->low.
    """
    if not candidates:
        return []
    model = _get_reranker()
    pairs = [[query, doc] for doc, _ in candidates]
    scores = model.predict(pairs)
    ranked = sorted(
        zip(candidates, scores), key=lambda x: float(x[1]), reverse=True
    )
    return [(doc, meta, float(score)) for ((doc, meta), score) in ranked]
