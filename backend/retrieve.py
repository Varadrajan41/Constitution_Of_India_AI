"""Retrieval layer: hybrid (both collections) + cross-encoder rerank.

Pipeline:
  1. embed the query once (bge-base, with query instruction)
  2. pull RETRIEVE_K candidates from EACH collection (naive + structured)
  3. merge; drop exact duplicates
  4. rerank the union with the cross-encoder
  5. select TOP_N, skipping passages that overlap an already-picked span
     (naive and structured cover the same text, so this removes near-dupes)

`single_retrieve` keeps the simple one-collection path for comparison.
"""
from __future__ import annotations

from backend import config

_chroma_client = None


def _client():
    """Reuse one PersistentClient — Chroma's shared system is not thread-safe."""
    global _chroma_client
    if _chroma_client is None:
        import chromadb

        _chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _chroma_client


def _collection(client, strategy: str):
    from backend.embeddings import BGEEmbeddingFunction

    name = config.collection_for(strategy)
    return client.get_collection(name=name, embedding_function=BGEEmbeddingFunction())


def single_retrieve(question: str, strategy: str = None, n_results: int = None):
    """Top-N from one collection (no rerank). Returns (docs, metas)."""
    from backend.embeddings import embed_query

    client = _client()
    col = _collection(client, strategy or config.DEFAULT_STRATEGY)
    res = col.query(
        query_embeddings=[embed_query(question)],
        n_results=n_results or config.N_RESULTS,
        include=["documents", "metadatas", "distances"],
    )
    return res["documents"][0], res["metadatas"][0]


def _spans_overlap(a, b, frac: float = 0.6) -> bool:
    a0, a1 = a
    b0, b1 = b
    if a0 is None or b0 is None:
        return False
    inter = max(0, min(a1, b1) - max(a0, b0))
    shorter = max(1, min(a1 - a0, b1 - b0))
    return inter / shorter >= frac


def hybrid_candidates(question: str, k: int = None):
    """Union of candidates from both collections, deduped by exact text."""
    from backend.embeddings import embed_query

    qvec = embed_query(question)
    client = _client()
    seen, cands = set(), []
    for strategy in ("naive", "structured"):
        try:
            col = _collection(client, strategy)
        except Exception:
            continue
        res = col.query(
            query_embeddings=[qvec],
            n_results=k or config.RETRIEVE_K,
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            key = doc.strip()
            if key in seen:
                continue
            seen.add(key)
            m = dict(meta)
            m["strategy"] = strategy
            cands.append((doc, m))
    return cands


def _select_top(ranked, top_n: int = None):
    """Apply rerank score floor + span dedupe."""
    top_n = top_n or config.TOP_N
    top_score = ranked[0][2] if ranked else 0.0
    floor = max(config.RERANK_MIN_SCORE, config.RERANK_KEEP_RATIO * top_score)

    selected = []
    for doc, meta, score in ranked:
        if config.RERANK_ENABLED and selected and score < floor:
            continue
        span = (meta.get("char_start"), meta.get("char_end"))
        if any(_spans_overlap(span, (m.get("char_start"), m.get("char_end")))
               for _, m, _ in selected):
            continue
        selected.append((doc, meta, score))
        if len(selected) >= top_n:
            break
    return selected


def retrieve(question: str, top_n: int = None):
    """Full hybrid + rerank retrieval. Returns list of (doc, meta, score)."""
    from backend.rerank import rerank

    cands = hybrid_candidates(question)
    if not cands:
        return []

    if config.RERANK_ENABLED:
        ranked = rerank(question, cands)
    else:
        ranked = [(d, m, 0.0) for d, m in cands]

    return _select_top(ranked, top_n)
