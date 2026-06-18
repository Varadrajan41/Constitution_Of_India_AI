"""Article citation parsing and grounding checks shared by critic + eval."""
from __future__ import annotations

import re

_CITE_RE = re.compile(
    r"[Aa]rticles?\s+((?:\d{1,3}[A-Z]{0,2})(?:\s*(?:,|and|&|to|/)\s*\d{1,3}[A-Z]{0,2})*)"
)

_ARTICLE_INDEX: list[tuple[str, int, int]] | None = None


def build_article_index():
    """Map character ranges -> Article number using structured article spans."""
    from backend import chunkers
    from backend.extract import extract

    spans = []
    for c in chunkers.structured_chunks(extract()):
        if c.metadata.get("type") == "article" and c.metadata.get("article"):
            spans.append((c.metadata["article"], c.start, c.end))
    return spans


def get_article_index():
    global _ARTICLE_INDEX
    if _ARTICLE_INDEX is None:
        _ARTICLE_INDEX = build_article_index()
    return _ARTICLE_INDEX


def articles_in_span(span, index) -> set[str]:
    cs, ce = span
    if cs is None or ce is None:
        return set()
    return {num for num, s, e in index if min(ce, e) - max(cs, s) > 0}


def cited_articles(text: str) -> set[str]:
    out = set()
    for m in _CITE_RE.finditer(text or ""):
        out.update(re.findall(r"\d{1,3}[A-Z]{0,2}", m.group(1)))
    return out


def grounded_articles(passages, index=None) -> set[str]:
    """Articles supported by retrieved passages (metadata span + in-text refs)."""
    index = index or get_article_index()
    arts = set()
    for doc, meta, _ in passages:
        arts |= articles_in_span((meta.get("char_start"), meta.get("char_end")), index)
        arts |= cited_articles(doc)
    return arts


def ungrounded_citations(text: str, passages, index=None) -> list[str]:
    """Article numbers cited in text but not supported by retrieved passages."""
    index = index or get_article_index()
    bad = cited_articles(text) - grounded_articles(passages, index)
    return sorted(bad)
