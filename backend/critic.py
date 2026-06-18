"""Post-generation critic: drop or rewrite answers with ungrounded Article citations."""
from __future__ import annotations

from backend import config
from backend.citations import grounded_articles, ungrounded_citations

CRITIC_REWRITE_TEMPLATE = """You are revising an answer about the Constitution of India.

The draft cites Article number(s) that do NOT appear in the retrieved context:
{ungrounded}

Article numbers that ARE supported by the retrieved context:
{grounded}

Rules for the rewrite:
1. Remove or rephrase any statement tied to the ungrounded Article number(s).
2. Keep correct citations that match the retrieved context.
3. You MAY keep a "General knowledge (not in retrieved text - verify):" section,
   but it must NOT cite specific Article numbers unless they are in the supported
   list above.
4. Do not invent new Article numbers.

Question: {question}

Draft answer:
{draft}

Rewritten answer:"""

FALLBACK_TEMPLATE = """Based on the retrieved constitutional text, I can only answer \
from the passages provided. The following is supported by the retrieved context:

{grounded_part}

General knowledge (not in retrieved text - verify): Additional details could not \
be verified against the retrieved passages alone. Please consult the full \
Constitution or authoritative commentary."""


def apply_critic(client, draft: str, passages, question: str) -> tuple[str, dict]:
    """Return (final_answer, meta). Meta includes rewrites and ungrounded removed."""
    meta = {"enabled": config.CRITIC_ENABLED, "rewrites": 0, "ungrounded": []}
    if not config.CRITIC_ENABLED or not (draft or "").strip():
        return draft, meta

    text = draft
    index = None
    for attempt in range(config.CRITIC_MAX_REWRITES + 1):
        bad = ungrounded_citations(text, passages, index)
        if not bad:
            meta["ungrounded"] = []
            return text, meta

        if not meta.get("fixed_ungrounded"):
            meta["fixed_ungrounded"] = bad
        meta["ungrounded"] = bad
        if attempt >= config.CRITIC_MAX_REWRITES:
            break

        grounded = sorted(grounded_articles(passages, index))
        try:
            resp = client.chat(
                model=config.OLLAMA_MODEL,
                messages=[{
                    "role": "user",
                    "content": CRITIC_REWRITE_TEMPLATE.format(
                        ungrounded=", ".join(bad),
                        grounded=", ".join(grounded) if grounded else "(none)",
                        question=question,
                        draft=text,
                    ),
                }],
                options={"temperature": 0.0},
            )
            revised = (resp["message"]["content"] or "").strip()
            if revised:
                text = revised
                meta["rewrites"] += 1
        except Exception:
            break

    # Still ungrounded after rewrites — return a safe fallback grounded in context.
    bad = ungrounded_citations(text, passages, index)
    if bad:
        meta["ungrounded"] = bad
        meta["fallback"] = True
        grounded_part = _grounded_excerpt(passages)
        if grounded_part:
            return FALLBACK_TEMPLATE.format(grounded_part=grounded_part), meta
        return (
            "General knowledge (not in retrieved text - verify): "
            "The retrieved passages do not fully support a cited answer to this "
            "question. Please verify against the official constitutional text."
        ), meta

    return text, meta


def _grounded_excerpt(passages, max_chars: int = 1200) -> str:
    parts = []
    n = 0
    for doc, meta, score in passages:
        art = meta.get("article")
        tag = f"Article {art}" if art else "Excerpt"
        block = f"**{tag}** (retrieval score {score:.2f}):\n{doc.strip()}"
        if n + len(block) > max_chars:
            break
        parts.append(block)
        n += len(block)
    return "\n\n".join(parts)
