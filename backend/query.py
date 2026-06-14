"""Answer questions over the Constitution: hybrid retrieval + rerank + Ollama.

Default path: hybrid (both collections) -> cross-encoder rerank -> graded prompt
that prefers the retrieved context but may add clearly-labeled general knowledge
when the context is insufficient (the LLM must never fabricate citations).

    python -m backend.query                          # hybrid + rerank
    python -m backend.query --no-hybrid --strategy naive
    python -m backend.query --no-rerank
    python -m backend.query --show                   # print retrieved passages

Exposes `answer(question, ...)` for the UI.
"""
import argparse

from backend import config, retrieve

# Graded prompt: context-first, with a *labeled* fallback to parametric knowledge.
SYSTEM_TEMPLATE = """You are a careful legal assistant for the Constitution of India.

Use the CONTEXT below (verbatim excerpts from the Constitution) as your primary
and authoritative source. Follow these rules strictly:

1. Prefer the CONTEXT. For every statement drawn from it, cite the exact Article
   number and its Part or Schedule (e.g. "Article 21, Part III").
2. Quote article numbers ONLY as they appear in the CONTEXT. Never invent or
   guess an Article/Part/Schedule number.
3. If the CONTEXT does not fully answer the question, you MAY add information
   from your own legal knowledge, but you MUST prefix that portion with:
   "General knowledge (not in retrieved text - verify):"
4. If you are unsure, say so explicitly rather than guessing.

CONTEXT:
{context}
"""


def build_context(passages) -> str:
    """passages: list of (doc, meta, score)."""
    parts = []
    for doc, meta, _ in passages:
        article = meta.get("article") or ""
        loc = meta.get("schedule") or meta.get("part") or ""
        tag = " ".join(t for t in [f"Article {article}" if article else "", loc] if t)
        parts.append(f"[{tag or 'Excerpt'}]\n{doc}")
    return "\n\n".join(parts)


def get_passages(question: str, hybrid: bool = None, strategy: str = None):
    """Return passages as list of (doc, meta, score)."""
    use_hybrid = config.HYBRID if hybrid is None else hybrid
    if use_hybrid:
        return retrieve.retrieve(question)
    docs, metas = retrieve.single_retrieve(question, strategy)
    return [(d, m, 0.0) for d, m in zip(docs, metas)]


def answer(question: str, hybrid: bool = None, strategy: str = None):
    """Return (answer_text, passages)."""
    import ollama

    passages = get_passages(question, hybrid=hybrid, strategy=strategy)
    context = build_context(passages)
    client = ollama.Client(host=config.OLLAMA_HOST)
    try:
        response = client.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)},
                {"role": "user", "content": question},
            ],
        )
    except ollama.ResponseError as exc:
        if "not found" in str(exc).lower():
            raise RuntimeError(
                f"Ollama model '{config.OLLAMA_MODEL}' is not installed.\n"
                f"  - Register SaulLM:   bash scripts/setup_saul.sh\n"
                f"  - Or use an existing model:  export OLLAMA_MODEL=<name>  "
                f"(see `ollama list`)"
            ) from None
        raise
    return response["message"]["content"], passages


def _sources_line(passages):
    arts = sorted(
        {m.get("article") for _, m, _ in passages
         if m.get("article") and m.get("type") == "article"}
    )
    return f"  sources: Articles {', '.join(arts)}" if arts else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hybrid", action="store_true", help="Use one collection.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking.")
    parser.add_argument("--strategy", choices=["naive", "structured"],
                        default=config.DEFAULT_STRATEGY)
    parser.add_argument("--show", action="store_true", help="Print retrieved passages.")
    args = parser.parse_args()

    if args.no_rerank:
        config.RERANK_ENABLED = False
    hybrid = not args.no_hybrid

    mode = "hybrid+rerank" if (hybrid and config.RERANK_ENABLED) else (
        "hybrid" if hybrid else args.strategy)
    print(f"Constitution RAG [{mode}] - ask a question (Ctrl+C to exit)")
    while True:
        try:
            question = input("\n[USER]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break
        if not question:
            continue
        try:
            text, passages = answer(question, hybrid=hybrid, strategy=args.strategy)
        except RuntimeError as exc:
            print(f"\n[setup needed]\n{exc}")
            break
        if args.show:
            print("\n--- retrieved ---")
            for doc, meta, score in passages:
                tag = meta.get("article") or meta.get("schedule") or meta.get("type")
                print(f"  [{meta.get('strategy','?')}|{tag}|score={score:.3f}] {doc[:90].strip()}...")
        print(f"\n[ASSISTANT]: {text}")
        line = _sources_line(passages)
        if line:
            print(f"\n{line}")


if __name__ == "__main__":
    main()
