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

# Turns a follow-up into a standalone query using recent conversation history.
REWRITE_TEMPLATE = """Given the conversation below and a follow-up question, \
rewrite the follow-up as a standalone question that can be understood without \
the conversation. Resolve pronouns and implicit references (e.g. "it", "that \
article") to what they refer to. Keep it concise. Output ONLY the rewritten \
question, with no preamble.

Conversation:
{history}

Follow-up: {question}
Standalone question:"""


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


def _render_history(history) -> str:
    turns = []
    for turn in (history or [])[-config.CHAT_HISTORY_TURNS:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            turns.append(f"{role.capitalize()}: {content}")
    return "\n".join(turns)


def rewrite_query(question: str, history=None) -> str:
    """Rewrite a follow-up into a standalone retrieval query using history.

    Returns the original question unchanged when rewriting is disabled, there is
    no history, or the rewrite call fails for any reason (never breaks retrieval).
    """
    history_text = _render_history(history)
    if not config.QUERY_REWRITE or not history_text:
        return question
    import ollama

    try:
        client = ollama.Client(host=config.OLLAMA_HOST)
        resp = client.chat(
            model=config.OLLAMA_MODEL,
            messages=[{
                "role": "user",
                "content": REWRITE_TEMPLATE.format(
                    history=history_text, question=question
                ),
            }],
            options={"temperature": 0.0},
        )
        rewritten = (resp["message"]["content"] or "").strip()
        return rewritten or question
    except Exception:
        return question


def _model_error(exc) -> RuntimeError | None:
    """Map an Ollama 'model not found' error to a friendly RuntimeError."""
    if "not found" in str(exc).lower():
        return RuntimeError(
            f"Ollama model '{config.OLLAMA_MODEL}' is not installed.\n"
            f"  - Register SaulLM:   bash scripts/setup_saul.sh\n"
            f"  - Or use an existing model:  export OLLAMA_MODEL=<name>  "
            f"(see `ollama list`)"
        )
    return None


def _build_messages(question: str, context: str, history=None):
    """System (context) + recent chat turns + the current question."""
    messages = [{"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)}]
    for turn in (history or [])[-config.CHAT_HISTORY_TURNS:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def _normalize_query(text: str) -> str:
    """Collapse whitespace and strip stray quotes for rewrite comparison."""
    return " ".join((text or "").strip().strip('"').strip("'").split())


def query_was_rewritten(original: str, search_query: str) -> bool:
    """True when retrieval used a different query than the user's message."""
    return _normalize_query(original) != _normalize_query(search_query)


def _prepare(question, hybrid, strategy, history):
    """Rewrite -> retrieve -> build prompt. Shared by answer/answer_stream
    (and, later, the critic). Returns (messages, passages, search_query)."""
    search_query = rewrite_query(question, history)
    passages = get_passages(search_query, hybrid=hybrid, strategy=strategy)
    messages = _build_messages(question, build_context(passages), history)
    return messages, passages, search_query


def answer(question: str, hybrid: bool = None, strategy: str = None, history=None):
    """Return (answer_text, passages, search_query).

    search_query is the query sent to retrieval (after optional rewrite).
    """
    import ollama

    messages, passages, search_query = _prepare(question, hybrid, strategy, history)
    client = ollama.Client(host=config.OLLAMA_HOST)
    try:
        response = client.chat(model=config.OLLAMA_MODEL, messages=messages)
    except ollama.ResponseError as exc:
        raise (_model_error(exc) or exc) from None
    return response["message"]["content"], passages, search_query


def answer_stream(question: str, hybrid: bool = None, strategy: str = None,
                  history=None):
    """Streaming variant. Returns (token_generator, passages, search_query).

    Retrieval/rewrite happen eagerly (so passages are known up front); only the
    LLM generation is streamed. Structured so a critic can later consume the
    non-streaming `answer()` for a draft and stream only the approved result.
    """
    messages, passages, search_query = _prepare(question, hybrid, strategy, history)

    def _tokens():
        import ollama

        client = ollama.Client(host=config.OLLAMA_HOST)
        try:
            for chunk in client.chat(
                model=config.OLLAMA_MODEL, messages=messages, stream=True
            ):
                yield chunk["message"]["content"]
        except ollama.ResponseError as exc:
            err = _model_error(exc)
            if err is None:
                raise
            yield f"\n\n**Setup needed:** {err}"

    return _tokens(), passages, search_query


def _print_rewritten(original: str, search_query: str):
    if query_was_rewritten(original, search_query):
        print(f"  rewritten: {search_query}")


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
    history = []
    while True:
        try:
            question = input("\n[USER]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break
        if not question:
            continue
        try:
            text, passages, search_query = answer(
                question, hybrid=hybrid, strategy=args.strategy, history=history
            )
        except RuntimeError as exc:
            print(f"\n[setup needed]\n{exc}")
            break
        _print_rewritten(question, search_query)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": text})
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
